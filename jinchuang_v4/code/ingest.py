"""
影像入库脚本：扫描图片目录 → 预处理 → 分类 → 提取特征 → 写入 FAISS 索引
支持 Flat 和 IVF 两种索引类型。

用法:
    python ingest.py --data_dir ./data --annotations ./data/annotations.csv  # 带标注入库
    python ingest.py --data_dir ./data                                       # 自动检测标注
    python ingest.py --data_dir ./test_images --force                        # 强制重建
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import argparse
import csv
import sys
import yaml
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# 将项目根目录加入路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.model import SigLIP2FeatureExtractor
from src.retrieval import SimilaritySearch
from src.classifier import ImageClassifier
from src.preprocessing import PreprocessingPipeline

# 图片类型 → 分类类别映射
IMAGE_TYPE_CATEGORY = {
    "face_signing": "面签照片",
    "id_card_front": "身份证",
    "id_card_back": "身份证",
    "contract": "合同",
    "bank_statement": "银行流水",
}


def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_annotations(annotation_path: str):
    """加载 annotations.csv 返回 {file_path: {loan_id, image_type, business_type, similar_group, is_similar_pair}}"""
    if not annotation_path or not os.path.exists(annotation_path):
        return None

    annotations = {}
    with open(annotation_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_path = row["file_path"].replace("\\", "/")
            image_type = row.get("image_type") or os.path.splitext(os.path.basename(file_path))[0]
            annotations[file_path] = {
                "loan_id": row.get("loan_id", ""),
                "image_type": image_type,
                "business_type": row.get("business_type", ""),
                "similar_group": row.get("similar_group", ""),
                "is_similar_pair": row.get("is_similar_pair", "0") == "1",
            }
    print(f"已加载标注信息: {len(annotations)} 条")
    return annotations


def get_image_files(data_dir: str, extensions: list[str]):
    """递归扫描目录下的所有图片文件"""
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"错误: 目录不存在 - {data_dir}")
        return []

    files = []
    for ext in extensions:
        files.extend(data_path.rglob(f"*{ext}"))
        files.extend(data_path.rglob(f"*{ext.upper()}"))

    seen = set()
    unique = []
    for f in files:
        if f not in seen:
            seen.add(f)
            unique.append(f)

    return sorted(unique)


def get_rel_path(file_path: Path, data_dir: str):
    """计算相对于 data_dir 的路径（统一用正斜杠）"""
    abs_data = str(Path(data_dir).resolve())
    abs_file = str(file_path.resolve())
    return abs_file.replace(abs_data, "").lstrip("\\/").replace("\\", "/")


def preprocess_image(img_path: str, image_size: int = 224, preprocessor=None, extractor=None):
    """加载并预处理单张图片（集成预处理链）"""
    try:
        img = Image.open(img_path).convert("RGB")
        if preprocessor:
            img = preprocessor(img)
        if extractor is not None:
            img_tensor = extractor.preprocess(img)
        else:
            img = img.resize((image_size, image_size))
            img_tensor = torch.tensor(np.array(img).transpose(2, 0, 1)).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0)
        return img, img_tensor
    except Exception as e:
        print(f"  读取失败 {img_path}: {e}")
        return None, None


def get_category_from_type(image_type: str):
    """根据 image_type 返回分类名称"""
    return IMAGE_TYPE_CATEGORY.get(image_type, "其他")


def ingest(config: dict, data_dir: str, force_rebuild: bool = False,
           index_type: str = None, annotations_file: str = None,
           sign_only: bool = False):
    """批量入库主函数"""
    # 索引类型
    if index_type is None:
        index_type = config["retrieval"].get("index_type", "flat")
    nlist = config["retrieval"].get("nlist", 100)

    extractor = SigLIP2FeatureExtractor(model_name=config["model"]["name"])  # 索引始终使用纯 SigLIP2 特征

    # 加载标注
    annotations = load_annotations(annotations_file)

    # 初始化分类器（当没有标注时使用零样本分类）
    classifier = ImageClassifier(
        model=extractor.model,
        processor=extractor.processor,
        categories=config["classifier"]["categories"],
        device=extractor.device,
    )

    # 初始化预处理链
    preprocessor = PreprocessingPipeline(config.get("preprocessing", {}))
    print(f"预处理链: {preprocessor.describe()}")

    # 初始化检索器
    searcher = SimilaritySearch(
        embedding_dim=config["model"]["embedding_dim"],
        index_type=index_type,
        nlist=nlist,
    )

    # 加载已有索引
    index_loaded = searcher.load()
    if force_rebuild:
        print(f"强制重建索引 ({index_type})，清空旧数据...")
        searcher = SimilaritySearch(
            embedding_dim=config["model"]["embedding_dim"],
            index_type=index_type,
            nlist=nlist,
        )
    elif index_loaded:
        print(f"已加载现有索引，共 {searcher.index.ntotal} 条记录，将追加新数据")
    else:
        print(f"未找到现有索引，将创建新索引 ({index_type})")

    # 扫描图片
    extensions = config["data"]["image_extensions"]
    image_files = get_image_files(data_dir, extensions)
    if not image_files:
        print(f"在 {data_dir} 下未找到任何图片（支持的格式: {extensions}）")
        return

    print(f"找到 {len(image_files)} 张图片，开始处理...")

    if sign_only and annotations:
        before_filter = len(image_files)
        image_files = [
            img_path for img_path in image_files
            if annotations.get(get_rel_path(img_path, data_dir), {}).get("image_type") == "face_signing"
        ]
        print(f"sign_only: {before_filter} -> {len(image_files)} face_signing images")

    total = len(image_files)
    success = 0
    skipped = 0
    category_counts = {}

    batch_embs = []
    batch_metadata = []

    # 第一次遍历：提取所有特征（IVF 需要先训练）
    if index_type == "ivf" and not (index_loaded and not force_rebuild):
        print("IVF 索引需要先训练聚类中心...")
        all_embs = []
        for img_path in tqdm(image_files, desc="提取训练样本"):
            img, img_tensor = preprocess_image(str(img_path), config["data"]["image_size"], preprocessor, extractor)
            if img_tensor is None:
                continue
            with torch.no_grad():
                feat = extractor.extract(img_tensor)
            feat_np = feat.cpu().numpy().astype(np.float32)
            feat_norm = np.linalg.norm(feat_np)
            if feat_norm > 0:
                feat_np = feat_np / feat_norm
            all_embs.append(feat_np[0])

        if all_embs:
            train_embs = np.stack(all_embs, axis=0)
            searcher.train(train_embs)
        print("IVF 训练完成")

    # 遍历入库（Flat 或 IVF 训练后第二轮）
    for img_path in tqdm(image_files, desc="处理中"):
        img, img_tensor = preprocess_image(str(img_path), config["data"]["image_size"], preprocessor, extractor)
        if img_tensor is None:
            skipped += 1
            continue

        # 获取图片元数据
        rel_path = get_rel_path(img_path, data_dir)
        annot = annotations.get(rel_path) if annotations else None

        if annot:
            # 使用标注信息
            cat_name = get_category_from_type(annot["image_type"])
            biz_id = annot["loan_id"]
            image_type = annot["image_type"]
            similar_group = annot.get("similar_group", "")
        else:
            # 回退到零样本分类
            cat_id, cat_name, scores = classifier.classify(img_tensor)
            biz_id = f"BIZ{success:04d}"
            image_type = "unknown"
            similar_group = ""

        category_counts[cat_name] = category_counts.get(cat_name, 0) + 1

        with torch.no_grad():
            feat = extractor.extract(img_tensor)
        feat_np = feat.cpu().numpy().astype(np.float32)
        feat_norm = np.linalg.norm(feat_np)
        if feat_norm > 0:
            feat_np = feat_np / feat_norm

        metadata = {
            "path": str(img_path),
            "rel_path": rel_path,
            "image_type": image_type,
            "cat_name": cat_name,
            "biz_id": biz_id,
            "loan_id": biz_id,
            "business_type": annot.get("business_type", "") if annot else "",
            "similar_group": similar_group,
        }
        batch_embs.append(feat_np[0])
        batch_metadata.append(metadata)
        success += 1

    # 批量写入索引
    if batch_embs:
        embeddings = np.stack(batch_embs, axis=0)
        searcher.add_embeddings(embeddings, batch_metadata)
        searcher.save()

    # 打印统计
    print(f"\n{'='*50}")
    print(f"入库完成！统计:")
    print(f"  索引类型: {index_type}")
    print(f"  总计扫描: {total} 张")
    print(f"  成功入库: {success} 张")
    print(f"  跳过(读取出错): {skipped} 张")
    print(f"  索引总记录: {searcher.index.ntotal} 条")
    print(f"\n各类别分布:")
    for cat_name, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat_name}: {count} 张")

    # 统计相似组信息
    if annotations:
        similar_groups = set()
        for a in annotations.values():
            if a.get("similar_group"):
                similar_groups.add(a["similar_group"])
        if similar_groups:
            print(f"\n相似组: {len(similar_groups)} 个（涉及 "
                  f"{sum(1 for a in annotations.values() if a['is_similar_pair'])} 张图片）")


def main():
    parser = argparse.ArgumentParser(description="金融影像批量入库工具")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="图片目录路径")
    parser.add_argument("--annotations", type=str, default=None,
                        help="annotations.csv 路径（自动检测 data/annotations.csv）")
    parser.add_argument("--force", action="store_true",
                        help="强制重建索引（清空旧数据）")
    parser.add_argument("--index_type", type=str, default=None,
                        choices=["flat", "ivf"],
                        help="索引类型（默认: config.yaml 中的配置）")
    parser.add_argument("--sign_only", action="store_true",
                        help="Only index face_signing images listed in annotations.csv")
    args = parser.parse_args()

    config = load_config()

    # 自动检测 annotations.csv
    if args.annotations is None:
        default_annot = os.path.join(args.data_dir, "annotations.csv")
        if os.path.exists(default_annot):
            args.annotations = default_annot

    ingest(config, args.data_dir, force_rebuild=args.force,
           index_type=args.index_type, annotations_file=args.annotations,
           sign_only=args.sign_only)


if __name__ == "__main__":
    main()
