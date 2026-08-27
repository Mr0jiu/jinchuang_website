"""
批量评估模块：分类准确率 + 相似度阈值分析（精确率/召回率/PR曲线）

功能：
    1. 评估零样本分类准确率
    2. 评估相似度检索在不同阈值下的精确率/召回率
    3. 自动选择最优阈值（F1-max）
    4. 输出评估报告 Markdown + PR 曲线图

使用方法：
    # 准备测试数据（按目录组织）:
    test_eval/
        same/          # 同一人/重复提交的图片对（正样本）
            pair_001_a.jpg
            pair_001_b.jpg
            pair_002_a.jpg
            pair_002_b.jpg
            ...
        diff/          # 不同人/不同业务的图片对（负样本）
            unrelated_001_a.jpg
            unrelated_001_b.jpg
            ...

    # 运行评估:
    python src/evaluate.py
    python src/evaluate.py --test_dir ./test_eval --thresholds 0.5 0.6 0.7 0.8 0.9
"""
import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

# 将项目根目录加入路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch
import numpy as np
from PIL import Image
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")  # 非交互后端
import matplotlib.pyplot as plt

try:
    from sklearn.metrics import precision_recall_curve, f1_score, accuracy_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_imports():
    """延迟导入，避免模块初始化时耗时长"""
    from src.model import SigLIP2FeatureExtractor
    from src.retrieval import SimilaritySearch
    from src.classifier import ImageClassifier
    return SigLIP2FeatureExtractor, SimilaritySearch, ImageClassifier


def load_image(path: str, image_size: int = 224):
    """加载并预处理单张图片"""
    img = Image.open(path).convert("RGB")
    img = img.resize((image_size, image_size))
    img_tensor = torch.tensor(np.array(img).transpose(2, 0, 1)).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor


def scan_test_pairs(test_dir: str):
    """扫描评估数据集，构造图片对列表

    目录结构:
        test_dir/
            same/       # 同一来源的图片对（如同一人面签）- 期望高相似度
            diff/       # 不同来源的图片对 - 期望低相似度

    Returns:
        pairs: list of (img1_path, img2_path, label)
            label = 1 for same, 0 for diff
    """
    test_path = Path(test_dir)
    if not test_path.exists():
        print(f"错误: 测试目录不存在 - {test_dir}")
        return [], 0, 0

    pairs = []
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    def collect_pairs(subdir: str, label: int):
        dir_path = test_path / subdir
        if not dir_path.exists():
            return []
        # 收集所有图片
        files = sorted([
            f for f in dir_path.iterdir()
            if f.is_file() and f.suffix.lower() in extensions
        ])
        # 两两配对（假设文件按 pair_xxx_a.jpg / pair_xxx_b.jpg 组织）
        sub_pairs = []
        seen = set()
        for f in files:
            stem = f.stem  # e.g., pair_001_a
            # 尝试提取 pair key
            parts = stem.rsplit("_", 1)
            if len(parts) == 2 and parts[1] in ("a", "b", "A", "B"):
                base = parts[0]
                counterpart = None
                for c in ("a", "b", "A", "B"):
                    candidate = dir_path / f"{base}_{c}{f.suffix}"
                    if candidate != f and candidate.exists():
                        counterpart = candidate
                        break
                if counterpart is not None:
                    key = tuple(sorted([str(f), str(counterpart)]))
                    if key not in seen:
                        seen.add(key)
                        sub_pairs.append((str(f), str(counterpart), label))
                    continue
            # 如果无法配对，跨文件配对（确保数量配对）
        return sub_pairs

    pairs.extend(collect_pairs("same", 1))
    pairs.extend(collect_pairs("diff", 0))

    same_count = sum(1 for _, _, l in pairs if l == 1)
    diff_count = sum(1 for _, _, l in pairs if l == 0)

    return pairs, same_count, diff_count


def evaluate_retrieval(pairs, extractor, searcher, config):
    """评估相似度检索性能，遍历不同阈值

    Returns:
        all_scores: (y_true, y_scores) 每个图片对的真实标签和相似度分数
        results_by_threshold: dict {threshold: {tp, fp, tn, fn, precision, recall, f1}}
    """
    y_true = []
    y_scores = []

    print(f"正在评估 {len(pairs)} 对图片...")
    for img1_path, img2_path, label in pairs:
        try:
            img1 = load_image(img1_path, config["data"]["image_size"])
            img2 = load_image(img2_path, config["data"]["image_size"])

            with torch.no_grad():
                feat1 = extractor.extract(img1)
                feat2 = extractor.extract(img2)

            feat1_np = feat1.cpu().numpy().astype(np.float32)
            feat2_np = feat2.cpu().numpy().astype(np.float32)

            # L2 归一化
            for f in [feat1_np, feat2_np]:
                norm = np.linalg.norm(f)
                if norm > 0:
                    f /= norm

            # 余弦相似度
            score = float(np.dot(feat1_np[0], feat2_np[0].T))
            y_true.append(label)
            y_scores.append(score)

        except Exception as e:
            print(f"  处理失败 {img1_path} | {img2_path}: {e}")
            continue

    if not y_true:
        return [], {}

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    # 遍历阈值
    thresholds = config["evaluate"]["thresholds"]
    results = {}

    for thresh in thresholds:
        y_pred = (y_scores >= thresh).astype(int)

        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        tn = int(np.sum((y_pred == 0) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0

        results[thresh] = {
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "accuracy": round(accuracy, 4),
        }

    return (y_true, y_scores), results


def evaluate_classification(pairs, classifier, config):
    """评估零样本分类准确率

    使用 test_eval/same/ 下的图片（全量图片，不配对），
    检查分类结果与目录名称是否一致。
    """
    test_path = Path(config["evaluate"]["test_dir"])
    if not test_path.exists():
        return None

    print("正在评估分类准确率...")
    correct = 0
    total = 0
    cat_correct = defaultdict(int)
    cat_total = defaultdict(int)

    # same/ 和 diff/ 目录下的图片都用来评估分类
    for subdir in ["same", "diff"]:
        dir_path = test_path / subdir
        if not dir_path.exists():
            continue
        # 从文件名推断真实类别
        for img_file in dir_path.iterdir():
            if not img_file.is_file():
                continue
            if img_file.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue

            try:
                img_tensor = load_image(str(img_file), config["data"]["image_size"])
                _, cat_name, _ = classifier.classify(img_tensor)

                # 从文件名前缀推断类别
                fname = img_file.stem.lower()
                if fname.startswith("sign"):
                    true_cat = "面签照片"
                elif fname.startswith("id"):
                    true_cat = "身份证"
                elif fname.startswith("certif"):
                    true_cat = "权证"
                elif fname.startswith("contract"):
                    true_cat = "合同"
                elif fname.startswith("other"):
                    true_cat = "其他"
                else:
                    continue

                cat_total[true_cat] += 1
                total += 1
                if cat_name == true_cat:
                    cat_correct[true_cat] += 1
                    correct += 1

            except Exception as e:
                continue

    if total == 0:
        return None

    return {
        "accuracy": round(correct / total, 4),
        "correct": correct,
        "total": total,
        "per_category": {
            cat: {
                "correct": cat_correct[cat],
                "total": cat_total[cat],
                "acc": round(cat_correct[cat] / cat_total[cat], 4) if cat_total[cat] > 0 else 0,
            }
            for cat in sorted(cat_total.keys())
        },
    }


def plot_pr_curve(y_true, y_scores, optimal_thresh, output_path: str):
    """绘制精确率-召回率曲线"""
    if HAS_SKLEARN:
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    else:
        # 手动计算 PR 曲线
        sorted_indices = np.argsort(y_scores)[::-1]
        y_sorted = y_true[sorted_indices]
        precisions = []
        recalls = []
        for i in range(len(y_sorted)):
            tp = np.sum(y_sorted[: i + 1])
            fp = (i + 1) - tp
            fn = np.sum(y_sorted[i + 1:])
            precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 1.0)
            recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        thresholds = y_scores[sorted_indices]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # PR 曲线
    ax1.plot(recalls, precisions, color="darkorange", lw=2, label="PR curve")
    ax1.axvline(x=recalls[np.argmin(np.abs(thresholds - optimal_thresh))] if len(thresholds) > 0 else 0.5,
                color="red", ls="--", label=f"Optimal threshold = {optimal_thresh}")
    ax1.set_xlabel("Recall")
    ax1.set_ylabel("Precision")
    ax1.set_title("Precision-Recall Curve")
    ax1.set_xlim([0.0, 1.05])
    ax1.set_ylim([0.0, 1.05])
    ax1.grid(alpha=0.3)
    ax1.legend()

    # 相似度分布直方图
    y_pos = y_scores[y_true == 1]
    y_neg = y_scores[y_true == 0]
    ax2.hist(y_pos, bins=20, alpha=0.6, color="green", label="Positive (same person)")
    ax2.hist(y_neg, bins=20, alpha=0.6, color="red", label="Negative (different)")
    ax2.axvline(x=optimal_thresh, color="blue", ls="--", label=f"Threshold = {optimal_thresh}")
    ax2.set_xlabel("Cosine Similarity")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Similarity Distribution")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"PR 曲线已保存至: {output_path}")


def generate_report(eval_result, class_result, config):
    """生成评估报告 Markdown"""
    (y_true, y_scores), thresholds_result = eval_result
    timestr = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 找最优阈值（F1 max）
    best_thresh = None
    best_f1 = -1
    for thresh, metrics in thresholds_result.items():
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_thresh = thresh

    lines = []
    lines.append("# 模型评估报告\n")
    lines.append(f"**生成时间**: {timestr}\n")
    lines.append(f"**模型**: SigLIP2-Base-Patch16-224 (google/siglip2-base-patch16-224)")
    lines.append(f"**索引类型**: FAISS IndexFlatIP (余弦相似度)")
    lines.append(f"**测试样本**: 正样本(同一人) {int(np.sum(y_true))} 对, "
                 f"负样本(不同人) {int(len(y_true) - np.sum(y_true))} 对\n")

    lines.append("---\n")
    lines.append("## 1. 相似度阈值分析\n")
    lines.append("| 阈值 | 精确率 | 召回率 | F1值 | 准确率 | TP | FP | TN | FN |")
    lines.append("|------|--------|--------|------|--------|----|----|----|----|")

    for thresh in sorted(thresholds_result.keys()):
        m = thresholds_result[thresh]
        marker = " ⬅️ 最优" if thresh == best_thresh else ""
        lines.append(
            f"| {thresh} | {m['precision']} | {m['recall']} | "
            f"{m['f1']}{marker} | {m['accuracy']} | "
            f"{m['tp']} | {m['fp']} | {m['tn']} | {m['fn']} |"
        )

    lines.append(f"\n**最优阈值**: {best_thresh} (F1 = {best_f1})\n")

    lines.append("---\n")
    lines.append("## 2. 阈值选取建议\n")
    lines.append(f"- **保守策略（低误报）**: 推荐阈值 {min(best_thresh + 0.05, 0.99) if best_thresh else 0.85} 以上，精确率更高")
    lines.append(f"- **均衡策略（推荐）**: 阈值 {best_thresh}，F1 最优")
    lines.append(f"- **宽松策略（低漏报）**: 推荐阈值 {max(best_thresh - 0.05, 0.3) if best_thresh else 0.70} 以下，召回率更高")
    lines.append("")
    lines.append("具体选择取决于业务需求：对重复提交 zero-tolerance 则选保守策略，")
    lines.append("人力充足可复核则选宽松策略。\n")

    # 分类结果
    if class_result:
        lines.append("---\n")
        lines.append("## 3. 影像分类评估\n")
        lines.append(f"**总体分类准确率**: {class_result['accuracy']} ({class_result['correct']}/{class_result['total']})\n")
        lines.append("| 类别 | 正确数 | 总数 | 准确率 |")
        lines.append("|------|--------|------|--------|")
        for cat, info in class_result["per_category"].items():
            lines.append(f"| {cat} | {info['correct']} | {info['total']} | {info['acc']} |")

    lines.append("\n---\n")
    lines.append(f"*报告由 evaluate.py 自动生成 ({timestr})*")

    return "\n".join(lines), best_thresh


def run_evaluation(config=None, test_dir=None, output_dir=None, thresholds=None):
    """评估入口函数，供命令行和 main.py 调用"""
    if config is None:
        config = load_config()

    if test_dir:
        config["evaluate"]["test_dir"] = test_dir
    if output_dir:
        config["evaluate"]["output_dir"] = output_dir
    if thresholds:
        config["evaluate"]["thresholds"] = thresholds

    # 确保输出目录
    out_path = Path(config["evaluate"]["output_dir"])
    out_path.mkdir(parents=True, exist_ok=True)

    # 扫描测试数据
    pairs, same_count, diff_count = scan_test_pairs(config["evaluate"]["test_dir"])
    if not pairs:
        print("错误: 未找到评估数据，请先准备测试集")
        return None

    print(f"找到 {len(pairs)} 对测试数据（正样本: {same_count}, 负样本: {diff_count}）")

    # 初始化组件
    SigLIP2FeatureExtractor, SimilaritySearch, ImageClassifier = ensure_imports()

    extractor = SigLIP2FeatureExtractor(model_name=config["model"]["name"])
    searcher = SimilaritySearch(embedding_dim=config["model"]["embedding_dim"])
    classifier = ImageClassifier(
        model=extractor.model,
        processor=extractor.processor,
        categories=config["classifier"]["categories"],
        device=extractor.device,
    )
    searcher.load()

    # 评估分类
    class_result = evaluate_classification(pairs, classifier, config)

    # 评估检索
    print("\n开始相似度检索评估...")
    eval_result, thresholds_result = evaluate_retrieval(pairs, extractor, searcher, config)

    if not eval_result:
        print("错误: 检索评估无结果")
        return None

    # 生成报告
    report_md, best_thresh = generate_report((eval_result, thresholds_result), class_result, config)

    # 保存报告
    report_path = out_path / "evaluation_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"\n评估报告已保存至: {report_path}")

    # 绘制 PR 曲线
    plot_path = out_path / "pr_curve.png"
    plot_pr_curve(
        eval_result[0], eval_result[1],
        best_thresh or 0.85,
        str(plot_path),
    )

    return {
        "report_path": str(report_path),
        "chart_path": str(plot_path),
        "best_threshold": best_thresh,
        "report_md": report_md,
        "thresholds": thresholds_result,
        "classification": class_result,
    }


def main():
    parser = argparse.ArgumentParser(
        description="金融影像相似度检测模型评估工具"
    )
    parser.add_argument("--test_dir", type=str, default=None,
                        help="评估数据集目录（默认: config.yaml 中的 test_dir）")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="评估输出目录（默认: config.yaml 中的 output_dir）")
    parser.add_argument("--thresholds", type=float, nargs="+", default=None,
                        help="要评估的阈值列表（默认: config.yaml 中的 thresholds）")
    args = parser.parse_args()

    config = load_config()
    result = run_evaluation(
        config=config,
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        thresholds=args.thresholds,
    )

    if result is None:
        sys.exit(1)


if __name__ == "__main__":
    main()
