"""
对比学习数据集：支持三元组（anchor/positive/negative）和数据对（same/diff）两种格式

目录约定:
    data_triplets/
        anchor/     # 每一张图作为一个锚点
        positive/   # 与 anchor 同人/同业务的图片（文件名前缀相同配对）
        negative/   # 不同人/不同业务的图片

    data_pairs/
        same/       # 正样本对
        diff/       # 负样本对

FraudAwareTripletDataset:
    基于 annotations.csv 的相似组（SG_XXX）构造三元组：
      - anchor: 面签照片
      - positive: 同一个 SG 的另一张面签（同一人重复提交欺诈场景）
      - negative: 另一个 SG 或普通借款人的面签
    训练时配合 DualMarginTripletLoss 使用，动态选择 m1/m2。
    数据增强用于为可能相同的正样本对增加光照/几何变化。
"""
import os
import torch
import random
import math
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
import csv


# ---------------------------------------------------------------------------
# 数据增强（轻量级在线增强，用于正样本对）
# ---------------------------------------------------------------------------

class LightAugmentation:
    """轻量级数据增强：为可能完全相同的正样本对增加多样性

    组合: 随机水平翻转 + 轻微旋转 + 色彩抖动 + 高斯模糊
    保持语义不变（仍然是面签照片）但改变光照/几何。
    """
    def __init__(self, image_size: int = 224, prob: float = 0.5):
        self.image_size = image_size
        self.prob = prob

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.prob:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < self.prob:
            angle = random.uniform(-8, 8)
            img = img.rotate(angle, resample=Image.BICUBIC, expand=False)
        if random.random() < self.prob:
            factor = random.uniform(0.85, 1.15)
            img = ImageEnhance.Brightness(img).enhance(factor)
        if random.random() < self.prob:
            factor = random.uniform(0.9, 1.1)
            img = ImageEnhance.Contrast(img).enhance(factor)
        return img


def tensor_from_pil(img: Image.Image, image_size: int = 224) -> torch.Tensor:
    """PIL Image → [C, H, W] 归一化张量"""
    img = img.resize((image_size, image_size))
    arr = np.array(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return torch.tensor(arr)


# ---------------------------------------------------------------------------
# 原有数据集（保持向后兼容）
# ---------------------------------------------------------------------------

class ContrastiveImageDataset(Dataset):
    """对比学习数据集：从三元组目录中构造训练样本

    每次返回 (anchor, positive, negative) 或 (image_a, image_b, label)
    """

    def __init__(self, data_dir: str, image_size: int = 224,
                 mode: str = "triplet", transform=None):
        """
        Args:
            data_dir: 数据目录
            image_size: 图片尺寸
            mode: "triplet" 返回 (anchor, pos, neg); "pair" 返回 (img_a, img_b, label)
            transform: 可选的在线数据增强
        """
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.mode = mode
        self.transform = transform

        if not self.data_dir.exists():
            raise FileNotFoundError(f"数据目录不存在: {data_dir}")

        if mode == "triplet":
            self._load_triplets()
        else:
            self._load_pairs()

    def _load_triplets(self):
        """加载三元组数据"""
        anchor_dir = self.data_dir / "anchor"
        pos_dir = self.data_dir / "positive"
        neg_dir = self.data_dir / "negative"

        if not all(d.exists() for d in [anchor_dir, pos_dir, neg_dir]):
            # 降级：从 same/diff 结构构造伪三元组
            print("  未找到标准三元组目录，尝试从 same/diff 构造...")
            self._build_triplets_from_pairs()
            return

        self.anchors = sorted([
            f for f in anchor_dir.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ])
        self.positives = sorted([
            f for f in pos_dir.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ])
        self.negatives = sorted([
            f for f in neg_dir.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ])

        # 如果数量不匹配，用最少那个
        self._len = min(len(self.anchors), len(self.positives), len(self.negatives))
        print(f"  三元组数据集: {self._len} 组 (anchor: {len(self.anchors)}, "
              f"pos: {len(self.positives)}, neg: {len(self.negatives)})")

    def _load_pairs(self):
        """从 same/diff 目录加载图片对"""
        same_dir = self.data_dir / "same"
        diff_dir = self.data_dir / "diff"

        self.pairs = []  # [(img1_path, img2_path, label)]

        if same_dir.exists():
            files = sorted([f for f in same_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])
            # 文件名前缀配对
            grouped = {}
            for f in files:
                parts = f.stem.rsplit("_", 1)
                if len(parts) == 2 and parts[1] in ("a", "b", "A", "B"):
                    base = parts[0]
                    grouped.setdefault(base, []).append(str(f))
            for base, pair_files in grouped.items():
                if len(pair_files) >= 2:
                    self.pairs.append((pair_files[0], pair_files[1], 1))

        if diff_dir.exists():
            files = sorted([f for f in diff_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])
            grouped = {}
            for f in files:
                parts = f.stem.rsplit("_", 1)
                if len(parts) == 2 and parts[1] in ("a", "b", "A", "B"):
                    base = parts[0]
                    grouped.setdefault(base, []).append(str(f))
            for base, pair_files in grouped.items():
                if len(pair_files) >= 2:
                    self.pairs.append((pair_files[0], pair_files[1], 0))

        random.shuffle(self.pairs)
        self._len = len(self.pairs)
        print(f"  图片对数据集: {self._len} 对 (same: {same_dir.exists()}, diff: {diff_dir.exists()})")

    def _build_triplets_from_pairs(self):
        """从图片对数据构造伪三元组"""
        same_dir = self.data_dir / "same"
        diff_dir = self.data_dir / "diff"

        anchors, positives, negatives = [], [], []

        if same_dir.exists():
            files = sorted([str(f) for f in same_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])
            # 偶数索引作为 anchor，奇数作为 positive
            for i in range(0, len(files) - 1, 2):
                anchors.append(files[i])
                positives.append(files[i + 1])

        if diff_dir.exists():
            negatives = [str(f) for f in diff_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
            # 取 b 文件（如果是配对的）
            neg_b = [f for f in negatives if f.rsplit("_", 1)[-1].rstrip("." + f.split(".")[-1]) in ("b", "B")]

        self.anchors = anchors
        self.positives = positives
        self.negatives = negatives if negatives else [random.choice(anchors + positives) for _ in range(len(anchors))]

        self._len = min(len(self.anchors), len(self.positives), len(self.negatives))
        print(f"  从图片对构造三元组: {self._len} 组")

    def _load_image(self, path: str):
        import numpy as np
        img = Image.open(path).convert("RGB")
        img = img.resize((self.image_size, self.image_size))
        tensor = torch.tensor(
            (np.array(img).transpose(2, 0, 1) / 255.0).astype(np.float32)
        )
        if self.transform:
            tensor = self.transform(tensor)
        return tensor

    def __len__(self):
        return max(self._len, 1)

    def __getitem__(self, idx):
        import numpy as np
        idx = idx % self._len

        if self.mode == "triplet":
            anchor = self._load_image(self.anchors[idx])
            positive = self._load_image(self.positives[idx])
            negative = self._load_image(self.negatives[idx % len(self.negatives)])
            return anchor, positive, negative
        else:
            img_a_path, img_b_path, label = self.pairs[idx]
            img_a = self._load_image(img_a_path)
            img_b = self._load_image(img_b_path)
            return img_a, img_b, torch.tensor(label, dtype=torch.float32)


# ---------------------------------------------------------------------------
# 基于 annotations.csv 的欺诈感知三元组数据集
# ---------------------------------------------------------------------------

class FraudAwareTripletDataset(Dataset):
    """基于 annotations.csv 构造三元组，配合 DualMarginTripletLoss 使用

    对每个 SG（相似组）中的面签照片作为 anchor:
      - positive: 同一 SG 中的另一张面签（重复提交欺诈正样本）
      - negative: 从其他 SG 或普通借款人面签中采样

    无 SG 的面签照片也可作为 anchor（此时 positive 是数据增强版本）。

    Args:
        data_dir: 数据根目录（含 loan_XXX/ 子目录）
        annotations_file: annotations.csv 路径
        image_size: 输出图片尺寸
        augment_prob: 正样本数据增强概率（为可能完全相同的复制图增加多样性）
    """
    def __init__(self, data_dir: str, annotations_file: str,
                 image_size: int = 224, augment_prob: float = 0.8):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.augment = LightAugmentation(image_size, prob=augment_prob)

        # 加载标注
        self._load_annotations(annotations_file)
        self._build_triplets()

    def _load_annotations(self, annotations_file: str):
        """加载 annotations.csv 按 loan_id 和 similar_group 组织"""
        self.loan_images = {}      # loan_id -> [{path, image_type, similar_group}]
        self.sg_images = {}        # similar_group -> [{path, loan_id}]
        self.all_sign_paths = []   # 所有面签路径（用于随机负采样）

        with open(annotations_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rel_path = row["file_path"].replace("\\", "/")
                full_path = self.data_dir / rel_path
                if not full_path.exists():
                    continue
                entry = {
                    "path": str(full_path),
                    "loan_id": row["loan_id"],
                    "image_type": row["image_type"],
                    "similar_group": row.get("similar_group", ""),
                    "is_similar_pair": row.get("is_similar_pair", "0") == "1",
                }
                # 按 loan_id 索引
                self.loan_images.setdefault(row["loan_id"], []).append(entry)
                # 按相似组索引
                sg = entry["similar_group"]
                if sg:
                    self.sg_images.setdefault(sg, []).append(entry)
                # 收集所有面签路径
                if row["image_type"] == "face_signing":
                    self.all_sign_paths.append(str(full_path))

        print(f"  标注加载: {len(self.loan_images)} 笔贷款, "
              f"{len(self.sg_images)} 个相似组, "
              f"{len(self.all_sign_paths)} 张面签")

    def _build_triplets(self):
        """构造三元组列表

        对每个相似组的每张面签:
          anchor = 该面签
          positive = 同 SG 的另一张面签（若同 SG 只有一张则用数据增强版本）
          negative = 随机另一张面签（不同 loan_id 且不在同一 SG）

        对无相似组的面签:
          anchor = 该面签
          positive = 数据增强版本（因为同人只有这一张）
          negative = 随机另一张面签
        """
        self.triplets = []  # [(anchor_path, positive_path, negative_path, is_fraud)]

        # 1. 有相似组的：同 SG 内构造
        for sg, members in self.sg_images.items():
            sign_members = [m for m in members if m["image_type"] == "face_signing"]
            if len(sign_members) < 2:
                continue
            # 对 SG 内每对面签都做 positive
            for i in range(len(sign_members)):
                anchor = sign_members[i]
                for j in range(len(sign_members)):
                    if i == j:
                        continue
                    positive = sign_members[j]
                    negative = self._sample_negative(sg)
                    self.triplets.append((
                        anchor["path"], positive["path"],
                        negative, True,
                    ))

        # 2. 无相似组的面签：anchor 的正样本来自数据增强
        for path in self.all_sign_paths:
            # 跳过已在 SG 中作为 anchor 的面签（避免重复）
            already_covered = set()
            for a, p, n, f in self.triplets:
                already_covered.add(a)
            if path in already_covered:
                continue
            negative = self._sample_negative(None)
            self.triplets.append((path, path, negative, False))

        # 如果三元组太少，补充跨贷款面签对（同 loan_id 的不同面签提升视角多样性）
        extra = 0
        while len(self.triplets) < 20 and extra < 100:
            # 随机两个不同贷款的面签作为 anchor-positive
            a_path = random.choice(self.all_sign_paths)
            n_path = self._sample_negative(None)
            self.triplets.append((a_path, a_path, n_path, False))
            extra += 1

        random.shuffle(self.triplets)
        print(f"  三元组总数: {len(self.triplets)} "
              f"(欺诈正样本: {sum(1 for _, _, _, f in self.triplets if f)})")

    def _sample_negative(self, exclude_sg):
        """从不同于 anchor 的 loan_id / SG 中采样负样本"""
        while True:
            neg_path = random.choice(self.all_sign_paths)
            neg_sg = None
            for sg, members in self.sg_images.items():
                for m in members:
                    if m["path"] == neg_path:
                        neg_sg = sg
                        break
            if exclude_sg is None or neg_sg != exclude_sg:
                return neg_path

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        anchor_path, pos_path, neg_path, is_fraud = self.triplets[idx]

        anchor_img = Image.open(anchor_path).convert("RGB")
        pos_img = Image.open(pos_path).convert("RGB")
        neg_img = Image.open(neg_path).convert("RGB")

        # 正样本应用数据增强
        if is_fraud:
            # 跨贷款的真实重复样本：增强增加多样性
            pos_img = self.augment(pos_img)

        anchor_t = tensor_from_pil(anchor_img, self.image_size)
        pos_t = tensor_from_pil(pos_img, self.image_size)
        neg_t = tensor_from_pil(neg_img, self.image_size)

        return anchor_t, pos_t, neg_t


# 快速测试
if __name__ == "__main__":
    import numpy as np

    # 用 test_eval 目录测试
    test_dir = "test_eval"
    os.makedirs(f"{test_dir}/same", exist_ok=True)
    os.makedirs(f"{test_dir}/diff", exist_ok=True)

    for i in range(4):
        Image.new("RGB", (224, 224), color=(240, 235, 220)).save(f"{test_dir}/same/pair_{i:03d}_a.jpg")
        Image.new("RGB", (224, 224), color=(235, 230, 215)).save(f"{test_dir}/same/pair_{i:03d}_b.jpg")
        Image.new("RGB", (224, 224), color=(255, 255, 255)).save(f"{test_dir}/diff/pair_{i:03d}_a.jpg")
        Image.new("RGB", (224, 224), color=(200, 200, 220)).save(f"{test_dir}/diff/pair_{i:03d}_b.jpg")

    ds = ContrastiveImageDataset(test_dir, mode="triplet")
    anchor, pos, neg = ds[0]
    print(f"三元组测试: anchor {anchor.shape}, pos {pos.shape}, neg {neg.shape}")

    ds2 = ContrastiveImageDataset(test_dir, mode="pair")
    a, b, label = ds2[0]
    print(f"图片对测试: a {a.shape}, b {b.shape}, label {label}")

    import shutil
    shutil.rmtree(test_dir, ignore_errors=True)
    print("数据集测试通过！")
