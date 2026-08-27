"""
对比学习训练脚本：支持标准 TripletMarginLoss 和分层双 Margin Loss

用法:
    # 标准模式（原有数据集目录）
    python src/train.py --data_dir ./data_triplets --epochs 20

    # 分层双 Margin 模式（基于 annotations.csv 的欺诈感知训练）
    python src/train.py --dual_margin --data_dir ./data --annotations ./data/annotations.csv --epochs 20

    # 快速测试
    python src/train.py --data_dir ./test_eval --epochs 2
"""
import os
import sys
import argparse
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class AdapterHead(nn.Module):
    """浅层适配器：线性投影 → ReLU → 线性投影"""
    def __init__(self, input_dim: int = 768, hidden_dim: int = 256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        return self.fc(x)


class SigLIP2TripletModel(nn.Module):
    """对比学习包装器：SigLIP2 特征提取 + 可选 adapter"""
    def __init__(self, vision_language_model, embedding_dim: int, freeze_backbone: bool = True, use_adapter: bool = True):
        super().__init__()
        self.vision_language_model = vision_language_model
        self.use_adapter = use_adapter

        if freeze_backbone:
            for param in self.vision_language_model.parameters():
                param.requires_grad_(False)

        if use_adapter:
            self.adapter = AdapterHead(input_dim=embedding_dim)

    def forward(self, images):
        features = self.vision_language_model.get_image_features(pixel_values=images)
        if not isinstance(features, torch.Tensor):
            features = features.pooler_output
        features = nn.functional.normalize(features, dim=-1)
        if self.use_adapter:
            features = self.adapter(features)
            features = nn.functional.normalize(features, dim=-1)
        return features


def train_triplet(config, model, train_loader, val_loader, device, use_dual_margin=False):
    """训练主循环"""
    epochs = config["training"]["epochs"]
    lr = config["training"]["learning_rate"]

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
    )

    if use_dual_margin:
        from src.losses import DualMarginTripletLossWithStats
        dm = config["training"]["dual_margin"]
        criterion = DualMarginTripletLossWithStats(
            m1=dm["m1"],
            m2=dm["m2"],
            sim_threshold=dm["sim_threshold"],
        )
        print(f"  Loss: DualMarginTripletLoss (m1={dm['m1']}, m2={dm['m2']}, "
              f"threshold={dm['sim_threshold']})")
    else:
        criterion = nn.TripletMarginLoss(margin=config["training"]["triplet_margin"])
        print(f"  Loss: TripletMarginLoss (margin={config['training']['triplet_margin']})")

    output_dir = Path(config["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    history = {"train_loss": [], "val_loss": []}
    if use_dual_margin:
        history["fraud_ratio"] = []

    for epoch in range(1, epochs + 1):
        # ---- 训练 ----
        model.train()
        if use_dual_margin:
            criterion.reset_stats()

        train_loss = 0.0
        train_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
        for batch in pbar:
            anchor, positive, negative = [b.to(device) for b in batch]

            optimizer.zero_grad()
            anchor_out = model(anchor)
            positive_out = model(positive)
            negative_out = model(negative)

            loss = criterion(anchor_out, positive_out, negative_out)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

            info = {"loss": f"{loss.item():.4f}"}
            if use_dual_margin:
                info["fraud"] = f"{criterion.get_ratio():.2%}"
            pbar.set_postfix(info)

        avg_train_loss = train_loss / max(train_batches, 1)
        history["train_loss"].append(avg_train_loss)

        if use_dual_margin:
            history["fraud_ratio"].append(criterion.get_ratio())

        # ---- 验证 ----
        model.eval()
        if use_dual_margin:
            criterion.reset_stats()

        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]"):
                anchor, positive, negative = [b.to(device) for b in batch]
                anchor_out = model(anchor)
                positive_out = model(positive)
                negative_out = model(negative)
                loss = criterion(anchor_out, positive_out, negative_out)
                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / max(val_batches, 1)
        history["val_loss"].append(avg_val_loss)

        print(f"  Epoch {epoch}: train_loss={avg_train_loss:.4f}, "
              f"val_loss={avg_val_loss:.4f}"
              + (f", fraud_ratio={criterion.get_ratio():.2%}" if use_dual_margin else ""))

        # 保存 checkpoint
        if epoch % 5 == 0 or epoch == epochs:
            ckpt_path = output_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "use_dual_margin": use_dual_margin,
            }, ckpt_path)

    # 保存最终模型
    final_path = output_dir / "final_model.pt"
    torch.save(model.state_dict(), final_path)
    print(f"\n模型已保存至: {final_path}")

    # 保存训练历史
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    return history


def main():
    parser = argparse.ArgumentParser(description="对比学习训练脚本")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="训练数据目录")
    parser.add_argument("--annotations", type=str, default=None,
                        help="annotations.csv 路径（双Margin模式需要）")
    parser.add_argument("--epochs", type=int, default=None,
                        help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="批大小")
    parser.add_argument("--freeze_backbone", action="store_true", default=None,
                        help="冻结 SigLIP2 backbone")
    parser.add_argument("--no_adapter", action="store_true",
                        help="不使用 adapter 头")
    parser.add_argument("--dual_margin", action="store_true",
                        help="启用分层双 Margin Loss（默认使用标准 TripletMarginLoss）")
    parser.add_argument("--device", type=str, default=None,
                        help="计算设备 (cpu/cuda)")
    args = parser.parse_args()

    config = load_config()

    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.freeze_backbone is not None:
        config["training"]["freeze_backbone"] = args.freeze_backbone

    device = torch.device(args.device if args.device else "cpu")

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"错误: 数据目录不存在: {args.data_dir}")
        return

    # 选择数据集
    if args.dual_margin:
        from src.dataset import FraudAwareTripletDataset
        annotations_file = args.annotations
        if annotations_file is None:
            default_annot = data_dir / "annotations.csv"
            if default_annot.exists():
                annotations_file = str(default_annot)
        if annotations_file is None or not Path(annotations_file).exists():
            print("错误: 双Margin模式需要 --annotations 指向 annotations.csv")
            return
        print(f"使用 FraudAwareTripletDataset (data_dir={data_dir}, "
              f"annotations={annotations_file})")
        dataset = FraudAwareTripletDataset(
            data_dir=str(data_dir),
            annotations_file=annotations_file,
            image_size=config["data"]["image_size"],
            augment_prob=config["training"].get("augment_prob", 0.8),
        )
    else:
        from src.dataset import ContrastiveImageDataset
        try:
            dataset = ContrastiveImageDataset(
                str(data_dir), image_size=config["data"]["image_size"], mode="triplet"
            )
        except (FileNotFoundError, KeyError):
            print("  三元组模式不可用，降级为图片对模式...")
            dataset = ContrastiveImageDataset(
                str(data_dir), image_size=config["data"]["image_size"], mode="pair"
            )

    if len(dataset) < 2:
        print("错误: 训练数据过少（至少需要 2 组）")
        return

    # 划分训练/验证集
    val_split = config["training"]["val_split"]
    val_size = max(1, int(len(dataset) * val_split))
    train_size = len(dataset) - val_size
    train_subset, val_subset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    batch_size = config["data"]["batch_size"]
    train_loader = DataLoader(train_subset, batch_size=min(batch_size, train_size),
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=min(batch_size, val_size),
                            shuffle=False, num_workers=0)

    print(f"训练集: {train_size} 组, 验证集: {val_size} 组, batch_size: {batch_size}")

    # 初始化模型
    from src.model import SigLIP2FeatureExtractor
    base_extractor = SigLIP2FeatureExtractor(device=device, model_name=config["model"]["name"])

    model = SigLIP2TripletModel(
        vision_language_model=base_extractor.model,
        embedding_dim=config["model"]["embedding_dim"],
        freeze_backbone=config["training"]["freeze_backbone"],
        use_adapter=not args.no_adapter,
    ).to(device)

    if config["training"]["freeze_backbone"]:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"  冻结 backbone: 训练参数 {trainable:,} / {total:,} "
              f"({100 * trainable / total:.1f}%)")

    # 开始训练
    print(f"\n开始训练 (epochs={config['training']['epochs']}, device={device})")
    use_dual_margin = args.dual_margin or config["training"]["dual_margin"]["enabled"]
    history = train_triplet(config, model, train_loader, val_loader, device,
                            use_dual_margin=use_dual_margin)

    print(f"\n训练完成! 最终 train_loss: {history['train_loss'][-1]:.4f}, "
          f"val_loss: {history['val_loss'][-1]:.4f}")


if __name__ == "__main__":
    main()
