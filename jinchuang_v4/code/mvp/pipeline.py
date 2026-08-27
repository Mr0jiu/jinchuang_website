from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageOps
from torch import nn
from transformers import AutoModel, AutoProcessor

from src.fraud_monitoring import attach_customer_identity, build_fraud_monitoring, load_annotations, write_monitoring_outputs
from src.risk_policy import ThresholdPolicy

IMAGE_TYPES = (
    "bank_statement",
    "contract",
    "face_signing",
    "id_card_back",
    "id_card_front",
)
DEFAULT_MODEL = "google/siglip2-base-patch16-224"
DEFAULT_HIGH_RISK_THRESHOLD = 0.97
DEFAULT_MEDIUM_RISK_THRESHOLD = 0.93


def find_dataset_root(repo_root: Path) -> Path:
    outer = next((path for path in repo_root.iterdir() if path.is_dir() and path.name.startswith("23-")), None)
    if outer is None:
        raise FileNotFoundError("Dataset directory starting with '23-' was not found.")
    inner = next((path for path in outer.iterdir() if path.is_dir()), None)
    if inner is None:
        raise FileNotFoundError(f"No dataset directory found under {outer}.")
    return inner


def find_annotations_file(dataset_root: Path) -> Path | None:
    candidate = dataset_root / "annotations.csv"
    return candidate if candidate.exists() else None


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def scan_dataset(dataset_root: Path) -> pd.DataFrame:
    rows = []
    for loan_dir in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        for path in sorted(loan_dir.glob("*")):
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            image_type = path.stem
            status, error, width, height = "ok", "", 0, 0
            try:
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    width, height = image.size
            except Exception as exc:
                status, error = "bad", str(exc)
            rows.append(
                {
                    "loan_id": loan_dir.name,
                    "image_type": image_type,
                    "path": str(path.resolve()),
                    "relative_path": str(path.relative_to(dataset_root)),
                    "size_bytes": path.stat().st_size,
                    "width": width,
                    "height": height,
                    "status": status,
                    "error": error,
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"No images found under {dataset_root}.")
    return frame


class SiglipEncoder:
    def __init__(self, model_name: str, device: torch.device):
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, torch_dtype="auto").to(device).eval()

    @torch.inference_mode()
    def encode(self, images: list[Image.Image], batch_size: int) -> np.ndarray:
        chunks = []
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size]
            inputs = self.processor(images=batch, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            features = self.model.get_image_features(**inputs)
            if not isinstance(features, torch.Tensor):
                features = features.pooler_output
            features = F.normalize(features.float(), dim=-1)
            chunks.append(features.cpu().numpy())
        return np.concatenate(chunks).astype("float32")

    def encode_paths(self, paths: list[str], batch_size: int) -> np.ndarray:
        chunks = []
        for start in range(0, len(paths), batch_size):
            images = []
            for path in paths[start : start + batch_size]:
                with Image.open(path) as image:
                    images.append(ImageOps.exif_transpose(image).convert("RGB"))
            chunks.append(self.encode(images, batch_size))
        return np.concatenate(chunks).astype("float32")

    def encode_paths_cached(
        self,
        paths: list[str],
        batch_size: int,
        cache_path: Path,
        throttle_seconds: float = 0.0,
    ) -> np.ndarray:
        expected_shape = (len(paths), self.model.config.vision_config.hidden_size)
        if cache_path.exists():
            cached = np.load(cache_path, mmap_mode="r")
            if cached.shape == expected_shape:
                print(f"  using cached embeddings: {cache_path}", flush=True)
                return np.asarray(cached, dtype="float32")
            print(f"  ignoring cache with unexpected shape {cached.shape}; expected {expected_shape}", flush=True)

        embeddings = np.lib.format.open_memmap(cache_path, mode="w+", dtype="float32", shape=expected_shape)
        total = len(paths)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            images = []
            for path in paths[start:end]:
                with Image.open(path) as image:
                    images.append(ImageOps.exif_transpose(image).convert("RGB"))
            embeddings[start:end] = self.encode(images, batch_size)
            embeddings.flush()
            print(f"  embeddings: {end}/{total}", flush=True)
            if throttle_seconds > 0:
                time.sleep(throttle_seconds)
        return np.asarray(embeddings, dtype="float32")


def group_split(frame: pd.DataFrame, seed: int) -> np.ndarray:
    groups = sorted(frame["loan_id"].unique())
    random.Random(seed).shuffle(groups)
    train_end = max(1, round(len(groups) * 0.70))
    val_end = max(train_end + 1, round(len(groups) * 0.85))
    assignment = {
        group: "train" if index < train_end else "val" if index < val_end else "test"
        for index, group in enumerate(groups)
    }
    return frame["loan_id"].map(assignment).to_numpy()


class LinearHead(nn.Module):
    def __init__(self, input_dim: int, classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, classes)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.linear(values)


def classification_metrics(labels: np.ndarray, predictions: np.ndarray, classes: list[str]) -> dict:
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    for truth, prediction in zip(labels, predictions):
        matrix[int(truth), int(prediction)] += 1
    per_class = {}
    f1_values = []
    for index, name in enumerate(classes):
        tp = matrix[index, index]
        fp = matrix[:, index].sum() - tp
        fn = matrix[index, :].sum() - tp
        precision = float(tp / (tp + fp)) if tp + fp else 0.0
        recall = float(tp / (tp + fn)) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1, "support": int(matrix[index].sum())}
    return {
        "accuracy": float(np.trace(matrix) / matrix.sum()) if matrix.sum() else 0.0,
        "macro_f1": float(np.mean(f1_values)),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }


def train_classifier(
    embeddings: np.ndarray,
    labels: np.ndarray,
    splits: np.ndarray,
    classes: list[str],
    output_dir: Path,
    seed: int,
    epochs: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    torch.manual_seed(seed)
    model = LinearHead(embeddings.shape[1], len(classes))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=0.01)
    x = torch.from_numpy(embeddings)
    y = torch.from_numpy(labels).long()
    train_mask = torch.from_numpy(splits == "train")

    best_state, best_val = None, -1.0
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x[train_mask]), y[train_mask])
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            predictions = model(x).argmax(dim=1).numpy()
        val_mask = splits == "val"
        val_score = classification_metrics(labels[val_mask], predictions[val_mask], classes)["macro_f1"]
        if val_score >= best_val:
            best_val = val_score
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(x)
        probabilities = logits.softmax(dim=1).numpy()
        predictions = logits.argmax(dim=1).numpy()
    torch.save({"state_dict": model.state_dict(), "classes": classes, "input_dim": embeddings.shape[1]}, output_dir / "classifier.pt")
    metrics = {
        split: classification_metrics(labels[splits == split], predictions[splits == split], classes)
        for split in ("train", "val", "test")
    }
    return predictions, probabilities, metrics


def build_faiss(
    embeddings: np.ndarray,
    frame: pd.DataFrame,
    output_dir: Path,
    top_k: int,
) -> pd.DataFrame:
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError("faiss-cpu is required. Run: python -m pip install -r requirements.txt") from exc

    vectors = np.ascontiguousarray(embeddings.astype("float32"))
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(output_dir / "face_signing.faiss"))

    k = min(top_k + 1, len(vectors))
    similarities, indices = index.search(vectors, k)
    rows = []
    for query_index in range(len(vectors)):
        rank = 0
        for similarity, neighbor_index in zip(similarities[query_index], indices[query_index]):
            if int(neighbor_index) == query_index:
                continue
            rank += 1
            rows.append(
                {
                    "query_loan_id": frame.iloc[query_index]["loan_id"],
                    "query_path": frame.iloc[query_index]["path"],
                    "match_loan_id": frame.iloc[int(neighbor_index)]["loan_id"],
                    "match_path": frame.iloc[int(neighbor_index)]["path"],
                    "rank": rank,
                    "cosine_similarity": float(similarity),
                }
            )
            if rank == top_k:
                break
    return pd.DataFrame(rows)


def make_positive_variants(paths: list[str]) -> list[Image.Image]:
    variants = []
    for index, path in enumerate(paths):
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            width, height = image.size
            border = max(2, round(min(width, height) * 0.025))
            image = image.crop((border, border, width - border, height - border)).resize((width, height))
            image = ImageEnhance.Brightness(image).enhance(0.92 if index % 2 else 1.08)
            variants.append(image)
    return variants


def threshold_experiment(
    encoder: SiglipEncoder,
    face_frame: pd.DataFrame,
    face_embeddings: np.ndarray,
    batch_size: int,
) -> tuple[pd.DataFrame, dict]:
    variants = make_positive_variants(face_frame["path"].tolist())
    variant_embeddings = encoder.encode(variants, batch_size)
    positive_scores = np.sum(face_embeddings * variant_embeddings, axis=1)

    similarity_matrix = face_embeddings @ face_embeddings.T
    negative_scores = similarity_matrix[np.triu_indices(len(face_embeddings), k=1)]
    labels = np.concatenate([np.ones(len(positive_scores), dtype=int), np.zeros(len(negative_scores), dtype=int)])
    scores = np.concatenate([positive_scores, negative_scores])

    rows = []
    for threshold in np.round(np.arange(0.50, 1.001, 0.01), 2):
        predictions = scores >= threshold
        tp = int(np.sum((predictions == 1) & (labels == 1)))
        fp = int(np.sum((predictions == 1) & (labels == 0)))
        fn = int(np.sum((predictions == 0) & (labels == 1)))
        tn = int(np.sum((predictions == 0) & (labels == 0)))
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        fpr = fp / (fp + tn) if fp + tn else 0.0
        rows.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "false_positive_rate": fpr,
                "review_count": int(predictions.sum()),
            }
        )
    report = pd.DataFrame(rows)
    best = report.iloc[report["f1"].idxmax()].to_dict()
    metadata = {
        "evaluation_type": "proxy",
        "note": "Positive pairs are deterministic crop/brightness variants; negatives are distinct loan photos. Replace with reviewed pair labels for final reporting.",
        "positive_pairs": len(positive_scores),
        "negative_pairs": len(negative_scores),
        "best_f1_threshold": best,
    }
    return report, metadata


def labeled_threshold_experiment(topk: pd.DataFrame, annotations: pd.DataFrame) -> tuple[pd.DataFrame, dict] | None:
    """Calibrate retrieval thresholds with competition labels, never use them at inference."""
    from src.fraud_monitoring import enrich_topk_with_business

    pairs = enrich_topk_with_business(topk, annotations)
    pairs = pairs[pairs["query_loan_id"].astype(str) != pairs["match_loan_id"].astype(str)]
    pairs = pairs.sort_values(["cosine_similarity", "rank"], ascending=[False, True]).drop_duplicates("pair_key")
    if pairs.empty:
        return None
    truth = (
        pairs["query_similar_group"].fillna("").astype(str).ne("")
        & pairs["query_similar_group"].fillna("").astype(str).eq(pairs["match_similar_group"].fillna("").astype(str))
    )
    if not truth.any():
        return None
    rows = []
    scores = pairs["cosine_similarity"].to_numpy()
    labels = truth.to_numpy()
    for threshold in np.round(np.arange(0.50, 1.001, 0.01), 2):
        selected = scores >= threshold
        tp, fp = int((selected & labels).sum()), int((selected & ~labels).sum())
        fn = int((~selected & labels).sum())
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({"threshold": threshold, "precision": precision, "recall": recall, "f1": f1, "review_count": int(selected.sum()), "tp": tp, "fp": fp, "fn": fn})
    report = pd.DataFrame(rows)
    best = report.iloc[report["f1"].idxmax()].to_dict()
    return report, {
        "evaluation_type": "competition_ground_truth",
        "note": "similar_group is used only as a de-identified offline ground truth for threshold calibration; it is never an online customer-relation feature.",
        "positive_pairs": int(labels.sum()), "negative_pairs": int((~labels).sum()), "best_f1_threshold": best,
    }


def assign_risk(topk: pd.DataFrame, high: float, medium: float) -> pd.DataFrame:
    result = topk.copy()
    result["risk_level"] = np.select(
        [result["cosine_similarity"] >= high, result["cosine_similarity"] >= medium],
        ["high", "medium"],
        default="low",
    )
    return result


def run(args: argparse.Namespace) -> None:
    started = time.time()
    repo_root = Path(args.repo_root).resolve()
    dataset_root = Path(args.dataset_root).resolve() if args.dataset_root else find_dataset_root(repo_root)
    output_dir = Path(args.output_dir).resolve() if args.output_dir else repo_root / "outputs" / "mvp"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/7] Scanning dataset: {dataset_root}")
    manifest = scan_dataset(dataset_root)
    manifest.to_csv(output_dir / "data_manifest.csv", index=False, encoding="utf-8-sig")
    valid = manifest[manifest["status"] == "ok"].reset_index(drop=True)
    unknown = sorted(set(valid["image_type"]) - set(IMAGE_TYPES))
    if unknown:
        raise ValueError(f"Unknown image types: {unknown}")

    device = choose_device(args.device)
    print(f"[2/7] Loading {args.model_name} on {device}")
    encoder = SiglipEncoder(args.model_name, device)
    print(f"[3/7] Extracting {len(valid)} image embeddings")
    embeddings = encoder.encode_paths_cached(
        valid["path"].tolist(),
        args.batch_size,
        output_dir / "image_embeddings.npy",
        args.throttle_seconds,
    )

    classes = list(IMAGE_TYPES)
    labels = valid["image_type"].map({name: index for index, name in enumerate(classes)}).to_numpy()
    splits = group_split(valid, args.seed)
    print("[4/7] Training classification head")
    predictions, probabilities, metrics = train_classifier(
        embeddings, labels, splits, classes, output_dir, args.seed, args.epochs
    )
    valid["split"] = splits
    valid["predicted_type"] = [classes[index] for index in predictions]
    valid["confidence"] = probabilities.max(axis=1)
    valid.to_csv(output_dir / "classification_predictions.csv", index=False, encoding="utf-8-sig")
    with (output_dir / "classification_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)

    face_mask = valid["predicted_type"].eq("face_signing").to_numpy()
    face_frame = valid[face_mask].reset_index(drop=True)
    face_embeddings = embeddings[face_mask]
    if len(face_frame) < 2:
        raise RuntimeError("Classifier selected fewer than two face-signing photos.")
    np.save(output_dir / "face_embeddings.npy", face_embeddings)
    face_frame.to_csv(output_dir / "face_manifest.csv", index=False, encoding="utf-8-sig")

    print(f"[5/7] Building FAISS index for {len(face_frame)} selected signing photos")
    topk = build_faiss(face_embeddings, face_frame, output_dir, args.top_k)
    annotations_path = find_annotations_file(dataset_root)
    annotations = load_annotations(annotations_path) if annotations_path else pd.DataFrame()
    if not annotations.empty and args.identity_map:
        annotations = attach_customer_identity(annotations, args.identity_map)
    calibrated = labeled_threshold_experiment(topk, annotations) if not annotations.empty else None
    if calibrated:
        print("[6/7] Running labeled threshold calibration")
        threshold_report, threshold_metadata = calibrated
    else:
        print("[6/7] Running proxy threshold experiment")
        threshold_report, threshold_metadata = threshold_experiment(encoder, face_frame, face_embeddings, args.batch_size)
    threshold_report.to_csv(output_dir / "threshold_experiment.csv", index=False, encoding="utf-8-sig")
    with (output_dir / "threshold_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(threshold_metadata, handle, ensure_ascii=False, indent=2)

    proxy_threshold = float(threshold_metadata["best_f1_threshold"]["threshold"])
    best_threshold = proxy_threshold if args.use_calibrated_high_threshold else args.high_risk_threshold
    medium_threshold = args.medium_risk_threshold
    risk_results = assign_risk(topk, best_threshold, medium_threshold)
    risk_results.to_csv(output_dir / "topk_results.csv", index=False, encoding="utf-8-sig")

    if annotations_path:
        print("[6b/7] Building fraud monitoring report")
        policy = ThresholdPolicy(
            enabled=True,
            same_customer=args.same_customer_threshold,
            cross_customer=args.cross_customer_threshold,
            default=best_threshold,
            high_risk=best_threshold,
            medium_risk=medium_threshold,
        )
        monitoring = build_fraud_monitoring(risk_results, annotations, policy)
        monitoring_summary = write_monitoring_outputs(monitoring, output_dir)
    else:
        monitoring_summary = {}

    summary = {
        "model_name": args.model_name,
        "device": str(device),
        "dataset_root": str(dataset_root),
        "annotations_path": str(annotations_path) if annotations_path else "",
        "identity_map_path": str(args.identity_map) if args.identity_map else "",
        "total_images": int(len(manifest)),
        "valid_images": int(len(valid)),
        "bad_images": int((manifest["status"] != "ok").sum()),
        "class_counts": dict(Counter(valid["image_type"])),
        "selected_face_signing": int(len(face_frame)),
        "top_k": args.top_k,
        "high_risk_threshold": best_threshold,
        "high_risk_threshold_source": "ground_truth_f1_calibration" if args.use_calibrated_high_threshold else "business_policy_initial_value",
        "medium_risk_threshold": medium_threshold,
        "cross_customer_threshold": args.cross_customer_threshold,
        "same_customer_threshold": args.same_customer_threshold,
        "fraud_monitoring_summary": monitoring_summary,
        "proxy_best_f1_threshold": proxy_threshold,
        "elapsed_seconds": round(time.time() - started, 2),
    }
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(f"[7/7] Complete. Outputs: {output_dir}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the financial image similarity MVP.")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--identity-map", default="", help="CSV from scripts/build_identity_map.py; contains hashed customer keys only.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--throttle-seconds", type=float, default=0.0, help="Sleep after each embedding batch to reduce system load.")
    parser.add_argument("--high-risk-threshold", type=float, default=DEFAULT_HIGH_RISK_THRESHOLD)
    parser.add_argument("--use-calibrated-high-threshold", action="store_true", help="Use the offline F1-selected threshold instead of the initial business threshold.")
    parser.add_argument("--medium-risk-threshold", type=float, default=DEFAULT_MEDIUM_RISK_THRESHOLD)
    parser.add_argument("--cross-customer-threshold", type=float, default=0.95)
    parser.add_argument("--same-customer-threshold", type=float, default=0.92)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
