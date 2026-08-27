"""Build a two-stage similarity detection and fraud typing pipeline.

Stage 1 detects whether a pair of face-signing photos belongs to the same
similar_group using image-only pair evidence.

Stage 2 explains risk type only for pairs predicted similar by Stage 1, using
business identity evidence such as customer hash, name match, same_iddd, and
edit_type. similar_group is used only as an offline label.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


IMAGE_FEATURE_COLUMNS = [
    "global_semantic_similarity",
    "subject_region_hist_similarity",
    "background_hist_similarity",
    "local_structure_orb_ratio",
    "dhash_similarity",
    "mirror_local_structure_orb_ratio",
    "mirror_subject_region_hist_similarity",
    "mirror_background_hist_similarity",
    "mirror_dhash_similarity",
    "equalized_dhash_similarity",
    "edge_dhash_similarity",
    "edge_hist_similarity",
    "rotated_dhash_similarity",
    "rotated_dhash_gain",
    "rotated_edge_dhash_similarity",
    "rotated_edge_dhash_gain",
    "brightness_delta",
    "contrast_delta",
    "rgb_mean_abs_delta",
    "rgb_mean_euclidean_delta",
    "lab_mean_abs_delta",
    "lab_delta_e",
    "lab_delta_e2000",
    "hsv_mean_abs_delta",
    "hsv_hist_similarity",
    "blur_ratio",
]


RENEWAL_EDIT_TYPES = {"bg", "hair", "shirt", "shirt_bg", "background", "clothes", "background_change", "hair_change", "clothes_change"}
MANIPULATION_EDIT_TYPES = {"brightness", "contrast", "rotate", "rotation", "crop", "mirror", "flip"}
REPEAT_FRAUD_TYPES = {"F"}
RENEWAL_FRAUD_TYPES = {"C", "T"}
STAGE2_DEFAULT_RENEWAL_THRESHOLD = 0.50
MIRROR_LOCAL_ORB_OVERRIDE_THRESHOLD = 0.95
MIRROR_DHASH_OVERRIDE_THRESHOLD = 0.98
MIRROR_PROBABILITY_FLOOR = 0.38
RECALL_FIRST_TARGET_RECALL = 0.95
RECALL_FIRST_MIN_PRECISION = 0.55
SOFT_MIRROR_SEMANTIC_FLOOR = 0.982
SOFT_MIRROR_LOCAL_ORB_THRESHOLD = 0.65
SOFT_MIRROR_DHASH_THRESHOLD = 0.84
SEMANTIC_VISUAL_SEMANTIC_FLOOR = 0.985
SEMANTIC_VISUAL_PROBABILITY_FLOOR = 0.12
SEMANTIC_VISUAL_DHASH_THRESHOLD = 0.76
SEMANTIC_VISUAL_EQUALIZED_DHASH_THRESHOLD = 0.78
SEMANTIC_VISUAL_EDGE_DHASH_THRESHOLD = 0.70
LOW_COLOR_DELTA_SEMANTIC_FLOOR = 0.982
LOW_COLOR_DELTA_E2000_MAX = 2.0
LOW_COLOR_DELTA_HSV_HIST_MIN = 0.97
CONTRAST_SHIFT_SEMANTIC_FLOOR = 0.982
CONTRAST_SHIFT_EQUALIZED_DHASH_MIN = 0.78
CONTRAST_SHIFT_EDGE_DHASH_MIN = 0.70
CONTRAST_SHIFT_BRIGHTNESS_DELTA_MIN = 8.0
CONTRAST_SHIFT_CONTRAST_DELTA_MIN = 6.0
CONTRAST_SHIFT_RGB_DELTA_MIN = 8.0
POSITIVE_REVIEW_DECISIONS = {"确认相似", "CSV漏标，确认相似", "相似", "similar", "true", "1", "yes", "positive", "pair_label_positive"}
NEGATIVE_REVIEW_DECISIONS = {"确认不相似", "误报/不采用", "不相似", "not_similar", "false", "0", "no"}


def normalize_name(value: object) -> str:
    return re.sub(r"\s+", "", str(value or "")).strip()


def dataset_loan_id_from_path(frame: pd.DataFrame) -> pd.Series:
    return frame["file_path"].fillna("").astype(str).str.replace("\\", "/", regex=False).str.split("/").str[0]


def pair_key(left: object, right: object) -> str:
    return "|".join(sorted((str(left), str(right))))


def collapse_symmetric_pairs(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    before_rows = len(frame)
    frame = frame[frame["query_loan_id"].astype(str) != frame["match_loan_id"].astype(str)].copy()
    after_self_rows = len(frame)
    sort_columns = [column for column in ["pair_key", "stage1_label", "reviewed_pair_label", "global_semantic_similarity", "rank"] if column in frame.columns]
    ascending = [True]
    for column in sort_columns[1:]:
        ascending.append(column == "rank")
    collapsed = (
        frame.sort_values(sort_columns, ascending=ascending, na_position="last")
        .drop_duplicates("pair_key", keep="first")
        .reset_index(drop=True)
    )
    return collapsed, {
        "rows_before_collapse": int(before_rows),
        "self_pair_rows_removed": int(before_rows - after_self_rows),
        "symmetric_direction_rows_removed": int(after_self_rows - len(collapsed)),
        "unique_undirected_pairs": int(len(collapsed)),
    }


def review_decision_to_label(value: object) -> int | None:
    text = str(value or "").strip()
    if text in POSITIVE_REVIEW_DECISIONS:
        return 1
    if text in NEGATIVE_REVIEW_DECISIONS:
        return 0
    return None


def load_pair_label_overrides(output_dir: Path) -> dict[str, int]:
    path = output_dir / "stage1_review.csv"
    if not path.exists():
        return {}
    reviews = pd.read_csv(path, dtype=str, encoding="utf-8-sig").fillna("")
    required = {"query_loan_id", "match_loan_id", "decision"}
    if not required.issubset(reviews.columns):
        return {}
    overrides: dict[str, int] = {}
    for row in reviews.itertuples(index=False):
        label = review_decision_to_label(getattr(row, "decision", ""))
        if label is None:
            continue
        overrides[pair_key(getattr(row, "query_loan_id"), getattr(row, "match_loan_id"))] = int(label)
    return overrides


def collapse_duplicate_metadata(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for _, group in frame[columns].groupby("dataset_loan_id", sort=False):
        merged = {}
        for column in columns:
            values = [str(value) for value in group[column].fillna("").tolist() if str(value)]
            if column == "similar_group":
                merged[column] = next((value for value in values if value), "")
            elif column == "is_similar_pair":
                merged[column] = "1" if "1" in values else (values[0] if values else "")
            else:
                merged[column] = max(values, key=len) if values else ""
        rows.append(merged)
    return pd.DataFrame(rows, columns=columns)


def choose_name_column(frame: pd.DataFrame) -> str:
    if "姓名" in frame.columns:
        return "姓名"
    if "base_from" in frame.columns:
        index = list(frame.columns).index("base_from")
        if index + 1 < len(frame.columns):
            return frame.columns[index + 1]
    raise ValueError("Could not infer name column")


def load_metadata(annotations_path: Path, output_dir: Path) -> pd.DataFrame:
    annotations = pd.read_csv(annotations_path, dtype=str, encoding="utf-8-sig").fillna("")
    name_column = choose_name_column(annotations)
    annotations = annotations.assign(dataset_loan_id=dataset_loan_id_from_path(annotations))
    # Older/manual-review annotation exports do not always contain every
    # optional Stage-2 field. Keep those files runnable; when fraud_type is
    # absent the Stage-2 classifier falls back to explicit renewal metadata
    # and the conservative repeat-review result.
    for optional_column in ["fraud_type", "edit_type", "base_from", "same_iddd"]:
        if optional_column not in annotations.columns:
            annotations[optional_column] = ""
    columns = ["dataset_loan_id", "file_path", "loan_id", "similar_group", "is_similar_pair", "fraud_type", "edit_type", "base_from", "same_iddd", name_column]
    metadata = collapse_duplicate_metadata(annotations, columns).rename(columns={"loan_id": "business_loan_id", name_column: "name"})
    metadata["name_norm"] = metadata["name"].map(normalize_name)
    metadata["loan_group_key"] = metadata["similar_group"].where(metadata["similar_group"].astype(str).ne(""), metadata["dataset_loan_id"])
    metadata["base_from_loan_id"] = (
        metadata["base_from"].fillna("").astype(str).str.replace("\\", "/", regex=False).str.split("/").str[0]
    )
    identity_path = output_dir / "customer_identity_map_from_annotations.csv"
    if identity_path.exists():
        identity = pd.read_csv(identity_path, dtype=str).fillna("")
        metadata = metadata.merge(
            identity[["dataset_loan_id", "customer_id_hash", "status"]],
            on="dataset_loan_id",
            how="left",
        )
    else:
        metadata["customer_id_hash"] = ""
        metadata["status"] = ""
    return metadata


def build_model() -> Pipeline:
    preprocess = ColumnTransformer(
        [
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                IMAGE_FEATURE_COLUMNS,
            )
        ]
    )
    classifier = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.06,
        l2_regularization=0.05,
        random_state=42,
    )
    return Pipeline([("preprocess", preprocess), ("classifier", classifier)])


def build_stage2_reuse_renewal_model(random_state: int) -> Pipeline:
    """Build the Stage-2 classifier for direct reuse versus normal renewal."""
    preprocess = ColumnTransformer(
        [
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                IMAGE_FEATURE_COLUMNS,
            )
        ]
    )
    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=500,
        random_state=random_state,
    )
    return Pipeline([("preprocess", preprocess), ("classifier", classifier)])


def best_threshold(y_true: pd.Series | np.ndarray, probabilities: np.ndarray) -> dict[str, float | int]:
    best: dict[str, float | int] | None = None
    y = np.asarray(y_true).astype(bool)
    for threshold in np.linspace(0.05, 0.95, 91):
        prediction = probabilities >= threshold
        tp = int((prediction & y).sum())
        fp = int((prediction & ~y).sum())
        fn = int((~prediction & y).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        row = {
            "threshold": float(threshold),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
        if best is None or row["f1"] > best["f1"]:
            best = row
    assert best is not None
    return best


def recall_first_threshold(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    target_recall: float,
    min_precision: float,
) -> dict[str, float | int | str]:
    best_recall: dict[str, float | int | str] | None = None
    best_f2: dict[str, float | int | str] | None = None
    y = np.asarray(y_true).astype(bool)
    for threshold in np.linspace(0.01, 0.95, 95):
        prediction = probabilities >= threshold
        tp = int((prediction & y).sum())
        fp = int((prediction & ~y).sum())
        fn = int((~prediction & y).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        beta2 = 4.0
        f2 = (1 + beta2) * precision * recall / (beta2 * precision + recall) if precision + recall else 0.0
        row: dict[str, float | int | str] = {
            "threshold": float(threshold),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "f2": float(f2),
            "selection_policy": "recall_first",
            "target_recall": float(target_recall),
            "min_precision": float(min_precision),
        }
        if recall >= target_recall and precision >= min_precision:
            if best_recall is None or row["precision"] > best_recall["precision"]:
                best_recall = row
        if best_f2 is None or row["f2"] > best_f2["f2"]:
            best_f2 = row
    assert best_f2 is not None
    if best_recall is not None:
        best_recall["selection_reason"] = "met_target_recall_and_min_precision"
        return best_recall
    best_f2["selection_reason"] = "fallback_best_f2"
    return best_f2


def metrics_at_threshold(y_true: pd.Series | np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float | int]:
    y = np.asarray(y_true).astype(int)
    prediction = (probabilities >= threshold).astype(int)
    return metrics_for_prediction(y, prediction, probabilities, threshold)


def metrics_for_prediction(
    y_true: pd.Series | np.ndarray,
    prediction: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    y = np.asarray(y_true).astype(int)
    prediction = np.asarray(prediction).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y, prediction, average="binary", zero_division=0)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, prediction)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc_score(y, probabilities)) if len(set(y)) > 1 else 0.0,
        "positive_predictions": int(prediction.sum()),
    }


def calibrate_stage2_renewal_threshold(
    labeled: pd.DataFrame,
    random_state: int,
    test_size: float = 0.25,
) -> tuple[float, dict[str, object]]:
    """Select the Stage-2 renewal threshold on a held-out validation split."""
    summary: dict[str, object] = {
        "selection_policy": "validation_best_f1",
        "fallback_threshold": STAGE2_DEFAULT_RENEWAL_THRESHOLD,
        "threshold_grid": "0.05..0.95 step 0.01",
        "validation_test_size": float(test_size),
    }
    if len(labeled) < 20 or labeled["stage2_renewal_label"].nunique() != 2:
        summary["selection_reason"] = "fallback_insufficient_labeled_rows_or_single_class"
        return STAGE2_DEFAULT_RENEWAL_THRESHOLD, summary

    min_class_count = int(labeled["stage2_renewal_label"].value_counts().min())
    if min_class_count < 2:
        summary["selection_reason"] = "fallback_min_class_count_below_2"
        return STAGE2_DEFAULT_RENEWAL_THRESHOLD, summary

    train, validation = train_test_split(
        labeled,
        test_size=test_size,
        random_state=random_state,
        stratify=labeled["stage2_renewal_label"],
    )
    model = build_stage2_reuse_renewal_model(random_state)
    model.fit(train[IMAGE_FEATURE_COLUMNS], train["stage2_renewal_label"])
    validation_probabilities = model.predict_proba(validation[IMAGE_FEATURE_COLUMNS])[:, 1]
    best = best_threshold(validation["stage2_renewal_label"], validation_probabilities)
    default_metrics = metrics_at_threshold(
        validation["stage2_renewal_label"],
        validation_probabilities,
        STAGE2_DEFAULT_RENEWAL_THRESHOLD,
    )
    summary.update(
        {
            "selection_reason": "heldout_validation_best_f1",
            "train_rows": int(len(train)),
            "validation_rows": int(len(validation)),
            "validation_class_counts": {
                str(key): int(value) for key, value in Counter(validation["stage2_renewal_label"]).items()
            },
            "best_threshold": best,
            "default_threshold_metrics": default_metrics,
        }
    )
    return float(best["threshold"]), summary


def numeric_feature(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    return frame.get(column, pd.Series(default, index=frame.index)).astype(float)


def visual_override_masks(frame: pd.DataFrame, probabilities: pd.Series | np.ndarray | None = None) -> dict[str, pd.Series]:
    mirror_orb = numeric_feature(frame, "mirror_local_structure_orb_ratio")
    mirror_hash = numeric_feature(frame, "mirror_dhash_similarity")
    semantic = numeric_feature(frame, "global_semantic_similarity")
    dhash = numeric_feature(frame, "dhash_similarity")
    equalized_dhash = numeric_feature(frame, "equalized_dhash_similarity")
    edge_dhash = numeric_feature(frame, "edge_dhash_similarity")
    lab_delta = numeric_feature(frame, "lab_delta_e2000", default=999.0)
    hsv_hist = numeric_feature(frame, "hsv_hist_similarity")
    brightness_delta = numeric_feature(frame, "brightness_delta")
    contrast_delta = numeric_feature(frame, "contrast_delta")
    rgb_delta = numeric_feature(frame, "rgb_mean_abs_delta")
    if probabilities is None:
        probability = numeric_feature(frame, "stage1_similarity_probability")
    else:
        probability = pd.Series(probabilities, index=frame.index).astype(float)
    strong_mirror = (
        (probability >= MIRROR_PROBABILITY_FLOOR)
        & (mirror_orb >= MIRROR_LOCAL_ORB_OVERRIDE_THRESHOLD)
        & (mirror_hash >= MIRROR_DHASH_OVERRIDE_THRESHOLD)
    )
    soft_mirror = (
        (semantic >= SOFT_MIRROR_SEMANTIC_FLOOR)
        & (mirror_orb >= SOFT_MIRROR_LOCAL_ORB_THRESHOLD)
        & (mirror_hash >= SOFT_MIRROR_DHASH_THRESHOLD)
    )
    semantic_visual = (
        (probability >= SEMANTIC_VISUAL_PROBABILITY_FLOOR)
        & (semantic >= SEMANTIC_VISUAL_SEMANTIC_FLOOR)
        & (
            (dhash >= SEMANTIC_VISUAL_DHASH_THRESHOLD)
            | (equalized_dhash >= SEMANTIC_VISUAL_EQUALIZED_DHASH_THRESHOLD)
            | (edge_dhash >= SEMANTIC_VISUAL_EDGE_DHASH_THRESHOLD)
            | ((lab_delta <= LOW_COLOR_DELTA_E2000_MAX) & (hsv_hist >= LOW_COLOR_DELTA_HSV_HIST_MIN))
        )
    )
    low_color_delta = (
        (semantic >= LOW_COLOR_DELTA_SEMANTIC_FLOOR)
        & (lab_delta <= LOW_COLOR_DELTA_E2000_MAX)
        & (hsv_hist >= LOW_COLOR_DELTA_HSV_HIST_MIN)
    )
    contrast_shift = (
        (semantic >= CONTRAST_SHIFT_SEMANTIC_FLOOR)
        & (equalized_dhash >= CONTRAST_SHIFT_EQUALIZED_DHASH_MIN)
        & (edge_dhash >= CONTRAST_SHIFT_EDGE_DHASH_MIN)
        & (
            (brightness_delta >= CONTRAST_SHIFT_BRIGHTNESS_DELTA_MIN)
            | (contrast_delta >= CONTRAST_SHIFT_CONTRAST_DELTA_MIN)
            | (rgb_delta >= CONTRAST_SHIFT_RGB_DELTA_MIN)
        )
    )
    return {
        "strong_mirror_evidence": strong_mirror,
        "soft_mirror_evidence": soft_mirror,
        "semantic_visual_evidence": semantic_visual,
        "low_color_delta_high_semantic": low_color_delta,
        "contrast_or_brightness_shift": contrast_shift,
    }


def visual_override_mask(frame: pd.DataFrame, probabilities: pd.Series | np.ndarray | None = None) -> pd.Series:
    combined = pd.Series(False, index=frame.index)
    for mask in visual_override_masks(frame, probabilities).values():
        combined = combined | mask
    return combined


def visual_override_reason(frame: pd.DataFrame, probabilities: pd.Series | np.ndarray | None = None) -> pd.Series:
    reasons = pd.Series("", index=frame.index, dtype=object)
    for reason, mask in visual_override_masks(frame, probabilities).items():
        reasons = reasons.mask(mask, reasons.where(reasons.eq(""), reasons + ";") + reason)
    return reasons


def make_group_split(frame: pd.DataFrame, test_size: float, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(random_state)
    all_groups = np.array(sorted(set(frame["query_loan_group_key"]) | set(frame["match_loan_group_key"])))
    rng.shuffle(all_groups)
    test_count = max(1, int(round(len(all_groups) * test_size)))
    test_groups = set(all_groups[:test_count])
    query_is_test = frame["query_loan_group_key"].isin(test_groups)
    match_is_test = frame["match_loan_group_key"].isin(test_groups)
    train = frame[~query_is_test & ~match_is_test].copy()
    test = frame[query_is_test & match_is_test].copy()
    dropped = frame[query_is_test ^ match_is_test].copy()
    return train, test, dropped


def add_stage2_reuse_renewal_classification(
    pairs: pd.DataFrame,
    random_state: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Classify same-customer Stage-1 matches as repeat reuse or normal renewal.

    Stage 1 remains unchanged and answers only whether the images are similar.
    This Stage-2 classifier consumes the already-computed image evidence. F
    samples represent direct copies/light transformations; C/T samples
    represent changed hair, clothing, or background. Explicit edit_type and
    base_from metadata takes precedence when it is available.
    """
    frame = pairs.copy()
    query_type = frame["query_fraud_type"].fillna("").astype(str).str.upper()
    match_type = frame["match_fraud_type"].fillna("").astype(str).str.upper()
    renewal_label = query_type.isin(RENEWAL_FRAUD_TYPES) | match_type.isin(RENEWAL_FRAUD_TYPES)
    repeat_label = query_type.isin(REPEAT_FRAUD_TYPES) | match_type.isin(REPEAT_FRAUD_TYPES)
    related_pair = frame["same_similar_group"].astype(bool) | frame["renewal_base_pair"].astype(bool)
    labeled = frame[related_pair & (renewal_label ^ repeat_label)].copy()
    labeled["stage2_renewal_label"] = renewal_label.loc[labeled.index].astype(int)

    frame["stage2_renewal_probability"] = 0.0
    summary: dict[str, object] = {
        "method": "explicit_metadata_or_default_repeat",
        "training_rows": int(len(labeled)),
        "training_class_counts": (
            {str(key): int(value) for key, value in Counter(labeled["stage2_renewal_label"]).items()}
            if not labeled.empty
            else {}
        ),
        "default_threshold": STAGE2_DEFAULT_RENEWAL_THRESHOLD,
        "input": "Stage-1 image evidence plus Stage-2 identity gating",
    }
    selected_threshold, threshold_summary = calibrate_stage2_renewal_threshold(labeled, random_state)
    summary["threshold"] = selected_threshold
    summary["threshold_calibration"] = threshold_summary
    if len(labeled) >= 20 and labeled["stage2_renewal_label"].nunique() == 2:
        model = build_stage2_reuse_renewal_model(random_state)
        model.fit(labeled[IMAGE_FEATURE_COLUMNS], labeled["stage2_renewal_label"])
        frame["stage2_renewal_probability"] = model.predict_proba(frame[IMAGE_FEATURE_COLUMNS])[:, 1]
        summary["method"] = "logistic_regression_image_evidence"

    same_customer = frame["same_iddd_pair"].astype(bool) | frame["id_match"].astype(bool)
    model_renewal = frame["stage2_renewal_probability"].ge(selected_threshold)
    frame["stage2_same_customer_type"] = np.select(
        [
            same_customer & frame["renewal_base_pair"].astype(bool),
            same_customer & model_renewal,
            same_customer,
        ],
        [
            "normal_renewal_similarity",
            "normal_renewal_similarity",
            "same_customer_repeat_review",
        ],
        default="",
    )
    frame["stage2_same_customer_reason"] = np.select(
        [
            same_customer & frame["renewal_base_pair"].astype(bool),
            same_customer & model_renewal,
            same_customer,
        ],
        [
            "explicit_edit_type_and_base_from",
            "changed_hair_clothes_background",
            "direct_copy_or_light_manipulation",
        ],
        default="identity_not_same_customer",
    )
    summary["same_customer_type_counts"] = {
        str(key): int(value) for key, value in Counter(frame.loc[same_customer, "stage2_same_customer_type"]).items()
    }
    return frame, summary


def explain_stage2(row: pd.Series) -> str:
    if not bool(row["stage1_predicted_similar"]):
        return "not_suspicious"
    if bool(row["id_conflict"]) and bool(row["name_match"]):
        return "same_name_cross_id_fraud"
    if bool(row["id_conflict"]):
        return "cross_customer_fraud"
    if bool(row["same_iddd_pair"]) or bool(row["id_match"]):
        return str(row["stage2_same_customer_type"])
    if bool(row["name_match"]):
        return "same_name_pending_identity"
    return "high_similarity_pending_identity"


def table_stage2_type(row: pd.Series) -> str:
    if bool(row["same_similar_group"]) or bool(row["renewal_base_pair"]):
        if bool(row["id_conflict"]):
            return "cross_customer_fraud"
        if bool(row["same_iddd_pair"]) or bool(row["id_match"]):
            return str(row["stage2_same_customer_type"])
        return "cross_customer_fraud"
    return "not_labeled_similar"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build two-stage similarity and fraud type reports.")
    parser.add_argument("--annotations", default="data/annotations.csv")
    parser.add_argument("--pair-report", default="outputs/mvp/pair_evidence_model_report.csv")
    parser.add_argument("--output-dir", default="outputs/mvp")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--threshold-policy", choices=["recall_first", "best_f1"], default="recall_first")
    parser.add_argument("--target-recall", type=float, default=RECALL_FIRST_TARGET_RECALL)
    parser.add_argument("--min-precision", type=float, default=RECALL_FIRST_MIN_PRECISION)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    pairs = pd.read_csv(args.pair_report, low_memory=False)
    metadata_columns = ["query_edit_type", "match_edit_type", "query_base_from", "match_base_from"]
    pairs = pairs.drop(columns=[column for column in metadata_columns if column in pairs.columns])
    metadata = load_metadata(Path(args.annotations), output_dir)
    left = metadata.add_prefix("query_")
    right = metadata.add_prefix("match_")
    pairs = pairs.merge(left, left_on="query_loan_id", right_on="query_dataset_loan_id", how="left")
    pairs = pairs.merge(right, left_on="match_loan_id", right_on="match_dataset_loan_id", how="left")
    pairs["query_edit_type_norm"] = pairs["query_edit_type"].fillna("").astype(str).str.lower()
    pairs["match_edit_type_norm"] = pairs["match_edit_type"].fillna("").astype(str).str.lower()
    pairs["renewal_base_pair"] = (
        pairs["query_edit_type_norm"].isin(RENEWAL_EDIT_TYPES)
        & pairs["query_base_from_loan_id"].fillna("").astype(str).eq(pairs["match_loan_id"].astype(str))
    ) | (
        pairs["match_edit_type_norm"].isin(RENEWAL_EDIT_TYPES)
        & pairs["match_base_from_loan_id"].fillna("").astype(str).eq(pairs["query_loan_id"].astype(str))
    )
    pairs["pair_key"] = [pair_key(a, b) for a, b in zip(pairs["query_loan_id"], pairs["match_loan_id"])]
    pair_label_overrides = load_pair_label_overrides(output_dir)
    pairs["reviewed_pair_label"] = pairs["pair_key"].map(pair_label_overrides)
    pairs["stage1_label"] = (pairs["same_similar_group"].astype(bool) | pairs["renewal_base_pair"]).astype(int)
    pairs.loc[pairs["reviewed_pair_label"].notna(), "stage1_label"] = pairs.loc[
        pairs["reviewed_pair_label"].notna(), "reviewed_pair_label"
    ].astype(int)
    pairs, symmetry_summary = collapse_symmetric_pairs(pairs)

    X_train, X_test, y_train, y_test = train_test_split(
        pairs[IMAGE_FEATURE_COLUMNS],
        pairs["stage1_label"],
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=pairs["stage1_label"],
    )
    pair_model = build_model()
    pair_model.fit(X_train, y_train)
    pair_test_probabilities = pair_model.predict_proba(X_test)[:, 1]
    if args.threshold_policy == "recall_first":
        pair_best = recall_first_threshold(y_test, pair_test_probabilities, args.target_recall, args.min_precision)
    else:
        pair_best = best_threshold(y_test, pair_test_probabilities)
    pair_threshold = float(pair_best["threshold"])
    pair_base_prediction = pair_test_probabilities >= pair_threshold
    pair_override = visual_override_mask(X_test, pair_test_probabilities).to_numpy()
    pair_metrics = metrics_for_prediction(y_test, pair_base_prediction | pair_override, pair_test_probabilities, pair_threshold)

    group_train, group_test, group_dropped = make_group_split(pairs, args.test_size, args.random_state)
    group_model = build_model()
    group_model.fit(group_train[IMAGE_FEATURE_COLUMNS], group_train["stage1_label"])
    group_test_probabilities = group_model.predict_proba(group_test[IMAGE_FEATURE_COLUMNS])[:, 1]
    if args.threshold_policy == "recall_first":
        group_best = recall_first_threshold(group_test["stage1_label"], group_test_probabilities, args.target_recall, args.min_precision)
    else:
        group_best = best_threshold(group_test["stage1_label"], group_test_probabilities)
    group_threshold = float(group_best["threshold"])
    group_base_prediction = group_test_probabilities >= group_threshold
    group_override = visual_override_mask(group_test, group_test_probabilities).to_numpy()
    group_metrics = metrics_for_prediction(group_test["stage1_label"], group_base_prediction | group_override, group_test_probabilities, group_threshold)

    final_model = build_model()
    final_model.fit(pairs[IMAGE_FEATURE_COLUMNS], pairs["stage1_label"])
    pairs["stage1_similarity_probability"] = final_model.predict_proba(pairs[IMAGE_FEATURE_COLUMNS])[:, 1]
    final_threshold = pair_threshold
    pairs["probability_predicted_similar"] = pairs["stage1_similarity_probability"] >= final_threshold
    pairs["visual_override_reason"] = visual_override_reason(pairs)
    pairs["visual_override_predicted_similar"] = pairs["visual_override_reason"].astype(bool)
    pairs["stage1_predicted_similar"] = pairs["probability_predicted_similar"] | pairs["visual_override_predicted_similar"]
    pairs["stage1_decision_source"] = np.select(
        [
            pairs["probability_predicted_similar"] & pairs["visual_override_predicted_similar"],
            pairs["probability_predicted_similar"],
            pairs["visual_override_predicted_similar"],
        ],
        ["probability_and_visual_override", "probability", "visual_override"],
        default="not_predicted",
    )
    pairs, stage2_reuse_renewal_summary = add_stage2_reuse_renewal_classification(pairs, args.random_state)
    pairs["stage2_predicted_type"] = pairs.apply(explain_stage2, axis=1)
    pairs["stage2_table_type"] = pairs.apply(table_stage2_type, axis=1)
    final_metrics = metrics_for_prediction(
        pairs["stage1_label"],
        pairs["stage1_predicted_similar"],
        pairs["stage1_similarity_probability"].to_numpy(),
        final_threshold,
    )
    final_label = pairs["stage1_label"].astype(bool)
    final_prediction = pairs["stage1_predicted_similar"].astype(bool)
    final_confusion = {
        "TP": int((final_label & final_prediction).sum()),
        "FP": int((~final_label & final_prediction).sum()),
        "FN": int((final_label & ~final_prediction).sum()),
        "TN": int((~final_label & ~final_prediction).sum()),
    }
    visual_reason_counts = Counter(
        reason
        for value in pairs.loc[pairs["visual_override_predicted_similar"], "visual_override_reason"].fillna("")
        for reason in str(value).split(";")
        if reason
    )

    stage1_columns = [
        "query_loan_id",
        "match_loan_id",
        "rank",
        *IMAGE_FEATURE_COLUMNS,
        "stage1_similarity_probability",
        "probability_predicted_similar",
        "visual_override_predicted_similar",
        "visual_override_reason",
        "stage1_decision_source",
        "stage1_predicted_similar",
        "stage1_label",
        "same_similar_group",
        "query_similar_group",
        "match_similar_group",
        "query_path",
        "match_path",
    ]
    stage2_columns = [
        "query_loan_id",
        "match_loan_id",
        "stage1_similarity_probability",
        "stage1_predicted_similar",
        "stage2_predicted_type",
        "stage2_table_type",
        "name_match",
        "id_match",
        "id_conflict",
        "same_iddd_pair",
        "renewal_base_pair",
        "stage2_renewal_probability",
        "stage2_same_customer_type",
        "stage2_same_customer_reason",
        "query_edit_type",
        "match_edit_type",
        "query_base_from",
        "match_base_from",
        "query_customer_id_hash",
        "match_customer_id_hash",
        "query_path",
        "match_path",
    ]
    stage1_path = output_dir / "stage1_similarity_report.csv"
    stage2_path = output_dir / "stage2_fraud_type_report.csv"
    pairs[stage1_columns].to_csv(stage1_path, index=False, encoding="utf-8-sig")
    pairs.loc[pairs["stage1_predicted_similar"], stage2_columns].to_csv(stage2_path, index=False, encoding="utf-8-sig")

    summary = {
        "stage1": {
            "purpose": "image-only similar_group detection",
            "operating_policy": {
                "name": args.threshold_policy,
                "risk_preference": "recall_first_false_positives_accepted_for_manual_review",
                "target_recall": float(args.target_recall),
                "min_precision": float(args.min_precision),
            },
            "pair_symmetry_handling": {
                "policy": "undirected_pair_canonicalization",
                "selection": "keep_highest_label_then_reviewed_then_global_similarity_then_best_rank",
                **symmetry_summary,
            },
            "image_features": IMAGE_FEATURE_COLUMNS,
            "label_definition": "same_similar_group OR renewal_base_pair, overridden by outputs/mvp/stage1_review.csv pair labels when present",
            "reviewed_pair_label_rows": int(pairs["reviewed_pair_label"].notna().sum()),
            "reviewed_pair_label_positive_rows": int(pairs["reviewed_pair_label"].eq(1).sum()),
            "reviewed_pair_label_negative_rows": int(pairs["reviewed_pair_label"].eq(0).sum()),
            "reviewed_pair_label_unique_pairs": int(len(pair_label_overrides)),
            "pair_level_split": {
                "rows_train": int(len(X_train)),
                "rows_test": int(len(X_test)),
                "best_threshold": pair_best,
                "metrics": pair_metrics,
                "visual_override_hits_test": int(pair_override.sum()),
            },
            "group_level_split": {
                "rows_train": int(len(group_train)),
                "rows_test": int(len(group_test)),
                "rows_dropped_cross_split": int(len(group_dropped)),
                "best_threshold": group_best,
                "metrics": group_metrics,
                "visual_override_hits_test": int(group_override.sum()),
            },
            "final_threshold_for_reports": final_threshold,
            "final_predicted_similar": int(pairs["stage1_predicted_similar"].sum()),
            "final_visual_override_predicted_similar": int(pairs["visual_override_predicted_similar"].sum()),
            "final_report_metrics_after_review_labels": {**final_confusion, **final_metrics},
            "final_visual_override_reason_counts": dict(visual_reason_counts),
            "visual_override_rules": {
                "strong_mirror_evidence": {
                    "stage1_similarity_probability_min": MIRROR_PROBABILITY_FLOOR,
                    "mirror_local_structure_orb_ratio_min": MIRROR_LOCAL_ORB_OVERRIDE_THRESHOLD,
                    "mirror_dhash_similarity_min": MIRROR_DHASH_OVERRIDE_THRESHOLD,
                },
                "soft_mirror_evidence": {
                    "global_semantic_similarity_min": SOFT_MIRROR_SEMANTIC_FLOOR,
                    "mirror_local_structure_orb_ratio_min": SOFT_MIRROR_LOCAL_ORB_THRESHOLD,
                    "mirror_dhash_similarity_min": SOFT_MIRROR_DHASH_THRESHOLD,
                },
                "semantic_visual_evidence": {
                    "stage1_similarity_probability_min": SEMANTIC_VISUAL_PROBABILITY_FLOOR,
                    "global_semantic_similarity_min": SEMANTIC_VISUAL_SEMANTIC_FLOOR,
                    "any_of": {
                        "dhash_similarity_min": SEMANTIC_VISUAL_DHASH_THRESHOLD,
                        "equalized_dhash_similarity_min": SEMANTIC_VISUAL_EQUALIZED_DHASH_THRESHOLD,
                        "edge_dhash_similarity_min": SEMANTIC_VISUAL_EDGE_DHASH_THRESHOLD,
                        "low_color_delta": {
                            "lab_delta_e2000_max": LOW_COLOR_DELTA_E2000_MAX,
                            "hsv_hist_similarity_min": LOW_COLOR_DELTA_HSV_HIST_MIN,
                        },
                    },
                },
                "low_color_delta_high_semantic": {
                    "global_semantic_similarity_min": LOW_COLOR_DELTA_SEMANTIC_FLOOR,
                    "lab_delta_e2000_max": LOW_COLOR_DELTA_E2000_MAX,
                    "hsv_hist_similarity_min": LOW_COLOR_DELTA_HSV_HIST_MIN,
                },
                "contrast_or_brightness_shift": {
                    "global_semantic_similarity_min": CONTRAST_SHIFT_SEMANTIC_FLOOR,
                    "equalized_dhash_similarity_min": CONTRAST_SHIFT_EQUALIZED_DHASH_MIN,
                    "edge_dhash_similarity_min": CONTRAST_SHIFT_EDGE_DHASH_MIN,
                    "any_delta_min": {
                        "brightness_delta": CONTRAST_SHIFT_BRIGHTNESS_DELTA_MIN,
                        "contrast_delta": CONTRAST_SHIFT_CONTRAST_DELTA_MIN,
                        "rgb_mean_abs_delta": CONTRAST_SHIFT_RGB_DELTA_MIN,
                    },
                },
            },
        },
        "stage2": {
            "purpose": "fraud/renewal type explanation for Stage-1 similar pairs",
            "reuse_vs_renewal_classifier": stage2_reuse_renewal_summary,
            "predicted_type_counts": dict(Counter(pairs.loc[pairs["stage1_predicted_similar"], "stage2_predicted_type"])),
            "table_type_counts_on_predicted_similar": dict(Counter(pairs.loc[pairs["stage1_predicted_similar"], "stage2_table_type"])),
        },
        "outputs": [str(stage1_path), str(stage2_path), str(output_dir / "two_stage_summary.json")],
    }
    summary_path = output_dir / "two_stage_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
