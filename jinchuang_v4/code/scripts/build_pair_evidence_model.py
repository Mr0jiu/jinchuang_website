"""Train a lightweight pair-evidence model for image similarity fraud signals.

The model works at the image-pair level. It combines global embedding
similarity, local structure, background, center-subject proxy, image quality,
and business identity features.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ID_CARD_PATTERN = re.compile(r"(?<!\d)(\d{17}[0-9Xx])(?![0-9Xx])")
RENEWAL_EDIT_TYPES = {"bg", "hair", "shirt", "shirt_bg", "background", "clothes", "background_change", "hair_change", "clothes_change"}
POSITIVE_REVIEW_DECISIONS = {"确认相似", "CSV漏标，确认相似", "相似", "similar", "true", "1", "yes", "positive", "pair_label_positive"}
NEGATIVE_REVIEW_DECISIONS = {"确认不相似", "误报/不采用", "不相似", "not_similar", "false", "0", "no"}


@dataclass
class ImageEvidence:
    path: str
    gray: np.ndarray
    center_gray: np.ndarray
    background_gray: np.ndarray
    hist: np.ndarray
    center_hist: np.ndarray
    background_hist: np.ndarray
    mirror_hist: np.ndarray
    mirror_center_hist: np.ndarray
    mirror_background_hist: np.ndarray
    dhash: np.ndarray
    mirror_dhash: np.ndarray
    equalized_dhash: np.ndarray
    edge_dhash: np.ndarray
    edge_hist: np.ndarray
    rotated_dhashes: list[np.ndarray]
    rotated_edge_dhashes: list[np.ndarray]
    rgb_mean: np.ndarray
    lab_mean: np.ndarray
    hsv_mean: np.ndarray
    hsv_hist: np.ndarray
    orb_desc: np.ndarray | None
    mirror_orb_desc: np.ndarray | None
    brightness: float
    contrast: float
    blur: float


def normalize_name(value: object) -> str:
    return re.sub(r"\s+", "", str(value or "")).strip()


def choose_name_column(frame: pd.DataFrame) -> str:
    if "姓名" in frame.columns:
        return "姓名"
    if "base_from" in frame.columns:
        index = list(frame.columns).index("base_from")
        if index + 1 < len(frame.columns):
            return frame.columns[index + 1]
    raise ValueError("Could not infer name column")


def choose_id_column(frame: pd.DataFrame) -> str:
    if "身份证号" in frame.columns:
        return "身份证号"
    best_column, best_count = "", 0
    for column in frame.columns:
        count = int(frame[column].fillna("").astype(str).map(lambda value: bool(ID_CARD_PATTERN.search(value))).sum())
        if count > best_count:
            best_column, best_count = column, count
    if not best_column:
        raise ValueError("Could not infer ID-card column")
    return best_column


def dataset_loan_id_from_path(frame: pd.DataFrame) -> pd.Series:
    return frame["file_path"].fillna("").astype(str).str.replace("\\", "/", regex=False).str.split("/").str[0]


def base_loan_id_from_path(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.replace("\\", "/", regex=False).str.split("/").str[0]


def pair_key(left: object, right: object) -> str:
    return "|".join(sorted((str(left), str(right))))


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
            elif column in {"is_similar_pair", "same_iddd"}:
                merged[column] = "1" if "1" in values else (values[0] if values else "")
            else:
                merged[column] = max(values, key=len) if values else ""
        rows.append(merged)
    return pd.DataFrame(rows, columns=columns)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


def histogram(gray: np.ndarray) -> np.ndarray:
    hist = cv2.calcHist([gray], [0], None, [32], [0, 256]).astype("float32").reshape(-1)
    total = float(hist.sum())
    return hist / total if total else hist


def hsv_histogram(hsv: np.ndarray) -> np.ndarray:
    hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256]).astype("float32").reshape(-1)
    total = float(hist.sum())
    return hist / total if total else hist


def dhash_bits(gray: np.ndarray) -> np.ndarray:
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    return (small[:, 1:] > small[:, :-1]).astype(np.uint8).reshape(-1)


def rotate_gray(gray: np.ndarray, angle: float) -> np.ndarray:
    height, width = gray.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    return cv2.warpAffine(gray, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def hamming_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return 1.0 - float(np.mean(a != b))


def load_image_evidence(path: str, orb: cv2.ORB) -> ImageEvidence:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized_gray = cv2.equalizeHist(gray)
    edges = cv2.Canny(gray, 80, 160)
    rotated_grays = [rotate_gray(gray, angle) for angle in (-15, -8, 8, 15)]
    rotated_edges = [cv2.Canny(rotated, 80, 160) for rotated in rotated_grays]
    mirror_gray = cv2.flip(gray, 1)
    center = gray[64:192, 64:192]
    mirror_center = mirror_gray[64:192, 64:192]
    top = gray[:56, :]
    bottom = gray[200:, :]
    left = gray[:, :56]
    right = gray[:, 200:]
    background = np.concatenate([top.reshape(-1), bottom.reshape(-1), left.reshape(-1), right.reshape(-1)]).reshape(-1, 1)
    mirror_top = mirror_gray[:56, :]
    mirror_bottom = mirror_gray[200:, :]
    mirror_left = mirror_gray[:, :56]
    mirror_right = mirror_gray[:, 200:]
    mirror_background = np.concatenate(
        [mirror_top.reshape(-1), mirror_bottom.reshape(-1), mirror_left.reshape(-1), mirror_right.reshape(-1)]
    ).reshape(-1, 1)
    _, desc = orb.detectAndCompute(gray, None)
    _, mirror_desc = orb.detectAndCompute(mirror_gray, None)
    return ImageEvidence(
        path=path,
        gray=gray,
        center_gray=center,
        background_gray=background,
        hist=histogram(gray),
        center_hist=histogram(center),
        background_hist=histogram(background),
        mirror_hist=histogram(mirror_gray),
        mirror_center_hist=histogram(mirror_center),
        mirror_background_hist=histogram(mirror_background),
        dhash=dhash_bits(gray),
        mirror_dhash=dhash_bits(mirror_gray),
        equalized_dhash=dhash_bits(equalized_gray),
        edge_dhash=dhash_bits(edges),
        edge_hist=histogram(edges),
        rotated_dhashes=[dhash_bits(rotated) for rotated in rotated_grays],
        rotated_edge_dhashes=[dhash_bits(rotated) for rotated in rotated_edges],
        rgb_mean=rgb.reshape(-1, 3).mean(axis=0),
        lab_mean=lab.reshape(-1, 3).mean(axis=0),
        hsv_mean=hsv.reshape(-1, 3).mean(axis=0),
        hsv_hist=hsv_histogram(hsv),
        orb_desc=desc,
        mirror_orb_desc=mirror_desc,
        brightness=float(gray.mean()),
        contrast=float(gray.std()),
        blur=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
    )


def orb_match_ratio(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None or len(a) < 2 or len(b) < 2:
        return 0.0
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(a, b)
    if not matches:
        return 0.0
    good = [match for match in matches if match.distance <= 48]
    return float(len(good) / max(len(a), len(b)))


def hist_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(cv2.compareHist(a.astype("float32"), b.astype("float32"), cv2.HISTCMP_CORREL))


def mean_abs_delta(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype("float32") - b.astype("float32"))))


def euclidean_delta(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a.astype("float32") - b.astype("float32")))


def opencv_lab_to_standard(lab_mean: np.ndarray) -> np.ndarray:
    lab = lab_mean.astype("float32")
    return np.array([lab[0] * 100.0 / 255.0, lab[1] - 128.0, lab[2] - 128.0], dtype="float32")


def ciede2000_delta(a: np.ndarray, b: np.ndarray) -> float:
    lab1 = opencv_lab_to_standard(a)
    lab2 = opencv_lab_to_standard(b)
    l1, a1, b1 = [float(value) for value in lab1]
    l2, a2, b2 = [float(value) for value in lab2]

    avg_lp = (l1 + l2) / 2.0
    c1 = np.sqrt(a1 * a1 + b1 * b1)
    c2 = np.sqrt(a2 * a2 + b2 * b2)
    avg_c = (c1 + c2) / 2.0
    g = 0.5 * (1.0 - np.sqrt((avg_c**7) / (avg_c**7 + 25.0**7)))
    a1p = (1.0 + g) * a1
    a2p = (1.0 + g) * a2
    c1p = np.sqrt(a1p * a1p + b1 * b1)
    c2p = np.sqrt(a2p * a2p + b2 * b2)
    avg_cp = (c1p + c2p) / 2.0

    h1p = np.degrees(np.arctan2(b1, a1p)) % 360.0 if c1p else 0.0
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360.0 if c2p else 0.0
    dhp = h2p - h1p
    if c1p * c2p and abs(dhp) > 180.0:
        dhp += 360.0 if dhp < 0.0 else -360.0
    delta_hp = 2.0 * np.sqrt(c1p * c2p) * np.sin(np.radians(dhp / 2.0))

    if not c1p * c2p:
        avg_hp = h1p + h2p
    elif abs(h1p - h2p) <= 180.0:
        avg_hp = (h1p + h2p) / 2.0
    else:
        avg_hp = (h1p + h2p + 360.0) / 2.0 if h1p + h2p < 360.0 else (h1p + h2p - 360.0) / 2.0

    delta_lp = l2 - l1
    delta_cp = c2p - c1p
    t = (
        1.0
        - 0.17 * np.cos(np.radians(avg_hp - 30.0))
        + 0.24 * np.cos(np.radians(2.0 * avg_hp))
        + 0.32 * np.cos(np.radians(3.0 * avg_hp + 6.0))
        - 0.20 * np.cos(np.radians(4.0 * avg_hp - 63.0))
    )
    delta_ro = 30.0 * np.exp(-(((avg_hp - 275.0) / 25.0) ** 2))
    rc = 2.0 * np.sqrt((avg_cp**7) / (avg_cp**7 + 25.0**7))
    sl = 1.0 + (0.015 * ((avg_lp - 50.0) ** 2)) / np.sqrt(20.0 + ((avg_lp - 50.0) ** 2))
    sc = 1.0 + 0.045 * avg_cp
    sh = 1.0 + 0.015 * avg_cp * t
    rt = -np.sin(np.radians(2.0 * delta_ro)) * rc
    return float(
        np.sqrt(
            (delta_lp / sl) ** 2
            + (delta_cp / sc) ** 2
            + (delta_hp / sh) ** 2
            + rt * (delta_cp / sc) * (delta_hp / sh)
        )
    )


def hsv_mean_abs_delta(a: np.ndarray, b: np.ndarray) -> float:
    hue_delta = abs(float(a[0]) - float(b[0]))
    hue_delta = min(hue_delta, 180.0 - hue_delta) * (255.0 / 180.0)
    saturation_delta = abs(float(a[1]) - float(b[1]))
    value_delta = abs(float(a[2]) - float(b[2]))
    return float(np.mean([hue_delta, saturation_delta, value_delta]))


def build_metadata(annotations_path: Path, output_dir: Path) -> pd.DataFrame:
    annotations = pd.read_csv(annotations_path, dtype=str).fillna("")
    identity = pd.read_csv(output_dir / "customer_identity_map_from_annotations.csv", dtype=str).fillna("")
    name_column = choose_name_column(annotations)
    annotations = annotations.assign(dataset_loan_id=dataset_loan_id_from_path(annotations))
    columns = ["dataset_loan_id", "loan_id", "similar_group", "is_similar_pair", "same_iddd", "edit_type", "base_from", name_column]
    meta = (
        collapse_duplicate_metadata(annotations, columns)
        .rename(columns={"loan_id": "business_loan_id", name_column: "name"})
    )
    meta["name_norm"] = meta["name"].map(normalize_name)
    meta["edit_type_norm"] = meta["edit_type"].fillna("").astype(str).str.lower()
    meta["base_from_loan_id"] = base_loan_id_from_path(meta["base_from"])
    meta = meta.merge(identity[["dataset_loan_id", "customer_id_hash", "status"]], on="dataset_loan_id", how="left")
    return meta


def fraud_type(row: pd.Series, probability_threshold: float) -> str:
    if float(row["similar_probability"]) < probability_threshold:
        return "not_suspicious"
    if bool(row["renewal_base_pair"]):
        return "same_customer_renewal_or_repeat"
    if bool(row["id_conflict"]) and bool(row["name_match"]):
        return "same_name_cross_id_fraud"
    if bool(row["id_conflict"]):
        return "cross_customer_fraud"
    if bool(row["id_match"]):
        return "same_customer_renewal_or_repeat"
    if bool(row["name_match"]):
        return "same_name_pending_identity"
    return "high_similarity_pending_identity"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a multi-dimensional pair evidence model.")
    parser.add_argument("--annotations", default="data/annotations.csv")
    parser.add_argument("--output-dir", default="outputs/mvp")
    parser.add_argument("--probability-threshold", type=float, default=0.38)
    parser.add_argument("--model", choices=["histgb", "logistic"], default="histgb")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    topk = pd.read_csv(output_dir / "topk_results.csv", dtype=str).fillna("")
    manifest = pd.read_csv(output_dir / "data_manifest.csv")
    embeddings = np.load(output_dir / "image_embeddings.npy")
    meta = build_metadata(Path(args.annotations), output_dir)

    face_manifest = manifest[manifest["image_type"].astype(str).eq("face_signing")].reset_index().rename(columns={"index": "embedding_index"})
    id_manifest = manifest[manifest["image_type"].astype(str).eq("id_card_front")].reset_index().rename(columns={"index": "embedding_index"})
    face_embedding = {str(row.loan_id): embeddings[int(row.embedding_index)].astype("float32") for row in face_manifest.itertuples(index=False)}
    id_embedding = {str(row.loan_id): embeddings[int(row.embedding_index)].astype("float32") for row in id_manifest.itertuples(index=False)}

    paths = sorted(set(topk["query_path"]) | set(topk["match_path"]))
    orb = cv2.ORB_create(nfeatures=700)
    image_cache = {path: load_image_evidence(path, orb) for path in paths}

    meta_left = meta.add_prefix("query_")
    meta_right = meta.add_prefix("match_")
    pairs = topk.rename(columns={"cosine_similarity": "global_semantic_similarity"})
    pairs = pairs.merge(meta_left, left_on="query_loan_id", right_on="query_dataset_loan_id", how="left")
    pairs = pairs.merge(meta_right, left_on="match_loan_id", right_on="match_dataset_loan_id", how="left")
    pairs["global_semantic_similarity"] = pairs["global_semantic_similarity"].astype(float)

    feature_rows = []
    for row in pairs.itertuples(index=False):
        query = image_cache[row.query_path]
        match = image_cache[row.match_path]
        qloan, mloan = str(row.query_loan_id), str(row.match_loan_id)
        qid, mid = str(row.query_customer_id_hash or ""), str(row.match_customer_id_hash or "")
        qname, mname = str(row.query_name_norm or ""), str(row.match_name_norm or "")
        same_group = bool(str(row.query_similar_group or "") and row.query_similar_group == row.match_similar_group)
        renewal_base_pair = (
            str(row.query_edit_type_norm or "") in RENEWAL_EDIT_TYPES
            and str(row.query_base_from_loan_id or "") == mloan
        ) or (
            str(row.match_edit_type_norm or "") in RENEWAL_EDIT_TYPES
            and str(row.match_base_from_loan_id or "") == qloan
        )
        id_match = bool(qid and mid and qid == mid)
        id_conflict = bool(qid and mid and qid != mid)
        name_match = bool(qname and mname and qname == mname)
        dhash_similarity = hamming_similarity(query.dhash, match.dhash)
        edge_dhash_similarity = hamming_similarity(query.edge_dhash, match.edge_dhash)
        rotated_dhash_similarity = max(
            [hamming_similarity(candidate, match.dhash) for candidate in query.rotated_dhashes]
            + [hamming_similarity(query.dhash, candidate) for candidate in match.rotated_dhashes]
            + [dhash_similarity]
        )
        rotated_edge_dhash_similarity = max(
            [hamming_similarity(candidate, match.edge_dhash) for candidate in query.rotated_edge_dhashes]
            + [hamming_similarity(query.edge_dhash, candidate) for candidate in match.rotated_edge_dhashes]
            + [edge_dhash_similarity]
        )
        feature_rows.append(
            {
                "query_loan_id": qloan,
                "match_loan_id": mloan,
                "rank": int(row.rank),
                "query_path": row.query_path,
                "match_path": row.match_path,
                "global_semantic_similarity": float(row.global_semantic_similarity),
                "id_card_semantic_similarity": cosine(id_embedding[qloan], id_embedding[mloan]) if qloan in id_embedding and mloan in id_embedding else np.nan,
                "subject_region_hist_similarity": hist_similarity(query.center_hist, match.center_hist),
                "background_hist_similarity": hist_similarity(query.background_hist, match.background_hist),
                "local_structure_orb_ratio": orb_match_ratio(query.orb_desc, match.orb_desc),
                "dhash_similarity": dhash_similarity,
                "mirror_local_structure_orb_ratio": max(
                    orb_match_ratio(query.mirror_orb_desc, match.orb_desc),
                    orb_match_ratio(query.orb_desc, match.mirror_orb_desc),
                ),
                "mirror_subject_region_hist_similarity": max(
                    hist_similarity(query.mirror_center_hist, match.center_hist),
                    hist_similarity(query.center_hist, match.mirror_center_hist),
                ),
                "mirror_background_hist_similarity": max(
                    hist_similarity(query.mirror_background_hist, match.background_hist),
                    hist_similarity(query.background_hist, match.mirror_background_hist),
                ),
                "mirror_dhash_similarity": max(
                    hamming_similarity(query.mirror_dhash, match.dhash),
                    hamming_similarity(query.dhash, match.mirror_dhash),
                ),
                "equalized_dhash_similarity": hamming_similarity(query.equalized_dhash, match.equalized_dhash),
                "edge_dhash_similarity": edge_dhash_similarity,
                "edge_hist_similarity": hist_similarity(query.edge_hist, match.edge_hist),
                "rotated_dhash_similarity": rotated_dhash_similarity,
                "rotated_dhash_gain": rotated_dhash_similarity - dhash_similarity,
                "rotated_edge_dhash_similarity": rotated_edge_dhash_similarity,
                "rotated_edge_dhash_gain": rotated_edge_dhash_similarity - edge_dhash_similarity,
                "brightness_delta": abs(query.brightness - match.brightness),
                "contrast_delta": abs(query.contrast - match.contrast),
                "rgb_mean_abs_delta": mean_abs_delta(query.rgb_mean, match.rgb_mean),
                "rgb_mean_euclidean_delta": euclidean_delta(query.rgb_mean, match.rgb_mean),
                "lab_mean_abs_delta": mean_abs_delta(query.lab_mean, match.lab_mean),
                "lab_delta_e": euclidean_delta(query.lab_mean, match.lab_mean),
                "lab_delta_e2000": ciede2000_delta(query.lab_mean, match.lab_mean),
                "hsv_mean_abs_delta": hsv_mean_abs_delta(query.hsv_mean, match.hsv_mean),
                "hsv_hist_similarity": hist_similarity(query.hsv_hist, match.hsv_hist),
                "blur_ratio": min(query.blur, match.blur) / max(query.blur, match.blur, 1e-6),
                "name_match": name_match,
                "id_match": id_match,
                "id_conflict": id_conflict,
                "same_similar_group": same_group,
                "renewal_base_pair": renewal_base_pair,
                "same_iddd_pair": str(row.query_same_iddd) == "1" or str(row.match_same_iddd) == "1",
                "query_edit_type": row.query_edit_type,
                "match_edit_type": row.match_edit_type,
                "query_base_from": row.query_base_from,
                "match_base_from": row.match_base_from,
            }
        )

    features = pd.DataFrame(feature_rows)
    feature_columns = [
        "global_semantic_similarity",
        "id_card_semantic_similarity",
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
        "name_match",
        "id_match",
        "id_conflict",
    ]
    pair_label_overrides = load_pair_label_overrides(output_dir)
    features["pair_key"] = [pair_key(a, b) for a, b in zip(features["query_loan_id"], features["match_loan_id"])]
    features["reviewed_pair_label"] = features["pair_key"].map(pair_label_overrides)
    y = (features["same_similar_group"].astype(bool) | features["renewal_base_pair"].astype(bool)).astype(int)
    y.loc[features["reviewed_pair_label"].notna()] = features.loc[
        features["reviewed_pair_label"].notna(), "reviewed_pair_label"
    ].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        features[feature_columns], y, test_size=0.25, random_state=42, stratify=y
    )
    numeric = [column for column in feature_columns if column not in {"name_match", "id_match", "id_conflict"}]
    boolean = ["name_match", "id_match", "id_conflict"]
    preprocess = ColumnTransformer(
        [
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
            ("bool", "passthrough", boolean),
        ]
    )
    if args.model == "histgb":
        classifier = HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.06,
            l2_regularization=0.05,
            random_state=42,
        )
    else:
        classifier = LogisticRegression(max_iter=1000, class_weight="balanced")
    model = Pipeline([("preprocess", preprocess), ("classifier", classifier)])
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(features[feature_columns])[:, 1]
    test_probabilities = model.predict_proba(X_test)[:, 1]
    test_predictions = (test_probabilities >= args.probability_threshold).astype(int)
    features["similar_probability"] = probabilities
    features["predicted_similar"] = features["similar_probability"] >= args.probability_threshold
    features["predicted_fraud_type"] = features.apply(fraud_type, axis=1, probability_threshold=args.probability_threshold)
    features["table_fraud_type"] = np.where(
        features["renewal_base_pair"],
        "same_customer_renewal_or_repeat",
        np.where(
            ~features["same_similar_group"],
            "not_labeled_similar",
            np.where(features["id_match"] | features["same_iddd_pair"], "same_customer_renewal_or_repeat", "cross_customer_fraud"),
        ),
    )

    report_path = output_dir / "pair_evidence_model_report.csv"
    features.to_csv(report_path, index=False, encoding="utf-8-sig")

    precision, recall, f1, _ = precision_recall_fscore_support(y_test, test_predictions, average="binary", zero_division=0)
    summary = {
        "rows": int(len(features)),
        "positive_pairs": int(y.sum()),
        "renewal_base_pairs": int(features["renewal_base_pair"].sum()),
        "label_definition": "same_similar_group OR renewal_base_pair, overridden by outputs/mvp/stage1_review.csv pair labels when present",
        "reviewed_pair_label_rows": int(features["reviewed_pair_label"].notna().sum()),
        "reviewed_pair_label_positive_rows": int(features["reviewed_pair_label"].eq(1).sum()),
        "reviewed_pair_label_negative_rows": int(features["reviewed_pair_label"].eq(0).sum()),
        "reviewed_pair_label_unique_pairs": int(len(pair_label_overrides)),
        "model": args.model,
        "probability_threshold": args.probability_threshold,
        "test_metrics": {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "roc_auc": float(roc_auc_score(y_test, test_probabilities)),
        },
        "predicted_fraud_type_counts": dict(Counter(features["predicted_fraud_type"])),
        "table_fraud_type_counts": dict(Counter(features["table_fraud_type"])),
        "feature_columns": feature_columns,
        "classification_report": classification_report(y_test, test_predictions, output_dict=True, zero_division=0),
        "outputs": [str(report_path), str(output_dir / "pair_evidence_model_summary.json")],
    }
    summary_path = output_dir / "pair_evidence_model_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
