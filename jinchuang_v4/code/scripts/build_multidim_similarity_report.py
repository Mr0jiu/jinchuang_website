"""Build a multi-dimensional similarity and fraud-type report.

Dimensions:
- face-signing image embedding similarity
- ID-card-front image embedding similarity
- name match
- ID-card hash match/conflict
- table ground truth from similar_group and same_iddd/customer hash
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ID_CARD_PATTERN = re.compile(r"(?<!\d)(\d{17}[0-9Xx])(?![0-9Xx])")
RENEWAL_EDIT_TYPES = {"bg", "hair", "shirt", "shirt_bg", "background", "clothes", "background_change", "hair_change", "clothes_change"}


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
    counts = {
        column: int(frame[column].fillna("").astype(str).map(lambda value: bool(ID_CARD_PATTERN.search(value))).sum())
        for column in frame.columns
    }
    best = max(counts, key=counts.get)
    if counts[best] == 0:
        raise ValueError("Could not infer ID-card column")
    return best


def dataset_loan_id_from_path(frame: pd.DataFrame) -> pd.Series:
    return frame["file_path"].fillna("").astype(str).str.replace("\\", "/", regex=False).str.split("/").str[0]


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


def base_loan_id_from_path(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.replace("\\", "/", regex=False).str.split("/").str[0]


def relation_type(row: pd.Series) -> str:
    if row["id_match"]:
        return "same_customer"
    if row["id_conflict"]:
        return "cross_customer"
    return "unknown_customer"


def predicted_fraud_type(row: pd.Series, face_threshold: float) -> str:
    if float(row["face_similarity"]) < face_threshold:
        return "not_suspicious"
    if row["renewal_base_pair"]:
        return "same_customer_renewal_or_repeat"
    if row["id_conflict"] and row["name_match"]:
        return "same_name_cross_id_fraud"
    if row["id_conflict"]:
        return "cross_customer_fraud"
    if row["id_match"]:
        return "same_customer_renewal_or_repeat"
    if row["name_match"]:
        return "same_name_pending_identity"
    return "high_similarity_pending_identity"


def table_fraud_type(row: pd.Series) -> str:
    if row["renewal_base_pair"]:
        return "same_customer_renewal_or_repeat"
    if not row["same_similar_group"]:
        return "not_labeled_similar"
    if row["id_match"] or row["same_iddd_pair"]:
        return "same_customer_renewal_or_repeat"
    if row["id_conflict"]:
        return "cross_customer_fraud"
    return "labeled_similar_unknown_identity"


def precision_recall_f1(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build multi-dimensional similarity report.")
    parser.add_argument("--annotations", default="data/annotations.csv")
    parser.add_argument("--output-dir", default="outputs/mvp")
    parser.add_argument("--face-threshold", type=float, default=0.98)
    parser.add_argument("--id-card-threshold", type=float, default=0.9995)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    annotations = pd.read_csv(args.annotations, dtype=str).fillna("")
    manifest = pd.read_csv(output_dir / "data_manifest.csv")
    embeddings = np.load(output_dir / "image_embeddings.npy")
    topk = pd.read_csv(output_dir / "topk_results.csv", dtype=str).fillna("")
    identity = pd.read_csv(output_dir / "customer_identity_map_from_annotations.csv", dtype=str).fillna("")

    name_column = choose_name_column(annotations)
    id_column = choose_id_column(annotations)
    annotations = annotations.assign(dataset_loan_id=dataset_loan_id_from_path(annotations))
    columns = ["dataset_loan_id", "loan_id", "similar_group", "is_similar_pair", "same_iddd", "edit_type", "base_from", name_column, id_column]
    meta = (
        collapse_duplicate_metadata(annotations, columns)
        .rename(columns={"loan_id": "business_loan_id", name_column: "name", id_column: "id_card_number"})
    )
    meta["name_norm"] = meta["name"].map(normalize_name)
    meta["edit_type_norm"] = meta["edit_type"].fillna("").astype(str).str.lower()
    meta["base_from_loan_id"] = base_loan_id_from_path(meta["base_from"])
    meta["has_id_text"] = meta["id_card_number"].astype(str).map(lambda value: bool(ID_CARD_PATTERN.search(value)))
    meta = meta.drop(columns=["id_card_number"])
    meta = meta.merge(identity[["dataset_loan_id", "customer_id_hash", "status"]], on="dataset_loan_id", how="left")

    id_manifest = manifest[manifest["image_type"].astype(str).eq("id_card_front")].reset_index().rename(columns={"index": "embedding_index"})
    id_vectors: dict[str, np.ndarray] = {}
    for row in id_manifest.itertuples(index=False):
        vector = embeddings[int(row.embedding_index)].astype("float32")
        id_vectors[str(row.loan_id)] = vector / (np.linalg.norm(vector) + 1e-12)

    left = meta.add_prefix("query_")
    right = meta.add_prefix("match_")
    result = topk.rename(columns={"cosine_similarity": "face_similarity"})
    result = result.merge(left, left_on="query_loan_id", right_on="query_dataset_loan_id", how="left")
    result = result.merge(right, left_on="match_loan_id", right_on="match_dataset_loan_id", how="left")
    result["face_similarity"] = result["face_similarity"].astype(float)

    result["id_card_similarity"] = [
        float(np.dot(id_vectors[q], id_vectors[m])) if q in id_vectors and m in id_vectors else np.nan
        for q, m in zip(result["query_loan_id"].astype(str), result["match_loan_id"].astype(str))
    ]
    result["name_match"] = (
        result["query_name_norm"].fillna("").ne("")
        & result["match_name_norm"].fillna("").ne("")
        & result["query_name_norm"].eq(result["match_name_norm"])
    )
    result["id_match"] = (
        result["query_customer_id_hash"].fillna("").ne("")
        & result["match_customer_id_hash"].fillna("").ne("")
        & result["query_customer_id_hash"].eq(result["match_customer_id_hash"])
    )
    result["id_conflict"] = (
        result["query_customer_id_hash"].fillna("").ne("")
        & result["match_customer_id_hash"].fillna("").ne("")
        & result["query_customer_id_hash"].ne(result["match_customer_id_hash"])
    )
    result["same_similar_group"] = (
        result["query_similar_group"].fillna("").ne("")
        & result["match_similar_group"].fillna("").ne("")
        & result["query_similar_group"].eq(result["match_similar_group"])
    )
    result["renewal_base_pair"] = (
        result["query_edit_type_norm"].isin(RENEWAL_EDIT_TYPES)
        & result["query_base_from_loan_id"].fillna("").astype(str).eq(result["match_loan_id"].astype(str))
    ) | (
        result["match_edit_type_norm"].isin(RENEWAL_EDIT_TYPES)
        & result["match_base_from_loan_id"].fillna("").astype(str).eq(result["query_loan_id"].astype(str))
    )
    result["same_iddd_pair"] = result["query_same_iddd"].astype(str).eq("1") | result["match_same_iddd"].astype(str).eq("1")
    result["customer_relation"] = result.apply(relation_type, axis=1)
    result["predicted_fraud_type"] = result.apply(predicted_fraud_type, axis=1, face_threshold=args.face_threshold)
    result["table_fraud_type"] = result.apply(table_fraud_type, axis=1)
    result["multidim_score"] = (
        0.70 * result["face_similarity"].clip(0, 1)
        + 0.10 * result["id_card_similarity"].fillna(0).clip(0, 1)
        + 0.10 * result["name_match"].astype(float)
        + 0.10 * (result["id_match"] | result["id_conflict"]).astype(float)
    )

    keep_columns = [
        "query_loan_id",
        "match_loan_id",
        "rank",
        "face_similarity",
        "id_card_similarity",
        "multidim_score",
        "query_name",
        "match_name",
        "name_match",
        "query_customer_id_hash",
        "match_customer_id_hash",
        "id_match",
        "id_conflict",
        "customer_relation",
        "same_similar_group",
        "renewal_base_pair",
        "same_iddd_pair",
        "query_edit_type",
        "match_edit_type",
        "query_base_from",
        "match_base_from",
        "predicted_fraud_type",
        "table_fraud_type",
        "query_path",
        "match_path",
    ]
    report_path = output_dir / "multidim_similarity_report.csv"
    result[keep_columns].to_csv(report_path, index=False, encoding="utf-8-sig")

    predicted_similar = result["face_similarity"] >= args.face_threshold
    actual_similar = result["same_similar_group"] | result["renewal_base_pair"]
    similar_metrics = precision_recall_f1(
        int((predicted_similar & actual_similar).sum()),
        int((predicted_similar & ~actual_similar).sum()),
        int((~predicted_similar & actual_similar).sum()),
    )

    predicted_cross_fraud = result["predicted_fraud_type"].isin(["cross_customer_fraud", "same_name_cross_id_fraud"])
    actual_cross_fraud = result["table_fraud_type"].eq("cross_customer_fraud")
    cross_metrics = precision_recall_f1(
        int((predicted_cross_fraud & actual_cross_fraud).sum()),
        int((predicted_cross_fraud & ~actual_cross_fraud).sum()),
        int((~predicted_cross_fraud & actual_cross_fraud).sum()),
    )

    summary = {
        "face_threshold": args.face_threshold,
        "id_card_threshold": args.id_card_threshold,
        "rows": int(len(result)),
        "name_column": name_column,
        "id_column": id_column,
        "renewal_base_pairs": int(result["renewal_base_pair"].sum()),
        "predicted_fraud_type_counts": dict(Counter(result["predicted_fraud_type"])),
        "table_fraud_type_counts": dict(Counter(result["table_fraud_type"])),
        "customer_relation_counts": dict(Counter(result["customer_relation"])),
        "similar_group_detection_metrics": similar_metrics,
        "cross_customer_fraud_metrics": cross_metrics,
        "outputs": [str(report_path), str(output_dir / "multidim_similarity_summary.json")],
    }
    summary_path = output_dir / "multidim_similarity_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
