"""Combine photo similarity, name match, and ID-card hash match."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd


ID_CARD_PATTERN = re.compile(r"(?<!\d)(\d{17}[0-9Xx])(?![0-9Xx])")
ID_CARD_CONTAINS_PATTERN = re.compile(r"(?<!\d)\d{17}[0-9Xx](?![0-9Xx])")


def dataset_loan_id_from_path(frame: pd.DataFrame) -> pd.Series:
    return frame["file_path"].fillna("").astype(str).str.replace("\\", "/", regex=False).str.split("/").str[0]


def choose_name_column(frame: pd.DataFrame) -> str:
    if "姓名" in frame.columns:
        return "姓名"
    if "base_from" in frame.columns:
        base_index = list(frame.columns).index("base_from")
        if base_index + 1 < len(frame.columns):
            return frame.columns[base_index + 1]
    raise ValueError("Could not infer name column.")


def choose_id_column(frame: pd.DataFrame) -> str:
    if "身份证号" in frame.columns:
        return "身份证号"
    best_column = ""
    best_count = 0
    for column in frame.columns:
        count = int(frame[column].fillna("").astype(str).map(lambda value: bool(ID_CARD_PATTERN.search(value))).sum())
        if count > best_count:
            best_column = column
            best_count = count
    if not best_column:
        raise ValueError("Could not infer ID-card column.")
    return best_column


def normalize_name(value: object) -> str:
    return re.sub(r"\s+", "", str(value or "")).strip()


def decision(row: pd.Series, photo_threshold: float) -> str:
    high_photo = float(row["cosine_similarity"]) >= photo_threshold
    if not high_photo:
        return "low_photo_similarity"
    if bool(row["id_match"]):
        return "same_customer_high_photo"
    if bool(row["id_conflict"]) and bool(row["name_match"]):
        return "identity_number_conflict_same_name"
    if bool(row["id_conflict"]):
        return "cross_customer_high_photo"
    if bool(row["name_match"]):
        return "same_name_pending_id"
    return "high_photo_pending_identity"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a triple similarity report.")
    parser.add_argument("--annotations", default="data/annotations.csv")
    parser.add_argument("--output-dir", default="outputs/mvp")
    parser.add_argument("--photo-threshold", type=float, default=0.98)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    annotations = pd.read_csv(args.annotations, dtype=str).fillna("")
    identity = pd.read_csv(output_dir / "customer_identity_map_from_annotations.csv", dtype=str).fillna("")
    topk = pd.read_csv(output_dir / "topk_results.csv", dtype=str).fillna("")

    name_column = choose_name_column(annotations)
    id_column = choose_id_column(annotations)
    annotations = annotations.assign(dataset_loan_id=dataset_loan_id_from_path(annotations))
    loan_text = (
        annotations[["dataset_loan_id", "loan_id", name_column, id_column]]
        .rename(columns={"loan_id": "business_loan_id", name_column: "name", id_column: "id_card_number"})
        .drop_duplicates("dataset_loan_id")
    )
    loan_text["name_norm"] = loan_text["name"].map(normalize_name)
    loan_text["has_id_text"] = loan_text["id_card_number"].astype(str).str.contains(ID_CARD_CONTAINS_PATTERN, regex=True)
    loan_text = loan_text.drop(columns=["id_card_number"])
    loan_text = loan_text.merge(identity[["dataset_loan_id", "customer_id_hash", "status"]], on="dataset_loan_id", how="left")

    left = loan_text.add_prefix("query_")
    right = loan_text.add_prefix("match_")
    result = topk.merge(left, left_on="query_loan_id", right_on="query_dataset_loan_id", how="left")
    result = result.merge(right, left_on="match_loan_id", right_on="match_dataset_loan_id", how="left")
    result["cosine_similarity"] = result["cosine_similarity"].astype(float)
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
    result["triple_decision"] = result.apply(decision, axis=1, photo_threshold=args.photo_threshold)

    keep_columns = [
        "query_loan_id",
        "match_loan_id",
        "rank",
        "cosine_similarity",
        "query_name",
        "match_name",
        "name_match",
        "query_customer_id_hash",
        "match_customer_id_hash",
        "query_status",
        "match_status",
        "id_match",
        "id_conflict",
        "triple_decision",
        "query_path",
        "match_path",
    ]
    output_path = output_dir / "triple_similarity_report.csv"
    result[keep_columns].to_csv(output_path, index=False, encoding="utf-8-sig")

    high = result[result["cosine_similarity"] >= args.photo_threshold]
    summary = {
        "photo_threshold": args.photo_threshold,
        "rows": int(len(result)),
        "high_photo_rows": int(len(high)),
        "name_column": name_column,
        "id_column": id_column,
        "decision_counts": dict(Counter(result["triple_decision"])),
        "high_photo_decision_counts": dict(Counter(high["triple_decision"])),
        "high_photo_name_match": int(high["name_match"].sum()),
        "high_photo_id_match": int(high["id_match"].sum()),
        "high_photo_id_conflict": int(high["id_conflict"].sum()),
        "outputs": [str(output_path), str(output_dir / "triple_similarity_summary.json")],
    }
    summary_path = output_dir / "triple_similarity_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
