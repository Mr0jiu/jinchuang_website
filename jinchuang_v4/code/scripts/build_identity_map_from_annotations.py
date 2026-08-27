"""Create a loan-to-customer hash map from annotation identity fields.

Raw ID-card numbers are read from the local annotation file, hashed with
IDENTITY_HASH_SALT, and are never written to disk.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.identity_resolver import extract_id_card_number, identity_hash


def infer_dataset_loan_id(frame: pd.DataFrame) -> pd.Series:
    if "file_path" in frame.columns:
        return frame["file_path"].fillna("").astype(str).str.replace("\\", "/", regex=False).str.split("/").str[0]
    return pd.Series([""] * len(frame), index=frame.index)


def choose_identity_column(frame: pd.DataFrame, explicit: str | None) -> str:
    if explicit:
        if explicit not in frame.columns:
            raise ValueError(f"Identity column not found: {explicit}")
        return explicit

    scores: list[tuple[int, str]] = []
    for column in frame.columns:
        values = frame[column].fillna("").astype(str)
        matches = values.map(lambda value: extract_id_card_number(value, allow_format_only=True) is not None)
        scores.append((int(matches.sum()), column))
    scores.sort(reverse=True)
    if not scores or scores[0][0] == 0:
        raise ValueError("No ID-card-like column found in annotations.")
    return scores[0][1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build customer identity hashes from an annotations CSV.")
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--identity-column", default="")
    parser.add_argument("--allow-format-only-id", action="store_true")
    args = parser.parse_args()

    frame = pd.read_csv(args.annotations, dtype=str).fillna("")
    identity_column = choose_identity_column(frame, args.identity_column or None)
    dataset_loan_ids = infer_dataset_loan_id(frame)
    business_loan_ids = frame["loan_id"].fillna("").astype(str) if "loan_id" in frame.columns else pd.Series([""] * len(frame), index=frame.index)

    records: dict[str, dict[str, str]] = {}
    conflicts: Counter[str] = Counter()
    for dataset_loan_id, business_loan_id, raw_value in zip(dataset_loan_ids, business_loan_ids, frame[identity_column].fillna("").astype(str)):
        dataset_loan_id = str(dataset_loan_id or "").strip()
        if not dataset_loan_id:
            continue
        strict_number = extract_id_card_number(raw_value)
        format_number = extract_id_card_number(raw_value, allow_format_only=args.allow_format_only_id)
        number = strict_number or format_number
        status = "matched" if strict_number else "matched_format_only" if format_number else "id_number_not_found"
        customer_id_hash = identity_hash(number) if number else ""
        next_record = {
            "dataset_loan_id": dataset_loan_id,
            "business_loan_id": str(business_loan_id or ""),
            "customer_id_hash": customer_id_hash,
            "status": status,
            "source": f"annotations:{identity_column}",
        }
        existing = records.get(dataset_loan_id)
        if existing and existing["customer_id_hash"] and customer_id_hash and existing["customer_id_hash"] != customer_id_hash:
            conflicts[dataset_loan_id] += 1
            existing["status"] = "conflicting_identity_values"
            continue
        if not existing or (customer_id_hash and not existing["customer_id_hash"]):
            records[dataset_loan_id] = next_record

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["dataset_loan_id", "business_loan_id", "customer_id_hash", "status", "source"]
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records[key] for key in sorted(records))

    status_counts = Counter(record["status"] for record in records.values())
    print(f"identity_column={identity_column}")
    print(f"wrote={len(records)} output={output}")
    print("status_counts=" + ", ".join(f"{key}:{value}" for key, value in sorted(status_counts.items())))
    print(f"conflicting_loans={len(conflicts)}")


if __name__ == "__main__":
    main()
