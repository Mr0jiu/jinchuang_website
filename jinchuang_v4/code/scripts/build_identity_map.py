"""Create a loan-to-customer hash map from ID-card front images.

Example:
  $env:IDENTITY_HASH_SALT='set-a-secret-in-your-vault'
  python scripts/build_identity_map.py --dataset-root <dataset> --output outputs/mvp/customer_identity_map.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.identity_resolver import extract_id_card_number, identity_hash, ocr_id_card_front


def main() -> None:
    parser = argparse.ArgumentParser(description="Build privacy-preserving customer ID hashes from ID-card front images.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-format-only-id", action="store_true", help="Competition/demo mode: accept an 18-character OCR number even if its official check digit is invalid.")
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()
    root, output = Path(args.dataset_root), Path(args.output)
    rows = []
    image_paths = sorted(root.rglob("id_card_front.*"))
    total = len(image_paths)
    for index, image_path in enumerate(image_paths, start=1):
        loan_id = image_path.parent.name
        try:
            ocr_text = ocr_id_card_front(image_path)
            number = extract_id_card_number(ocr_text, allow_format_only=args.allow_format_only_id)
            if number:
                strict = extract_id_card_number(ocr_text)
                status = "matched" if strict else "matched_format_only"
                rows.append({"dataset_loan_id": loan_id, "customer_id_hash": identity_hash(number), "status": status})
            else:
                rows.append({"dataset_loan_id": loan_id, "customer_id_hash": "", "status": "id_number_not_found"})
        except Exception as exc:
            rows.append({"dataset_loan_id": loan_id, "customer_id_hash": "", "status": f"ocr_failed:{type(exc).__name__}"})
        if args.progress_every and (index == total or index % args.progress_every == 0):
            matched = sum(row["customer_id_hash"] != "" for row in rows)
            print(f"identity: {index}/{total}; matched={matched}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset_loan_id", "customer_id_hash", "status"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} records to {output}; matched={sum(row['status'] == 'matched' for row in rows)}")


if __name__ == "__main__":
    main()
