"""Sweep face-similarity thresholds for the multi-dimensional report."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep multi-dimensional similarity thresholds.")
    parser.add_argument("--annotations", default="data/annotations.csv")
    parser.add_argument("--output-dir", default="outputs/mvp")
    parser.add_argument("--thresholds", nargs="*", type=float, default=[0.98, 0.985, 0.99, 0.995, 0.998])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    rows = []
    for threshold in args.thresholds:
        subprocess.run(
            [
                sys.executable,
                "scripts/build_multidim_similarity_report.py",
                "--annotations",
                args.annotations,
                "--output-dir",
                args.output_dir,
                "--face-threshold",
                str(threshold),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        summary = json.loads((output_dir / "multidim_similarity_summary.json").read_text(encoding="utf-8"))
        similar = summary["similar_group_detection_metrics"]
        cross = summary["cross_customer_fraud_metrics"]
        rows.append(
            {
                "face_threshold": threshold,
                "similar_precision": similar["precision"],
                "similar_recall": similar["recall"],
                "similar_f1": similar["f1"],
                "similar_tp": similar["tp"],
                "similar_fp": similar["fp"],
                "similar_fn": similar["fn"],
                "cross_fraud_precision": cross["precision"],
                "cross_fraud_recall": cross["recall"],
                "cross_fraud_f1": cross["f1"],
                "cross_fraud_tp": cross["tp"],
                "cross_fraud_fp": cross["fp"],
                "cross_fraud_fn": cross["fn"],
            }
        )

    result = pd.DataFrame(rows)
    sweep_path = output_dir / "multidim_threshold_sweep.csv"
    result.to_csv(sweep_path, index=False, encoding="utf-8-sig")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
