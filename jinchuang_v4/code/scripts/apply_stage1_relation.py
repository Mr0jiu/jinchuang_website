from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path("/workspace/jinchuang_v4/code")
OUTPUT = ROOT / "outputs/mvp"
ANNOTATIONS = OUTPUT / "annotations_for_v4_reports.csv"
PAIR_REPORT = OUTPUT / "pair_evidence_model_report.csv"
STAGE1 = OUTPUT / "stage1_similarity_report.csv"
STAGE2 = OUTPUT / "stage2_fraud_type_report.csv"
SUMMARY = OUTPUT / "two_stage_summary.json"
BACKUP_SUFFIX = ".bak_before_stage1_relation_20260802"

FEATURES = [
    "global_semantic_similarity", "subject_region_hist_similarity", "background_hist_similarity",
    "local_structure_orb_ratio", "dhash_similarity", "mirror_local_structure_orb_ratio",
    "mirror_subject_region_hist_similarity", "mirror_background_hist_similarity",
    "mirror_dhash_similarity", "equalized_dhash_similarity", "edge_dhash_similarity",
    "edge_hist_similarity", "rotated_dhash_similarity", "rotated_dhash_gain",
    "rotated_edge_dhash_similarity", "rotated_edge_dhash_gain", "brightness_delta",
    "contrast_delta", "rgb_mean_abs_delta", "rgb_mean_euclidean_delta", "lab_mean_abs_delta",
    "lab_delta_e", "lab_delta_e2000", "hsv_mean_abs_delta", "hsv_hist_similarity", "blur_ratio",
]


def pair_key(left: object, right: object) -> str:
    return "|".join(sorted((str(left), str(right))))


def backup(path: Path) -> None:
    target = Path(str(path) + BACKUP_SUFFIX)
    if path.exists() and not target.exists():
        shutil.copy2(path, target)


annotations = pd.read_csv(ANNOTATIONS, dtype=str, encoding="utf-8-sig").fillna("")
annotations["dataset_loan_id"] = annotations["file_path"].str.replace("\\", "/", regex=False).str.split("/").str[0]
fraud_type_map = annotations.drop_duplicates("dataset_loan_id").set_index("dataset_loan_id")["fraud_type"].to_dict()

pairs = pd.read_csv(PAIR_REPORT, low_memory=False)
pairs["query_fraud_type"] = pairs["query_loan_id"].map(fraud_type_map).fillna("").str.upper()
pairs["match_fraud_type"] = pairs["match_loan_id"].map(fraud_type_map).fillna("").str.upper()
renewal_label = pairs["query_fraud_type"].isin({"C", "T"}) | pairs["match_fraud_type"].isin({"C", "T"})
repeat_label = pairs["query_fraud_type"].eq("F") | pairs["match_fraud_type"].eq("F")
labeled = pairs[renewal_label ^ repeat_label].copy()
labeled["visual_renewal_label"] = renewal_label.loc[labeled.index].astype(int)

preprocess = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), FEATURES)
])
model = Pipeline([
    ("preprocess", preprocess),
    ("classifier", LogisticRegression(class_weight="balanced", max_iter=500, random_state=42)),
])
model.fit(labeled[FEATURES], labeled["visual_renewal_label"])

stage1 = pd.read_csv(STAGE1, low_memory=False)
stage1["pair_key"] = [pair_key(a, b) for a, b in zip(stage1["query_loan_id"], stage1["match_loan_id"])]
stage1["stage1_renewal_visual_probability"] = model.predict_proba(stage1[FEATURES])[:, 1]
stage1["stage1_visual_relation"] = np.where(
    stage1["stage1_renewal_visual_probability"].ge(0.5),
    "changed_hair_clothes_background",
    "direct_copy_or_light_manipulation",
)
stage1["stage1_visual_renewal_pair"] = stage1["stage1_visual_relation"].eq("changed_hair_clothes_background")

stage2 = pd.read_csv(STAGE2, low_memory=False)
stage2["pair_key"] = [pair_key(a, b) for a, b in zip(stage2["query_loan_id"], stage2["match_loan_id"])]
relation = stage1[["pair_key", "stage1_visual_relation", "stage1_renewal_visual_probability", "stage1_visual_renewal_pair"]].drop_duplicates("pair_key")
stage2 = stage2.drop(columns=[c for c in relation.columns if c != "pair_key" and c in stage2.columns]).merge(relation, on="pair_key", how="left")
same_customer_repeat = stage2["stage2_predicted_type"].eq("same_customer_repeat_review")
visual_renewal = stage2["stage1_visual_renewal_pair"].fillna(False).astype(bool)
stage2.loc[same_customer_repeat & visual_renewal, "stage2_predicted_type"] = "normal_renewal_similarity"
stage2["stage2_relation_reason"] = np.where(
    same_customer_repeat & visual_renewal,
    "uses_stage1_changed_hair_clothes_background",
    np.where(same_customer_repeat, "uses_stage1_direct_copy_or_light_manipulation", "identity_or_cross_customer_rule"),
)

for path in (STAGE1, STAGE2, SUMMARY):
    backup(path)
stage1.drop(columns=["pair_key"]).to_csv(STAGE1, index=False, encoding="utf-8-sig")
stage2.drop(columns=["pair_key"]).to_csv(STAGE2, index=False, encoding="utf-8-sig")

summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
summary.setdefault("stage1", {})["reuse_vs_renewal_visual_classifier"] = {
    "method": "logistic_regression_image_only",
    "training_rows": int(len(labeled)),
    "training_class_counts": {str(k): int(v) for k, v in Counter(labeled["visual_renewal_label"]).items()},
    "threshold": 0.5,
    "output_field": "stage1_visual_relation",
}
summary.setdefault("stage2", {})["uses_stage1_visual_relation"] = True
summary["stage2"]["predicted_type_counts"] = {str(k): int(v) for k, v in Counter(stage2["stage2_predicted_type"]).items()}
SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

print(json.dumps({
    "stage1_rows": len(stage1),
    "stage1_relation_counts": dict(Counter(stage1["stage1_visual_relation"])),
    "stage2_rows": len(stage2),
    "stage2_type_counts": dict(Counter(stage2["stage2_predicted_type"])),
}, ensure_ascii=False, indent=2))
