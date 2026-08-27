from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd

from src.risk_policy import ThresholdPolicy


FRAUD_TYPE_LABELS = {
    "cross_customer_fraud": "cross-customer suspected fraud",
    "same_customer_repeat": "same-customer repeat submission",
    "cross_customer_candidate": "high-similarity candidate pending customer verification",
    "normal_low_risk": "normal low-risk candidate",
}

FRAUD_TYPE_LABELS_ZH = {
    "cross_customer_fraud": "陌生人跨客户疑似欺诈",
    "same_customer_repeat": "同客户重复提交",
    "cross_customer_candidate": "高相似待客户关系核验",
    "normal_low_risk": "低风险候选",
}

RELATION_LABELS_ZH = {
    "self": "同一笔业务",
    "same_customer": "同客户",
    "cross_customer": "陌生人/跨客户",
    "unknown": "客户关系待核验",
}

RECOMMENDED_ACTIONS_ZH = {
    "cross_customer_fraud": "进入反欺诈复核，重点核验身份、面签场景和贷款上下文。",
    "same_customer_repeat": "进入运营/合规复核，确认是否为续贷、补件或重复提交。",
    "cross_customer_candidate": "先通过客户主数据核验是否为不同客户，再决定是否进入反欺诈复核。",
    "normal_low_risk": "保留为审计证据，低优先级抽检。",
}

FRAUD_SCORE_LEVELS_ZH = {
    "critical": "极高",
    "high": "高",
    "medium": "中",
    "low": "低",
}


def pair_key(left: str, right: str) -> str:
    return "::".join(sorted([str(left), str(right)]))


def load_annotations(path: Path | str | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    annotation_path = Path(path)
    if not annotation_path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(annotation_path)
    if "file_path" in frame.columns:
        normalized_paths = frame["file_path"].fillna("").astype(str).str.replace("\\", "/", regex=False)
        frame["dataset_loan_id"] = (
            normalized_paths.str.split("/").str[0]
        )
        if "image_type" not in frame.columns:
            frame["image_type"] = normalized_paths.str.rsplit("/", n=1).str[-1].str.rsplit(".", n=1).str[0]
    return frame


def attach_customer_identity(annotations: pd.DataFrame, identity_map_path: Path | str | None) -> pd.DataFrame:
    """Attach a hashed customer key; raw ID numbers are never accepted here."""
    if annotations.empty or not identity_map_path or not Path(identity_map_path).exists():
        return annotations
    identity_map = pd.read_csv(identity_map_path, dtype=str).fillna("")
    required = {"dataset_loan_id", "customer_id_hash"}
    if not required.issubset(identity_map.columns):
        raise ValueError(f"Identity map must include {sorted(required)}")
    keep = ["dataset_loan_id", "customer_id_hash"] + (["status"] if "status" in identity_map.columns else [])
    identity_map = identity_map[keep].drop_duplicates("dataset_loan_id").rename(columns={"customer_id_hash": "customer_id", "status": "customer_id_status"})
    return annotations.merge(identity_map, on="dataset_loan_id", how="left")


def face_business_frame(annotations: pd.DataFrame) -> pd.DataFrame:
    columns = ["dataset_loan_id", "business_loan_id", "business_type", "customer_id", "customer_id_status", "similar_group", "is_similar_pair"]
    if annotations.empty or "image_type" not in annotations.columns:
        return pd.DataFrame(columns=columns)

    face = annotations[annotations["image_type"].eq("face_signing")].copy()
    if face.empty:
        return pd.DataFrame(columns=columns)
    if "dataset_loan_id" not in face.columns:
        face["dataset_loan_id"] = face.get("loan_id", "").astype(str)

    customer_values = face["customer_id"] if "customer_id" in face.columns else (
        face["customer_no"] if "customer_no" in face.columns else pd.Series("", index=face.index)
    )
    loan_values = face["loan_id"] if "loan_id" in face.columns else pd.Series("", index=face.index)
    business_values = face["business_type"] if "business_type" in face.columns else pd.Series("", index=face.index)
    similar_values = face["similar_group"] if "similar_group" in face.columns else pd.Series("", index=face.index)
    pair_values = face["is_similar_pair"] if "is_similar_pair" in face.columns else pd.Series(0, index=face.index)
    result = pd.DataFrame(
        {
            "dataset_loan_id": face["dataset_loan_id"].astype(str),
            "business_loan_id": loan_values.fillna("").astype(str),
            "business_type": business_values.fillna("").astype(str),
            "customer_id": customer_values.fillna("").astype(str),
            "customer_id_status": face.get("customer_id_status", pd.Series("", index=face.index)).fillna("").astype(str),
            "similar_group": similar_values.fillna("").astype(str),
            "is_similar_pair": pair_values,
        }
    )
    return result.drop_duplicates("dataset_loan_id")


def enrich_topk_with_business(topk: pd.DataFrame, annotations: pd.DataFrame) -> pd.DataFrame:
    enriched = topk.copy()
    enriched["pair_key"] = [pair_key(a, b) for a, b in zip(enriched["query_loan_id"], enriched["match_loan_id"])]
    loans = face_business_frame(annotations)
    if loans.empty:
        for column in (
            "query_business_loan_id",
            "query_business_type",
            "query_customer_id",
            "query_customer_id_status",
            "query_similar_group",
            "match_business_loan_id",
            "match_business_type",
            "match_customer_id",
            "match_customer_id_status",
            "match_similar_group",
        ):
            enriched[column] = ""
        return enriched

    enriched = enriched.merge(
        loans.add_prefix("query_"),
        left_on="query_loan_id",
        right_on="query_dataset_loan_id",
        how="left",
    )
    enriched = enriched.merge(
        loans.add_prefix("match_"),
        left_on="match_loan_id",
        right_on="match_dataset_loan_id",
        how="left",
    )
    for column in (
        "query_business_loan_id",
        "query_business_type",
        "query_customer_id",
        "query_customer_id_status",
        "query_similar_group",
        "match_business_loan_id",
        "match_business_type",
        "match_customer_id",
        "match_customer_id_status",
        "match_similar_group",
    ):
        if column not in enriched.columns:
            enriched[column] = ""
        enriched[column] = enriched[column].fillna("").astype(str)
    return enriched


def infer_customer_relation(row: pd.Series) -> tuple[str, str]:
    if str(row.get("query_loan_id", "")) == str(row.get("match_loan_id", "")):
        return "self", "loan_id"
    query_customer = str(row.get("query_customer_id", "") or "")
    match_customer = str(row.get("match_customer_id", "") or "")
    if query_customer and match_customer:
        weak = "matched_format_only" in {str(row.get("query_customer_id_status", "")), str(row.get("match_customer_id_status", ""))}
        source = "customer_id_format_only" if weak else "customer_id"
        return ("same_customer" if query_customer == match_customer else "cross_customer"), source
    return "unknown", "customer_id_unavailable"


def classify_monitoring_row(row: pd.Series, policy: ThresholdPolicy) -> dict:
    relation, relation_source = infer_customer_relation(row)
    score = float(row["cosine_similarity"])
    if relation == "self":
        threshold = 1.0
    elif policy.enabled and relation == "same_customer":
        threshold = policy.same_customer
    elif policy.enabled:
        threshold = policy.cross_customer
    else:
        threshold = policy.default

    is_suspicious = bool(relation != "self" and score >= threshold)
    # A format-only competition identifier can support triage, but must not be
    # presented as a confirmed cross-customer conclusion.
    if relation == "cross_customer" and is_suspicious and relation_source == "customer_id":
        fraud_type = "cross_customer_fraud"
    elif relation == "cross_customer" and is_suspicious:
        fraud_type = "cross_customer_candidate"
    elif relation == "same_customer" and is_suspicious:
        fraud_type = "same_customer_repeat"
    elif relation == "unknown" and is_suspicious:
        fraud_type = "cross_customer_candidate"
    else:
        fraud_type = "normal_low_risk"

    if is_suspicious and score >= policy.high_risk:
        risk_level = "high"
        priority = "urgent"
    elif is_suspicious and score >= policy.medium_risk:
        risk_level = "medium"
        priority = "standard"
    elif is_suspicious:
        risk_level = "low"
        priority = "low"
    else:
        risk_level = "low"
        priority = "low"

    return {
        "customer_relation": relation,
        "customer_relation_source": relation_source,
        "identity_evidence_level": "strong" if relation_source == "customer_id" else "weak" if relation_source == "customer_id_format_only" else "unavailable",
        "customer_relation_label": RELATION_LABELS_ZH.get(relation, relation),
        "monitor_threshold": threshold,
        "is_suspicious": is_suspicious,
        "fraud_type": fraud_type,
        "fraud_type_label": FRAUD_TYPE_LABELS[fraud_type],
        "fraud_type_label_zh": FRAUD_TYPE_LABELS_ZH[fraud_type],
        "monitor_risk_level": risk_level,
        "review_priority": priority,
        "recommended_action_zh": RECOMMENDED_ACTIONS_ZH[fraud_type],
    }


def _find(parent: dict[str, str], item: str) -> str:
    parent.setdefault(item, item)
    if parent[item] != item:
        parent[item] = _find(parent, parent[item])
    return parent[item]


def _union(parent: dict[str, str], left: str, right: str) -> None:
    left_root = _find(parent, left)
    right_root = _find(parent, right)
    if left_root != right_root:
        parent[right_root] = left_root


def add_fraud_graph_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if result.empty:
        return result

    suspicious = result[result["is_suspicious"].astype(bool)]
    degree = Counter()
    parent: dict[str, str] = {}
    for row in suspicious.itertuples():
        left = str(row.query_loan_id)
        right = str(row.match_loan_id)
        degree[left] += 1
        degree[right] += 1
        _union(parent, left, right)

    component_members: dict[str, set[str]] = {}
    for loan_id in degree:
        root = _find(parent, loan_id)
        component_members.setdefault(root, set()).add(loan_id)

    component_ids = {root: f"RISK_CLUSTER_{index:03d}" for index, root in enumerate(sorted(component_members), start=1)}
    component_sizes = {root: len(members) for root, members in component_members.items()}

    query_degree = []
    match_degree = []
    cluster_id = []
    cluster_size = []
    cross_business_scene = []
    fraud_scores = []
    similarity_components = []
    margin_components = []
    relation_components = []
    business_components = []
    degree_components = []
    cluster_components = []
    score_levels = []
    innovation_tags = []

    for row in result.itertuples():
        query = str(row.query_loan_id)
        match = str(row.match_loan_id)
        query_degree.append(int(degree.get(query, 0)))
        match_degree.append(int(degree.get(match, 0)))

        if bool(row.is_suspicious):
            root = _find(parent, query)
            cluster_id.append(component_ids.get(root, ""))
            cluster_size.append(int(component_sizes.get(root, 1)))
        else:
            cluster_id.append("")
            cluster_size.append(0)

        query_business = str(getattr(row, "query_business_type", "") or "")
        match_business = str(getattr(row, "match_business_type", "") or "")
        is_cross_business = bool(query_business and match_business and query_business != match_business)
        cross_business_scene.append(is_cross_business)

        threshold = float(row.monitor_threshold) if float(row.monitor_threshold) > 0 else 1.0
        normalized_margin = max(0.0, (float(row.cosine_similarity) - threshold) / max(1.0 - threshold, 1e-6))
        relation_bonus = 0.18 if row.customer_relation == "cross_customer" else 0.08 if row.customer_relation == "same_customer" else 0.0
        business_bonus = 0.06 if is_cross_business else 0.0
        graph_bonus = min(0.18, 0.04 * max(degree.get(query, 0), degree.get(match, 0)))
        cluster_bonus = min(0.12, 0.02 * max(cluster_size[-1] - 2, 0))
        # The score is deliberately decomposed into auditable components.  A
        # reviewer can therefore distinguish a very similar pair from a pair
        # whose priority is raised mainly by its business and graph context.
        similarity_component = 0.52 * float(row.cosine_similarity)
        margin_component = 0.24 * normalized_margin
        score = min(1.0, similarity_component + margin_component + relation_bonus + business_bonus + graph_bonus + cluster_bonus)
        if not bool(row.is_suspicious):
            score = min(score, 0.49)
        fraud_scores.append(round(score, 4))
        similarity_components.append(round(similarity_component, 4))
        margin_components.append(round(margin_component, 4))
        relation_components.append(round(relation_bonus, 4))
        business_components.append(round(business_bonus, 4))
        degree_components.append(round(graph_bonus, 4))
        cluster_components.append(round(cluster_bonus, 4))

        if score >= 0.90:
            level = "critical"
        elif score >= 0.78:
            level = "high"
        elif score >= 0.60:
            level = "medium"
        else:
            level = "low"
        score_levels.append(level)

        tags = []
        if row.customer_relation == "cross_customer" and bool(row.is_suspicious):
            tags.append("跨客户高相似")
        if row.customer_relation == "same_customer" and bool(row.is_suspicious):
            tags.append("同客户重复")
        if is_cross_business:
            tags.append("跨产品复用")
        if cluster_size[-1] >= 3:
            tags.append("风险关系簇")
        if max(degree.get(query, 0), degree.get(match, 0)) >= 3:
            tags.append("高连接节点")
        if score >= 0.90:
            tags.append("极高综合欺诈分")
        innovation_tags.append("、".join(tags) if tags else "常规相似候选")

    result["query_risk_degree"] = query_degree
    result["match_risk_degree"] = match_degree
    result["risk_cluster_id"] = cluster_id
    result["risk_cluster_size"] = cluster_size
    result["cross_business_scene"] = cross_business_scene
    result["fraud_score"] = fraud_scores
    result["score_component_similarity"] = similarity_components
    result["score_component_threshold_margin"] = margin_components
    result["score_component_customer_relation"] = relation_components
    result["score_component_cross_product"] = business_components
    result["score_component_node_degree"] = degree_components
    result["score_component_cluster_size"] = cluster_components
    result["fraud_score_level"] = score_levels
    result["fraud_score_level_zh"] = [FRAUD_SCORE_LEVELS_ZH[level] for level in score_levels]
    result["innovation_tags"] = innovation_tags
    return result


def build_fraud_monitoring(topk: pd.DataFrame, annotations: pd.DataFrame, policy: ThresholdPolicy) -> pd.DataFrame:
    if topk.empty:
        return pd.DataFrame()

    enriched = enrich_topk_with_business(topk, annotations)
    enriched = (
        enriched.sort_values(["cosine_similarity", "rank"], ascending=[False, True])
        .drop_duplicates("pair_key")
        .reset_index(drop=True)
    )
    assessments = pd.DataFrame([classify_monitoring_row(row, policy) for _, row in enriched.iterrows()])
    result = pd.concat([enriched, assessments], axis=1)
    result["score_gap_to_threshold"] = result["cosine_similarity"] - result["monitor_threshold"]
    result = add_fraud_graph_features(result)
    return result.sort_values(
        ["is_suspicious", "fraud_score", "monitor_risk_level", "cosine_similarity"],
        ascending=[False, False, True, False],
    ).reset_index(drop=True)


def build_risk_graph(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return presentation-ready risk graph nodes and suspicious similarity edges.

    The graph stays intentionally tabular: it can be loaded by Streamlit,
    NetworkX/Neo4j, or a downstream graph database without coupling the batch
    pipeline to one graph vendor.
    """
    node_columns = ["loan_id", "business_loan_id", "business_type", "similar_group", "risk_degree", "risk_cluster_id", "risk_cluster_size", "max_fraud_score"]
    edge_columns = ["source_loan_id", "target_loan_id", "cosine_similarity", "fraud_type", "fraud_score", "risk_cluster_id", "cross_business_scene", "innovation_tags"]
    if frame.empty or "is_suspicious" not in frame:
        return pd.DataFrame(columns=node_columns), pd.DataFrame(columns=edge_columns)

    suspicious = frame[frame["is_suspicious"].astype(bool)].copy()
    if suspicious.empty:
        return pd.DataFrame(columns=node_columns), pd.DataFrame(columns=edge_columns)

    edges = pd.DataFrame({
        "source_loan_id": suspicious["query_loan_id"].astype(str),
        "target_loan_id": suspicious["match_loan_id"].astype(str),
        "cosine_similarity": suspicious["cosine_similarity"],
        "fraud_type": suspicious["fraud_type"],
        "fraud_score": suspicious["fraud_score"],
        "risk_cluster_id": suspicious["risk_cluster_id"],
        "cross_business_scene": suspicious["cross_business_scene"],
        "innovation_tags": suspicious["innovation_tags"],
    }).drop_duplicates(["source_loan_id", "target_loan_id"])

    node_records: list[dict] = []
    for row in suspicious.itertuples():
        for side in ("query", "match"):
            node_records.append({
                "loan_id": str(getattr(row, f"{side}_loan_id")),
                "business_loan_id": str(getattr(row, f"{side}_business_loan_id", "") or ""),
                "business_type": str(getattr(row, f"{side}_business_type", "") or ""),
                "similar_group": str(getattr(row, f"{side}_similar_group", "") or ""),
                "risk_degree": int(getattr(row, f"{side}_risk_degree", 0)),
                "risk_cluster_id": str(getattr(row, "risk_cluster_id", "") or ""),
                "risk_cluster_size": int(getattr(row, "risk_cluster_size", 0)),
                "max_fraud_score": float(getattr(row, "fraud_score", 0.0)),
            })
    nodes = pd.DataFrame(node_records).groupby("loan_id", as_index=False).agg(
        business_loan_id=("business_loan_id", "first"),
        business_type=("business_type", "first"),
        similar_group=("similar_group", "first"),
        risk_degree=("risk_degree", "max"),
        risk_cluster_id=("risk_cluster_id", "first"),
        risk_cluster_size=("risk_cluster_size", "max"),
        max_fraud_score=("max_fraud_score", "max"),
    )
    return nodes[node_columns], edges[edge_columns]


def summarize_monitoring(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {
            "total_pairs": 0,
            "suspicious_pairs": 0,
            "by_fraud_type": {},
            "by_relation": {},
            "by_priority": {},
        }
    suspicious = frame[frame["is_suspicious"]]
    clusters = suspicious[suspicious.get("risk_cluster_id", "").astype(str).ne("")] if not suspicious.empty else suspicious
    return {
        "total_pairs": int(len(frame)),
        "suspicious_pairs": int(len(suspicious)),
        "by_fraud_type": dict(Counter(suspicious["fraud_type"])),
        "pending_customer_verification": int((suspicious["fraud_type"] == "cross_customer_candidate").sum()),
        "by_relation": dict(Counter(frame["customer_relation"])),
        "by_priority": dict(Counter(suspicious["review_priority"])),
        "by_fraud_score_level": dict(Counter(suspicious.get("fraud_score_level", []))),
        "risk_cluster_count": int(clusters["risk_cluster_id"].nunique()) if not clusters.empty else 0,
        "max_risk_cluster_size": int(clusters["risk_cluster_size"].max()) if not clusters.empty else 0,
        "cross_business_suspicious": int(suspicious.get("cross_business_scene", pd.Series(dtype=bool)).sum()),
        "critical_alerts": int((suspicious.get("fraud_score", pd.Series(dtype=float)) >= 0.90).sum()),
    }


def write_monitoring_outputs(frame: pd.DataFrame, output_dir: Path, name: str = "fraud_monitoring") -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{name}.csv"
    summary_path = output_dir / f"{name}_summary.json"
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    graph_nodes, graph_edges = build_risk_graph(frame)
    graph_nodes.to_csv(output_dir / "risk_graph_nodes.csv", index=False, encoding="utf-8-sig")
    graph_edges.to_csv(output_dir / "risk_graph_edges.csv", index=False, encoding="utf-8-sig")
    summary = summarize_monitoring(frame)
    summary["risk_graph_nodes"] = int(len(graph_nodes))
    summary["risk_graph_edges"] = int(len(graph_edges))
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return summary
