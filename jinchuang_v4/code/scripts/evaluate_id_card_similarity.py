"""Run a simple ID-card-front embedding similarity baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ID-card-front embedding nearest neighbors.")
    parser.add_argument("--output-dir", default="outputs/mvp")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    manifest = pd.read_csv(output_dir / "data_manifest.csv")
    embeddings = np.load(output_dir / "image_embeddings.npy")
    identity = pd.read_csv(output_dir / "customer_identity_map_from_annotations.csv", dtype=str).fillna("")
    id_map = identity.set_index("dataset_loan_id")["customer_id_hash"].to_dict()
    status_map = identity.set_index("dataset_loan_id")["status"].to_dict()

    subset = manifest[manifest["image_type"].astype(str).eq("id_card_front")].reset_index()
    subset = subset.rename(columns={"index": "embedding_index"})
    vectors = embeddings[subset["embedding_index"].to_numpy()].astype("float32")
    vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12)

    nn = NearestNeighbors(n_neighbors=args.top_k + 1, metric="cosine")
    nn.fit(vectors)
    distances, indices = nn.kneighbors(vectors)

    rows = []
    for query_index, query_row in subset.iterrows():
        query_loan = str(query_row["loan_id"])
        query_customer = id_map.get(query_loan, "")
        for rank, match_index in enumerate(indices[query_index][1:], start=1):
            match_row = subset.iloc[int(match_index)]
            match_loan = str(match_row["loan_id"])
            match_customer = id_map.get(match_loan, "")
            relation = "unknown"
            if query_customer and match_customer:
                relation = "same_customer" if query_customer == match_customer else "cross_customer"
            rows.append(
                {
                    "query_loan_id": query_loan,
                    "query_path": query_row["path"],
                    "match_loan_id": match_loan,
                    "match_path": match_row["path"],
                    "rank": rank,
                    "cosine_similarity": 1.0 - float(distances[query_index][rank]),
                    "query_customer_id": query_customer,
                    "match_customer_id": match_customer,
                    "query_customer_id_status": status_map.get(query_loan, ""),
                    "match_customer_id_status": status_map.get(match_loan, ""),
                    "customer_relation": relation,
                }
            )

    result = pd.DataFrame(rows)
    result_path = output_dir / "id_card_front_topk_with_identity.csv"
    result.to_csv(result_path, index=False, encoding="utf-8-sig")

    summary = {
        "image_type": "id_card_front",
        "items": int(len(subset)),
        "topk_rows": int(len(result)),
        "relation_counts": result["customer_relation"].value_counts().to_dict(),
        "top1_relation_counts": result[result["rank"].eq(1)]["customer_relation"].value_counts().to_dict(),
        "at_0_95": result[result["cosine_similarity"] >= 0.95]["customer_relation"].value_counts().to_dict(),
        "at_0_98": result[result["cosine_similarity"] >= 0.98]["customer_relation"].value_counts().to_dict(),
        "same_customer_similarity": result[result["customer_relation"].eq("same_customer")]["cosine_similarity"].describe().to_dict(),
        "cross_customer_similarity": result[result["customer_relation"].eq("cross_customer")]["cosine_similarity"].describe().to_dict(),
        "outputs": [str(result_path), str(output_dir / "id_card_front_similarity_summary.json")],
    }
    summary_path = output_dir / "id_card_front_similarity_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
