#!/usr/bin/env python3
"""补录语料中未入库的贷款：FAISS 索引里有影像但 loans 表无记录的目录。

背景：语料 loan_001~ 等 74 个目录在 FAISS 面签索引中，但从未录入 loans 表，
检测命中它们时无法映射业务号、页面影像 404。本脚本从 annotations.csv 取
业务号（个人身份字段缺失则留空/NULL），提取面签特征后插入 loans 表，
并按 0.97 阈值对全库特征重新聚类（相似→F+组号，其余→R）。

用法：python backfill_loans.py            # 补录 + 重聚类
      python backfill_loans.py --dry-run  # 只看待补录清单
"""
from __future__ import annotations

import csv
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "web1"))
import model_bridge  # noqa: E402

DATA_DIR = ROOT.parent / "jinchuang_v4" / "extracted" / "data"
ANNOTATIONS = DATA_DIR / "annotations.csv"
HIGH_SIM_THRESHOLD = 0.97


def detect_db_path() -> Path:
    local = ROOT / "data.db"
    if local.exists():
        return local
    shared = ROOT.parent / "web1" / "data.db"
    return shared if shared.exists() else local


def main(dry_run: bool = False):
    db_path = detect_db_path()
    with sqlite3.connect(db_path) as conn:
        known_dirs = {r[0] for r in conn.execute(
            "select image_dir from loans where image_dir is not null and image_dir!=''")}
        known_ids = {r[0] for r in conn.execute("select loan_id from loans")}

        with open(ANNOTATIONS, "r", encoding="utf-8-sig", newline="") as handle:
            pending = [
                row for row in csv.DictReader(handle)
                if row["file_path"].split("/")[0] not in known_dirs
                and row["loan_id"] not in known_ids
            ]
        print(f"待补录贷款: {len(pending)} 笔（语料有影像、loans 表无记录）")
        if not pending:
            return
        if dry_run:
            for row in pending[:10]:
                print("  ", row["loan_id"], row["file_path"])
            return

        now = datetime.now().astimezone().isoformat(timespec="seconds")
        inserted = 0
        for row in pending:
            image_dir = row["file_path"].split("/")[0]
            # 个人身份/合同字段缺失时用空串或 NULL（表允许），核心是业务号+影像目录+特征
            conn.execute("""
                insert into loans(loan_id,customer_id,business_type,auto_group,verify_status,
                    status,image_dir,face_feature,party_a,loan_amount,loan_term,
                    loan_purpose,repayment_method,created_at)
                values(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                row["loan_id"], (row["身份证号"].strip() or None), "首贷", "", "N",
                "正常", image_dir, None, row["甲方名称"].strip(), row["借款金额"].strip(),
                row["借款期限(月)"].strip(), row["借款用途"].strip(),
                row["还款方式"].strip(), now,
            ))
            inserted += 1
        conn.commit()
        print(f"已插入 loans: {inserted} 笔（verify_status=N，待聚类判定）")

        # 提取面签特征（语料每贷款一张 face_signing.jpg）
        missing_feature = conn.execute(
            "select loan_id, image_dir from loans where face_feature is null"
        ).fetchall()
        print(f"待提取特征: {len(missing_feature)} 笔")
        n_feat = 0
        for loan_id, image_dir in missing_feature:
            path = DATA_DIR / Path(image_dir).name / "face_signing.jpg"
            if path.is_file():
                feat = model_bridge.extract_feature_from_path(str(path))
                if feat is not None:
                    conn.execute(
                        "update loans set face_feature=? where loan_id=?", (feat, loan_id))
                    n_feat += 1
        conn.commit()
        print(f"已提取特征: {n_feat} 笔")

        # 全库特征按 0.97 重新聚类：相似簇→F+组号，孤立→R（人工结论 C/B 不动）
        features = {r[0]: r[1] for r in conn.execute(
            "select loan_id, face_feature from loans "
            "where face_feature is not null and length(face_feature)>0") if r[1]}
        cur_status = {r[0]: (r[1] or "N") for r in conn.execute(
            "select loan_id, verify_status from loans")}
        cur_group = {r[0]: (r[1] or "") for r in conn.execute(
            "select loan_id, auto_group from loans")}
        assignments = model_bridge.compute_similar_groups(
            features, cur_status=cur_status, cur_group=cur_group,
            threshold=HIGH_SIM_THRESHOLD)
        for loan_id, (status, group) in assignments.items():
            conn.execute(
                "update loans set verify_status=?, auto_group=? where loan_id=?",
                (status, group, loan_id))
        conn.commit()
        n_f = sum(1 for s, _ in assignments.values() if s == "F")
        n_r = sum(1 for s, _ in assignments.values() if s == "R")
        groups = len({g for _, g in assignments.values() if g})
        total = conn.execute("select count(*) from loans").fetchone()[0]
        dist = dict(conn.execute(
            "select verify_status, count(*) from loans group by verify_status").fetchall())
        print(f"重聚类完成: 相似(F)={n_f} 正常(R)={n_r} 相似组={groups}")
        print(f"loans 总数: {total} | verify_status 分布: {dist}")


if __name__ == "__main__":
    main(dry_run="--dry-run" in sys.argv)
