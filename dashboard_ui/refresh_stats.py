#!/usr/bin/env python3
"""首页指标统计脚本：重算 Tab1 三张缓存表并落库。

表与口径
--------
    dashboard_overview         KPI、面签概况与客户授信结构快照（单行）
    dashboard_similarity_dist  相似度分布折线图（94%~98% 每 0.5 一档，共 9 档，影像级）
    dashboard_loan_behavior    贷款行为饼图（同客户复用/跨客户复用/正常）
    dashboard_purpose_dist     相似交易×业务类型堆叠柱状图（商户易贷/锡微贷/消费贷）

客户授信结构（贷款级）：每笔贷款按客户名下贷款笔数归属——
客户名下 ≥2 笔 → 该客户全部贷款计「同客户多笔授信」，否则「单笔授信」；
总数 = 贷款笔数（customer_id 为空的贷款归入单笔授信）。

统计对象 = FAISS 面签索引全量语料（建索引时仅收面签影像，天然排除身份证/
合同/流水等其他类别）；模型不可用时退回 loans.face_feature 子集。

影像级口径（2026-08-23 修正）
----
相似度分布的每一档 = 「存在至少一条 ≥该档 相似对的面签影像数」，同客户/
跨客户为其互斥细分（同客户优先）。三档数值均 ≤ 面签总数，不再出现
“相似对总数超过面签数量”的对级计数。高相似率 = ≥0.97 档涉及面签占比。

用法
----
    python refresh_stats.py     # 独立重算并打印摘要
页面「刷新数据」按钮 → POST /api/dashboard/refresh → 调用本脚本同一入口。
"""
from __future__ import annotations

import sqlite3
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "web1"))
import model_bridge  # noqa: E402

# 相似度分布折线图档位：94%~98%，每 0.5 个百分点一档（含两端），共 9 档
SIM_THRESHOLDS = (0.94, 0.945, 0.95, 0.955, 0.96, 0.965, 0.97, 0.975, 0.98)
# 高相似基准档：KPI（高相似率/跨客户套用）、贷款行为、涉及订单均按此阈值，
# 独立于柱状图展示档位（展示档位调整不影响业务口径）
HIGH_SIM_THRESHOLD = 0.97
BEHAVIOR_KEYS = ("same_customer_reuse", "cross_customer_reuse", "normal")
# 业务类型（loans.业务类型 列，由 backfill_business_type.py 分配）：
# 商户易贷=指定借款用途，锡微贷/消费贷=其余随机
BIZ_KEYS = ("商户易贷", "锡微贷", "消费贷")
BIZ_COLUMN = "业务类型"


def detect_db_path() -> Path:
    """数据库定位：dashboard_ui/data.db 优先（独立副本），否则沿用 web1 的库。"""
    local = ROOT / "data.db"
    if local.exists():
        return local
    shared = ROOT.parent / "web1" / "data.db"
    return shared if shared.exists() else local


def ensure_tables(conn):
    conn.execute("""
        create table if not exists dashboard_overview(
            id integer primary key check(id=1),
            customers integer not null,
            loans integer not null,
            total_images integer not null,
            face_images integer not null,
            pending_review integer not null,
            involved_loans integer not null,
            feature_loans integer not null,
            total_pairs integer not null,
            high_similar_pairs integer not null,
            high_similar_rate real not null,
            same_customer_pairs integer not null,
            cross_customer_pairs integer not null,
            multi_loan_customers integer not null,
            single_loan_customers integer not null,
            updated_at text not null
        )
    """)
    conn.execute("""
        create table if not exists dashboard_similarity_dist(
            threshold real primary key,
            pair_count integer not null,
            same_customer_pairs integer not null,
            cross_customer_pairs integer not null,
            updated_at text not null
        )
    """)
    conn.execute("""
        create table if not exists dashboard_loan_behavior(
            behavior text primary key,
            loan_count integer not null,
            updated_at text not null
        )
    """)
    conn.execute("""
        create table if not exists dashboard_purpose_dist(
            biz_type text primary key,
            same_count integer not null,
            cross_count integer not null,
            updated_at text not null
        )
    """)
    conn.execute("""
        create table if not exists dashboard_credit_behavior(
            credit text not null,
            behavior text not null,
            loan_count integer not null,
            updated_at text not null,
            primary key(credit, behavior)
        )
    """)
    # 兼容旧库：dashboard_overview 缺客户授信结构列时补列（默认 0，等待重算）
    ov_cols = [r[1] for r in conn.execute("pragma table_info(dashboard_overview)")]
    for _col in ("multi_loan_customers", "single_loan_customers", "normal_multi_loan"):
        if _col not in ov_cols:
            conn.execute(f"alter table dashboard_overview add column {_col} integer not null default 0")


def _load_feature_matrix(conn):
    """loans.face_feature 兜底语料：返回 (L2 归一化矩阵, 客户口径列表)。"""
    rows = conn.execute("""
        select coalesce(customer_id, loan_id), face_feature
        from loans where face_feature is not null and length(face_feature) > 0
    """).fetchall()
    if not rows:
        return None, []
    feats = np.array([np.frombuffer(r[1], dtype=np.float32) for r in rows], dtype=np.float32)
    feats = feats / np.clip(np.linalg.norm(feats, axis=1, keepdims=True), 1e-9, None)
    return feats, [r[0] for r in rows]


def _group_credit_behavior(conn):
    """读 dashboard_credit_behavior，按 credit 分组返回 [(credit, [(behavior,count),...]), ...]。"""
    rows = conn.execute(
        "select credit, behavior, loan_count from dashboard_credit_behavior"
    ).fetchall()
    grouped: dict[str, list] = {"multi": [], "single": []}
    for c, b, n in rows:
        grouped.setdefault(c, []).append((b, n))
    return [(c, cells) for c, cells in grouped.items()]


def read_tables(db_path: Path | None = None) -> dict:
    """读取三张缓存表，组装 /api/stats 的 dashboard 字段；无数据返回 {}。"""
    db_path = db_path or detect_db_path()
    with sqlite3.connect(db_path) as conn:
        ensure_tables(conn)
        row = conn.execute("""
            select customers,loans,total_images,face_images,pending_review,involved_loans,
                   feature_loans,total_pairs,high_similar_pairs,high_similar_rate,
                   same_customer_pairs,cross_customer_pairs,
                   multi_loan_customers,single_loan_customers,normal_multi_loan,updated_at
            from dashboard_overview where id=1
        """).fetchone()
        if not row:
            return {}
        keys = ("customers", "loans", "total_images", "face_images", "pending_review",
                "involved_loans", "feature_loans", "total_pairs", "high_similar_pairs",
                "high_similar_rate", "same_customer_pairs", "cross_customer_pairs",
                "multi_loan_customers", "single_loan_customers", "normal_multi_loan", "updated_at")
        dash = dict(zip(keys, row))
        dash["customer_credit"] = {
            "multi_loan": dash["multi_loan_customers"],
            "single_loan": dash["single_loan_customers"],
        }
        dash["credit_mosaic"] = {
            c: {b: n for b, n in cells}
            for c, cells in _group_credit_behavior(conn)
        }
        dash["similarity_dist"] = [
            {"threshold": r[0], "pair_count": r[1],
             "same_customer_pairs": r[2], "cross_customer_pairs": r[3]}
            for r in conn.execute(
                "select threshold,pair_count,same_customer_pairs,cross_customer_pairs "
                "from dashboard_similarity_dist order by threshold"
            ).fetchall()
        ]
        dash["loan_behavior"] = {
            r[0]: r[1] for r in conn.execute(
                "select behavior,loan_count from dashboard_loan_behavior"
            ).fetchall()
        }
        # 正常贷款中的多笔授信数量（贷款分布饼图芝麻点数据）
        dash["loan_behavior"]["normal_multi_loan"] = dash["normal_multi_loan"]
        dash["purpose_dist"] = [
            {"biz_type": r[0], "same_count": r[1], "cross_count": r[2]}
            for r in conn.execute(
                "select biz_type,same_count,cross_count from dashboard_purpose_dist "
                "order by case biz_type when '商户易贷' then 1 when '锡微贷' then 2 else 3 end"
            ).fetchall()
        ]
        # 全量贷款 × 业务类型（总览「贷款结构」紧凑柱图）：简单 GROUP BY 实时统计，
        # 已知类型按 BIZ_KEYS 顺序返回，未知/未登记类型按数量追加在后
        if BIZ_COLUMN in [c[1] for c in conn.execute("pragma table_info(loans)")]:
            rows = conn.execute(
                f'''select coalesce(nullif("{BIZ_COLUMN}",''),'未登记'), count(*)
                    from loans group by coalesce(nullif("{BIZ_COLUMN}",''),'未登记')'''
            ).fetchall()
        else:
            rows = []
        cnt = {b: c for b, c in rows}
        extras = sorted(((b, c) for b, c in rows if b not in BIZ_KEYS), key=lambda x: -x[1])
        dash["loan_structure"] = [
            {"biz_type": b, "loan_count": cnt.get(b, 0)} for b in BIZ_KEYS
        ] + [{"biz_type": b, "loan_count": c} for b, c in extras]
        return dash


def refresh_dashboard_stats(db_path: Path | None = None) -> dict:
    """重算三张缓存表并写库，返回与页面字段同形的 dict。"""
    db_path = db_path or detect_db_path()
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    with sqlite3.connect(db_path) as conn:
        ensure_tables(conn)
        customers = conn.execute("select count(*) from customers").fetchone()[0]
        loans = conn.execute("select count(*) from loans").fetchone()[0]
        pending = conn.execute("select count(*) from loans where verify_status='F'").fetchone()[0]
        # 客户授信结构（贷款级，总数=贷款笔数）：
        # 每笔贷款按其客户名下贷款笔数归属——客户名下 ≥2 笔 → 该贷款计「同客户多笔授信」，
        # 否则 → 「单笔授信」；customer_id 为空的贷款无法归属多笔客户，归入单笔授信
        cust2cnt: dict[str, int] = {}
        for (cid,) in conn.execute(
            "select customer_id from loans "
            "where customer_id is not null and trim(customer_id) != ''"
        ):
            cust2cnt[cid] = cust2cnt.get(cid, 0) + 1
        credit = {"multi_loan": 0, "single_loan": 0}
        for (cid,) in conn.execute("select customer_id from loans"):
            cid = (cid or "").strip()
            if cid and cust2cnt.get(cid, 0) >= 2:
                credit["multi_loan"] += 1
            else:
                credit["single_loan"] += 1
        # 影像目录(loan_001) → 客户口径/业务号，供同跨客户拆分与涉及贷款统计
        img2cust, img2loan = {}, {}
        for loan_id, image_dir, cust in conn.execute(
            "select loan_id, coalesce(image_dir,''), coalesce(customer_id,loan_id) from loans"
        ):
            if image_dir:
                img2cust[image_dir] = cust
                img2loan[image_dir] = loan_id
        # 业务类型映射（loans.业务类型，backfill_business_type.py 分配）；
        # 列尚未建立时降级为空映射，堆叠图统计自然为 0
        if BIZ_COLUMN in [c[1] for c in conn.execute("pragma table_info(loans)")]:
            loan2biz = {r[0]: (r[1] or "") for r in conn.execute(
                f'select loan_id, "{BIZ_COLUMN}" from loans')}
        else:
            loan2biz = {}

    # 首选 FAISS 全量面签语料；模型不可用时退回库内特征子集（口径降级但可用）
    try:
        vecs, manifest = model_bridge.get_face_corpus()
    except Exception:
        traceback.print_exc()
        vecs, manifest = None, []
    if vecs is None:
        with sqlite3.connect(db_path) as conn:
            feats, cust = _load_feature_matrix(conn)
        vecs, manifest = feats, [{"loan_id": c} for c in (cust or [])]

    n = len(manifest)
    total_pairs = n * (n - 1) // 2
    dist_rows = []
    high_images = same_images = cross_images = 0
    involved = 0
    rate = 0.0
    behavior = {k: 0 for k in BEHAVIOR_KEYS}
    purpose = {b: {"same": 0, "cross": 0} for b in BIZ_KEYS}
    # 客户授信结构 × 行为 马赛克单元格；normal_multi = 正常贷款中多笔授信笔数
    mosaic = {c: {b: 0 for b in BEHAVIOR_KEYS} for c in ("multi", "single")}
    normal_multi = 0
    if n >= 2 and vecs is not None:
        cust_arr = np.array([
            img2cust.get(m.get("loan_id", ""), m.get("loan_id", "")) for m in manifest
        ])
        sims = vecs @ vecs.T
        same = cust_arr[:, None] == cust_arr[None, :]
        # 影像级分布：每档 = 存在 ≥该档 相似对的面签数（同/跨客户互斥细分，同客户优先）
        for th in SIM_THRESHOLDS:
            hit = sims >= th
            np.fill_diagonal(hit, False)
            has_same = (hit & same).any(axis=1)
            has_cross = (hit & ~same).any(axis=1)
            dist_rows.append((
                th, int((has_same | has_cross).sum()),
                int(has_same.sum()), int((has_cross & ~has_same).sum()),
            ))
        # 高相似基准档（0.97）：KPI、贷款行为、涉及订单全部用这一档
        high_row = next((r for r in dist_rows if r[0] == HIGH_SIM_THRESHOLD), dist_rows[-1])
        high_images, same_images, cross_images = high_row[1:]
        hit = sims >= HIGH_SIM_THRESHOLD
        np.fill_diagonal(hit, False)
        has_same = (hit & same).any(axis=1)
        has_cross = (hit & ~same).any(axis=1)
        behavior = {
            "same_customer_reuse": int(has_same.sum()),
            "cross_customer_reuse": int((has_cross & ~has_same).sum()),
            "normal": int((~(has_same | has_cross)).sum()),
        }
        involved_imgs = has_same | has_cross
        rate = float(involved_imgs.sum()) / n if n else 0.0
        involved = len({
            img2loan[m.get("loan_id", "")]
            for i, m in enumerate(manifest)
            if involved_imgs[i] and img2loan.get(m.get("loan_id", ""))
        })
        # 相似交易 × 业务类型（仅可疑贷款：同客户优先，其次跨客户；正常不计入）
        # 客户授信结构 × 行为 马赛克单元格 + 正常贷款中多笔授信数量（芝麻点数据）
        for i, m in enumerate(manifest):
            lid = img2loan.get(m.get("loan_id", ""))
            biz = loan2biz.get(lid) if lid else None
            if biz in purpose:
                if has_same[i]:
                    purpose[biz]["same"] += 1
                elif has_cross[i]:
                    purpose[biz]["cross"] += 1
            beh = "same_customer_reuse" if has_same[i] else \
                ("cross_customer_reuse" if has_cross[i] else "normal")
            cust = img2cust.get(m.get("loan_id", ""))
            cred = "multi" if (cust and cust2cnt.get(cust, 0) >= 2) else "single"
            mosaic[cred][beh] += 1
            if beh == "normal" and cred == "multi":
                normal_multi += 1

    with sqlite3.connect(db_path) as conn:
        ensure_tables(conn)
        conn.execute("delete from dashboard_overview")
        conn.execute("""
            insert into dashboard_overview(id,customers,loans,total_images,face_images,
                pending_review,involved_loans,feature_loans,total_pairs,high_similar_pairs,
                high_similar_rate,same_customer_pairs,cross_customer_pairs,
                multi_loan_customers,single_loan_customers,normal_multi_loan,updated_at)
            values(1,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (customers, loans, loans * 5, n, pending, involved, n, total_pairs,
              high_images, rate, same_images, cross_images,
              credit["multi_loan"], credit["single_loan"], normal_multi, now))
        conn.execute("delete from dashboard_similarity_dist")
        conn.executemany(
            "insert into dashboard_similarity_dist(threshold,pair_count,same_customer_pairs,"
            "cross_customer_pairs,updated_at) values(?,?,?,?,?)",
            [(th, pc, sp, cp, now) for th, pc, sp, cp in dist_rows],
        )
        conn.execute("delete from dashboard_loan_behavior")
        conn.executemany(
            "insert into dashboard_loan_behavior(behavior,loan_count,updated_at) values(?,?,?)",
            [(k, behavior[k], now) for k in BEHAVIOR_KEYS],
        )
        conn.execute("delete from dashboard_purpose_dist")
        conn.executemany(
            "insert into dashboard_purpose_dist(biz_type,same_count,cross_count,updated_at) "
            "values(?,?,?,?)",
            [(b, purpose[b]["same"], purpose[b]["cross"], now) for b in BIZ_KEYS],
        )
        conn.execute("delete from dashboard_credit_behavior")
        conn.executemany(
            "insert into dashboard_credit_behavior(credit,behavior,loan_count,updated_at) "
            "values(?,?,?,?)",
            [(c, b, mosaic[c][b], now)
             for c in ("multi", "single") for b in BEHAVIOR_KEYS],
        )
        conn.commit()
        return read_tables(db_path)


def main():
    dash = refresh_dashboard_stats()
    print("=== 首页指标统计（已写缓存表）===")
    print(f"客户 {dash['customers']} | 贷款 {dash['loans']} | 面签影像 {dash['face_images']}"
          f" | 待复核 {dash['pending_review']} | 涉及贷款 {dash['involved_loans']}")
    print(f"高相似率(≥{HIGH_SIM_THRESHOLD}) {dash['high_similar_rate']*100:.2f}%"
          f" | 同客户 {dash['same_customer_pairs']} | 跨客户 {dash['cross_customer_pairs']}")
    print("客户授信结构:", dash.get("customer_credit"))
    print(f"正常贷款中多笔授信: {dash.get('loan_behavior', {}).get('normal_multi_loan', 0)} 笔")
    print("授信结构 × 行为 马赛克:", dash.get("credit_mosaic"))
    for row in dash["similarity_dist"]:
        print(f"  ≥{row['threshold']:.3f}: {row['pair_count']} 张面签"
              f"（同客户 {row['same_customer_pairs']} · 跨客户 {row['cross_customer_pairs']}）")
    print("贷款行为:", dash["loan_behavior"])
    for row in dash.get("purpose_dist", []):
        print(f"  {row['biz_type']}: 同客户 {row['same_count']} · 跨客户 {row['cross_count']}"
              f" · 合计 {row['same_count'] + row['cross_count']}")


if __name__ == "__main__":
    main()
