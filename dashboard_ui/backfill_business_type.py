#!/usr/bin/env python3
"""loans 表末尾新增「业务类型」列并按借款用途分配。

分配规则（2026-08-24 与产品确认）：
    商户易贷 —— 借款用途 ∈ {房屋装修, 扩大生产, 教育进修, 企业流动资金}
    锡微贷   —— 其余用途（含用途为空的补录贷款）随机分配
    消费贷   —— 其余用途随机分配（与锡微贷各约 50%）

随机分配使用固定种子，保证多次重跑结果一致；已赋值的行不覆盖（幂等），
只补 NULL，便于后续新增贷款后追加执行。

用法：python backfill_business_type.py           # 建列 + 分配
      python backfill_business_type.py --dry-run # 只看统计不写库
"""
from __future__ import annotations

import random
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def detect_db_path() -> Path:
    local = ROOT / "data.db"
    if local.exists():
        return local
    shared = ROOT.parent / "web1" / "data.db"
    return shared if shared.exists() else local


COLUMN = "业务类型"
MERCHANT_PURPOSES = ("房屋装修", "扩大生产", "教育进修", "企业流动资金")
BIZ_TYPES = ("商户易贷", "锡微贷", "消费贷")
RANDOM_SEED = 20260824  # 固定种子：重跑不改变既有随机分配结果


def main(dry_run: bool = False):
    db_path = detect_db_path()
    rng = random.Random(RANDOM_SEED)
    with sqlite3.connect(db_path) as conn:
        cols = [c[1] for c in conn.execute("pragma table_info(loans)")]
        if COLUMN not in cols:
            if dry_run:
                print(f"[dry-run] 将新增列 loans.{COLUMN}（表末尾）")
            else:
                conn.execute(f'alter table loans add column "{COLUMN}" text')
                conn.commit()
                print(f"已新增列 loans.{COLUMN}（表末尾）")
        else:
            print(f"列 loans.{COLUMN} 已存在")

        # 幂等：只分配尚未赋值的行
        placeholders = ",".join("?" for _ in MERCHANT_PURPOSES)
        todo_merchant = conn.execute(f"""
            select loan_id from loans
            where "{COLUMN}" is null and loan_purpose in ({placeholders})
        """, MERCHANT_PURPOSES).fetchall()
        todo_random = conn.execute(f"""
            select loan_id, loan_purpose from loans
            where "{COLUMN}" is null
              and (loan_purpose is null or loan_purpose = ''
                   or loan_purpose not in ({placeholders}))
        """, MERCHANT_PURPOSES).fetchall()
        print(f"待分配：商户易贷 {len(todo_merchant)} 笔 · 随机(锡微贷/消费贷) {len(todo_random)} 笔")
        if dry_run:
            return

        for (loan_id,) in todo_merchant:
            conn.execute(
                f'update loans set "{COLUMN}"=? where loan_id=?', ("商户易贷", loan_id))
        # 随机分配按 loan_id 排序后进行，配合固定种子保证可复现
        for loan_id, _purpose in sorted(todo_random):
            biz = "锡微贷" if rng.random() < 0.5 else "消费贷"
            conn.execute(
                f'update loans set "{COLUMN}"=? where loan_id=?', (biz, loan_id))
        conn.commit()

        dist = conn.execute(
            f'select "{COLUMN}", count(*) from loans group by "{COLUMN}" order by 2 desc'
        ).fetchall()
        print("业务类型分布:", dict(dist))
        # 交叉核对：商户易贷的用途构成
        cross = conn.execute(f"""
            select loan_purpose, count(*) from loans
            where "{COLUMN}"='商户易贷' group by loan_purpose order by 2 desc
        """).fetchall()
        print("商户易贷用途构成:", dict(cross))


if __name__ == "__main__":
    main(dry_run="--dry-run" in sys.argv)
