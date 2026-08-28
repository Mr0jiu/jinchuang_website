#!/usr/bin/env python3
"""独立蓝白前端服务：保留原模型桥与数据库，避免 Gradio DOM 干扰布局。"""
from __future__ import annotations

import io
import json
import os
import sqlite3
import csv
import hashlib
import secrets
import sys
import threading
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

import numpy as np

ROOT = Path(__file__).resolve().parent
UI_DIR = ROOT / "modern_ui"
if not (UI_DIR / "index.html").exists():
    UI_DIR = ROOT

# model_bridge.py 与本服务同源（老版网页 web1/ 目录共用一份，避免两份漂移）；
# dashboard_ui 内没有副本时自动把 web1/ 加入 sys.path。
try:
    import model_bridge
except ModuleNotFoundError:
    _WEB1_DIR = ROOT.parent / "web1"
    if not (_WEB1_DIR / "model_bridge.py").is_file():
        raise
    sys.path.insert(0, str(_WEB1_DIR))
    import model_bridge

import evidence_bridge


def _detect_db_path() -> Path:
    """数据库定位：dashboard_ui/data.db 优先（独立副本），否则沿用 web1 的库。"""
    local = ROOT / "data.db"
    if local.exists():
        return local
    shared = ROOT.parent / "web1" / "data.db"
    if shared.exists():
        return shared
    return local


DB_PATH = _detect_db_path()


def _detect_data_dir() -> Path:
    """影像根目录：环境变量 DATA_DIR 优先，否则向上找 jinchuang_v4/extracted/data，
    兜底保留云端 Linux 路径（云端容器内依然命中）。"""
    env = os.environ.get("DATA_DIR", "").strip()
    if env:
        return Path(env)
    cur = ROOT
    for _ in range(6):
        cand = cur / "jinchuang_v4" / "extracted" / "data"
        if cand.is_dir():
            return cand
        if cur.parent == cur:
            break
        cur = cur.parent
    return Path("/workspace/jinchuang_v4/extracted/data")


DATA_DIR = _detect_data_dir()
SAME_CUSTOMER_DIR = ROOT.parent / "outputs" / "same_customer_experiment"
_annotation_customer_cache: dict[str, str] | None = None


def _annotation_customer_names() -> dict[str, str]:
    """Load loan-id/name pairs from the source annotations as a display fallback."""
    global _annotation_customer_cache
    if _annotation_customer_cache is not None:
        return _annotation_customer_cache
    result: dict[str, str] = {}
    for path in (DATA_DIR / "annotations_filtered_with_identity.csv", DATA_DIR / "annotations.csv"):
        if not path.is_file():
            continue
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    loan_id = (row.get("loan_id") or "").strip()
                    name = (row.get("姓名") or "").strip()
                    if loan_id and name and loan_id not in result:
                        result[loan_id] = name
        except (OSError, UnicodeError, csv.Error):
            continue
        if result:
            break
    _annotation_customer_cache = result
    return result


def _fill_annotation_customer_names(rows: list[dict]) -> list[dict]:
    names = _annotation_customer_names()
    for row in rows:
        if not str(row.get("customer_name") or "").strip() or row.get("customer_name") == "未登记":
            row["customer_name"] = names.get(str(row.get("loan_id") or ""), "未登记")
    return rows
# 同客户区分实验产物候选目录：正式 outputs 目录优先，本机缺失时回退 teamfix 归档目录
_SAME_CUSTOMER_CANDIDATES = [SAME_CUSTOMER_DIR, ROOT / "teamfix"]
# Tab2「保存到库」的影像落盘目录：每批上传按保存时间建独立文件夹
SAVED_DIR = ROOT / "saved_images"
OUTPUT_DIRS = [
    ROOT.parent / "jinchuang_v4" / "code" / "outputs" / "mvp",
    ROOT.parent / "outputs" / "mvp",
    ROOT.parent / "submission_ready" / "01_source_code" / "outputs" / "mvp",
    ROOT.parent / "submission_ready" / "03_model_artifacts" / "outputs" / "mvp",
    Path("/workspace/jinchuang_v4/code/outputs/mvp"),
    Path("/workspace/web1/outputs/mvp"),
]
# Stage2 风险类型展示分桶：把多种原始类型归并为 3 个业务口径，供报告页柱状图展示
STAGE2_DISPLAY_TYPES = [
    ("跨客户套用", {"cross_customer_fraud", "same_name_cross_id_fraud", "high_similarity_pending_identity"}),
    ("同客户复用", {"same_customer_repeat_review"}),
    ("同客户多笔授信", {"normal_renewal_similarity"}),
]
IMAGE_FILES = {
    "face": "face_signing.jpg", "id_front": "id_card_front.jpg",
    "id_back": "id_card_back.jpg", "contract": "contract.jpg",
    "bank": "bank_statement.jpg",
}
_rebuild_state = {"running": False, "progress": 0, "message": "尚未运行", "result": None, "error": ""}

app = FastAPI(title="金融影像智能相似度检测系统")
app.mount("/assets", StaticFiles(directory=UI_DIR), name="assets")


def _output_dir() -> Path | None:
    for path in OUTPUT_DIRS:
        if (path / "run_summary.json").exists():
            return path
    return None


def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _same_customer_results() -> dict:
    """读取同客户多笔授信/重复使用区分实验的正式结果。

    优先正式 outputs 目录；本机尚未生成该目录时回退 teamfix（报告归档处），
    results.json 放在任一候选目录均可被读取。"""
    for directory in _SAME_CUSTOMER_CANDIDATES:
        data = _read_json(directory / "results.json", {})
        if data:
            return data
    return {}


def _same_customer_artifact(filename: str) -> Path | None:
    """按候选目录查找同客户区分实验产物文件（docx / md 报告等）。"""
    for directory in _SAME_CUSTOMER_CANDIDATES:
        path = directory / filename
        if path.is_file():
            return path
    return None


# ---------- 风险策略（阈值管理） ----------
DEFAULT_RISK_POLICY = {"high_risk_threshold": 0.97}
LEGACY_REVIEW_BUFFER_THRESHOLD = 0.93  # 兼容历史字段，不参与判定
_policy_eval_cache = None
_policy_eval_lock = threading.Lock()


class PolicyRequest(BaseModel):
    high_risk_threshold: float
    operator: str = "system"


def _ensure_policy_tables() -> None:
    """创建单例运行策略和不可变更的发布审计记录。"""
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            create table if not exists risk_policy(
                id integer primary key check(id = 1),
                high_risk_threshold real not null,
                review_buffer_threshold real not null,
                stage1_visual_threshold real,
                stage2_risk_threshold real,
                version integer not null default 2,
                updated_at text not null,
                operator text not null default 'system'
            )
        """)
        conn.execute("""
            create table if not exists policy_audit(
                id integer primary key autoincrement,
                version integer not null,
                high_risk_threshold real not null,
                review_buffer_threshold real not null,
                stage1_visual_threshold real,
                stage2_risk_threshold real,
                operator text not null,
                created_at text not null
            )
        """)
        conn.execute("""
            insert or ignore into risk_policy(
                id, high_risk_threshold, review_buffer_threshold,
                version, updated_at, operator
            ) values(1, ?, ?, 2, ?, 'system')
        """, (
            DEFAULT_RISK_POLICY["high_risk_threshold"],
            LEGACY_REVIEW_BUFFER_THRESHOLD,
            now,
        ))
        # 兼容已存在的 data.db：保留旧字段作审计，新增两阶段字段
        policy_columns = {row[1] for row in conn.execute("pragma table_info(risk_policy)")}
        audit_columns = {row[1] for row in conn.execute("pragma table_info(policy_audit)")}
        for col in ("stage1_visual_threshold", "stage2_risk_threshold"):
            if col not in policy_columns:
                conn.execute(f"alter table risk_policy add column {col} real")
            if col not in audit_columns:
                conn.execute(f"alter table policy_audit add column {col} real")
        conn.commit()


def _get_policy() -> dict:
    _ensure_policy_tables()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("select * from risk_policy where id=1").fetchone()
    policy = dict(row) if row else {
        **DEFAULT_RISK_POLICY,
        "version": 2, "updated_at": "", "operator": "system",
    }
    policy["high_risk_threshold"] = float(
        policy.get("high_risk_threshold") or DEFAULT_RISK_POLICY["high_risk_threshold"]
    )
    for k in ("review_buffer_threshold", "stage1_visual_threshold", "stage2_risk_threshold"):
        policy.pop(k, None)
    policy["policy_version"] = f"V3.{policy['version']}"
    return policy


def _apply_policy(row: dict, policy: dict | None = None) -> dict:
    """按唯一的疑似风险阈值进行二分类。"""
    policy = policy or _get_policy()
    score = float(row.get("score", 0) or 0)
    relationship = str(row.get("relationship", "") or "")
    high = float(policy["high_risk_threshold"])
    if relationship == "self":
        decision, risk_level, needs_review = "正常排除", "low", False
    elif score >= high:
        decision, risk_level, needs_review = "疑似风险", "high", True
    else:
        decision, risk_level, needs_review = "正常排除", "low", False
    return {
        **row,
        "threshold": high,
        "high_risk_threshold": high,
        "is_suspicious": needs_review,
        "is_high_risk": decision == "疑似风险",
        "needs_review": needs_review,
        "decision": decision,
        "risk_level": risk_level,
        "policy_version": policy["policy_version"],
    }


def _policy_artifacts() -> tuple:
    output_candidates = [
        ROOT.parent / "submission_ready" / "03_model_artifacts" / "outputs" / "mvp",
        ROOT.parent / "jinchuang_v4" / "code" / "outputs" / "mvp",
        ROOT.parent / "outputs" / "mvp",
    ]
    output = next((p for p in output_candidates
                   if (p / "face_signing.faiss").is_file()
                   and (p / "face_manifest.csv").is_file()), None)
    annotation_candidates = [
        ROOT.parent / "data" / "annotations_filtered_with_identity.csv",
        ROOT.parent / "data" / "annotations.csv",
        DATA_DIR / "annotations_filtered_with_identity.csv",
        DATA_DIR / "annotations.csv",
    ]
    annotations = next((p for p in annotation_candidates if p.is_file()), None)
    if output is None or annotations is None:
        raise FileNotFoundError("缺少 FAISS 索引、面签清单或相似组标注文件")
    return output / "face_signing.faiss", output / "face_manifest.csv", annotations


def _load_policy_eval_cache() -> dict:
    """以清单中的全部面签样本做离线回放，候选库保持为完整 FAISS 索引。"""
    global _policy_eval_cache
    with _policy_eval_lock:
        if _policy_eval_cache is not None:
            return _policy_eval_cache
        import faiss
        index_path, manifest_path, annotations_path = _policy_artifacts()
        index = faiss.read_index(str(index_path))
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            manifest = list(csv.DictReader(handle))
        with annotations_path.open("r", encoding="utf-8-sig", newline="") as handle:
            annotations = list(csv.DictReader(handle))
        if index.ntotal != len(manifest):
            raise RuntimeError(f"索引数量 {index.ntotal} 与清单数量 {len(manifest)} 不一致")
        group_by_dir = {}
        for row in annotations:
            rel = str(row.get("file_path", "") or "").replace("\\", "/").strip("/")
            directory = rel.split("/", 1)[0] if rel else ""
            if directory:
                group_by_dir[directory] = str(row.get("similar_group", "") or "")
        # 策略影响估计展示全量业务口径：3254 条面签样本，而不是仅用于
        # 模型泛化评估的 488 条 test 样本。
        query_ids = list(range(index.ntotal))
        evaluation_split = "all"
        query_vectors = np.vstack([index.reconstruct(i) for i in query_ids]).astype("float32")
        scores, neighbors = index.search(query_vectors, 6)
        max_scores = []
        truths = []
        comparison_count = 0
        for query_id, row_scores, row_neighbors in zip(query_ids, scores, neighbors):
            best = float("-inf")
            for score, neighbor_id in zip(row_scores, row_neighbors):
                if neighbor_id < 0 or int(neighbor_id) == query_id:
                    continue
                best = max(best, float(score))
                comparison_count += 1
            max_scores.append(best)
            directory = str(manifest[query_id].get("loan_id", "") or "")
            truths.append(bool(group_by_dir.get(directory, "")))
        _policy_eval_cache = {
            "scores": np.asarray(max_scores, dtype="float32"),
            "truths": np.asarray(truths, dtype=bool),
            "sample_count": len(query_ids),
            "comparison_count": comparison_count,
            "evaluation_split": evaluation_split,
            "index_source": str(index_path),
            "annotations_source": str(annotations_path),
        }
        return _policy_eval_cache


def _threshold_metrics(cache: dict, threshold: float) -> dict:
    predicted = cache["scores"] >= threshold
    truth = cache["truths"]
    tp = int(np.sum(predicted & truth))
    tn = int(np.sum(~predicted & ~truth))
    fp = int(np.sum(predicted & ~truth))
    fn = int(np.sum(~predicted & truth))
    total = int(truth.size)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision is not None and recall is not None and precision + recall else None)
    return {
        "threshold": threshold,
        "accuracy": (tp + tn) / total if total else None,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "hits": int(np.sum(predicted)),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def _float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bool_text(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _stage2_display_counts(raw_counts: Counter | dict) -> dict[str, int]:
    """把 Stage2 原始类型计数按业务分桶归并，返回 {业务桶: 笔数}。"""
    return {
        label: int(sum(raw_counts.get(key, 0) for key in keys))
        for label, keys in STAGE2_DISPLAY_TYPES
    }


# ---------------------------------------------------------------------------
# Tab1 指标缓存表：统计逻辑独立在 refresh_stats.py（可脚本运行），
# 「刷新数据」按钮(/api/dashboard/refresh)与本模块共用同一入口。
# ---------------------------------------------------------------------------
from refresh_stats import (  # noqa: E402
    HIGH_SIM_THRESHOLD,
    read_tables as _read_dashboard_tables_db,
    refresh_dashboard_stats,
)
import refresh_stats  # noqa: E402


def _refresh_dashboard_tables() -> dict:
    """重算三张缓存表（委托 refresh_stats 脚本）。"""
    return refresh_dashboard_stats(DB_PATH)


def _dashboard_payload() -> dict:
    """读取缓存表；客户数/贷款数与库内现状不一致时自动重算。

    面签语料随 loans 表增减，客户/贷款计数即可覆盖常规变更；
    语料重建等少数场景由页面「刷新数据」按钮强制重算。
    """
    if not DB_PATH.exists():
        return {}
    with sqlite3.connect(DB_PATH) as conn:
        try:
            loans_now = conn.execute("select count(*) from loans").fetchone()[0]
            customers_now = conn.execute("select count(*) from customers").fetchone()[0]
        except sqlite3.Error:
            return _refresh_dashboard_tables()
    dash = _read_dashboard_tables_db(DB_PATH)
    if not dash or dash.get("loans") != loans_now or dash.get("customers") != customers_now:
        try:
            dash = _refresh_dashboard_tables()
        except Exception:
            traceback.print_exc()
    return dash


def _mvp_dashboard_payload() -> dict:
    """首页模型 KPI：统一读取当前正式 MVP 汇总，而不是 SQLite 缓存。"""
    output = ROOT.parent / "jinchuang_v4" / "code" / "outputs" / "mvp"
    run = _read_json(output / "run_summary.json", {})
    monitoring = _read_json(output / "fraud_monitoring_summary.json", {})
    if not run or not monitoring:
        return {}
    customer_count = None
    identity_path = output / "customer_identity_map_from_annotations.csv"
    if identity_path.exists():
        try:
            with identity_path.open("r", encoding="utf-8-sig", newline="") as handle:
                customer_count = len({
                    row.get("customer_id_hash", "")
                    for row in csv.DictReader(handle)
                    if row.get("customer_id_hash", "")
                })
        except Exception:
            customer_count = None
    total_pairs = int(monitoring.get("total_pairs", 0) or 0)
    suspicious_pairs = int(monitoring.get("suspicious_pairs", 0) or 0)
    priority = monitoring.get("by_priority", {}) or {}
    mtimes = [
        (output / name).stat().st_mtime
        for name in ("run_summary.json", "fraud_monitoring_summary.json")
        if (output / name).exists()
    ]
    return {
        "customers": customer_count,
        "loans": int(run.get("selected_face_signing", 0) or 0),
        "total_images": int(run.get("total_images", 0) or 0),
        "face_images": int(run.get("selected_face_signing", 0) or 0),
        "pending_review": int(priority.get("urgent", 0) or 0),
        "involved_loans": int(monitoring.get("risk_graph_nodes", 0) or 0),
        "feature_loans": int(run.get("selected_face_signing", 0) or 0),
        "total_pairs": total_pairs,
        "high_similar_pairs": suspicious_pairs,
        "high_similar_rate": suspicious_pairs / total_pairs if total_pairs else 0,
        "source": str(output),
        "updated_at": datetime.fromtimestamp(max(mtimes)).astimezone().isoformat(timespec="seconds") if mtimes else "",
        "similarity_dist": [],
        "loan_structure": {},
        "purpose_dist": {},
    }



def _sample_csv(path: Path, limit: int = 6, sort_key: str | None = None, reverse: bool = True):
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
    if sort_key:
        rows.sort(key=lambda row: _float(row.get(sort_key)), reverse=reverse)
    return rows[:limit]


def _ensure_auth_tables():
    """用户表（账号为主键）+ 会话表 + 影像保存操作记录表。

    users 结构要求：username 主键，保存账号/密码(盐+哈希)/姓名(name)/职位(position)。
    兼容迁移：检测到旧结构（id 自增主键、display_name/role 列）时平移数据重建，
    旧账号的盐与密码哈希原样保留（老密码继续有效），旧会话全部作废。
    """
    with sqlite3.connect(DB_PATH) as conn:
        old_cols = []
        if conn.execute(
            "select 1 from sqlite_master where type='table' and name='users'"
        ).fetchone():
            old_cols = [r[1] for r in conn.execute("pragma table_info(users)")]
        if old_cols and "name" not in old_cols:  # 旧结构 → 迁移
            conn.execute("alter table users rename to users_old")
            _create_users_table(conn)
            conn.execute("""
                insert or ignore into users(username,password_salt,password_hash,name,position,created_at)
                select username,password_salt,password_hash,display_name,role,created_at from users_old
            """)
            conn.execute("drop table users_old")
        elif not old_cols:
            _create_users_table(conn)
        # 会话表：旧版挂 user_id，统一重建为挂 username（旧登录态作废）
        conn.execute("drop table if exists auth_sessions")
        conn.execute("""
            create table auth_sessions(
                token text primary key,
                username text not null,
                created_at text not null,
                foreign key(username) references users(username)
            )
        """)
        # 影像保存操作记录：时间为主键，操作用户名为副键（索引）
        conn.execute("""
            create table if not exists operation_records(
                created_at text primary key,
                username text not null,
                folder text not null,
                file_count integer default 0
            )
        """)
        conn.execute(
            "create index if not exists idx_operation_records_username on operation_records(username)"
        )
        # 预设演示账号
        if not conn.execute("select 1 from users where username='demo'").fetchone():
            salt = secrets.token_hex(16)
            conn.execute(
                "insert into users(username,password_salt,password_hash,name,position,created_at)"
                " values(?,?,?,?,?,?)",
                ("demo", salt, _hash_password("demo123", salt), "演示专员", "风控专员",
                 datetime.now().astimezone().isoformat(timespec="seconds")),
            )
        conn.commit()


def _create_users_table(conn):
    conn.execute("""
        create table users(
            username text primary key,
            password_salt text not null,
            password_hash text not null,
            name text not null,
            position text not null default '风控专员',
            created_at text not null
        )
    """)


def _create_user(username: str, password: str, display_name: str = "", role: str = "风控专员"):
    salt = secrets.token_hex(16)
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    display_name = (display_name or username).strip()
    role = (role or "风控专员").strip()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("""
            insert into users(username,password_salt,password_hash,name,position,created_at)
            values(?,?,?,?,?,?)
        """, (username, salt, _hash_password(password, salt), display_name, role, now))
        conn.commit()
        return conn.execute(
            "select username,name,position from users where username=?", (username,)
        ).fetchone()


def _hash_password(password: str, salt: str) -> str:
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), 120_000)
    return digest.hex()


def _public_user(row) -> dict:
    # display_name/role 为前端既有字段名，语义分别对应新表的 name/position
    return {"username": row["username"], "display_name": row["name"], "role": row["position"]}


@app.get("/")
def index():
    return FileResponse(UI_DIR / "index.html")


# 模型后台预热状态：/api/health 不阻塞、不强制加载（模型首次调用才懒加载），
# 由 __main__ 中的后台线程提前加载，保证网页秒开、检测时模型已热。
_MODEL_STATUS = {"ready": False, "total": 0}


@app.get("/api/health")
def health():
    return {"ok": True, "model": _MODEL_STATUS["ready"], "index_total": _MODEL_STATUS["total"]}


class AuthRequest(BaseModel):
    username: str
    password: str
    display_name: str = ""
    role: str = "风控专员"


class TokenRequest(BaseModel):
    token: str = ""


def _auth_payload(row) -> dict:
    token = secrets.token_urlsafe(32)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "insert into auth_sessions(token,username,created_at) values(?,?,?)",
            (token, row["username"], datetime.now().astimezone().isoformat(timespec="seconds")),
        )
        conn.commit()
    return {"ok": True, "token": token, "user": _public_user(row)}


@app.post("/api/auth/register")
def register(req: AuthRequest):
    username = req.username.strip()
    display_name = (req.display_name or username).strip()
    role = (req.role or "风控专员").strip()
    if len(username) < 3:
        raise HTTPException(400, "账号至少 3 个字符")
    if len(req.password) < 6:
        raise HTTPException(400, "密码至少 6 位")
    _ensure_auth_tables()
    try:
        row = _create_user(username, req.password, display_name, role)
    except sqlite3.IntegrityError:
        raise HTTPException(400, "账号已存在") from None
    return _auth_payload(row)


@app.post("/api/auth/login")
def login(req: AuthRequest):
    """严格匹配：在 users 表中查 username + 密码哈希，任一不匹配返回 401 登录失败。"""
    username = req.username.strip()
    _ensure_auth_tables()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "select username,password_salt,password_hash from users where username=?",
            (username,),
        ).fetchone()
        if not row or _hash_password(req.password, row["password_salt"]) != row["password_hash"]:
            raise HTTPException(401, "登录失败")
        public_row = conn.execute(
            "select username,name,position from users where username=?", (username,)
        ).fetchone()
    return _auth_payload(public_row)


@app.post("/api/auth/me")
def current_user(req: TokenRequest):
    _ensure_auth_tables()
    if not req.token:
        return {"ok": False, "user": None}
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("""
            select u.username,u.name,u.position
            from auth_sessions s join users u on u.username=s.username
            where s.token=?
        """, (req.token,)).fetchone()
    return {"ok": bool(row), "user": _public_user(row) if row else None}


@app.post("/api/auth/logout")
def logout(req: TokenRequest):
    _ensure_auth_tables()
    if req.token:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("delete from auth_sessions where token=?", (req.token,))
            conn.commit()
    return {"ok": True}


@app.post("/api/save-to-library")
async def save_to_library(files: list[UploadFile] = File(...), username: str = Form("")):
    """Tab2「保存到库」：上传影像存入 saved_images/<保存时间>/，并写操作记录。

    操作记录表 operation_records：created_at（保存时间，主键）+ username（副键）
    + folder（保存文件夹名）+ file_count。
    """
    username = (username or "").strip()
    if not username:
        raise HTTPException(401, "请先登录")
    real_files = [f for f in files if f.filename]
    if not real_files:
        raise HTTPException(400, "没有收到影像")
    ts = datetime.now().astimezone()
    folder = ts.strftime("%Y%m%d_%H%M%S")
    target = SAVED_DIR / folder
    target.mkdir(parents=True, exist_ok=True)
    saved = 0
    for f in real_files:
        raw = await f.read()
        # Path(...).name 剥离路径成分，防 ../ 路径穿越
        (target / Path(f.filename).name).write_bytes(raw)
        saved += 1
    created_at = ts.isoformat(timespec="seconds")
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "insert or replace into operation_records(created_at,username,folder,file_count)"
            " values(?,?,?,?)",
            (created_at, username, folder, saved),
        )
        conn.commit()
    return {"ok": True, "folder": folder, "saved": saved, "created_at": created_at}


@app.get("/api/operation-records")
def get_operation_records():
    """影像保存操作记录（按时间倒序）；前端按当前用户分组：自己的在前。"""
    rows = []
    if DB_PATH.exists():
        _ensure_auth_tables()
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            rows = [dict(x) for x in conn.execute("""
                select o.created_at, o.username, o.folder, o.file_count,
                       u.name, u.position
                from operation_records o left join users u on u.username = o.username
                order by o.created_at desc
                limit 200
            """).fetchall()]
    return {"items": rows}


@app.get("/api/stats")
def stats():
    result = {"customers": 0, "loans": 0, "danger": 0, "groups": 0, "feature_ready": 0, "statuses": {}}
    if DB_PATH.exists():
        with sqlite3.connect(DB_PATH) as conn:
            for key, sql in {
                "customers": "select count(*) from customers",
                "loans": "select count(*) from loans",
                "danger": "select count(*) from loans where verify_status in ('F','C')",
                "groups": "select count(distinct auto_group) from loans where auto_group is not null and auto_group != ''",
                "feature_ready": "select count(*) from loans where face_feature is not null",
            }.items():
                try:
                    result[key] = conn.execute(sql).fetchone()[0]
                except sqlite3.Error:
                    pass
            try:
                result["statuses"] = dict(conn.execute(
                    "select coalesce(verify_status,'N'), count(*) from loans group by coalesce(verify_status,'N')"
                ).fetchall())
            except sqlite3.Error:
                pass
            try:
                result["business_types"] = dict(conn.execute(
                    "select coalesce(nullif(business_type,''),'未登记'), count(*) from loans group by coalesce(nullif(business_type,''),'未登记')"
                ).fetchall())
            except sqlite3.Error:
                result["business_types"] = {}
    # Tab1 展示指标：读 dashboard_* 缓存表（库内计数变化时自动重算）
    dash = _dashboard_payload()
    if dash:
        result["dashboard"] = dash
    mvp_dash = _mvp_dashboard_payload()
    if mvp_dash:
        result["mvp_dashboard"] = mvp_dash
    # 客户授信结构（马赛克图数据）：同客户多笔授信 vs 单笔授信 × 行为交叉
    # 统计由 refresh_stats.py 计算并落库到 dashboard_overview / dashboard_credit_behavior，
    # 随「刷新数据」一起刷新
    result["customer_credit"] = (dash or {}).get("customer_credit") or {}
    result["credit_mosaic"] = (dash or {}).get("credit_mosaic") or {}
    result["model_metrics"] = model_bridge.get_model_metrics()
    result["sources"] = {
        "database": str(DB_PATH),
        "model": "SigLIP2 + FAISS 当前服务",
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    return result


@app.post("/api/dashboard/refresh")
def dashboard_refresh():
    """手动刷新 Tab1 缓存表（页面「刷新数据」按钮调用）。"""
    try:
        dash = _refresh_dashboard_tables()
        return {"ok": bool(dash), "dashboard": dash}
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(500, f"刷新失败: {exc}") from exc


@app.get("/api/suspicious")
def suspicious(limit: int = 100):
    rows = []
    if not DB_PATH.exists():
        return {"items": rows, "source": "SQLite loans/customers"}
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        # 业务类型优先取 loans."业务类型"（商户易贷/锡微贷/消费贷），旧库无此列时回退 business_type
        cols = [c[1] for c in conn.execute("pragma table_info(loans)")]
        biz_expr = 'coalesce(nullif(l."业务类型",\'\'),\'\')' if "业务类型" in cols \
            else "coalesce(l.business_type,'')"
        query = f"""
        select l.loan_id, coalesce(nullif(trim(c.name),''),'未登记') customer_name,
               {biz_expr} business_type,
               coalesce(l.verify_status,'N') verify_status,
               coalesce(l.auto_group,'') auto_group,
               coalesce(l.status,'') status,
               coalesce(l.created_at,'') created_at
        from loans l left join customers c on c.customer_id=l.customer_id
        where l.verify_status in ('F','C','B')
        order by case l.verify_status when 'C' then 0 when 'F' then 1 else 2 end,
                 l.created_at desc, l.loan_id desc limit ?
        """
        rows = _fill_annotation_customer_names([dict(x) for x in conn.execute(query, (min(max(limit, 1), 500),)).fetchall()])
    return {"items": rows, "source": "SQLite loans + customers", "score_note": "历史贷款表未存储逐笔相似度分数"}


@app.get("/api/loans/search")
def search_loans(q: str = "", limit: int = 200):
    """Tab3 身份查询：按客户姓名 / 证件号(customer_id) / 贷款编号 / 业务类型模糊匹配 loans 表。"""
    q = (q or "").strip()
    rows = []
    if not q:
        return {"items": rows, "error": "请输入查询内容"}
    if not DB_PATH.exists():
        return {"items": rows, "error": "数据库不存在"}
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cols = [c[1] for c in conn.execute("pragma table_info(loans)")]
        biz_expr = 'coalesce(nullif(l."业务类型",\'\'),\'\')' if "业务类型" in cols \
            else "coalesce(l.business_type,'')"
        like = f"%{q}%"
        annotation_ids = [loan_id for loan_id, name in _annotation_customer_names().items()
                          if q.casefold() in name.casefold()]
        name_fallback = (f" or l.loan_id in ({','.join('?' for _ in annotation_ids)})"
                         if annotation_ids else "")
        query = f"""
        select l.loan_id, coalesce(nullif(trim(c.name),''),'未登记') customer_name,
               coalesce(l.customer_id,'') customer_id,
               {biz_expr} business_type,
               coalesce(l.verify_status,'N') verify_status,
               coalesce(l.auto_group,'') auto_group,
               coalesce(l.status,'') status,
               coalesce(l.created_at,'') created_at
        from loans l left join customers c on c.customer_id=l.customer_id
        where l.loan_id like ? or l.customer_id like ?
              or coalesce(c.name,'') like ? or {biz_expr} like ?{name_fallback}
        order by l.created_at desc, l.loan_id desc limit ?
        """
        params = [like, like, like, like, *annotation_ids, min(max(limit, 1), 500)]
        rows = _fill_annotation_customer_names([dict(x) for x in conn.execute(query, params).fetchall()])
    return {"items": rows, "source": "SQLite loans + customers"}


@app.get("/api/loans/all")
def all_loans(limit: int = 3500):
    """Tab3 完整数据库：返回全部贷款条目（含未验证/正常贷款），供分页展示。"""
    rows = []
    if not DB_PATH.exists():
        return {"items": rows, "source": "SQLite loans + customers"}
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cols = [c[1] for c in conn.execute("pragma table_info(loans)")]
        biz_expr = 'coalesce(nullif(l."业务类型",\'\'),\'\')' if "业务类型" in cols \
            else "coalesce(l.business_type,'')"
        query = f"""
        select l.loan_id, coalesce(nullif(trim(c.name),''),'未登记') customer_name,
               {biz_expr} business_type,
               coalesce(l.verify_status,'N') verify_status,
               coalesce(l.auto_group,'') auto_group,
               coalesce(l.status,'') status,
               coalesce(l.created_at,'') created_at
        from loans l left join customers c on c.customer_id=l.customer_id
        order by case when nullif(trim(c.name),'') is null then 1 else 0 end,
                 l.created_at desc, l.loan_id desc limit ?
        """
        rows = _fill_annotation_customer_names([dict(x) for x in conn.execute(query, (min(max(limit, 1), 5000),)).fetchall()])
    return {"items": rows, "source": "SQLite loans + customers", "total": len(rows)}


# ---------- 风险策略 API ----------
@app.get("/api/policy")
def get_policy():
    return _get_policy()


@app.put("/api/policy")
def update_policy(req: PolicyRequest):
    high = float(req.high_risk_threshold)
    if not (0.0 <= high <= 1.0):
        raise HTTPException(400, "疑似风险判定阈值必须在 0 到 1 之间")
    operator = (req.operator or "system").strip()[:64]
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    _ensure_policy_tables()
    with sqlite3.connect(DB_PATH) as conn:
        current = conn.execute(
            "select version, review_buffer_threshold from risk_policy where id=1"
        ).fetchone()
        version = int(current[0] if current else 2) + 1
        legacy_buffer = float(current[1] if current else LEGACY_REVIEW_BUFFER_THRESHOLD)
        conn.execute("""
            update risk_policy
            set high_risk_threshold=?, version=?, updated_at=?, operator=? where id=1
        """, (high, version, now, operator))
        conn.execute("""
            insert into policy_audit(
                version, high_risk_threshold, review_buffer_threshold, operator, created_at
            ) values(?,?,?,?,?)
        """, (version, high, legacy_buffer, operator, now))
        conn.commit()
    return {"ok": True, **_get_policy()}


@app.get("/api/policy/impact")
def policy_impact(high_risk_threshold: float):
    high = float(high_risk_threshold)
    if not (0.0 <= high <= 1.0):
        raise HTTPException(400, "疑似风险判定阈值必须在 0 到 1 之间")
    try:
        cache = _load_policy_eval_cache()
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(503, f"真实阈值评估不可用: {exc}") from exc
    curve_thresholds = sorted(set([
        0.90, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99,
        round(high, 4),
    ]))
    return {
        "available": True,
        "sample_count": cache["sample_count"],
        "comparison_count": cache["comparison_count"],
        "evaluation_split": cache["evaluation_split"],
        "sample_scope": "全量面签样本对完整 FAISS 库的 Top-5 离线回放",
        "high_risk": _threshold_metrics(cache, high),
        "curve": [_threshold_metrics(cache, t) for t in curve_thresholds],
        "sources": {
            "index": cache["index_source"],
            "annotations": cache["annotations_source"],
        },
    }


# ---------- 同客户区分实验 API ----------
@app.get("/api/same-customer")
def same_customer_experiment():
    data = _same_customer_results()
    if not data:
        return {
            "available": False,
            "error": "尚未生成同客户区分实验结果，请运行 experiments/run_same_customer_experiment.py",
        }
    groups = data.get("groups", [])
    summary = {k: v for k, v in data.items() if k != "groups"}
    summary["available"] = True
    summary["items"] = [
        {
            "group": row.get("group", ""),
            "customer": row.get("customer", {}),
            "original": row.get("original", {}),
            "counts": row.get("counts", {}),
            "candidate_count": len(row.get("candidates", [])),
        }
        for row in groups
    ]
    return summary


@app.get("/api/same-customer/groups/{group_id}")
def same_customer_group(group_id: str):
    data = _same_customer_results()
    for row in data.get("groups", []):
        if row.get("group") == group_id:
            return {"available": True, **row}
    raise HTTPException(404, "未找到该同客户相似组")


@app.get("/api/same-customer/report")
def same_customer_report():
    path = _same_customer_artifact("同客户高相似影像区分实验报告.docx")
    if path is None:
        raise HTTPException(404, "同客户区分实验报告尚未生成")
    return FileResponse(
        path,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename="同客户高相似影像区分实验报告.docx",
    )


@app.get("/api/same-customer/report.md")
def same_customer_markdown_report():
    path = _same_customer_artifact("report.md")
    if path is None:
        raise HTTPException(404, "同客户区分 Markdown 报告尚未生成")
    return FileResponse(path, media_type="text/markdown; charset=utf-8",
                        filename="同客户高相似影像区分实验报告.md")


@app.get("/api/operations")
def operations(limit: int = 100):
    rows = []
    if DB_PATH.exists():
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            try:
                rows = [dict(x) for x in conn.execute(
                    "select created_at, operator, action, auto_group, loan_ids, note from reviews order by created_at desc, id desc limit ?",
                    (min(max(limit, 1), 500),),
                ).fetchall()]
            except sqlite3.Error:
                pass
    return {"items": rows, "source": "SQLite reviews"}


def report():
    """基础报告数据（metrics + 阈值）。真正的 /api/report 路由由 report_v2 提供，
    此处保留为普通函数供 report_v2 调用，避免重复注册导致 report_v2 被遮蔽。"""
    metrics = model_bridge.get_model_metrics()
    thresholds = {}
    candidates = [
        ROOT.parent / "jinchuang_v4" / "code" / "outputs" / "mvp" / "run_summary.json",
        Path("/workspace/jinchuang_v4/code/outputs/mvp/run_summary.json"),
    ]
    for path in candidates:
        if path.exists():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                thresholds = {k: raw.get(k) for k in (
                    "high_risk_threshold", "medium_risk_threshold",
                    "cross_customer_threshold", "same_customer_threshold",
                    "high_risk_threshold_source", "model_name",
                )}
                thresholds["source_file"] = str(path)
                break
            except Exception:
                continue
    return {"metrics": metrics, "thresholds": thresholds, "source": "outputs/mvp + 当前模型桥"}


@app.get("/api/experiments")
def experiments():
    output = _output_dir()
    if not output:
        return {"available": False, "error": "未找到 outputs/mvp 实验结果目录"}

    summary = _read_json(output / "run_summary.json", {})
    class_metrics = _read_json(output / "classification_metrics.json", {})
    threshold_meta = _read_json(output / "threshold_metadata.json", {})
    two_stage = _read_json(output / "two_stage_summary.json", {})
    monitoring = _read_json(output / "fraud_monitoring_summary.json", summary.get("fraud_monitoring_summary", {}))

    stage1_metrics = two_stage.get("stage1", {}).get("pair_level_split", {}).get("metrics", {})
    group_metrics = two_stage.get("stage1", {}).get("group_level_split", {}).get("metrics", {})
    stage1_summary = two_stage.get("stage1", {})
    stage2_summary = two_stage.get("stage2", {})
    best_f1 = threshold_meta.get("best_f1_threshold", {})
    classifier_test = class_metrics.get("test", {})

    threshold_rows = []
    threshold_path = output / "threshold_experiment.csv"
    if threshold_path.exists():
        with threshold_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                threshold_rows.append({
                    "threshold": _float(row.get("threshold")),
                    "precision": _float(row.get("precision")),
                    "recall": _float(row.get("recall")),
                    "f1": _float(row.get("f1")),
                    "review_count": int(_float(row.get("review_count"))),
                    "tp": int(_float(row.get("tp"))),
                    "fp": int(_float(row.get("fp"))),
                    "fn": int(_float(row.get("fn"))),
                })

    stage1_counts = Counter()
    stage1_examples = []
    stage1_path = output / "stage1_similarity_report.csv"
    if stage1_path.exists():
        with stage1_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                pred = _bool_text(row.get("stage1_predicted_similar"))
                label = _bool_text(row.get("stage1_label"))
                error_type = "TP" if pred and label else "FP" if pred else "FN" if label else "TN"
                stage1_counts[error_type] += 1
                if pred and len(stage1_examples) < 8:
                    stage1_examples.append({
                        "query_loan_id": row.get("query_loan_id", ""),
                        "match_loan_id": row.get("match_loan_id", ""),
                        "probability": _float(row.get("stage1_similarity_probability")),
                        "decision_source": row.get("stage1_decision_source", ""),
                        "global_similarity": _float(row.get("global_semantic_similarity")),
                        "subject_similarity": _float(row.get("subject_region_hist_similarity")),
                        "background_similarity": _float(row.get("background_hist_similarity")),
                        "local_structure": _float(row.get("local_structure_orb_ratio")),
                        "error_type": error_type,
                    })

    stage2_counts = Counter()
    stage2_path = output / "stage2_fraud_type_report.csv"
    if stage2_path.exists():
        with stage2_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                stage2_counts[row.get("stage2_predicted_type") or "unknown"] += 1
    # CSV 缺失时回退 two_stage_summary.json 的 predicted_type_counts，保证 Stage2 分桶有数
    if not stage2_counts:
        for k, v in (two_stage.get("stage2", {}).get("predicted_type_counts", {}) or {}).items():
            if isinstance(v, (int, float)):
                stage2_counts[k] += int(v)

    risk_examples = []
    fraud_path = output / "fraud_monitoring.csv"
    if fraud_path.exists():
        risk_examples = [
            {
                "query_loan_id": row.get("query_loan_id", ""),
                "match_loan_id": row.get("match_loan_id", ""),
                "cosine_similarity": _float(row.get("cosine_similarity")),
                "fraud_score": _float(row.get("fraud_score")),
                "risk_cluster_id": row.get("risk_cluster_id", ""),
                "risk_cluster_size": int(_float(row.get("risk_cluster_size"))),
                "priority": row.get("review_priority", ""),
                "risk_level": row.get("fraud_score_level_zh") or row.get("monitor_risk_level", ""),
                "action": row.get("recommended_action_zh", ""),
                "tags": row.get("innovation_tags", ""),
            }
            for row in _sample_csv(fraud_path, limit=6, sort_key="fraud_score")
        ]

    return {
        "available": True,
        "source": str(output),
        "summary": {
            "model_name": summary.get("model_name", "SigLIP2"),
            "device": summary.get("device", "-"),
            "total_images": summary.get("total_images", 0),
            "selected_face_signing": summary.get("selected_face_signing", 0),
            "elapsed_seconds": summary.get("elapsed_seconds", 0),
            "class_counts": summary.get("class_counts", {}),
        },
        "classifier": {
            "accuracy": classifier_test.get("accuracy"),
            "macro_f1": classifier_test.get("macro_f1"),
            "per_class": classifier_test.get("per_class", {}),
        },
        "stage1": {
            "metrics": stage1_metrics,
            "group_metrics": group_metrics,
            "counts": dict(stage1_counts),
            "examples": stage1_examples,
            "final_predicted_similar": stage1_summary.get("final_predicted_similar", 0),
            "unique_pairs": stage1_summary.get("pair_symmetry_handling", {}).get("unique_undirected_pairs", 0),
            "features": stage1_summary.get("image_features", []),
        },
        "stage2": {
            "summary": stage2_summary,
            "type_counts": _stage2_display_counts(stage2_counts),
            "raw_type_counts": dict(stage2_counts),
        },
        "threshold": {
            "metadata": threshold_meta,
            "best_f1": best_f1,
            "rows": threshold_rows,
        },
        "monitoring": {
            "summary": monitoring,
            "examples": risk_examples,
        },
    }


@app.get("/api/groups")
def groups():
    items = []
    if DB_PATH.exists():
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            items = [dict(x) for x in conn.execute("""
                select auto_group,
                       count(*) loan_count,
                       sum(case when verify_status='F' then 1 else 0 end) pending_count,
                       sum(case when verify_status='C' then 1 else 0 end) danger_count,
                       sum(case when verify_status='B' then 1 else 0 end) excluded_count
                from loans where auto_group is not null and auto_group != ''
                group by auto_group order by pending_count desc, danger_count desc, auto_group
            """).fetchall()]
    return {"items": items, "source": "SQLite loans.auto_group"}


@app.get("/api/groups/{group_id}")
def group_detail(group_id: str):
    if not DB_PATH.exists():
        return {"items": []}
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = [dict(x) for x in conn.execute("""
            select l.loan_id, l.image_dir, l.business_type, l.verify_status,
                   l.status, l.created_at, coalesce(c.name,'未登记') customer_name,
                   coalesce(c.id_card,'') id_card
            from loans l left join customers c on c.customer_id=l.customer_id
            where l.auto_group=? order by l.loan_id
        """, (group_id,)).fetchall()]
    for row in rows:
        row["images"] = {k: f"/api/images/{row['loan_id']}/{k}" for k in IMAGE_FILES}
    return {"group": group_id, "items": rows, "source": "SQLite + DATA_DIR"}


@app.get("/api/images/{loan_id}/{kind}")
def loan_image(loan_id: str, kind: str):
    if kind not in IMAGE_FILES:
        raise HTTPException(404, "未知图片类型")
    image_dir = ""
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute("select image_dir from loans where loan_id=?", (loan_id,)).fetchone()
        if row:
            image_dir = row[0] or ""
    if not image_dir:
        # 兜底：语料中少量目录（loan_001 等 74 个）在 FAISS 索引里但未录入 loans 表，
        # 检测命中时前端会拿目录名来取图；按目录名直接在语料根下查找，
        # 下面的 resolve 校验仍然拦截路径穿越
        image_dir = loan_id
    safe_dir = Path(image_dir).name
    path = (DATA_DIR / safe_dir / IMAGE_FILES[kind]).resolve()
    if DATA_DIR.resolve() not in path.parents or not path.is_file():
        raise HTTPException(404, "图片不存在")
    return FileResponse(path)


class ReviewRequest(BaseModel):
    group: str
    loan_ids: list[str]
    action: str
    operator: str = "风控专员"
    note: str = ""


@app.post("/api/review")
def save_review(req: ReviewRequest):
    if req.action not in ("confirm_danger", "exclude_danger"):
        raise HTTPException(400, "无效复核动作")
    if not req.loan_ids:
        raise HTTPException(400, "请至少选择一笔贷款")
    new_status = "C" if req.action == "confirm_danger" else "B"
    business_status = "待重新上传" if new_status == "C" else "正常"
    placeholders = ",".join("?" for _ in req.loan_ids)
    with sqlite3.connect(DB_PATH) as conn:
        valid = [x[0] for x in conn.execute(
            f"select loan_id from loans where auto_group=? and loan_id in ({placeholders})",
            [req.group, *req.loan_ids],
        ).fetchall()]
        if not valid:
            raise HTTPException(400, "所选贷款不属于该相似组")
        conn.execute(
            f"update loans set verify_status=?, status=? where loan_id in ({','.join('?' for _ in valid)})",
            [new_status, business_status, *valid],
        )
        image_dirs = [x[0] for x in conn.execute(
            f"select image_dir from loans where loan_id in ({','.join('?' for _ in valid)})", valid
        ).fetchall()]
        conn.execute("""
            insert into reviews(auto_group,loan_ids,image_dirs,action,operator,note,created_at)
            values(?,?,?,?,?,?,?)
        """, (req.group, ",".join(valid), ",".join(image_dirs), req.action,
              req.operator, req.note, datetime.now().astimezone().isoformat(timespec="seconds")))
        conn.commit()
    return {"ok": True, "updated": len(valid), "verify_status": new_status, "status": business_status}


@app.get("/api/export/suspicious.csv")
def export_suspicious():
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["贷款编号", "客户名称", "业务类型", "验证状态", "相似组", "业务状态", "创建时间"])
    with sqlite3.connect(DB_PATH) as conn:
        for row in conn.execute("""
            select l.loan_id,coalesce(c.name,''),l.business_type,l.verify_status,l.auto_group,l.status,l.created_at
            from loans l left join customers c on c.customer_id=l.customer_id
            where l.verify_status in ('F','C') order by l.verify_status,l.created_at desc
        """):
            writer.writerow(row)
    payload = output.getvalue().encode("utf-8-sig")
    return StreamingResponse(iter([payload]), media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=suspicious_loans.csv"})


def _run_rebuild():
    _rebuild_state.update(running=True, progress=1, message="扫描未验证贷款", result=None, error="")
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            pending = [dict(x) for x in conn.execute(
                "select loan_id,image_dir from loans where verify_status='N' and face_feature is null"
            ).fetchall()]
            status_rows = conn.execute("select loan_id,verify_status,auto_group from loans").fetchall()
        new_features = {}
        total = len(pending)
        for i, row in enumerate(pending):
            _rebuild_state.update(progress=5 + int(70 * (i / max(total, 1))), message=f"提取特征 {i+1}/{total}")
            path = DATA_DIR / Path(row["image_dir"]).name / IMAGE_FILES["face"]
            if path.is_file():
                feat = model_bridge.extract_feature_from_path(str(path))
                if feat is not None:
                    new_features[row["loan_id"]] = feat
        with sqlite3.connect(DB_PATH) as conn:
            for loan_id, feat in new_features.items():
                conn.execute("update loans set face_feature=? where loan_id=?", (feat, loan_id))
            conn.commit()
            all_features = {r[0]: r[1] for r in conn.execute(
                "select loan_id,face_feature from loans where face_feature is not null"
            ).fetchall() if r[1]}
        _rebuild_state.update(progress=82, message="全库聚类与相似分组")
        cur_status = {r[0]: r[1] or "N" for r in status_rows}
        cur_group = {r[0]: r[2] or "" for r in status_rows}
        assignments = model_bridge.compute_similar_groups(
            all_features, cur_status=cur_status, cur_group=cur_group,
            threshold=float(_get_policy()["high_risk_threshold"]),
        )
        with sqlite3.connect(DB_PATH) as conn:
            for loan_id, (status, group) in assignments.items():
                conn.execute("update loans set verify_status=?,auto_group=? where loan_id=?", (status, group, loan_id))
            conn.commit()
        # 特征与分组已变化，同步重算 Tab1 缓存表（失败不影响重建结果上报）
        try:
            _refresh_dashboard_tables()
        except Exception:
            traceback.print_exc()
        result = {"new_features": len(new_features), "all_features": len(all_features),
                  "groups": len({g for s, g in assignments.values() if g}),
                  "pending_review": sum(1 for s, g in assignments.values() if s == "F")}
        _rebuild_state.update(running=False, progress=100, message="重建完成", result=result)
    except Exception as exc:
        traceback.print_exc()
        _rebuild_state.update(running=False, message="重建失败", error=str(exc))


@app.post("/api/rebuild")
def start_rebuild():
    if _rebuild_state["running"]:
        return _rebuild_state
    threading.Thread(target=_run_rebuild, daemon=True).start()
    return _rebuild_state


@app.get("/api/rebuild/status")
def rebuild_status():
    return _rebuild_state


# ---------------------------------------------------------------------------
# Tab2 上传临时表：上传影像数量不可预期，明细（类型/检测结果/图片数据）
# 落 detect_uploads 表，每次上传刷新旧批次；页面左侧列表点击行可回看
# 预览图与对应检测结果
# ---------------------------------------------------------------------------
def _ensure_detect_uploads(conn):
    conn.execute("""
        create table if not exists detect_uploads(
            id integer primary key autoincrement,
            batch_id text not null,
            filename text not null,
            relative_path text not null default '',
            image_type text not null default '',
            category text not null default '',
            is_sign_photo integer not null default 0,
            status text not null default '',
            top_score real,
            result_json text not null default '{}',
            image blob,
            created_at text not null
        )
    """)
    conn.execute(
        "create index if not exists idx_detect_uploads_batch on detect_uploads(batch_id)"
    )


def _record_detect_uploads(batch_id: str, rows: list[dict]):
    """写入本批上传明细：清除其他批次（临时表只留最新一批），本批追加/覆盖。"""
    if not batch_id or not rows:
        return
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    with sqlite3.connect(DB_PATH) as conn:
        _ensure_detect_uploads(conn)
        conn.execute("delete from detect_uploads where batch_id<>?", (batch_id,))
        conn.executemany(
            "insert into detect_uploads(batch_id,filename,relative_path,image_type,category,"
            "is_sign_photo,status,top_score,result_json,image,created_at) "
            "values(?,?,?,?,?,?,?,?,?,?,?)",
            [(batch_id, r["filename"], r.get("relative_path", ""), r.get("image_type", ""),
              r.get("category", ""), int(bool(r.get("is_sign_photo"))), r.get("status", ""),
              r.get("top_score"), json.dumps(r.get("result", {}), ensure_ascii=False),
              r.get("image"), now) for r in rows],
        )
        conn.commit()


@app.get("/api/detect-uploads")
def detect_uploads_list(batch_id: str = ""):
    """Tab2 本批上传明细（不含图片数据），左侧列表渲染与点击回看用。"""
    with sqlite3.connect(DB_PATH) as conn:
        _ensure_detect_uploads(conn)
        sql = ("select id,batch_id,filename,relative_path,image_type,category,is_sign_photo,"
               "status,top_score,result_json from detect_uploads")
        params = ()
        if batch_id:
            sql += " where batch_id=?"
            params = (batch_id,)
        raw_rows = conn.execute(sql + " order by id", params).fetchall()
    items = [{
        "id": r[0], "batch_id": r[1], "filename": r[2], "relative_path": r[3],
        "image_type": r[4], "category": r[5], "is_sign_photo": bool(r[6]),
        "status": r[7], "top_score": r[8], "result": json.loads(r[9] or "{}"),
    } for r in raw_rows]
    return {"items": items}


@app.get("/api/detect-uploads/{row_id}/image")
def detect_uploads_image(row_id: int):
    """Tab2 回看：返回上传影像原图（上传时的字节，避免依赖本地文件）。"""
    with sqlite3.connect(DB_PATH) as conn:
        _ensure_detect_uploads(conn)
        row = conn.execute(
            "select image from detect_uploads where id=?", (row_id,)
        ).fetchone()
    if not row or not row[0]:
        raise HTTPException(404, "影像不存在或已随新一批上传清除")
    return Response(content=row[0], media_type="image/jpeg")


@app.post("/api/detect")
async def detect(file: UploadFile = File(...), batch_id: str = Form(default="")):
    try:
        raw = await file.read()
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        # 前置类型闸门：规范命名的非面签影像（身份证/合同/流水）直接跳过
        # 特征提取与检索；命名不明的影像仍交给 search_similar 内部分类
        typed = model_bridge.classify_image_type(filename=file.filename or "")
        if typed["image_type"] and not typed["is_sign_photo"]:
            skip = {
                "available": True,
                "error": "",
                "category": typed["category"],
                "is_sign_photo": False,
                "status": f"非面签照片（{typed['category']}），跳过特征提取与相似检索",
                "results": [],
            }
            if batch_id:
                _record_detect_uploads(batch_id, [{
                    "filename": file.filename or "",
                    "image_type": typed["image_type"],
                    "category": typed["category"],
                    "is_sign_photo": False,
                    "status": "跳过",
                    "result": skip, "image": raw,
                }])
            return skip
        result = model_bridge.search_similar(image, top_k=5, query_loan_id="")
        policy = _get_policy()
        safe = {
            "available": result.get("available", False),
            "error": result.get("error", ""),
            "category": result.get("category", "未知"),
            "is_sign_photo": result.get("is_sign_photo", False),
            "status": result.get("status", "检测完成"),
            "policy": policy,
            "results": [],
        }
        groups_by_loan = {}
        if DB_PATH.exists():
            loan_ids = [str(row.get("loan_id", "")) for row in result.get("results", [])[:5] if row.get("loan_id")]
            if loan_ids:
                placeholders = ",".join("?" for _ in loan_ids)
                with sqlite3.connect(DB_PATH) as conn:
                    # FAISS 清单里的 loan_id 是影像目录名（loan_001），而 loans 表主键是
                    # LN2024xxxx 业务号，二者通过 image_dir 对齐；查到即统一改写为业务号。
                    for db_id, image_dir, grp in conn.execute(
                        f"select loan_id, coalesce(image_dir,''), coalesce(auto_group,'') "
                        f"from loans where loan_id in ({placeholders}) or image_dir in ({placeholders})",
                        loan_ids + loan_ids,
                    ).fetchall():
                        groups_by_loan[db_id] = (db_id, grp)
                        if image_dir:
                            groups_by_loan[image_dir] = (db_id, grp)
        for row in result.get("results", [])[:5]:
            raw_id = str(row.get("loan_id", "-"))
            db_id, group = groups_by_loan.get(raw_id, (raw_id, ""))
            # 26 维证据指标：对 FAISS 前五目标计算图像证据并标注超阈值维度
            evidence = {"available": False, "flagged": [], "error": None}
            # manifest 里的 path 是云端路径(/workspace/...)，本地不存在；
            # 用 DATA_DIR + loan_id(即 image_dir 名) 构造本地路径，与 /api/images 一致
            local_match = DATA_DIR / raw_id / IMAGE_FILES["face"]
            if local_match.is_file():
                evidence = evidence_bridge.compute_pair_evidence(
                    image, str(local_match), float(row.get("score", 0))
                )
            else:
                evidence["error"] = f"匹配图不存在: {local_match}"
            safe["results"].append(_apply_policy({
                "score": float(row.get("score", 0)),
                "loan_id": db_id,
                "auto_group": group,
                "relationship": str(row.get("relationship", "")),
                "evidence": {
                    "available": evidence.get("available", False),
                    "flagged": evidence.get("flagged", []),
                    "error": evidence.get("error"),
                },
            }, policy))
        safe["needs_review"] = any(row["needs_review"] for row in safe["results"])
        if batch_id:
            suspicious = [r for r in safe["results"] if r.get("is_suspicious")]
            _record_detect_uploads(batch_id, [{
                "filename": file.filename or "",
                "image_type": "face_signing" if safe.get("is_sign_photo") else "",
                "category": safe.get("category", ""),
                "is_sign_photo": bool(safe.get("is_sign_photo")),
                "status": "命中可疑" if suspicious else "未命中",
                "top_score": safe["results"][0]["score"] if safe["results"] else None,
                "result": safe, "image": raw,
            }])
        return safe
    except Exception as exc:
        return JSONResponse({"available": False, "error": str(exc)}, status_code=500)


def _group_key(path: str, fallback: str) -> str:
    """webkitRelativePath 首段作为样本组号；无目录层的散图统一归入 __ungrouped__。"""
    normalized = (path or fallback or "").replace("\\", "/").strip("/")
    if "/" not in normalized:
        return "__ungrouped__"
    return normalized.split("/", 1)[0] or "__ungrouped__"


def _map_loan_groups(loan_ids: list[str]) -> dict[str, tuple[str, str]]:
    """FAISS 清单 loan_id（影像目录名 loan_001）→ (loans 业务号, auto_group)。

    与 /api/detect 相同的对齐规则：loan_id 或 image_dir 任一命中即映射，
    查不到的保持原值、组号为空。"""
    mapping: dict[str, tuple[str, str]] = {}
    if not (DB_PATH.exists() and loan_ids):
        return mapping
    placeholders = ",".join("?" for _ in loan_ids)
    with sqlite3.connect(DB_PATH) as conn:
        for db_id, image_dir, grp in conn.execute(
            f"select loan_id, coalesce(image_dir,''), coalesce(auto_group,'') "
            f"from loans where loan_id in ({placeholders}) or image_dir in ({placeholders})",
            loan_ids + loan_ids,
        ).fetchall():
            mapping[db_id] = (db_id, grp)
            if image_dir:
                mapping[image_dir] = (db_id, grp)
    return mapping


@app.post("/api/detect-group")
async def detect_group(
    files: list[UploadFile] = File(...),
    relative_paths: list[str] = Form(default=[]),
    batch_id: str = Form(default=""),
):
    """文件夹上传按样本组检测。

    浏览器文件夹上传时每个文件附带 webkitRelativePath，路径首段即样本组号。
    每组依次完成：类型闸门（仅面签进入特征提取）→ 面签照片筛选 → 组内一致性
    校验（多张面签相似度）→ 选清晰度/完整度最高的一张 → 走既有 FAISS 相似检索。
    """
    try:
        policy = _get_policy()
        buckets: dict[str, list[dict]] = {}
        for index, file in enumerate(files):
            raw = await file.read()
            image = Image.open(io.BytesIO(raw)).convert("RGB")
            rel = relative_paths[index] if index < len(relative_paths) else file.filename
            key = _group_key(rel, file.filename or f"file_{index}")
            buckets.setdefault(key, []).append({
                "name": file.filename or f"file_{index}",
                "relative_path": rel or file.filename or f"file_{index}",
                "image": image,
                "raw": raw,
            })

        groups = []
        upload_rows = []
        for key, items in buckets.items():
            result = model_bridge.analyze_multi_signing_photos(items, top_k=5, query_loan_id="")
            # 检索结果对齐业务号与相似组（与 /api/detect 同规则）
            raw_ids = [str(r.get("loan_id", "")) for r in result.get("results", []) if r.get("loan_id")]
            loan_map = _map_loan_groups(raw_ids)
            mapped = []
            for row in result.get("results", [])[:5]:
                raw_id = str(row.get("loan_id", "-"))
                db_id, group = loan_map.get(raw_id, (raw_id, ""))
                mapped.append(_apply_policy({
                    "score": float(row.get("score", 0)),
                    "loan_id": db_id,
                    "auto_group": group,
                    "similar_group": str(row.get("similar_group", "")),
                    "relationship": str(row.get("relationship", "")),
                }, policy))
            safe_candidates = [
                {k: v for k, v in row.items() if not k.startswith("_")}
                for row in result.get("candidates", [])
            ]
            selected = result.get("selected")
            if selected:
                selected = {k: v for k, v in selected.items() if not k.startswith("_")}
            group_payload = {
                "group": key,
                "file_count": len(items),
                "available": bool(result.get("available", False)),
                "error": result.get("error", ""),
                "status": result.get("status", ""),
                "needs_review": any(row["needs_review"] for row in mapped)
                                or bool(result.get("needs_review", False)),
                "policy": policy,
                "same_person": result.get("same_person"),
                "same_shoot_or_reuse": result.get("same_shoot_or_reuse"),
                "min_internal_similarity": result.get("min_internal_similarity"),
                "avg_internal_similarity": result.get("avg_internal_similarity"),
                "selected": selected,
                "selected_reason": result.get("selected_reason", ""),
                "category": result.get("category", ""),
                "is_sign_photo": bool(result.get("is_sign_photo", False)),
                "detect_status": result.get("detect_status", ""),
                "candidates": safe_candidates,
                "results": mapped,
            }
            groups.append(group_payload)

            # 本组每个文件落临时表：非面签记跳过，面签记组级判定与检索结果
            top_score = mapped[0]["score"] if mapped else None
            group_status = ("组内命中" if any(r["is_suspicious"] for r in mapped)
                            else "组内需复核" if group_payload["needs_review"] else "正常")
            for item in items:
                cand = next((c for c in safe_candidates
                             if c.get("relative_path") == item["relative_path"]
                             and c.get("name") == item["name"]), {})
                is_sign = bool(cand.get("is_sign_photo"))
                upload_rows.append({
                    "filename": item["name"],
                    "relative_path": item["relative_path"],
                    "image_type": cand.get("image_type", ""),
                    "category": cand.get("category", group_payload["category"]),
                    "is_sign_photo": is_sign,
                    "status": group_status if is_sign else "跳过",
                    "top_score": top_score if is_sign else None,
                    "result": group_payload if is_sign else {
                        "category": cand.get("category", ""),
                        "is_sign_photo": False,
                        "status": f"非面签照片（{cand.get('category', '未知')}），跳过特征提取",
                        "results": [],
                    },
                    "image": item.get("raw"),
                })

        if batch_id:
            _record_detect_uploads(batch_id, upload_rows)
        return {"available": True, "group_count": len(groups), "groups": groups}
    except Exception as exc:
        traceback.print_exc()
        return JSONResponse({"available": False, "error": str(exc)}, status_code=500)


# ---------------------------------------------------------------------------
# Tab5：保存到库 — 查看/删除图像，删除记录
# ---------------------------------------------------------------------------
@app.get("/api/saved-images/{folder}")
def saved_images_list(folder: str):
    """列出某个保存文件夹中的影像文件名与预览 URL。folder 需精确匹配，防穿越。"""
    safe_folder = Path(folder).name  # 只取目录名，剥离 ../
    folder_path = (SAVED_DIR / safe_folder).resolve()
    if SAVED_DIR.resolve() not in folder_path.parents and folder_path != SAVED_DIR.resolve():
        raise HTTPException(403, "路径越界")
    if not folder_path.is_dir():
        raise HTTPException(404, "文件夹不存在")
    files = []
    for p in sorted(folder_path.iterdir()):
        if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"):
            files.append({
                "name": p.name,
                "size": p.stat().st_size,
                "url": f"/api/saved-images/{safe_folder}/{p.name}",
            })
    return {"folder": safe_folder, "files": files}


@app.get("/api/saved-images/{folder}/{filename}")
def saved_images_file(folder: str, filename: str):
    """读取单张保存影像，URL 路径规则与 /api/images 一致。"""
    safe_folder = Path(folder).name
    safe_name = Path(filename).name
    file_path = (SAVED_DIR / safe_folder / safe_name).resolve()
    if SAVED_DIR.resolve() not in file_path.parents:
        raise HTTPException(403, "路径越界")
    if not file_path.is_file():
        raise HTTPException(404, "影像不存在")
    return FileResponse(file_path)


@app.delete("/api/saved-images/{folder}/{filename}")
def saved_images_delete(folder: str, filename: str):
    """从保存到库文件夹中删除单张影像。"""
    safe_folder = Path(folder).name
    safe_name = Path(filename).name
    file_path = (SAVED_DIR / safe_folder / safe_name).resolve()
    if SAVED_DIR.resolve() not in file_path.parents:
        raise HTTPException(403, "路径越界")
    if not file_path.is_file():
        raise HTTPException(404, "影像不存在")
    file_path.unlink()
    # 更新 operation_records 中的 file_count（仅当文件夹还在）
    folder_path = file_path.parent
    remaining = len([p for p in folder_path.iterdir() if p.is_file()]) if folder_path.is_dir() else 0
    created_at_like = f"%{safe_folder}%"
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "update operation_records set file_count=? where folder=?",
                (remaining, safe_folder),
            )
            conn.commit()
    except sqlite3.Error:
        pass
    return {"ok": True, "deleted": safe_name, "remaining_files": remaining}


@app.delete("/api/operation-records/{folder}")
def delete_operation_record(folder: str):
    """删除保存记录：删除整个影像文件夹 + 删除 operation_records 行。"""
    safe_folder = Path(folder).name
    folder_path = (SAVED_DIR / safe_folder).resolve()
    if SAVED_DIR.resolve() not in folder_path.parents and folder_path != SAVED_DIR.resolve():
        raise HTTPException(403, "路径越界")
    deleted_count = 0
    if folder_path.is_dir():
        for p in folder_path.iterdir():
            try:
                if p.is_file():
                    p.unlink()
                    deleted_count += 1
            except OSError:
                pass
        try:
            folder_path.rmdir()
        except OSError:
            pass
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            "delete from operation_records where folder=?", (safe_folder,)
        )
        conn.commit()
        deleted_rows = cur.rowcount
    return {"ok": True, "folder": safe_folder, "deleted_files": deleted_count, "deleted_records": deleted_rows}


# ---------------------------------------------------------------------------
# /api/report 扩展：在 metrics 基础上追加 26 维证据阈值详情与阶段样例
# ---------------------------------------------------------------------------
_original_report = report
@app.get("/api/report")
def report_v2():
    base = _original_report() if callable(_original_report) else {"metrics": {}, "thresholds": {}, "source": ""}
    exp = experiments()
    if not isinstance(base, dict):
        return base
    # 26 维证据定义：仅把 EVIDENCE_THRESHOLDS 的阈值说明发给前端渲染
    evidence_defs = []
    try:
        from evidence_bridge import EVIDENCE_THRESHOLDS
        for k, spec in EVIDENCE_THRESHOLDS.items():
            evidence_defs.append({
                "name": k,
                "threshold": spec["threshold"],
                "direction": spec["dir"],
                "label": spec["label"],
            })
    except Exception:
        pass
    base["evidence"] = evidence_defs
    base["experiments_available"] = exp.get("available", False) if isinstance(exp, dict) else False
    base["stage1_examples"] = (exp.get("stage1") or {}).get("examples", []) if isinstance(exp, dict) else []
    base["threshold_rows"] = ((exp.get("threshold") or {}).get("rows", []))[:10] if isinstance(exp, dict) else []
    base["stage2_type_counts"] = (exp.get("stage2") or {}).get("type_counts", {}) if isinstance(exp, dict) else {}
    base["monitoring"] = (exp.get("monitoring") or {}).get("summary", {}) if isinstance(exp, dict) else base.get("monitoring", {})
    base["summary_detail"] = (exp.get("summary") or {}) if isinstance(exp, dict) else {}
    # 统一报告页与实验页的产物来源：两者必须使用同一份当前数据版本。
    # 旧版 base metrics 可能来自历史模型桥/旧 FAISS 产物，导致 support、F1、
    # 监控配对数与实验页不一致；实验接口已按 OUTPUT_DIRS 选定当前正式产物。
    if isinstance(exp, dict) and exp.get("available"):
        classifier = exp.get("classifier") or {}
        stage1 = exp.get("stage1") or {}
        pair = stage1.get("metrics") or {}
        group = stage1.get("group_metrics") or {}
        base["metrics"] = {
            "available": True,
            "classifier": {
                "available": True,
                "accuracy": classifier.get("accuracy"),
                "macro_f1": classifier.get("macro_f1"),
                "support": sum(int(v.get("support", 0)) for v in (classifier.get("per_class") or {}).values() if isinstance(v, dict)),
            },
            "two_stage": {
                "available": True,
                "pair_precision": pair.get("precision"),
                "pair_recall": pair.get("recall"),
                "pair_f1": pair.get("f1"),
                "pair_roc_auc": pair.get("roc_auc"),
                "group_precision": group.get("precision"),
                "group_recall": group.get("recall"),
                "group_f1": group.get("f1"),
            },
            "fraud_monitoring": (exp.get("monitoring") or {}).get("summary", {}),
        }
    # 指标回退：实验数据可用但模型指标缺失时，用实验的 classifier/stage1 指标填充，
    # 保证报告页 KPI 与 Stage1 阈值表在有实验产物时也能展示真实数据
    if isinstance(exp, dict) and exp.get("available") and not (base.get("metrics") or {}).get("available"):
        classifier = exp.get("classifier") or {}
        stage1 = exp.get("stage1") or {}
        pair = stage1.get("metrics") or {}
        group = stage1.get("group_metrics") or {}
        base["metrics"] = {
            "available": True,
            "classifier": {
                "available": True,
                "accuracy": classifier.get("accuracy"),
                "macro_f1": classifier.get("macro_f1"),
                "support": sum(
                    int(v.get("support", 0)) for v in (classifier.get("per_class") or {}).values()
                    if isinstance(v, dict)
                ),
            },
            "two_stage": {
                "available": True,
                "pair_precision": pair.get("precision"),
                "pair_recall": pair.get("recall"),
                "pair_f1": pair.get("f1"),
                "pair_roc_auc": pair.get("roc_auc"),
                "group_precision": group.get("precision"),
                "group_recall": group.get("recall"),
                "group_f1": group.get("f1"),
            },
            "fraud_monitoring": {"available": True, **base.get("monitoring", {})},
        }
    # 加入当前生效的风险策略阈值
    try:
        policy = _get_policy()
        base.setdefault("thresholds", {}).update({
            "high_risk_threshold": policy["high_risk_threshold"],
            "policy_version": policy["policy_version"],
            "policy_updated_at": policy["updated_at"],
            "policy_operator": policy["operator"],
            "source_file": f"{DB_PATH}#risk_policy",
        })
    except Exception:
        pass
    return base


# ---------------------------------------------------------------------------
# /api/suspicious 扩展：补充同组贷款计数（便于辐射图按组筛选）
# ---------------------------------------------------------------------------
_original_suspicious = suspicious
@app.get("/api/suspicious")
def suspicious_v2(limit: int = 100):
    base = _original_suspicious(limit)
    if not isinstance(base, dict) or "items" not in base:
        return base
    items = base["items"]
    # 给每笔补充同组贷款数量（不含自身）
    group_counter = {}
    for it in items:
        g = it.get("auto_group") or ""
        if g:
            group_counter[g] = group_counter.get(g, 0) + 1
    for it in items:
        g = it.get("auto_group") or ""
        it["group_size"] = group_counter.get(g, 0)
    base["groups_count"] = len(group_counter)
    return base


if __name__ == "__main__":
    import uvicorn
    print(f"[server] UI_DIR   = {UI_DIR}")
    print(f"[server] DB_PATH  = {DB_PATH} (exists={DB_PATH.exists()})")
    print(f"[server] DATA_DIR = {DATA_DIR} (exists={DATA_DIR.is_dir()})")

    def _warm_model():
        """后台预热模型：不阻塞 uvicorn 启动与 /api/health，加载完成前检测请求会等待同一把锁。"""
        try:
            info = model_bridge.get_index_stats()
            _MODEL_STATUS["ready"] = True
            _MODEL_STATUS["total"] = info.get("total", 0)
        except Exception as e:
            _MODEL_STATUS["ready"] = False
            print(f"[server] 模型后台预热失败（网页仍可用，检测时再试加载）: {e}")

    threading.Thread(target=_warm_model, daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=5173)
