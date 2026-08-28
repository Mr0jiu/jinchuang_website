"""
模型桥接层：把 jinchuang_v4/code 的模型能力接入网页
=================================================

职责：
- 懒加载 SigLIP2 特征提取器 + 分类器 + FAISS 索引（首次调用才加载，避免启动卡顿）
- 提供三个核心能力：
    1) search_similar(image, top_k)  上传图片 → 特征提取 → FAISS 检索相似面签
    2) get_index_stats()             FAISS 索引统计
    3) get_model_metrics()           读取 outputs/mvp 的真实评估指标
- 兼容两种 FAISS 索引格式：
    a) src/retrieval.py 的 SimilaritySearch（checkpoints/faiss_index.bin + _meta.pkl）
    b) mvp/pipeline.py 的 face_signing.faiss + face_manifest.csv

路径配置：
- 通过环境变量 MODEL_CODE_DIR / DATA_DIR / FAISS_INDEX 覆盖默认路径
- 本地缺依赖（torch/faiss）时 is_available()=False，网页降级为纯数据库模式
"""
from __future__ import annotations

import os
import json
import pickle
import threading
import hashlib
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# 路径配置（网页→模型代码的桥梁）
# ---------------------------------------------------------------------------
# 自适应定位：从 website 目录向上搜索，找到 jinchuang_v4/code 的那一层
# 兼容两种部署结构：
#   本地: 金创/git/jinchuang/website      + 金创/jinchuang_v4/code   (website 多嵌套两层)
#   云端: <root>/website                   + <root>/jinchuang_v4/code (两者平级)
_WEBSITE_DIR = Path(__file__).resolve().parent


def _find_sibling(name: str, *sub: str) -> Path | None:
    """从 website 向上找 jinchuang_v4 兄弟目录，返回首个命中的 <它>/<sub> 路径。

    Args:
        name: 兄弟目录名（"jinchuang_v4"）
        sub: 子路径片段（如 ("code",)）
    Returns:
        命中的绝对路径，或 None
    """
    cur = _WEBSITE_DIR
    for _ in range(6):  # 最多向上 6 层，足够覆盖常见嵌套
        candidate = cur / name
        if candidate.is_dir():
            return candidate.joinpath(*sub)
        if cur.parent == cur:  # 到根了
            break
        cur = cur.parent
    return None


# 模型代码根目录（环境变量优先，否则自适应搜索）
_env_code = os.environ.get("MODEL_CODE_DIR", "")
_default_code = (
    Path(_env_code) if _env_code
    else _find_sibling("jinchuang_v4", "code")
)
MODEL_CODE_DIR = _default_code if _default_code else (
    _WEBSITE_DIR.parent.parent.parent / "jinchuang_v4" / "code"  # 兜底（保留本地旧行为）
)

# 图片根目录（环境变量优先，否则自适应搜索）
_env_data = os.environ.get("DATA_DIR", "")
_default_data = (
    Path(_env_data) if _env_data
    else _find_sibling("jinchuang_v4", "extracted")
)
DATA_DIR = _default_data if _default_data else (
    _WEBSITE_DIR.parent.parent.parent / "jinchuang_v4" / "extracted"  # 兜底
)

# 模型评估产物目录：环境变量优先；本仓库提交包输出优先于旧 outputs/mvp。
_env_output = os.environ.get("MVP_OUTPUT_DIR", "").strip()
_output_candidates = [
    Path(_env_output) if _env_output else None,
    _WEBSITE_DIR.parent / "submission_ready" / "03_model_artifacts" / "outputs" / "mvp",
    _WEBSITE_DIR.parent / "submission_ready" / "01_source_code" / "outputs" / "mvp",
    MODEL_CODE_DIR / "outputs" / "mvp",
    _WEBSITE_DIR.parent / "outputs" / "mvp",
]
MVP_OUTPUT_DIR = next(
    (path for path in _output_candidates if path and path.exists()),
    MODEL_CODE_DIR / "outputs" / "mvp",
)

# FAISS 索引优先级：环境变量 > SimilaritySearch 默认 > MVP face_signing.faiss
_FAISS_INDEX_ENV = os.environ.get("FAISS_INDEX", "")
DEFAULT_FAISS_INDEX = (
    Path(_FAISS_INDEX_ENV) if _FAISS_INDEX_ENV
    else MODEL_CODE_DIR / "checkpoints" / "faiss_index.bin"
)
MVP_FAISS_INDEX = MVP_OUTPUT_DIR / "face_signing.faiss"
MVP_MANIFEST = MVP_OUTPUT_DIR / "face_manifest.csv"


# ---------------------------------------------------------------------------
# 懒加载全局状态
# ---------------------------------------------------------------------------
_state: dict[str, Any] = {"ready": False}

# 模型懒加载锁：后台预热线程与首次检测请求可能并发进入 _try_init，
# 保证同一时刻只有一个线程真正执行加载，避免重复加载占用双份内存。
_LOAD_LOCK = threading.Lock()


def _try_init() -> bool:
    """懒加载模型组件。成功返回 True，已加载直接返回。

    线程安全：后台预热线程与首次检测请求可能并发进入，用锁保证只加载一次。
    """
    if _state["ready"]:
        return True
    with _LOAD_LOCK:
        if _state["ready"]:
            return True
        return _load_once()


def _load_once() -> bool:
    """实际执行模型加载（仅由 _try_init 在持锁状态下调用）。"""
    # 检查目录是否存在
    if not MODEL_CODE_DIR.is_dir():
        print(f"[bridge] 模型代码目录不存在: {MODEL_CODE_DIR}")
        return False

    try:
        import sys
        if str(MODEL_CODE_DIR) not in sys.path:
            sys.path.insert(0, str(MODEL_CODE_DIR))

        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

        import yaml
        import numpy as np
        import torch
        from PIL import Image as _PILImage  # noqa: F401

        # 加载模型配置
        cfg_path = MODEL_CODE_DIR / "config.yaml"
        with open(cfg_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # 加载 SigLIP2 特征提取器 + 分类器 + 预处理
        from src.model import SigLIP2FeatureExtractor
        from src.classifier import ImageClassifier
        from src.preprocessing import PreprocessingPipeline

        # 模型权重来源：优先本地快照（如仓库根 siglip2/，config.json + model.safetensors
        # 齐全即视为可用），避免离线/弱网时从 HF 下载 1.5GB；找不到再回退 config 里的
        # 仓库名在线加载（云端行为不变）。
        model_source = config["model"]["name"]
        _local_candidates = []
        _env_local = os.environ.get("SIGLIP2_LOCAL_DIR", "").strip()
        if _env_local:
            _local_candidates.append(Path(_env_local))
        _local_candidates.append(_WEBSITE_DIR.parent / "siglip2")
        for _cand in _local_candidates:
            if (_cand / "config.json").is_file() and (_cand / "model.safetensors").is_file():
                model_source = str(_cand)
                print(f"[bridge] 使用本地模型快照: {_cand}")
                break

        _device = "cuda" if torch.cuda.is_available() else "cpu"
        extractor = SigLIP2FeatureExtractor(model_name=model_source, device=_device)
        print(f"[bridge] SigLIP2 device = {_device}")
        classifier = ImageClassifier(
            model=extractor.model,
            processor=extractor.processor,
            categories=config["classifier"]["categories"],
            device=extractor.device,
        )
        preprocessor = PreprocessingPipeline(config.get("preprocessing", {}))

        # 加载 FAISS 索引 + 元数据
        faiss_index, metadata, index_source = _load_faiss_index()

        # 构建 loan_id → similar_group 映射（差异化阈值用）
        loan_to_sg: dict[str, str] = {}
        for m in metadata:
            sg = m.get("similar_group", "")
            loan = m.get("loan_id", "")
            if sg and loan:
                loan_to_sg[loan] = sg

        _state.update({
            "ready": True,
            "config": config,
            "extractor": extractor,
            "classifier": classifier,
            "preprocessor": preprocessor,
            "faiss_index": faiss_index,
            "metadata": metadata,
            "index_source": index_source,
            "loan_to_sg": loan_to_sg,
            "np": np,
            "torch": torch,
        })
        print(f"[bridge] 模型加载完成 | 索引来源: {index_source} | "
              f"记录数: {faiss_index.ntotal if faiss_index else 0}")
        return True
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[bridge] 模型加载失败，网页将以纯数据库模式运行: {e}")
        return False


def _load_faiss_index():
    """加载 FAISS 索引，兼容两种格式。

    Returns:
        (index, metadata_list, source_desc)
    """
    import faiss

    # 格式 A: SimilaritySearch（faiss_index.bin + _meta.pkl）
    if DEFAULT_FAISS_INDEX.exists():
        index = faiss.read_index(str(DEFAULT_FAISS_INDEX))
        meta_path = str(DEFAULT_FAISS_INDEX).replace(".bin", "_meta.pkl")
        metadata = []
        if os.path.exists(meta_path):
            with open(meta_path, "rb") as f:
                data = pickle.load(f)
            metadata = data.get("metadata", []) if isinstance(data, dict) else data
        return index, metadata, f"SimilaritySearch({DEFAULT_FAISS_INDEX.name})"

    # 格式 B: MVP 管线（face_signing.faiss + face_manifest.csv）
    if MVP_FAISS_INDEX.exists():
        index = faiss.read_index(str(MVP_FAISS_INDEX))
        metadata = _load_manifest_metadata()
        return index, metadata, f"MVP({MVP_FAISS_INDEX.name})"

    return None, [], "未找到索引"


def _load_manifest_metadata() -> list[dict]:
    """从 face_manifest.csv 构造元数据（MVP 格式）。"""
    if not MVP_MANIFEST.exists():
        return []
    import csv
    metadata = []
    # utf-8-sig：该清单由云端写出时带 BOM，若按裸 utf-8 读，首列键会变成
    # "﻿loan_id"，导致所有记录的 loan_id 取值为空、复核跳转与缩略图失效。
    with open(MVP_MANIFEST, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metadata.append({
                "loan_id": row.get("loan_id", ""),
                "path": row.get("path", ""),
                "cat_name": "面签照片",
                "image_type": "face_signing",
                "similar_group": row.get("similar_group", ""),
                "business_type": row.get("business_type", ""),
            })
    return metadata


# ---------------------------------------------------------------------------
# 公共 API
# ---------------------------------------------------------------------------
def is_available() -> bool:
    """探测模型是否可用（不强制加载）。"""
    return _try_init()


def get_status() -> dict:
    """返回模型状态摘要（供网页状态栏展示）。"""
    if not _try_init():
        return {
            "available": False,
            "reason": f"模型代码目录或依赖缺失: {MODEL_CODE_DIR}",
            "index_size": 0,
        }
    idx = _state["faiss_index"]
    return {
        "available": True,
        "index_size": idx.ntotal if idx is not None else 0,
        "index_source": _state.get("index_source", ""),
        "model_name": _state["config"]["model"]["name"],
    }


def search_similar(
    image,
    top_k: int = 5,
    query_loan_id: str = "",
    force_sign_photo: bool = False,
) -> dict:
    """上传图片 → 预处理 → 分类 → 特征提取 → FAISS 检索。

    Args:
        image: PIL.Image 或图片路径字符串
        top_k: 返回前 K 条
        query_loan_id: 查询图片所属 loan_id（用于差异化阈值，空则自动识别）
        force_sign_photo: 调用方已完成类型闸门时，强制按面签照继续检索。

    Returns:
        {
            "category": str,            # 分类名称
            "is_sign_photo": bool,     # 是否面签照片
            "sign_confidence": float,
            "results": [                # 相似结果（保留自身/同图命中，风险判定单独给出）
                {"score": float, "loan_id": str, "path": str,
                 "similar_group": str, "relationship": str, "is_suspicious": bool}
            ],
            "status": str,              # 状态摘要
            "error": str | None,
        }
    """
    if not _try_init():
        return {"available": False, "error": "模型不可用", "results": []}

    try:
        from PIL import Image
        np = _state["np"]
        torch = _state["torch"]
        extractor = _state["extractor"]
        classifier = _state["classifier"]
        preprocessor = _state["preprocessor"]
        config = _state["config"]
        idx = _state["faiss_index"]
        metadata = _state["metadata"]
        loan_to_sg = _state["loan_to_sg"]

        if idx is None or idx.ntotal == 0:
            return {"available": True, "error": "FAISS 索引为空",
                    "results": [], "status": "索引为空"}

        # 读图
        if isinstance(image, str):
            image = Image.open(image)
        image = preprocessor(image).convert("RGB")
        img_tensor = torch.tensor(
            np.array(image.resize((224, 224))).transpose(2, 0, 1)
        ).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(extractor.device)

        # 分类
        _, cat_name, _ = classifier.classify(img_tensor)
        is_sign, sign_conf = classifier.is_sign_photo(img_tensor)
        if force_sign_photo:
            is_sign = True
            cat_name = "面签照片"

        if not is_sign:
            return {
                "available": True,
                "category": cat_name,
                "is_sign_photo": False,
                "sign_confidence": float(sign_conf),
                "results": [],
                "status": f"非面签照片（{cat_name}），跳过相似度检测",
                "error": None,
            }

        # 特征提取
        with torch.no_grad():
            feat = extractor.extract(img_tensor)
        feat_np = feat.cpu().numpy().astype(np.float32)
        norm = np.linalg.norm(feat_np)
        if norm > 0:
            feat_np = feat_np / norm

        # 检索（多取几条用于去重和补足展示结果）
        import faiss
        k = min(top_k + 6, idx.ntotal)
        sims, inds = idx.search(feat_np.astype(np.float32), k)

        # 自动识别 loan_id
        if not query_loan_id:
            for s, i in zip(sims[0], inds[0]):
                if float(s) >= 0.999 and 0 <= i < len(metadata):
                    query_loan_id = metadata[i].get("loan_id", "")
                    break

        dyn = config["retrieval"].get("dynamic_threshold", {})
        use_dynamic = dyn.get("enabled", False)
        single_threshold = config["retrieval"]["similarity_threshold"]

        results = []
        seen_paths = set()
        for s, i in zip(sims[0], inds[0]):
            if i < 0 or i >= len(metadata):
                continue
            score = float(s)
            m = metadata[i]
            path = m.get("path", "")

            # 路径去重
            if path in seen_paths:
                continue
            seen_paths.add(path)

            # 阈值判定
            if use_dynamic:
                threshold, rel = _effective_threshold(
                    query_loan_id, m, dyn, single_threshold, loan_to_sg
                )
            else:
                threshold, rel = single_threshold, ""

            is_suspicious = bool(score >= threshold and rel != "self")
            results.append({
                "score": score,
                "loan_id": m.get("loan_id", ""),
                "path": path,
                "similar_group": m.get("similar_group", ""),
                "business_type": m.get("business_type", ""),
                "relationship": rel,
                "threshold": threshold,
                "is_suspicious": is_suspicious,
            })
            if len(results) >= top_k:
                break

        # 状态
        suspicious = [r for r in results if r["is_suspicious"]]
        if not results:
            status = "无相似结果"
        elif suspicious:
            status = f"发现 {len(suspicious)} 条可疑相似"
        elif any(r.get("relationship") == "self" for r in results):
            status = "命中库内同图/自身记录（不计为跨客户风险）"
        else:
            max_s = max(r["score"] for r in results)
            status = f"最高相似度 {max_s:.3f}（低于阈值，安全）"

        return {
            "available": True,
            "category": cat_name,
            "is_sign_photo": True,
            "sign_confidence": float(sign_conf),
            "results": results,
            "status": status,
            "query_loan_id": query_loan_id,
            "error": None,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"available": True, "error": str(e), "results": [], "status": f"出错: {e}"}


def _effective_threshold(query_loan_id, result_meta, dyn, default, loan_to_sg):
    """差异化阈值（复刻 main.py 逻辑）。"""
    result_loan = result_meta.get("loan_id", "")
    result_sg = result_meta.get("similar_group", "")

    if query_loan_id and query_loan_id == result_loan:
        return 1.0, "self"

    query_sg = loan_to_sg.get(query_loan_id, "") if query_loan_id else ""
    if query_sg and result_sg and query_sg == result_sg:
        return dyn.get("same_customer", 0.92), "same_customer"

    return dyn.get("fraud", 0.75), "cross_customer"


def get_index_stats() -> dict:
    """返回 FAISS 索引统计。"""
    if not _try_init():
        return {"available": False, "total": 0}

    idx = _state["faiss_index"]
    metadata = _state["metadata"]
    if idx is None:
        return {"available": True, "total": 0, "category_distribution": {}}

    from collections import Counter
    cat_counter = Counter(m.get("cat_name", "未知") for m in metadata)
    return {
        "available": True,
        "total": idx.ntotal,
        "index_source": _state.get("index_source", ""),
        "category_distribution": dict(cat_counter.most_common()),
        "similar_groups": len({m.get("similar_group", "")
                               for m in metadata if m.get("similar_group")}),
    }


def get_face_corpus():
    """返回全量面签语料 (特征矩阵, manifest 元数据列表)，供首页统计等使用。

    矩阵按行 L2 归一化（与检索同口径），行序与 metadata 一一对应。
    模型或索引不可用时返回 (None, [])。语料只含面签影像（索引构建时
    仅收面签），不混入身份证/合同/流水等其他类别。
    """
    if not _try_init():
        return None, []
    idx = _state["faiss_index"]
    np = _state["np"]
    metadata = _state["metadata"]
    if idx is None or idx.ntotal == 0 or not metadata:
        return None, []
    import faiss
    vecs = np.stack([idx.reconstruct(i) for i in range(idx.ntotal)]).astype(np.float32)
    vecs = vecs / np.clip(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9, None)
    return vecs, metadata


# ---------------------------------------------------------------------------
# 五类影像类型判定（智能检测多图上传的前置闸门：只有面签进入特征提取）
# ---------------------------------------------------------------------------
# 五类口径与影像目录文件名一致；TYPE_LABELS 提供前端展示的中文名。
IMAGE_TYPES = ("face_signing", "idcardfront", "idcardback", "contract", "bank_statement")
TYPE_LABELS = {
    "face_signing": "面签照片",
    "idcardfront": "身份证正面",
    "idcardback": "身份证背面",
    "contract": "合同",
    "bank_statement": "银行流水",
}
# 文件名规则：语料与业务上传普遍使用规范命名（face_signing.jpg 等），
# 比零样本分类器更稳（分类器曾把身份证判成面签），故规则优先。
_FILENAME_RULES = (
    ("face_signing", "face_signing"), ("facesigning", "face_signing"), ("面签", "face_signing"),
    ("id_card_front", "idcardfront"), ("idcardfront", "idcardfront"), ("身份证正面", "idcardfront"),
    ("id_card_back", "idcardback"), ("idcardback", "idcardback"), ("身份证背面", "idcardback"),
    ("contract", "contract"), ("合同", "contract"),
    ("bank_statement", "bank_statement"), ("bankstatement", "bank_statement"), ("流水", "bank_statement"),
)
# 分类器中文类别 → 五类口径（身份证不分正反面，统一记正面档，正反面靠文件名区分）
_CLASSIFIER_TYPE_MAP = {
    "面签照片": "face_signing",
    "身份证": "idcardfront",
    "合同": "contract",
    "银行流水": "bank_statement",
}


def classify_image_type(image=None, filename: str = "") -> dict:
    """判定影像五类类型：bank_statement/contract/face_signing/idcardback/idcardfront。

    判定顺序：文件名规范命名规则（不依赖模型，最稳）→ SigLIP2 零样本分类器
    （现有能力兜底）。只有 face_signing 是面签，其余四类均不进入特征提取。

    Args:
        image: PIL.Image；仅当文件名规则未命中时才需要（走分类器）
        filename: 文件名或相对路径（取其规范命名子串）

    Returns:
        {"image_type": 五类之一 | None(无法判定), "category": 中文类别名,
         "is_sign_photo": bool, "source": "filename" | "model" | "none"}
    """
    normalized = (filename or "").replace("\\", "/").lower()
    for token, image_type in _FILENAME_RULES:
        if token in normalized:
            return {"image_type": image_type, "category": TYPE_LABELS[image_type],
                    "is_sign_photo": image_type == "face_signing", "source": "filename"}

    if image is None:
        return {"image_type": None, "category": "", "is_sign_photo": False, "source": "none"}
    if not _try_init():
        return {"image_type": None, "category": "", "is_sign_photo": False, "source": "none"}

    classifier = _state["classifier"]
    torch = _state["torch"]
    np = _state["np"]
    extractor = _state["extractor"]
    preprocessor = _state["preprocessor"]
    processed = preprocessor(image.convert("RGB")).convert("RGB")
    tensor = torch.tensor(
        np.array(processed.resize((224, 224))).transpose(2, 0, 1)
    ).float() / 255.0
    tensor = tensor.unsqueeze(0).to(extractor.device)
    _, cat_name, _ = classifier.classify(tensor)
    image_type = _CLASSIFIER_TYPE_MAP.get(str(cat_name))
    if image_type is None:
        return {"image_type": None, "category": str(cat_name), "is_sign_photo": False,
                "source": "model"}
    return {"image_type": image_type, "category": TYPE_LABELS[image_type],
            "is_sign_photo": image_type == "face_signing", "source": "model"}


def get_model_metrics() -> dict:
    """读取 outputs/mvp 下真实评估指标。"""
    def _f(v):
        """指标值统一转 float。

        outputs/mvp 的 JSON 里部分指标存成字符串（如 '0.9760914760914761'），
        若原样透传给网页，Tab3 回调里对这些值做 :.4f 格式化、或交给 matplotlib
        绘图，会抛 TypeError/ValueError，导致回调崩溃、Gradio 返回空响应，
        前端报 "Unexpected end of JSON input"。统一转数值；空值/异常值降级为 0.0。
        """
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0

    metrics = {"available": False}

    # 分类器指标
    cls_path = MVP_OUTPUT_DIR / "classification_metrics.json"
    if cls_path.exists():
        try:
            with open(cls_path, "r", encoding="utf-8") as f:
                cls_data = json.load(f)
            test = cls_data.get("test", {})
            metrics["classifier"] = {
                "available": True,
                "accuracy": _f(test.get("accuracy")),
                "macro_f1": _f(test.get("macro_f1")),
                "support": sum(
                    c.get("support", 0) for c in test.get("per_class", {}).values()
                ),
            }
        except Exception as e:
            metrics["classifier"] = {"available": False, "error": str(e)}
    else:
        metrics["classifier"] = {"available": False}

    # 两阶段管线指标
    ts_path = MVP_OUTPUT_DIR / "two_stage_summary.json"
    if ts_path.exists():
        try:
            with open(ts_path, "r", encoding="utf-8") as f:
                ts = json.load(f)
            pair = ts.get("stage1", {}).get("pair_level_split", {}).get("metrics", {})
            group = ts.get("stage1", {}).get("group_level_split", {}).get("metrics", {})
            metrics["two_stage"] = {
                "available": True,
                "pair_precision": _f(pair.get("precision")),
                "pair_recall": _f(pair.get("recall")),
                "pair_f1": _f(pair.get("f1")),
                "pair_roc_auc": _f(pair.get("roc_auc")),
                "group_precision": _f(group.get("precision")),
                "group_recall": _f(group.get("recall")),
                "group_f1": _f(group.get("f1")),
            }
        except Exception as e:
            metrics["two_stage"] = {"available": False, "error": str(e)}
    else:
        metrics["two_stage"] = {"available": False}

    # 欺诈监控统计
    fm_path = MVP_OUTPUT_DIR / "fraud_monitoring_summary.json"
    if fm_path.exists():
        try:
            with open(fm_path, "r", encoding="utf-8") as f:
                fm = json.load(f)
            metrics["fraud_monitoring"] = {"available": True, **fm}
        except Exception as e:
            metrics["fraud_monitoring"] = {"available": False, "error": str(e)}
    else:
        metrics["fraud_monitoring"] = {"available": False}

    metrics["available"] = any(
        v.get("available") for v in [
            metrics.get("classifier", {}),
            metrics.get("two_stage", {}),
            metrics.get("fraud_monitoring", {}),
        ] if isinstance(v, dict)
    )
    return metrics


# ---------------------------------------------------------------------------
# 特征提取与相似分组（Tab2 实时流程：照片入库 → 特征提取 → 相似分组 → 复核）
# ---------------------------------------------------------------------------
def extract_feature_from_path(image_path: str) -> bytes | None:
    """对单张图片路径提取 SigLIP2 768 维特征，返回 L2 归一化后的 float32 bytes。

    供 Tab2 的"批量特征提取"按钮逐张调用。模型不可用时返回 None。

    Args:
        image_path: 图片绝对路径

    Returns:
        768 维 float32（L2 归一化）的 bytes，可直接写入 loans.face_feature；失败返回 None
    """
    if not _try_init():
        return None
    try:
        from PIL import Image
        np = _state["np"]
        torch = _state["torch"]
        extractor = _state["extractor"]
        preprocessor = _state["preprocessor"]

        if not image_path or not os.path.isfile(image_path):
            return None

        image = Image.open(image_path)
        image = preprocessor(image).convert("RGB")
        img_tensor = torch.tensor(
            np.array(image.resize((224, 224))).transpose(2, 0, 1)
        ).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)

        with torch.no_grad():
            feat = extractor.extract(img_tensor)
        feat_np = feat.cpu().numpy().astype(np.float32)
        norm = np.linalg.norm(feat_np)
        if norm > 0:
            feat_np = feat_np / norm
        return feat_np.tobytes()
    except Exception as e:
        print(f"[bridge] 特征提取失败 {image_path}: {e}")
        return None


def compute_similar_groups(
    features: dict[str, bytes],
    cur_status: dict[str, str] | None = None,
    cur_group: dict[str, str] | None = None,
    threshold: float = 0.75,
) -> dict[str, tuple[str, str]]:
    """对全库已提取特征的面签做两两相似度聚类，产出每个 loan 的新 (verify_status, auto_group)。

    稳定性设计（Tab2 增量流程的命脉）：
    - 组号锁定：簇内若已有带 auto_group 的成员（F/B/C），沿用其组号（多个旧组被新 N
      桥接合并时取字典序最小者为主号）；仅当簇内全是 N/R（全新簇）才分配新号，
      且新号一次分配后不再随成员变化重算 → 幂等、组号不漂移。
    - R 升级：原 R 被新数据桥接进簇时升级为 F（解决「新 N 撞 R」的自洽问题）。
    - 人工结论不动：已是 B/C 的保持原 status（仅 auto_group 可能因组号合并而变）。

    Args:
        features: {loan_id: face_feature_bytes}，全库特征（N 的新提 + F/R 的已存）
        cur_status: {loan_id: 当前 verify_status}，用于判断锁定 / 升级 / 保持
        cur_group: {loan_id: 当前 auto_group}，用于沿用旧组号
        threshold: 相似度阈值，>= 视为相似（默认 0.75，对应 Stage-1 fraud 阈值）

    Returns:
        {loan_id: (new_verify_status, new_auto_group)}；features 为空时返回 {}。
    """
    cur_status = cur_status or {}
    cur_group = cur_group or {}
    if not features:
        return {}

    # 解码特征向量
    import numpy as _np
    vecs: dict[str, _np.ndarray] = {}
    for lid, blob in features.items():
        try:
            v = _np.frombuffer(blob, dtype=_np.float32)
            if v.size > 0:
                vecs[lid] = v
        except Exception:
            continue
    if not vecs:
        return {}

    lids = list(vecs.keys())

    # 并查集
    parent = {lid: lid for lid in lids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # 两两比对（数据规模有限，O(n²) 可接受）
    for i in range(len(lids)):
        for j in range(i + 1, len(lids)):
            a, b = lids[i], lids[j]
            sim = float(_np.dot(vecs[a], vecs[b]))  # 已 L2 归一化，点积即余弦
            if sim >= threshold:
                union(a, b)

    # 聚簇
    clusters: dict[str, list[str]] = {}
    for lid in lids:
        clusters.setdefault(find(lid), []).append(lid)

    result: dict[str, tuple[str, str]] = {}
    for members in clusters.values():
        members_sorted = sorted(members)
        if len(members_sorted) >= 2:
            # 组号锁定：簇内已有的 auto_group（来自 F/B/C 成员）沿用
            existing = sorted({
                cur_group.get(m, "")
                for m in members_sorted
                if cur_status.get(m, "N") in ("F", "B", "C") and cur_group.get(m, "")
            })
            if existing:
                group_code = existing[0]                 # 多旧组合并时取字典序最小者为主号
            else:
                # 全新簇：首次分配新号（仅此一次，之后锁定，不再随成员变化重算）
                digest = hashlib.md5(",".join(members_sorted).encode("utf-8")).hexdigest()[:8]
                group_code = f"SG_{digest}"
            for m in members_sorted:
                st = cur_status.get(m, "N")
                if st == "N":
                    new_st = "F"                          # 新数据落入相似组
                elif st == "R":
                    new_st = "F"                          # R 被桥接，升级
                else:  # F / B / C
                    new_st = st                           # 保持（人工 B/C 不被自动覆盖）
                result[m] = (new_st, group_code)
        else:
            # 孤立点
            (m,) = members_sorted
            st = cur_status.get(m, "N")
            if st in ("N", "R"):
                result[m] = ("R", "")                     # 新经验证无相似 / R 保持
            else:  # F/B/C 孤立（理论上不应出现，防御性保持）
                result[m] = (st, cur_group.get(m, ""))
    return result


# ---------------------------------------------------------------------------
# 组级面签分析（智能检测文件夹上传：一组多图 → 面签筛选 → 组内一致性 → 选优检索）
# ---------------------------------------------------------------------------
def _image_quality_score(image) -> dict[str, float]:
    """图像完整度/清晰度打分，用于从同组面签中挑选最佳一张。

    五维加权：分辨率(百万像素) 0.30 + 清晰度(Laplacian 方差) 0.28 + 曝光 0.18
    + 对比度 0.14 + 长宽比 0.10。OpenCV 缺失时清晰度退化为对比度平方。

    Args:
        image: PIL.Image（建议传预处理后的图）

    Returns:
        {"quality": 0~1 综合分, 各维子分, "width", "height"}
    """
    from PIL import ImageStat

    rgb = image.convert("RGB")
    width, height = rgb.size
    pixels = width * height
    gray = rgb.convert("L")
    stat = ImageStat.Stat(gray)
    brightness = float(stat.mean[0])
    contrast = float(stat.stddev[0])

    try:
        import cv2
        import numpy as _np

        arr = _np.array(gray)
        sharpness = float(cv2.Laplacian(arr, cv2.CV_64F).var())
    except Exception:
        sharpness = contrast * contrast

    megapixel_score = min(1.0, pixels / (1280 * 960))
    sharpness_score = min(1.0, sharpness / 450.0)
    contrast_score = min(1.0, contrast / 64.0)
    exposure_score = max(0.0, 1.0 - abs(brightness - 128.0) / 128.0)
    aspect = min(width, height) / max(width, height) if max(width, height) else 0.0
    aspect_score = min(1.0, aspect / 0.55)
    quality = (
        0.30 * megapixel_score
        + 0.28 * sharpness_score
        + 0.18 * exposure_score
        + 0.14 * contrast_score
        + 0.10 * aspect_score
    )
    return {
        "quality": round(float(quality), 4),
        "megapixel_score": round(float(megapixel_score), 4),
        "sharpness_score": round(float(sharpness_score), 4),
        "exposure_score": round(float(exposure_score), 4),
        "contrast_score": round(float(contrast_score), 4),
        "aspect_score": round(float(aspect_score), 4),
        "width": float(width),
        "height": float(height),
    }


def analyze_multi_signing_photos(
    items: list[dict[str, Any]],
    top_k: int = 5,
    query_loan_id: str = "",
) -> dict[str, Any]:
    """从同一贷款样本的多张照片中自动选出最佳面签照并做相似检索。

    流程（对应网页「文件夹按样本组检测」）：
        1) 前置类型闸门：逐张判定五类类型（文件名规则优先、分类器兜底；
           items 可带 image_type 提示直接采信），只有 face_signing 是面签，
           其余四类只记录类别与质量分，不进入特征提取；
        2) 组内面签特征两两余弦：min ≥ 0.78 视为同人，min ≥ 0.92 疑似同次拍摄/素材复用，
           多张面签且低于 0.78 时标记 needs_review 交人工核验；
        3) 在通过筛选的候选中选质量分最高的一张（并列取清晰度/面积），
           调 search_similar 完成 FAISS 检索。

    Args:
        items: [{"name": 文件名, "relative_path": 相对路径, "image": PIL.Image,
                 "image_type": 可选类型提示}, ...]
        top_k: 检索返回条数
        query_loan_id: 查询图所属 loan_id（差异化阈值用，上传场景留空）

    Returns:
        {"available", "candidates": 组内逐张明细(含 image_type), "selected": 选中照,
         "results": 检索结果, "same_person", "same_shoot_or_reuse",
         "min/avg_internal_similarity", ...}
    """
    if not _try_init():
        return {"available": False, "error": "模型不可用", "candidates": []}

    classifier = _state["classifier"]
    np = _state["np"]
    torch = _state["torch"]
    extractor = _state["extractor"]
    preprocessor = _state["preprocessor"]
    candidates: list[dict[str, Any]] = []
    vectors: list[Any] = []

    for item in items:
        image = item["image"].convert("RGB")
        # 前置类型闸门：显式提示 > 文件名规则 > 分类器
        hint = item.get("image_type")
        if hint in IMAGE_TYPES:
            image_type, category, type_source = hint, TYPE_LABELS[hint], "hint"
        else:
            typed = classify_image_type(
                image=image,
                filename=f"{item.get('name', '')} {item.get('relative_path', '')}",
            )
            image_type = typed["image_type"]
            category = typed["category"] or "未识别"
            type_source = typed["source"]
        is_sign = image_type == "face_signing"
        processed = preprocessor(image).convert("RGB")
        tensor = torch.tensor(
            np.array(processed.resize((224, 224))).transpose(2, 0, 1)
        ).float() / 255.0
        tensor = tensor.unsqueeze(0).to(extractor.device)
        if is_sign and type_source == "model":
            _, sign_confidence = classifier.is_sign_photo(tensor)
        else:
            sign_confidence = 1.0 if is_sign else 0.0
        quality = _image_quality_score(processed)
        row = {
            "name": item.get("name", ""),
            "relative_path": item.get("relative_path", ""),
            "image_type": image_type or "",
            "category": category,
            "sign_confidence": float(sign_confidence),
            "is_sign_photo": bool(is_sign),
            **quality,
        }
        if is_sign:
            with torch.no_grad():
                feat = extractor.extract(tensor)
            feat_np = feat.cpu().numpy().astype(np.float32)
            norm = np.linalg.norm(feat_np)
            if norm > 0:
                feat_np = feat_np / norm
            vectors.append(feat_np[0])
            row["_candidate_index"] = len(vectors) - 1
        candidates.append(row)

    sign_candidates = [row for row in candidates if row["is_sign_photo"]]
    if not sign_candidates:
        return {
            "available": True,
            "error": None,
            "status": "组内未识别到面签照片",
            "needs_review": True,
            "candidates": candidates,
            "selected": None,
            "results": [],
        }

    # 组内一致性：面签特征两两余弦（已 L2 归一化，点积即余弦）
    pair_scores: list[float] = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            pair_scores.append(float(np.dot(vectors[i], vectors[j])))
    min_pair = min(pair_scores) if pair_scores else 1.0
    avg_pair = sum(pair_scores) / len(pair_scores) if pair_scores else 1.0

    same_person = bool(min_pair >= 0.78)
    same_shoot_or_reuse = bool(min_pair >= 0.92)
    needs_review = bool(len(sign_candidates) > 1 and not same_person)

    selected = max(
        sign_candidates,
        key=lambda row: (
            row["quality"],
            row["sign_confidence"],
            row["width"] * row["height"],
        ),
    )
    selected_image = next(
        (
            item["image"]
            for item in items
            if item.get("relative_path", "") == selected.get("relative_path", "")
            and item.get("name", "") == selected.get("name", "")
        ),
        None,
    )
    detect_result = (
        search_similar(
            selected_image,
            top_k=top_k,
            query_loan_id=query_loan_id,
            force_sign_photo=True,
        )
        if selected_image is not None
        else {"results": [], "status": "选中照片读取失败", "category": "", "is_sign_photo": True}
    )

    for row in candidates:
        row.pop("_candidate_index", None)

    return {
        "available": True,
        "error": None,
        "status": "组级检测完成",
        "needs_review": needs_review,
        "same_person": same_person,
        "same_shoot_or_reuse": same_shoot_or_reuse,
        "min_internal_similarity": round(float(min_pair), 4),
        "avg_internal_similarity": round(float(avg_pair), 4),
        "candidates": candidates,
        "selected": selected,
        "selected_reason": "在通过面签筛选且组内身份一致的候选中，选择完整度/清晰度得分最高的一张",
        "results": detect_result.get("results", []),
        "detect_status": detect_result.get("status", ""),
        "category": detect_result.get("category", selected.get("category", "")),
        "is_sign_photo": detect_result.get("is_sign_photo", True),
    }


if __name__ == "__main__":
    # 自检
    print(f"模型代码目录: {MODEL_CODE_DIR} (存在: {MODEL_CODE_DIR.is_dir()})")
    print(f"数据目录:     {DATA_DIR} (存在: {DATA_DIR.is_dir()})")
    print(f"MVP 输出:     {MVP_OUTPUT_DIR} (存在: {MVP_OUTPUT_DIR.is_dir()})")
    print(f"FAISS 索引:   {DEFAULT_FAISS_INDEX} (存在: {DEFAULT_FAISS_INDEX.exists()})")
    print(f"MVP FAISS:    {MVP_FAISS_INDEX} (存在: {MVP_FAISS_INDEX.exists()})")
    print()
    print("索引统计:", get_index_stats())
    print("模型指标:", get_model_metrics())
