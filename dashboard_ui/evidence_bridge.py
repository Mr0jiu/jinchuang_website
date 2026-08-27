"""26 维图像证据指标计算（Tab2 检测增强）

算法复刻自 jinchuang_v4/code/scripts/build_pair_evidence_model.py，
阈值取自 build_two_stage_pipeline.py 的 visual_override_masks 常量。
仅对有明确单指标阈值的维度做"超阈值"标注，供前端文字提示。
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# 阈值定义（取自 build_two_stage_pipeline.py L68-88 的 visual_override_masks）
# ---------------------------------------------------------------------------
EVIDENCE_THRESHOLDS = {
    # 相似度类: 值 >= 阈值 → 达到相似水平
    "global_semantic_similarity":       {"threshold": 0.985, "dir": "ge", "label": "画面整体高度相似"},
    "dhash_similarity":                 {"threshold": 0.76,  "dir": "ge", "label": "图像特征高度相似"},
    "equalized_dhash_similarity":       {"threshold": 0.78,  "dir": "ge", "label": "调亮后特征仍相似"},
    "edge_dhash_similarity":            {"threshold": 0.70,  "dir": "ge", "label": "人物轮廓高度相似"},
    "hsv_hist_similarity":              {"threshold": 0.97,  "dir": "ge", "label": "色彩分布高度一致"},
    "mirror_local_structure_orb_ratio": {"threshold": 0.65,  "dir": "ge", "label": "左右翻转后结构仍匹配"},
    "mirror_dhash_similarity":          {"threshold": 0.84,  "dir": "ge", "label": "左右翻转后特征仍相似"},
    # 差异类: 值 <= 阈值 → 色彩一致（疑似同源）
    "lab_delta_e2000":                  {"threshold": 2.0,   "dir": "le", "label": "色彩几乎无差异（疑似同一张图）"},
    # 差异类: 值 >= 阈值 → 亮度/对比度偏移（疑似篡改）
    "brightness_delta":                 {"threshold": 8.0,   "dir": "ge", "label": "亮度差异明显（疑似调过亮度）"},
    "contrast_delta":                   {"threshold": 6.0,   "dir": "ge", "label": "清晰度差异明显"},
    "rgb_mean_abs_delta":               {"threshold": 8.0,   "dir": "ge", "label": "整体色调差异明显"},
}


# ---------------------------------------------------------------------------
# 单图证据提取辅助函数
# ---------------------------------------------------------------------------
def _histogram(gray):
    import cv2
    hist = cv2.calcHist([gray], [0], None, [32], [0, 256]).astype("float32").reshape(-1)
    total = float(hist.sum())
    return hist / total if total else hist


def _hsv_histogram(hsv):
    import cv2
    hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256]).astype("float32").reshape(-1)
    total = float(hist.sum())
    return hist / total if total else hist


def _dhash_bits(gray):
    import cv2
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    return (small[:, 1:] > small[:, :-1]).astype("uint8").reshape(-1)


def _rotate_gray(gray, angle):
    import cv2
    h, w = gray.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def _hamming(a, b):
    import numpy as np
    return 1.0 - float(np.mean(a != b))


def _hist_sim(a, b):
    import numpy as np
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


def _orb_ratio(desc_a, desc_b):
    """ORB 描述符匹配比例（复刻 build_pair_evidence_model.orb_match_ratio）。"""
    if desc_a is None or desc_b is None or len(desc_a) == 0 or len(desc_b) == 0:
        return 0.0
    import cv2
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(desc_a, desc_b, k=2)
    good = 0
    for pair in matches:
        if len(pair) >= 2 and pair[0].distance < 0.75 * pair[1].distance:
            good += 1
    return float(good) / float(max(len(desc_a), 1))


def _mean_abs(a, b):
    import numpy as np
    return float(np.mean(np.abs(a - b)))


def _euclidean(a, b):
    import numpy as np
    return float(np.linalg.norm(a - b))


def _ciede2000(lab_a, lab_b):
    """CIEDE2000 色差（复刻 build_pair_evidence_model.ciede2000_delta）。"""
    import numpy as np
    try:
        from skimage.color import deltaE_ciede2000
        return float(deltaE_ciede2000(lab_a.reshape(1, 1, 3).astype("float64"),
                                     lab_b.reshape(1, 1, 3).astype("float64"))[0, 0])
    except Exception:
        return _euclidean(lab_a, lab_b)


def _compute_image_evidence(img_bgr):
    """从 BGR 图像（已 resize 到 256x256）提取单图证据特征。"""
    import cv2
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    edges = cv2.Canny(gray, 80, 160)
    mirror_gray = cv2.flip(gray, 1)
    rotated_grays = [_rotate_gray(gray, a) for a in (-15, -8, 8, 15)]
    rotated_edges = [cv2.Canny(r, 80, 160) for r in rotated_grays]
    center = gray[64:192, 64:192]
    mirror_center = mirror_gray[64:192, 64:192]
    top, bottom = gray[:56, :], gray[200:, :]
    left, right = gray[:, :56], gray[:, 200:]
    bg = __concat_bg(top, bottom, left, right)
    m_top, m_bottom = mirror_gray[:56, :], mirror_gray[200:, :]
    m_left, m_right = mirror_gray[:, :56], mirror_gray[:, 200:]
    m_bg = __concat_bg(m_top, m_bottom, m_left, m_right)
    orb = cv2.ORB_create(nfeatures=700)
    _, desc = orb.detectAndCompute(gray, None)
    _, m_desc = orb.detectAndCompute(mirror_gray, None)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return {
        "center_hist": _histogram(center), "background_hist": _histogram(bg),
        "mirror_center_hist": _histogram(mirror_center), "mirror_background_hist": _histogram(m_bg),
        "dhash": _dhash_bits(gray), "mirror_dhash": _dhash_bits(mirror_gray),
        "equalized_dhash": _dhash_bits(equalized), "edge_dhash": _dhash_bits(edges),
        "edge_hist": _histogram(edges), "hsv_hist": _hsv_histogram(hsv),
        "rotated_dhashes": [_dhash_bits(r) for r in rotated_grays],
        "rotated_edge_dhashes": [_dhash_bits(r) for r in rotated_edges],
        "rgb_mean": rgb.reshape(-1, 3).mean(axis=0), "lab_mean": lab.reshape(-1, 3).mean(axis=0),
        "hsv_mean": hsv.reshape(-1, 3).mean(axis=0),
        "orb_desc": desc, "mirror_orb_desc": m_desc,
        "brightness": float(gray.mean()), "contrast": float(gray.std()), "blur": blur,
    }


def __concat_bg(top, bottom, left, right):
    import numpy as np
    return np.concatenate([top.reshape(-1), bottom.reshape(-1),
                           left.reshape(-1), right.reshape(-1)]).reshape(-1, 1)


def _compute_pair_features(q_ev, m_ev, global_sim):
    """计算 26 维指标（复刻 build_pair_evidence_model.py 的组装逻辑）。"""
    dhash_sim = _hamming(q_ev["dhash"], m_ev["dhash"])
    edge_dhash_sim = _hamming(q_ev["edge_dhash"], m_ev["edge_dhash"])
    rot_dhash = max(
        [_hamming(c, m_ev["dhash"]) for c in q_ev["rotated_dhashes"]]
        + [_hamming(q_ev["dhash"], c) for c in m_ev["rotated_dhashes"]]
        + [dhash_sim]
    )
    rot_edge_dhash = max(
        [_hamming(c, m_ev["edge_dhash"]) for c in q_ev["rotated_edge_dhashes"]]
        + [_hamming(q_ev["edge_dhash"], c) for c in m_ev["rotated_edge_dhashes"]]
        + [edge_dhash_sim]
    )
    return {
        "global_semantic_similarity": float(global_sim),
        "subject_region_hist_similarity": _hist_sim(q_ev["center_hist"], m_ev["center_hist"]),
        "background_hist_similarity": _hist_sim(q_ev["background_hist"], m_ev["background_hist"]),
        "local_structure_orb_ratio": _orb_ratio(q_ev["orb_desc"], m_ev["orb_desc"]),
        "dhash_similarity": dhash_sim,
        "mirror_local_structure_orb_ratio": max(
            _orb_ratio(q_ev["mirror_orb_desc"], m_ev["orb_desc"]),
            _orb_ratio(q_ev["orb_desc"], m_ev["mirror_orb_desc"]),
        ),
        "mirror_subject_region_hist_similarity": max(
            _hist_sim(q_ev["mirror_center_hist"], m_ev["center_hist"]),
            _hist_sim(q_ev["center_hist"], m_ev["mirror_center_hist"]),
        ),
        "mirror_background_hist_similarity": max(
            _hist_sim(q_ev["mirror_background_hist"], m_ev["background_hist"]),
            _hist_sim(q_ev["background_hist"], m_ev["mirror_background_hist"]),
        ),
        "mirror_dhash_similarity": max(
            _hamming(q_ev["mirror_dhash"], m_ev["dhash"]),
            _hamming(q_ev["dhash"], m_ev["mirror_dhash"]),
        ),
        "equalized_dhash_similarity": _hamming(q_ev["equalized_dhash"], m_ev["equalized_dhash"]),
        "edge_dhash_similarity": edge_dhash_sim,
        "edge_hist_similarity": _hist_sim(q_ev["edge_hist"], m_ev["edge_hist"]),
        "rotated_dhash_similarity": rot_dhash,
        "rotated_dhash_gain": rot_dhash - dhash_sim,
        "rotated_edge_dhash_similarity": rot_edge_dhash,
        "rotated_edge_dhash_gain": rot_edge_dhash - edge_dhash_sim,
        "brightness_delta": abs(q_ev["brightness"] - m_ev["brightness"]),
        "contrast_delta": abs(q_ev["contrast"] - m_ev["contrast"]),
        "rgb_mean_abs_delta": _mean_abs(q_ev["rgb_mean"], m_ev["rgb_mean"]),
        "rgb_mean_euclidean_delta": _euclidean(q_ev["rgb_mean"], m_ev["rgb_mean"]),
        "lab_mean_abs_delta": _mean_abs(q_ev["lab_mean"], m_ev["lab_mean"]),
        "lab_delta_e": _euclidean(q_ev["lab_mean"], m_ev["lab_mean"]),
        "lab_delta_e2000": _ciede2000(q_ev["lab_mean"], m_ev["lab_mean"]),
        "hsv_mean_abs_delta": _mean_abs(q_ev["hsv_mean"], m_ev["hsv_mean"]),
        "hsv_hist_similarity": _hist_sim(q_ev["hsv_hist"], m_ev["hsv_hist"]),
        "blur_ratio": min(q_ev["blur"], m_ev["blur"]) / max(q_ev["blur"], m_ev["blur"], 1e-6),
    }


def compute_pair_evidence(query_image, match_path, global_sim):
    """对一对 (查询图, 匹配图) 计算 26 维证据指标并标注超阈值维度。

    Args:
        query_image: PIL.Image 或图片路径（查询图）
        match_path: 匹配图路径字符串
        global_sim: FAISS 余弦相似度（作为 global_semantic_similarity）

    Returns:
        {
            "available": bool,
            "features": {指标名: 值, ...},   # 26 维
            "flagged": [{"name","value","threshold","label"}, ...],  # 超阈值维度
            "error": str | None,
        }
    """
    try:
        import cv2
        from PIL import Image

        # 加载查询图
        if isinstance(query_image, str):
            q_pil = Image.open(query_image)
        else:
            q_pil = query_image
        import numpy as np
        q_bgr = cv2.cvtColor(np.array(q_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
        q_bgr = cv2.resize(q_bgr, (256, 256), interpolation=cv2.INTER_AREA)

        # 加载匹配图
        m_bgr = cv2.imread(str(match_path), cv2.IMREAD_COLOR)
        if m_bgr is None:
            return {"available": False, "features": {}, "flagged": [], "error": f"无法读取: {match_path}"}
        m_bgr = cv2.resize(m_bgr, (256, 256), interpolation=cv2.INTER_AREA)

        q_ev = _compute_image_evidence(q_bgr)
        m_ev = _compute_image_evidence(m_bgr)
        feats = _compute_pair_features(q_ev, m_ev, global_sim)

        flagged = []
        for name, spec in EVIDENCE_THRESHOLDS.items():
            val = feats.get(name, 0.0)
            hit = val >= spec["threshold"] if spec["dir"] == "ge" else val <= spec["threshold"]
            if hit:
                flagged.append({
                    "name": name,
                    "value": round(float(val), 4),
                    "threshold": spec["threshold"],
                    "label": spec["label"],
                })
        return {"available": True, "features": feats, "flagged": flagged, "error": None}
    except Exception as e:
        return {"available": False, "features": {}, "flagged": [], "error": str(e)}
