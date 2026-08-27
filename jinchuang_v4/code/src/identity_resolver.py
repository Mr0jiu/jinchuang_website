"""Privacy-preserving identity-key extraction from ID-card front images.

Only a salted SHA-256 value is persisted.  The raw ID number and raw OCR text
are kept in memory for the duration of extraction and are never written to the
identity mapping file.
"""
from __future__ import annotations

import hashlib
import os
import re
from functools import lru_cache
from pathlib import Path


ID_CARD_PATTERN = re.compile(r"(?<!\d)(\d{17}[0-9Xx])(?![0-9Xx])")


def validate_chinese_id_card(value: str) -> bool:
    """Validate format and GB 11643 check digit for an 18-digit ID number."""
    candidate = value.upper().strip()
    if not re.fullmatch(r"\d{17}[0-9X]", candidate):
        return False
    weights = (7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2)
    checks = "10X98765432"
    return checks[sum(int(digit) * weight for digit, weight in zip(candidate[:17], weights)) % 11] == candidate[-1]


def extract_id_card_number(ocr_text: str, allow_format_only: bool = False) -> str | None:
    compact = re.sub(r"\s+", "", ocr_text)
    for match in ID_CARD_PATTERN.finditer(compact):
        candidate = match.group(1).upper()
        if validate_chinese_id_card(candidate):
            return candidate
        if allow_format_only:
            return candidate
    return None


def identity_hash(id_card_number: str, salt: str | None = None) -> str:
    secret = salt or os.getenv("IDENTITY_HASH_SALT")
    if not secret:
        raise ValueError("Set IDENTITY_HASH_SALT before generating identity hashes.")
    return hashlib.sha256(f"{secret}:{id_card_number.upper()}".encode("utf-8")).hexdigest()


@lru_cache(maxsize=1)
def _rapid_ocr():
    from rapidocr_onnxruntime import RapidOCR
    return RapidOCR()


def ocr_id_card_front(image_path: str | Path) -> str:
    """Run lightweight local OCR first; use PaddleOCR only as a fallback."""
    try:
        result, _ = _rapid_ocr()(str(image_path))
        text = " ".join(str(item[1]) for item in result or [])
        if text:
            return text
    except ImportError:
        pass
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise RuntimeError("PaddleOCR is required: install the optional identity OCR dependencies.") from exc
    # PaddleOCR 2.x exposes ``ocr`` while 3.x exposes ``predict``.  Support
    # both so the extraction job stays reproducible across local environments.
    engine = PaddleOCR(lang="ch")
    if hasattr(engine, "predict"):
        result = engine.predict(str(image_path))
        texts = []
        for item in result:
            payload = item.json if hasattr(item, "json") else item
            if callable(payload):
                payload = payload()
            payload = payload.get("res", payload) if isinstance(payload, dict) else {}
            texts.extend(payload.get("rec_texts", []))
        return " ".join(str(text) for text in texts)
    result = engine.ocr(str(image_path), cls=True)
    return " ".join(item[1][0] for block in result or [] for item in (block or []))
