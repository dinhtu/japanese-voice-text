"""Lab / Experiment API endpoints for testing Romaji conversion via romkan2 and MADLAD-400 translation.

Completely isolated from existing production endpoints (/api/pronunciation).
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.parse
import urllib.request
from typing import Any

from fastapi import APIRouter, Form, HTTPException

import romkan2
from app.services.pitch_accent import pitch_accent_patterns

logger = logging.getLogger(__name__)

router = APIRouter()


# ==============================================================================
# Helper Functions
# ==============================================================================

def _get_pitch_pattern_safe(text: str) -> dict[str, Any]:
    """Extracts pitch accent pattern using the existing UniDic pitch accent engine."""
    try:
        patterns = pitch_accent_patterns(text)
        if patterns:
            primary = patterns[0]
            reading = "".join(m.mora for m in primary)
            moras = [{"mora": m.mora, "pitch": m.pitch, "phrase": m.phrase} for m in primary]
            return {
                "reading": reading,
                "mora_count": len(primary),
                "pattern": moras,
                "pitches_str": "".join(m.pitch for m in primary),
                "all_patterns_count": len(patterns),
                "error": None,
            }
    except Exception as exc:
        return {
            "reading": "",
            "mora_count": 0,
            "pattern": [],
            "pitches_str": "",
            "all_patterns_count": 0,
            "error": str(exc),
        }
    return {
        "reading": "",
        "mora_count": 0,
        "pattern": [],
        "pitches_str": "",
        "all_patterns_count": 0,
        "error": "No pattern generated",
    }


def _call_madlad400_api(
    text: str,
    base_url: str = "http://localhost:8001",
    api_key: str = "",
    timeout: float = 3.0,
) -> dict[str, Any]:
    """Call local MADLAD-400 MT API (/api/v1/translate)."""
    url_str = str(base_url).strip() if base_url else "http://localhost:8001"
    key_str = str(api_key).strip() if api_key else ""
    clean_url = url_str.rstrip("/") + "/api/v1/translate"

    data = urllib.parse.urlencode({
        "text": text,
        "source_language": "vi",
        "target_language": "ja",
        "wait": "true",
    }).encode("utf-8")

    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    if key_str:
        headers["X-API-Key"] = key_str

    start_time = time.time()
    req = urllib.request.Request(clean_url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            elapsed = time.time() - start_time
            raw = resp.read().decode("utf-8")
            res_json = json.loads(raw)
            translation = res_json.get("translation", "").strip()
            return {
                "success": True,
                "translation": translation,
                "elapsed_seconds": round(elapsed, 3),
                "raw_response": res_json,
                "error": None,
            }
    except urllib.error.URLError as e:
        return {
            "success": False,
            "translation": "",
            "elapsed_seconds": round(time.time() - start_time, 3),
            "raw_response": None,
            "error": f"Connection to MADLAD-400 failed ({clean_url}): {e.reason}",
        }
    except Exception as e:
        return {
            "success": False,
            "translation": "",
            "elapsed_seconds": round(time.time() - start_time, 3),
            "raw_response": None,
            "error": f"MADLAD-400 error: {e}",
        }


# ==============================================================================
# Lab API Routes
# ==============================================================================

@router.get("/status")
def get_lab_status() -> dict[str, Any]:
    """Check availability of testing services (romkan2, MADLAD-400)."""
    madlad_url = os.getenv("MADLAD_API_URL", "http://localhost:8001").rstrip("/")
    madlad_online = False
    try:
        req = urllib.request.Request(f"{madlad_url}/health", headers={"User-Agent": "LabCheck/1.0"})
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            if resp.status == 200:
                madlad_online = True
    except Exception:
        madlad_online = False

    return {
        "status": "ok",
        "romkan2_installed": True,
        "madlad_url": madlad_url,
        "madlad_online": madlad_online,
    }


@router.post("/romaji-to-kana")
def test_romkan2_conversion(text: str = Form(..., min_length=1)) -> dict[str, Any]:
    """Convert Romaji to Kana using romkan2 library and extract Pitch Accent using existing engine."""
    text_clean = text.strip()
    if not text_clean:
        raise HTTPException(status_code=400, detail="Text is empty")

    t0 = time.time()
    try:
        # 1. Use romkan2 to convert Romaji into Hiragana & Katakana
        hiragana = romkan2.to_hiragana(text_clean)
        katakana = romkan2.to_katakana(text_clean)
        elapsed_ms = round((time.time() - t0) * 1000, 2)

        # 2. Extract Pitch Accent from the resulting Hiragana using the existing UniDic pitch accent pipeline
        pitch_accent = _get_pitch_pattern_safe(hiragana)

        return {
            "success": True,
            "input_romaji": text_clean,
            "hiragana": hiragana,
            "katakana": katakana,
            "pitch_accent": pitch_accent,
            "elapsed_ms": elapsed_ms,
            "error": None,
        }
    except Exception as e:
        logger.exception("romkan2 conversion error")
        return {
            "success": False,
            "input_romaji": text_clean,
            "hiragana": "",
            "katakana": "",
            "pitch_accent": None,
            "elapsed_ms": 0,
            "error": str(e),
        }


@router.post("/madlad-translate")
def test_madlad_translation(
    text: str = Form(..., min_length=1),
    api_url: str = Form("http://localhost:8001"),
    api_key: str = Form(""),
) -> dict[str, Any]:
    """Test MADLAD-400 translation from Vietnamese to Japanese and extract Pitch Accent."""
    text_clean = text.strip()
    if not text_clean:
        raise HTTPException(status_code=400, detail="Text is empty")

    madlad_res = _call_madlad400_api(text_clean, base_url=api_url, api_key=api_key)

    pitch_info = None
    if madlad_res.get("success") and madlad_res.get("translation"):
        pitch_info = _get_pitch_pattern_safe(madlad_res["translation"])

    return {
        "input_vietnamese": text_clean,
        "api_url": api_url,
        "madlad_result": madlad_res,
        "pitch_accent": pitch_info,
    }
