"""Download the PASQA checkpoint + config into models/pasqa/.

The 753MB pickle is not in git. After this script finishes, /evaluate
picks it up automatically (see app.core.config DEFAULT_PASQA_CHECKPOINT).

    python scripts/download_pasqa.py
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "models" / "pasqa"
HF = "https://huggingface.co/ly-corporation/PASQA/resolve/main"
FILES = (
    ("checkpoint-100000steps.pkl", 700_000_000),  # ~753MB; size is a sanity floor
    ("config.yml", 200),
)


def _download(name: str, min_bytes: int) -> Path:
    dest = DEST / name
    url = f"{HF}/{name}"
    print(f"Downloading {name} …")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    size = tmp.stat().st_size
    if size < min_bytes:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{name} is only {size} bytes; download looks truncated")
    tmp.replace(dest)
    print(f"  -> {dest} ({size:,} bytes)")
    return dest


def main() -> int:
    try:
        for name, min_bytes in FILES:
            dest = DEST / name
            if dest.exists() and dest.stat().st_size >= min_bytes:
                print(f"Already have {dest} ({dest.stat().st_size:,} bytes)")
                continue
            _download(name, min_bytes)
    except Exception as e:  # noqa: BLE001
        print(f"Download failed: {e}", file=sys.stderr)
        print("Manual: hf download ly-corporation/PASQA --local-dir models/pasqa", file=sys.stderr)
        return 1
    print("PASQA weights ready. Restart the API; first /evaluate loads s3prl wav2vec2.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
