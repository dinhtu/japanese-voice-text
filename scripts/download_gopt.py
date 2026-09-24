"""Download the official MIT GOPT Librispeech checkpoint into models/gopt/.

The file is ~121KB (not in git). /en evaluate also fetches it on first use.

    python scripts/download_gopt.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.english_gopt import DEFAULT_CHECKPOINT, download_gopt_weights  # noqa: E402


def main() -> int:
    try:
        dest = download_gopt_weights(DEFAULT_CHECKPOINT)
    except Exception as e:  # noqa: BLE001
        print(f"Download failed: {e}", file=sys.stderr)
        print(
            "Manual: curl -L -o models/gopt/best_audio_model.pth "
            "https://github.com/YuanGongND/gopt/raw/master/"
            "pretrained_models/gopt_librispeech/best_audio_model.pth",
            file=sys.stderr,
        )
        return 1
    print(f"GOPT weights ready: {dest} ({dest.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
