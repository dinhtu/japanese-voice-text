"""Regression guard for the Korean transcriber (app/services/korean_text.py).

Compares transcribe_korean() with the hand-made `kana` (IPA) and `romaji`
(Revised Romanization) of the Korean exercise JSON files and prints the
exact-match rates. The data itself is not always consistent (좋습니다 is
written both joseumnida and josseumnida), so 100% is not the goal: a rate
that drops after a rule change is.

    python scripts/eval_korean_transcription.py <exercises dir> [--show N] [--by-category]

<exercises dir>: a folder searched recursively for *.json files whose
category targets Korean (e.g. inf_jlpt_admin/database/data/exercises, or
one ko_* folder of it).
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.korean_text import transcribe_korean  # noqa: E402


def load_rows(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.rglob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        category = data.get("category") or {}
        code = str(data.get("category_code") or category.get("code") or path.parent.name)
        if category.get("target_language") != "ko" and not code.startswith("ko_"):
            continue
        for ex in data.get("exercises") or []:
            if isinstance(ex, dict) and ex.get("text"):
                rows.append({"cat": code, "text": ex["text"], "kana": ex.get("kana") or "",
                             "romaji": ex.get("romaji") or ""})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("root", type=Path, help="exercises folder (searched recursively)")
    parser.add_argument("--show", type=int, default=0, help="print N random mismatches")
    parser.add_argument("--by-category", action="store_true", help="also print the rates per category")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    rows = load_rows(args.root)
    if not rows:
        print(f"No Korean exercises under {args.root}")
        return 1

    totals = collections.Counter()
    per_cat: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    misses = []
    for r in rows:
        tr = transcribe_korean(r["text"])
        ipa, rr = tr.ipa or "", tr.romanization or ""
        hit = {
            "rows": 1,
            "unsupported": not tr.supported,
            "kana": ipa == r["kana"],
            "romaji": rr == r["romaji"],
            "romaji_ci": rr.lower() == r["romaji"].lower(),
        }
        hit["both"] = hit["kana"] and hit["romaji"]
        a, b = ipa.strip("[]").split(), r["kana"].strip("[]").split()
        totals["words"] += len(b)
        if len(a) == len(b):
            totals["words_ok"] += sum(x == y for x, y in zip(a, b))
        for key, value in hit.items():
            totals[key] += value
            per_cat[r["cat"]][key] += value
        if not hit["both"]:
            misses.append((r, ipa, rr))

    n = totals["rows"]

    def pct(key: str, total: int | None = None) -> str:
        total = total or n
        return f"{totals[key]}/{total} = {totals[key] / total:.1%}"

    print(f"Korean exercises: {n} rows in {len(per_cat)} categories ({totals['unsupported']} unsupported)")
    print(f"kana   (IPA) exact match:          {pct('kana')}")
    print(f"romaji (RR)  exact match:          {pct('romaji')}")
    print(f"romaji (RR)  case-insensitive:     {pct('romaji_ci')}")
    print(f"kana and romaji both:              {pct('both')}")
    print(f"kana word-level match:             {pct('words_ok', totals['words'])}")
    if args.by_category:
        for cat in sorted(per_cat):
            c = per_cat[cat]
            print(f"  {cat:34s} kana {c['kana'] / c['rows']:6.1%}  romaji {c['romaji'] / c['rows']:6.1%}"
                  f"  (n={c['rows']})")
    if args.show:
        random.seed(7)
        for r, ipa, rr in random.sample(misses, min(args.show, len(misses))):
            print(f"\n  text   {r['text']}\n  kana   {r['kana']}\n  ours   {ipa}\n"
                  f"  romaji {r['romaji']}\n  ours   {rr}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
