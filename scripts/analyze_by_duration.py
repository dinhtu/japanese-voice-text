"""Analyze KER by duration bucket and show 30 representative samples."""
import json
import sys

import cutlet

katsu = cutlet.Cutlet(ensure_ascii=False)


def to_kana(text: str) -> str:
    """Convert kanji text to hiragana using cutlet/fugashi."""
    import fugashi
    tagger = fugashi.Tagger()
    result = []
    for word in tagger(text):
        reading = word.feature.kana
        if reading:
            # Convert katakana to hiragana
            result.append("".join(
                chr(ord(c) - 0x60) if 0x30A1 <= ord(c) <= 0x30F6 else c
                for c in reading
            ))
        else:
            result.append(word.surface)
    return "".join(result)


def compute_ker(ref: str, hyp: str) -> float:
    """Compute Kana Error Rate using edit distance."""
    ref_chars = list(ref.replace(" ", ""))
    hyp_chars = list(hyp.replace(" ", ""))
    n = len(ref_chars)
    m = len(hyp_chars)
    if n == 0:
        return 1.0 if m > 0 else 0.0
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_chars[i - 1] == hyp_chars[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[n][m] / n


def main():
    with open("/tmp/jsut_eval_results.json") as f:
        results = json.load(f)

    # Convert ref to kana and compute KER
    for r in results:
        r["ref_kana"] = to_kana(r["ref"])
        r["ker"] = compute_ker(r["ref_kana"], r["pred"])

    # Duration buckets
    buckets = {
        "< 3s": [r for r in results if r["duration"] < 3],
        "3-5s": [r for r in results if 3 <= r["duration"] < 5],
        "5-8s": [r for r in results if 5 <= r["duration"] < 8],
        "8s+": [r for r in results if r["duration"] >= 8],
    }

    print("=" * 70)
    print("Duration vs KER Summary")
    print("=" * 70)
    for name, items in buckets.items():
        if not items:
            continue
        avg_ker = sum(r["ker"] for r in items) / len(items)
        median_ker = sorted(r["ker"] for r in items)[len(items) // 2]
        good = sum(1 for r in items if r["ker"] == 0)
        print(f"  {name:>6s}: {len(items):4d} samples, "
              f"avg KER={avg_ker*100:.1f}%, median={median_ker*100:.1f}%, "
              f"perfect={good} ({good/len(items)*100:.0f}%)")

    total_ker = sum(r["ker"] for r in results) / len(results)
    print(f"  {'Total':>6s}: {len(results):4d} samples, avg KER={total_ker*100:.1f}%")

    # Show 30 samples: 5 best + 5 worst from each of short (<3s), medium (3-8s), long (8s+)
    print("\n" + "=" * 70)
    print("30 Representative Samples (sorted by duration)")
    print("=" * 70)

    categories = [
        ("Short (<3s)", [r for r in results if r["duration"] < 3]),
        ("Medium (3-8s)", [r for r in results if 3 <= r["duration"] < 8]),
        ("Long (8s+)", [r for r in results if r["duration"] >= 8]),
    ]

    for cat_name, items in categories:
        if not items:
            continue
        sorted_by_ker = sorted(items, key=lambda x: x["ker"])
        best5 = sorted_by_ker[:5]
        worst5 = sorted_by_ker[-5:]

        print(f"\n--- {cat_name} ---")
        print(f"  [GOOD examples]")
        for r in best5:
            marker = "OK" if r["ker"] == 0 else f"KER={r['ker']*100:.0f}%"
            print(f"  {r['duration']:5.1f}s [{marker:>8s}] ref: {r['ref_kana']}")
            print(f"                       pred: {r['pred']}")

        print(f"  [BAD examples]")
        for r in worst5:
            print(f"  {r['duration']:5.1f}s [KER={r['ker']*100:.0f}%] ref: {r['ref_kana']}")
            print(f"                       pred: {r['pred']}")


if __name__ == "__main__":
    main()
