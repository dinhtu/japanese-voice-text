"""Evaluation metrics for Japanese ASR.

Includes:
  - edit_distance: Levenshtein distance
  - phoneme_error_rate (PER): For InterCTC phoneme output
  - kana_error_rate (KER): For final kana output (primary metric)
  - confusion analysis: Per-group error breakdown
"""


# Phoneme groups for confusion analysis (InterCTC)
PHONEME_GROUPS = {
    "母音 (vowels)": {"a", "i", "u", "e", "o"},
    "破裂音 (plosives)": {"k", "g", "t", "d", "p", "b"},
    "摩擦音 (fricatives)": {"s", "z", "sh", "h", "f"},
    "鼻音 (nasals)": {"m", "n", "N", "ny"},
    "流音 (liquids)": {"r"},
    "半母音 (glides)": {"w", "y"},
    "破擦音 (affricates)": {"ch", "ts", "j"},
    "拗音 (palatalized)": {"ky", "gy", "hy", "my", "ry", "by", "py"},
    "促音 (geminate)": {"q"},
}

# Kana groups for confusion analysis (Final CTC)
KANA_GROUPS = {
    "母音 (vowels)": set("あいうえお"),
    "か行": set("かきくけこ"),
    "さ行": set("さしすせそ"),
    "た行": set("たちつてと"),
    "な行": set("なにぬねの"),
    "は行": set("はひふへほ"),
    "ま行": set("まみむめも"),
    "や行": set("やゆよ"),
    "ら行": set("らりるれろ"),
    "わ行": set("わをん"),
    "濁音 (dakuten)": set("がぎぐげござじずぜぞだぢづでどばびぶべぼ"),
    "半濁音 (handakuten)": set("ぱぴぷぺぽ"),
    "小書き (small)": set("ぁぃぅぇぉっゃゅょゎ"),
    "長音": {"ー"},
}


def _build_group_lookup(groups: dict[str, set[str]]) -> dict[str, str]:
    lookup = {}
    for group_name, members in groups.items():
        for token in members:
            lookup[token] = group_name
    return lookup


_PHONEME_TO_GROUP = _build_group_lookup(PHONEME_GROUPS)
_KANA_TO_GROUP = _build_group_lookup(KANA_GROUPS)


def edit_distance(ref: list, hyp: list) -> int:
    """Levenshtein edit distance between two token lists."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    return d[n][m]


def edit_ops(ref: list, hyp: list) -> list[tuple[str, int, int]]:
    """Compute edit operations (backtrace) between ref and hyp."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)

    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1):
            if ref[i - 1] != hyp[j - 1]:
                ops.append(("sub", i - 1, j - 1))
            i -= 1
            j -= 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            ops.append(("del", i - 1, j))
            i -= 1
        else:
            ops.append(("ins", i, j - 1))
            j -= 1

    return list(reversed(ops))


def phoneme_error_rate(ref_phonemes: str, hyp_phonemes: str) -> float:
    """Phoneme Error Rate: edit distance on phoneme tokens / reference length."""
    ref = ref_phonemes.split()
    hyp = hyp_phonemes.split()
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    return edit_distance(ref, hyp) / len(ref)


def kana_error_rate(ref_kana: str, hyp_kana: str) -> float:
    """Kana Error Rate: edit distance on kana characters / reference length.

    Input strings are raw kana (not space-separated).
    Spaces (word boundaries) are treated as tokens.
    """
    ref = list(ref_kana.replace(" ", "\u3000"))  # preserve word boundaries
    hyp = list(hyp_kana.replace(" ", "\u3000"))
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    return edit_distance(ref, hyp) / len(ref)


def confusion_analysis(
    ref_list: list[str],
    hyp_list: list[str],
    token_to_group: dict[str, str],
    groups: dict[str, set[str]],
    split_fn=None,
) -> dict:
    """Analyze errors by token group across multiple samples.

    Args:
        ref_list: List of reference strings.
        hyp_list: List of hypothesis strings.
        token_to_group: Token → group name mapping.
        groups: Group name → set of tokens.
        split_fn: Function to split string into tokens. Default: str.split().
    """
    if split_fn is None:
        split_fn = str.split

    group_errors = {}
    for group_name in groups:
        group_errors[group_name] = {"sub": 0, "del": 0, "ins": 0, "total": 0, "ref_count": 0}
    group_errors["unknown"] = {"sub": 0, "del": 0, "ins": 0, "total": 0, "ref_count": 0}

    confusion_counts: dict[tuple[str, str], int] = {}
    total_edits = 0
    total_ref_len = 0

    for ref_str, hyp_str in zip(ref_list, hyp_list):
        ref = split_fn(ref_str)
        hyp = split_fn(hyp_str)
        total_ref_len += len(ref)
        total_edits += edit_distance(ref, hyp)

        for token in ref:
            group = token_to_group.get(token, "unknown")
            group_errors[group]["ref_count"] += 1

        ops = edit_ops(ref, hyp)
        for op_type, ref_idx, hyp_idx in ops:
            if op_type == "sub":
                ref_token = ref[ref_idx]
                hyp_token = hyp[hyp_idx]
                group = token_to_group.get(ref_token, "unknown")
                group_errors[group]["sub"] += 1
                group_errors[group]["total"] += 1
                pair = (ref_token, hyp_token)
                confusion_counts[pair] = confusion_counts.get(pair, 0) + 1
            elif op_type == "del":
                ref_token = ref[ref_idx]
                group = token_to_group.get(ref_token, "unknown")
                group_errors[group]["del"] += 1
                group_errors[group]["total"] += 1
            elif op_type == "ins":
                hyp_token = hyp[hyp_idx]
                group = token_to_group.get(hyp_token, "unknown")
                group_errors[group]["ins"] += 1
                group_errors[group]["total"] += 1

    for group_name, errors in group_errors.items():
        ref_count = errors["ref_count"]
        errors["er"] = errors["total"] / ref_count if ref_count > 0 else 0.0

    group_errors = {k: v for k, v in group_errors.items() if v["ref_count"] > 0 or v["total"] > 0}

    confusion_pairs = sorted(
        [(r, h, c) for (r, h), c in confusion_counts.items()],
        key=lambda x: x[2],
        reverse=True,
    )

    return {
        "groups": group_errors,
        "confusion_pairs": confusion_pairs[:20],
        "overall_er": total_edits / max(total_ref_len, 1),
        "total_ref_tokens": total_ref_len,
        "total_edits": total_edits,
    }


def phoneme_confusion_analysis(ref_list: list[str], hyp_list: list[str]) -> dict:
    """Analyze phoneme errors by phoneme group."""
    return confusion_analysis(ref_list, hyp_list, _PHONEME_TO_GROUP, PHONEME_GROUPS)


def kana_confusion_analysis(ref_list: list[str], hyp_list: list[str]) -> dict:
    """Analyze kana errors by kana group."""
    return confusion_analysis(
        ref_list, hyp_list, _KANA_TO_GROUP, KANA_GROUPS,
        split_fn=list,  # split into individual characters
    )


def format_confusion_report(analysis: dict, metric_name: str = "ER") -> str:
    """Format confusion analysis as a human-readable report."""
    lines = []
    lines.append("=" * 60)
    lines.append("Group Confusion Analysis")
    lines.append(f"Overall {metric_name}: {analysis['overall_er']:.4f} "
                 f"({analysis['total_edits']} edits / {analysis['total_ref_tokens']} ref)")
    lines.append("=" * 60)

    sorted_groups = sorted(
        analysis["groups"].items(),
        key=lambda x: x[1]["er"],
        reverse=True,
    )

    cols = f"{'Group':<24} {metric_name:>6} {'Sub':>5} {'Del':>5} {'Ins':>5} {'Tot':>5} {'Ref':>5}"
    lines.append(f"\n{cols}")
    lines.append("-" * 60)
    for group_name, errors in sorted_groups:
        lines.append(
            f"{group_name:<25} {errors['er']:>5.1%} "
            f"{errors['sub']:>5} {errors['del']:>5} {errors['ins']:>5} "
            f"{errors['total']:>6} {errors['ref_count']:>6}"
        )

    if analysis["confusion_pairs"]:
        lines.append("\nTop confusion pairs (ref -> hyp):")
        lines.append("-" * 40)
        for ref_token, hyp_token, count in analysis["confusion_pairs"][:10]:
            lines.append(f"  {ref_token:>4} -> {hyp_token:<4}  ({count} times)")

    return "\n".join(lines)
