"""Turns this app's already-measured pronunciation facts (score, per-mora
correctness, sokuon/chouon timing, pitch-accent direction match) into a
short, natural-language Vietnamese coaching comment, via a locally-run LLM
through Ollama (https://ollama.com) -- see .env.example for OLLAMA_HOST /
OLLAMA_MODEL, and the README for install/pull instructions. Nothing here
is sent to a third party: the request goes to whatever OLLAMA_HOST points
at, normally http://localhost:11434 on the same machine.

## Why an LLM at all, and why grounded

A hand-written template ("o 'tsu' hay ngat han mot nhip...") only covers
issues someone thought to template in advance, and reads noticeably
robotic once two or three issues need combining into one paragraph. An
LLM can write a single coherent comment instead. But an LLM given free
rein (or the raw audio) will confidently describe problems that were
never actually measured -- so this module never gives the model audio and
never asks it to judge anything itself: it is handed only the `facts`
this app already computed deterministically (score_pronunciation,
app.services.prosody_issues.detect_duration_issues, and the pitch-accent
direction match computed below) and instructed to describe *only* those
facts, in natural Vietnamese. The measurement stays 100% deterministic
and testable (see tests/test_coaching.py for the parts that don't need
Ollama itself); the model's only job is phrasing.

## Limitation

A local model can still ignore "only describe these facts" occasionally --
smaller/faster models drift more than larger ones (this is exactly why
the system prompt repeats the constraint and gives a style example rather
than trusting a single instruction). This module does not fact-check the
generated text against `facts` before returning it: treat the comment as
a best-effort coaching aid to read alongside the verified score, mora
grid and pitch chart, not as a verified report on its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from app.services.prosody_issues import DurationIssue

if TYPE_CHECKING:
    from app.core.config import Settings
    from app.services.pitch_accent import MoraPitch
    from app.services.scoring import PronunciationError

SYSTEM_PROMPT = """\
Bạn là một giáo viên dạy phát âm tiếng Nhật cho người Việt, đang nhận xét một \
lượt học viên đọc to một câu tiếng Nhật. Bạn sẽ nhận được một khối dữ liệu mô \
tả CHÍNH XÁC những gì đã đo được từ lượt đọc đó.

QUY TẮC BẮT BUỘC:
- Chỉ nhận xét đúng những gì có trong dữ liệu được cung cấp. Tuyệt đối không \
suy đoán hay bịa thêm lỗi không có trong dữ liệu.
- Nếu dữ liệu cho thấy không có vấn đề gì đáng kể, hãy khen ngắn gọn và thành \
thật - đừng cố bịa ra một lỗi để góp ý cho có.
- Viết bằng tiếng Việt tự nhiên, giọng điệu khích lệ như một giáo viên, KHÔNG \
liệt kê kiểu bullet point, không nhắc lại số liệu thô (ví dụ đừng nói "0.03 \
giây" hay "9/12 mora") - hãy diễn đạt lại bằng lời tự nhiên.
- Độ dài: 2-4 câu.
- Nếu có vấn đề về âm ngắt (っ) hoặc âm kéo dài (ー), hãy giải thích ngắn gọn \
CÁCH sửa, đúng tinh thần ví dụ mẫu dưới đây.

VÍ DỤ VĂN PHONG MONG MUỐN (chỉ tham khảo giọng điệu, không copy nguyên văn):
"Ngữ điệu cả câu rất tự nhiên. Còn một chỗ: âm ngắt hơi ngắn. Ở 「っ」 hãy \
ngắt hẳn một nhịp - im lặng đúng bằng một âm tiết, rồi mới bật ra 「と」. \
Người Việt thường nối liền nên nghe thành "choto"."\
"""


class CoachingUnavailableError(RuntimeError):
    """Ollama could not be reached, the model isn't pulled, or the `ollama`
    package isn't installed. Distinct from a bug in this app's own code --
    routes.py turns this into a specific, actionable HTTP error."""


@dataclass
class PronunciationFacts:
    """Everything the coaching prompt is allowed to talk about -- nothing
    else. If it isn't a field here, the model was never told about it."""

    text: str
    score: int
    level: str
    total_morae: int
    wrong_morae: list[str] = field(default_factory=list)
    duration_issues: list[DurationIssue] = field(default_factory=list)
    pitch_matched: int | None = None
    pitch_total: int | None = None


def pitch_direction_match(
    moras: list["MoraPitch"], points: list[dict[str, Any]]
) -> tuple[int, int]:
    """How many voiced learner pitch points landed on the expected side of
    zero for their mora's reference H/L label (semitone > 0 for "H", < 0
    for "L") -- both already relative to each accent phrase's own H/L
    midpoint, see app.services.pitch_extraction. A tie (exactly 0.0)
    counts as a miss: genuinely ambiguous, not a match.

    `moras` and `points` must be the same length and positionally aligned
    (app.services.pitch_accent.pitch_accent_pattern's output, and
    app.services.pitch_extraction.extract_pitch_for_windows's output, for
    the same target text). Returns (0, 0) if nothing lines up or nothing
    is voiced.
    """
    if len(moras) != len(points):
        return (0, 0)
    matched = 0
    total = 0
    for mora, point in zip(moras, points):
        semitone = point.get("semitone")
        if not point.get("voiced") or semitone is None:
            continue
        total += 1
        if (mora.pitch == "H" and semitone > 0) or (mora.pitch == "L" and semitone < 0):
            matched += 1
    return matched, total


def build_facts(
    text: str,
    score: int,
    level: str,
    moras: list["MoraPitch"],
    errors: list["PronunciationError"],
    duration_issues: list[DurationIssue],
    pitch_points: list[dict[str, Any]] | None,
) -> PronunciationFacts:
    """Assemble the facts block from data every caller of this already
    computes elsewhere (score_pronunciation's errors, prosody_issues'
    duration_issues, extract_pitch_for_windows' points) -- this function
    does no measurement of its own, only reshapes what's already real."""
    wrong_morae: list[str] = []
    for e in errors:
        if e.type == "sub":
            wrong_morae.append(f"{e.target}→{e.recognized}")
        elif e.type == "del":
            wrong_morae.append(f"{e.target}→(mất âm)")
        elif e.type == "ins":
            wrong_morae.append(f"(âm thừa){e.recognized}")

    pitch_matched = pitch_total = None
    if pitch_points is not None:
        pitch_matched, pitch_total = pitch_direction_match(moras, pitch_points)

    return PronunciationFacts(
        text=text,
        score=score,
        level=level,
        total_morae=len(moras),
        wrong_morae=wrong_morae,
        duration_issues=duration_issues,
        pitch_matched=pitch_matched,
        pitch_total=pitch_total,
    )


def _format_facts(facts: PronunciationFacts) -> str:
    lines = [
        f"Câu mục tiêu: {facts.text}",
        f"Điểm tổng: {facts.score}/100 (mức: {facts.level})",
        f"Số mora trong câu: {facts.total_morae}",
    ]

    if facts.wrong_morae:
        lines.append(
            "Mora đọc sai (mục tiêu→nghe được): " + ", ".join(facts.wrong_morae)
        )
    else:
        lines.append("Không có mora nào đọc sai (khớp 100% với văn bản mục tiêu).")

    if facts.duration_issues:
        for issue in facts.duration_issues:
            kind_vn = "âm ngắt 「っ」" if issue.kind == "short_sokuon" else "âm kéo dài 「ー」"
            prev_ = issue.prev_mora or "?"
            next_ = issue.next_mora or "?"
            ratio_pct = round(100 * issue.measured_seconds / issue.expected_seconds)
            lines.append(
                f"Vấn đề thời lượng: {kind_vn} nằm giữa 「{prev_}」và「{next_}」"
                f" chỉ kéo dài khoảng {ratio_pct}% so với một mora bình thường "
                f"trong chính bản ghi này ({issue.measured_seconds:.2f}s so với "
                f"{issue.expected_seconds:.2f}s) - bị đọc rút ngắn rõ rệt."
            )
    else:
        lines.append("Không phát hiện vấn đề về thời lượng ngắt/kéo dài.")

    if facts.pitch_total:
        lines.append(
            f"Cao độ: {facts.pitch_matched}/{facts.pitch_total} mora có audio "
            f"khớp đúng hướng cao/thấp so với mẫu."
        )
    else:
        lines.append("Không có đủ dữ liệu cao độ để đánh giá.")

    return "\n".join(lines)


async def generate_comment(facts: PronunciationFacts, settings: "Settings") -> str:
    """Call the local Ollama model to phrase `facts` as a natural
    Vietnamese coaching comment.

    Raises:
        CoachingUnavailableError: the `ollama` package isn't installed,
            Ollama isn't reachable at settings.ollama_host, the model
            named in settings.ollama_model hasn't been pulled, or the
            model returned nothing usable.
    """
    try:
        from ollama import AsyncClient, ResponseError
    except ImportError as e:
        raise CoachingUnavailableError(
            "Thiếu package 'ollama'. Cài bằng: pip install ollama"
        ) from e

    client = AsyncClient(host=settings.ollama_host, timeout=settings.ollama_timeout_s)
    try:
        response = await client.chat(
            model=settings.ollama_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _format_facts(facts)},
            ],
            think=False,
            stream=False,
            options={"temperature": settings.ollama_temperature, "num_predict": 300},
        )
    except ResponseError as e:
        if e.status_code == 404:
            raise CoachingUnavailableError(
                f"Model '{settings.ollama_model}' chưa được tải về Ollama. "
                f"Chạy: ollama pull {settings.ollama_model}"
            ) from e
        raise CoachingUnavailableError(f"Ollama báo lỗi: {e}") from e
    except CoachingUnavailableError:
        raise
    except Exception as e:  # noqa: BLE001 -- connection refused, timeout, DNS, etc.
        raise CoachingUnavailableError(
            f"Không kết nối được tới Ollama tại {settings.ollama_host}. "
            f"Đã chạy `ollama serve` chưa? (chi tiết: {e})"
        ) from e

    content = (response.get("message", {}) or {}).get("content", "").strip()
    if not content:
        raise CoachingUnavailableError("Ollama trả về nội dung rỗng.")
    return content
