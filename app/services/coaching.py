"""Turns this app's already-measured pronunciation facts (score, per-mora
correctness, sokuon/chouon timing, pitch-accent direction match) into a
short, natural-language coaching comment -- split into an "assessment" and
a "suggestion" (see CoachingComment) and written in the caller's requested
language, see SUPPORTED_LANGUAGES -- via a locally-run LLM through Ollama
(https://ollama.com) -- see .env.example for OLLAMA_HOST / OLLAMA_MODEL,
and the README for install/pull instructions. Nothing here is sent to a
third party: the request goes to whatever OLLAMA_HOST points at, normally
http://localhost:11434 on the same machine.

The `facts` block handed to the model (see _format_facts) stays
Vietnamese-labeled regardless of the requested output language -- it is
never shown to the end user, only read by the model, and the system
prompt for each language explicitly tells it to write its answer in that
language no matter what language the data itself is labeled in.

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

## Why structured JSON output

The frontend shows the assessment ("nhan xet") and the practice suggestion
("goi y tap luyen") as two separately labeled sections rather than one
paragraph, so the model is constrained (via Ollama's `format` parameter,
see COMMENT_JSON_SCHEMA) to return a small JSON object with exactly those
two keys instead of free text this module would otherwise have to split
itself with no reliable boundary between the two.

## Limitation

A local model can still ignore "only describe these facts" occasionally --
smaller/faster models drift more than larger ones (this is exactly why
the system prompt repeats the constraint and gives a style example rather
than trusting a single instruction). The `format` schema constrains the
JSON *shape* the model returns, not the *truthfulness* of its content --
this module does not fact-check the generated text against `facts` before
returning it: treat the comment as a best-effort coaching aid to read
alongside the verified score, mora grid and pitch chart, not as a
verified report on its own.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from app.services.prosody_issues import DurationIssue

if TYPE_CHECKING:
    from app.core.config import Settings
    from app.services.pitch_accent import MoraPitch
    from app.services.scoring import PronunciationError

DEFAULT_LANG = "vi"

# One full system prompt per supported output language, rather than one
# prompt plus a "write in {lang}" bolt-on -- a translated instruction set
# keeps the rules (grounding, JSON shape, no bullet points, no raw numbers,
# style example) natural in that language instead of reading like a
# translation of a translation once relayed through the model.
SYSTEM_PROMPTS: dict[str, str] = {
    "vi": """\
Bạn là một giáo viên dạy phát âm tiếng Nhật cho người Việt, đang nhận xét một \
lượt học viên đọc to một câu tiếng Nhật. Bạn sẽ nhận được một khối dữ liệu mô \
tả CHÍNH XÁC những gì đã đo được từ lượt đọc đó.

QUY TẮC BẮT BUỘC:
- Chỉ nhận xét đúng những gì có trong dữ liệu được cung cấp. Tuyệt đối không \
suy đoán hay bịa thêm lỗi không có trong dữ liệu.
- Trả lời DUY NHẤT một object JSON hợp lệ, không kèm lời giải thích, không \
kèm markdown, đúng 2 khóa: "assessment" và "suggestion".
- "assessment": nhận xét tổng quan 1-2 câu về lượt đọc này - ngữ điệu và độ \
chính xác nhìn chung thế nào. Nếu dữ liệu cho thấy không có vấn đề gì đáng \
kể, hãy khen ngắn gọn và thành thật ở đây.
- "suggestion": gợi ý luyện tập CỤ THỂ, 1-2 câu, chỉ nêu CÁCH sửa lỗi lớn \
nhất nếu có - ví dụ cách ngắt âm 「っ」 hay giữ âm kéo dài 「ー」. Nếu không \
có lỗi nào đáng để gợi ý luyện tập, để "suggestion" là chuỗi rỗng "".
- Cả hai trường viết bằng tiếng Việt tự nhiên, giọng điệu khích lệ như một \
giáo viên, KHÔNG liệt kê kiểu bullet point, không nhắc lại số liệu thô (ví \
dụ đừng nói "0.03 giây" hay "9/12 mora") - hãy diễn đạt lại bằng lời tự \
nhiên.

VÍ DỤ ĐẦU RA MONG MUỐN (chỉ tham khảo giọng điệu và cấu trúc, không copy \
nguyên văn):
{"assessment": "Ngữ điệu cả câu rất tự nhiên, hầu hết các mora đều đọc \
đúng.", "suggestion": "Còn một chỗ: âm ngắt hơi ngắn. Ở 「っ」 hãy ngắt hẳn \
một nhịp - im lặng đúng bằng một âm tiết, rồi mới bật ra 「と」. Người Việt \
thường nối liền nên nghe thành \\"choto\\"."}\
""",
    "en": """\
You are a Japanese pronunciation teacher giving feedback to a \
Vietnamese-speaking learner on one attempt at reading a Japanese sentence \
aloud. You will be given a block of data describing EXACTLY what was \
measured from that attempt.

MANDATORY RULES:
- Only comment on what is actually in the data provided. Never guess or \
invent an issue that isn't in the data.
- Reply with ONLY a single valid JSON object, no explanation, no markdown, \
with exactly 2 keys: "assessment" and "suggestion".
- "assessment": a 1-2 sentence overall assessment of this attempt -- how \
the intonation and accuracy sounded overall. If the data shows nothing \
significant wrong, give brief, sincere praise here.
- "suggestion": a CONCRETE 1-2 sentence practice suggestion, naming HOW to \
fix the single biggest issue if there is one -- for example how to hold a \
glottal stop (っ) or a long vowel (ー). If there is nothing worth \
suggesting, leave "suggestion" as the empty string "".
- Write both fields in natural, encouraging English, in a teacher's voice. \
Do NOT use bullet points, and do not repeat raw numbers (for example, \
don't say "0.03 seconds" or "9 out of 12 morae") -- rephrase them in \
natural words instead.
- Write your entire response in English, regardless of what language the \
data block itself is labeled in.

EXAMPLE OF THE DESIRED OUTPUT (for tone and structure only -- do not copy \
it verbatim):
{"assessment": "The overall intonation of the sentence sounds very \
natural, and most morae were pronounced correctly.", "suggestion": "One \
thing to work on: the pause is a little short. At 「っ」, hold a full beat \
of silence -- exactly the length of one syllable -- before releasing into \
「と」. Vietnamese speakers often run the two sounds together, so it comes \
out sounding like 'choto' instead of 'chotto'."}\
""",
}

# Same output languages, but the sentence being practised is English.
ENGLISH_TARGET_PROMPTS: dict[str, str] = {
    "vi": """\
Bạn là một giáo viên dạy phát âm tiếng Anh cho người Việt, đang nhận xét một \
lượt học viên đọc to một câu tiếng Anh. Bạn sẽ nhận được một khối dữ liệu mô \
tả CHÍNH XÁC những gì đã đo được từ lượt đọc đó.

QUY TẮC BẮT BUỘC:
- Chỉ nhận xét đúng những gì có trong dữ liệu được cung cấp. Tuyệt đối không \
suy đoán hay bịa thêm lỗi không có trong dữ liệu.
- Không nhắc tới mora, âm ngắt 「っ」, âm kéo dài 「ー」 hay tiếng Nhật — \
đơn vị ở đây là từ tiếng Anh và trọng âm từ (H = từ nội dung, L = từ chức năng).
- Trả lời DUY NHẤT một object JSON hợp lệ, không kèm lời giải thích, không \
kèm markdown, đúng 2 khóa: "assessment" và "suggestion".
- "assessment": nhận xét tổng quan 1-2 câu. Nếu không có vấn đề đáng kể, \
khen ngắn gọn và thành thật.
- "suggestion": gợi ý luyện tập CỤ THỂ 1-2 câu về từ đọc sai hoặc trọng âm. \
Nếu không có lỗi đáng kể, để "suggestion" là chuỗi rỗng "".
- Cả hai trường viết bằng tiếng Việt tự nhiên, giọng khích lệ, không bullet, \
không nhắc số liệu thô.

VÍ DỤ:
{"assessment": "Câu nghe khá rõ, hầu hết các từ khớp với mẫu.", \
"suggestion": "Từ 'beautiful' bị nuốt âm giữa. Đọc đủ ba âm bea-u-ti-ful, \
nhấn nhẹ âm đầu."}\
""",
    "en": """\
You are an English pronunciation teacher giving feedback to a \
Vietnamese-speaking learner on one attempt at reading an English sentence \
aloud. You will be given a block of data describing EXACTLY what was \
measured from that attempt.

MANDATORY RULES:
- Only comment on what is actually in the data provided. Never invent issues.
- Do not mention morae, Japanese っ/ー, or Japanese pitch accent. Units here \
are English words and lexical stress (H = content word, L = function word).
- Reply with ONLY a single valid JSON object, no markdown, keys \
"assessment" and "suggestion".
- "assessment": 1-2 sentences overall. Praise briefly if nothing is wrong.
- "suggestion": 1-2 concrete sentences on the worst word or stress issue, \
or "" if nothing is worth suggesting.
- Write both fields in natural encouraging English. No bullets, no raw numbers.

EXAMPLE:
{"assessment": "The sentence is mostly clear and most words match the target.", \
"suggestion": "Slow down on 'beautiful' and keep all three syllables."}\
""",
}

SUPPORTED_LANGUAGES: tuple[str, ...] = tuple(SYSTEM_PROMPTS)

# JSON schema passed to Ollama's `format` parameter (see generate_comment)
# so the model is constrained to emit exactly this shape instead of free
# text -- this is what makes the assessment/suggestion split in the UI
# reliable rather than something this module has to parse out of prose
# after the fact. It constrains shape only, not truthfulness -- see the
# module docstring's Limitation section.
COMMENT_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "assessment": {"type": "string"},
        "suggestion": {"type": "string"},
    },
    "required": ["assessment", "suggestion"],
}


def resolve_system_prompt(lang: str, target_lang: str = "ja") -> str:
    """System prompt for `lang` (e.g. "vi", "en") -- this, not
    _format_facts, is what actually controls the generated comment's
    language; see the module docstring.

    `target_lang` is the language of the practised sentence (ja | en).

    Raises:
        ValueError: `lang` isn't one of SUPPORTED_LANGUAGES.
    """
    table = ENGLISH_TARGET_PROMPTS if target_lang == "en" else SYSTEM_PROMPTS
    try:
        return table[lang]
    except KeyError:
        raise ValueError(
            f"Unsupported lang '{lang}'. Supported: {', '.join(SUPPORTED_LANGUAGES)}"
        ) from None


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


@dataclass
class CoachingComment:
    """The two-part coaching comment the frontend displays as separate
    sections -- see templates/index.html's coach panel. `suggestion` is
    the empty string when the model had nothing notable to suggest."""

    assessment: str
    suggestion: str


def pitch_direction_match(
    moras: list["MoraPitch"], points: list[dict[str, Any]]
) -> tuple[int, int]:
    """How many voiced learner points sit on the expected side of their
    own accent-phrase mean (H above that mean, L below).

    Semitones from pitch_extraction are relative to the *clip* mean, so
    a later High after natural declination can be negative in absolute
    terms and still be the high mora of its phrase. Comparing to the
    phrase-local mean (not to 0) keeps that from looking like a miss.
    A tie with the phrase mean counts as a miss.

    `moras` and `points` must be the same length and positionally aligned.
    Returns (0, 0) if nothing lines up or nothing is voiced.
    """
    if len(moras) != len(points):
        return (0, 0)

    phrase_values: dict[int, list[float]] = {}
    for mora, point in zip(moras, points):
        semitone = point.get("semitone")
        if not point.get("voiced") or semitone is None:
            continue
        phrase_values.setdefault(int(getattr(mora, "phrase", 0)), []).append(float(semitone))
    phrase_mean = {
        p: (sum(vals) / len(vals) if len(vals) >= 2 else 0.0)
        for p, vals in phrase_values.items()
        if vals
    }

    matched = 0
    total = 0
    for mora, point in zip(moras, points):
        semitone = point.get("semitone")
        if not point.get("voiced") or semitone is None:
            continue
        total += 1
        mean = phrase_mean.get(int(getattr(mora, "phrase", 0)))
        if mean is None:
            continue
        if (mora.pitch == "H" and float(semitone) > mean) or (
            mora.pitch == "L" and float(semitone) < mean
        ):
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


def _format_facts(facts: PronunciationFacts, target_lang: str = "ja") -> str:
    unit = "từ" if target_lang == "en" else "mora"
    lines = [
        f"Câu mục tiêu: {facts.text}",
        f"Điểm tổng: {facts.score}/100 (mức: {facts.level})",
        f"Số {unit} trong câu: {facts.total_morae}",
    ]
    if target_lang == "en":
        lines.append(
            "Đây là câu tiếng Anh. Đơn vị là từ (không phải mora tiếng Nhật)."
        )

    if facts.wrong_morae:
        lines.append(
            f"{unit.capitalize()} đọc sai (mục tiêu→nghe được): "
            + ", ".join(facts.wrong_morae)
        )
    else:
        lines.append(
            f"Không có {unit} nào đọc sai (khớp 100% với văn bản mục tiêu)."
        )

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
            f"Cao độ: {facts.pitch_matched}/{facts.pitch_total} {unit} có audio "
            f"khớp đúng hướng cao/thấp so với mẫu."
        )
    else:
        lines.append("Không có đủ dữ liệu cao độ để đánh giá.")

    return "\n".join(lines)


def _parse_comment(content: str) -> CoachingComment:
    """Parse the model's JSON response (constrained by COMMENT_JSON_SCHEMA)
    into a CoachingComment.

    Falls back gracefully if the model still returns non-JSON text despite
    the `format` constraint (smaller/local models can drift -- see the
    module docstring's Limitation section): the whole response becomes the
    assessment and suggestion is left empty, which degrades to the same
    single-block look the UI had before this split, instead of losing the
    comment entirely.
    """
    try:
        data = json.loads(content)
        assessment = str(data.get("assessment", "") or "").strip()
        suggestion = str(data.get("suggestion", "") or "").strip()
    except (json.JSONDecodeError, AttributeError):
        assessment = content.strip()
        suggestion = ""

    if not assessment and not suggestion:
        raise CoachingUnavailableError("Ollama trả về nội dung rỗng.")
    return CoachingComment(assessment=assessment, suggestion=suggestion)


async def generate_comment(
    facts: PronunciationFacts,
    settings: "Settings",
    lang: str = DEFAULT_LANG,
    target_lang: str = "ja",
) -> CoachingComment:
    """Call the local Ollama model to phrase `facts` as a natural, two-part
    coaching comment (see CoachingComment), written in `lang` (see
    SUPPORTED_LANGUAGES).

    Raises:
        ValueError: `lang` isn't supported (see resolve_system_prompt).
            Callers with an HTTP boundary (routes.py) should validate
            `lang` before doing any other work so this never fires after
            an expensive ASR call -- this check is a defensive second
            layer, not the primary one.
        CoachingUnavailableError: the `ollama` package isn't installed,
            Ollama isn't reachable at settings.ollama_host, the model
            named in settings.ollama_model hasn't been pulled, or the
            model returned nothing usable.
    """
    system_prompt = resolve_system_prompt(lang, target_lang=target_lang)

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
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": _format_facts(facts, target_lang=target_lang)},
            ],
            think=False,
            stream=False,
            format=COMMENT_JSON_SCHEMA,
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
    return _parse_comment(content)
