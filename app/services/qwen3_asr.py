"""Qwen3-ASR through Ollama (EN_OLLAMA_ASR_MODEL, default frozenlab/qwen3-asr).

All calls go over /api/chat with the audio as a 16 kHz mono PCM16 WAV,
base64 in the user message's `images` field (Ollama has no audio field).

* transcribe() -- free decoding: what the learner said, the language,
  per-word confidence and, per generated token, the top-20 alternatives
  (Ollama `logprobs` / `top_logprobs`, server >= 0.12.11). The system
  prompt asks for a verbatim transcript (measured: it then keeps "waiter"
  for "water" instead of repairing it); it never contains the target
  sentence, which would bias the transcript toward it. Given the caller's
  target language, a window the model labels as another one (accented
  English heard as "Japanese" and written in kana) is decoded again with
  the reply prefilled "language English<asr_text>" -- only then: forced on
  silence or noise it makes up an English sentence where free decoding
  answers "language None".

* word_evidence() -- pronunciation evidence for a TARGET word from the
  distribution at the transcript token where it starts (in the transcript
  as heard: after repeats and insertions). Qwen3-ASR's language model pulls
  misreadings toward the fluent target ("grass" is still written "glass",
  P = 0.9986), so P(target) alone says little; the margin to the best
  rival word, log P(target) - log P(rival) ("grass" 0.13 % -> 6.6 nats; a
  clean "glass" has no rival in the top 20 -> > 20 nats), is the evidence.
  Word level only: Ollama has no phoneme model and no timestamps.

Everything raises AsrError with a report-style status.
"""

from __future__ import annotations

import base64
import io
import logging
import math
import re
import time
import unicodedata
import wave
from urllib.parse import urlsplit
from typing import TYPE_CHECKING, Any, Callable, Sequence

import numpy as np

if TYPE_CHECKING:
    from app.core.config import Settings

logger = logging.getLogger(__name__)

ASR_TAG = "<asr_text>"
# Generic -- never the target sentence.
VERBATIM_PROMPT = (
    "Transcribe exactly what is pronounced, including mispronunciations and wrong words. "
    "Do not correct grammar or meaning."
)
# Qwen3-ASR audio is processed in 30 s chunks; the model page allows ~60 s.
WINDOW_S = 30.0
SAMPLE_RATE = 16000
KEEP_ALIVE = "10m"
NUM_PREDICT = 512
TOP_LOGPROBS = 20  # Ollama's maximum
# Margin (nats) between the target word and its best rival that counts as
# a clean word: 100 points at or above, linear to 0 at 0. Measured on TTS:
# clean words > 20, "grass" for glass 6.6, "rise" for rice 4.2, "waiter"
# for water 0.8. Not calibrated on learner speech.
MARGIN_FULL = 10.0
FILLERS = frozenset({"uh", "um", "er", "ah", "eh", "hm", "hmm", "mm", "erm", "uhm"})
TRIM_PAD_S = 0.3
CUT_SEARCH_S = 5.0
_LETTERS = re.compile(r"[^a-z]")


class AsrError(Exception):
    """An Ollama problem, already mapped to a report status."""

    def __init__(self, status: str, message: str):
        super().__init__(message)
        self.status = status


# --- audio -----------------------------------------------------------------

def wav_bytes(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Mono PCM16 WAV with a full RIFF header (raw PCM is not recognised)."""
    pcm = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes((pcm * 32767.0).astype("<i2").tobytes())
    return buf.getvalue()


def _frame_rms(samples: np.ndarray, frame: int) -> np.ndarray:
    n = len(samples) // frame
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    x = np.asarray(samples[: n * frame], dtype=np.float32).reshape(n, frame)
    return np.sqrt(np.mean(x * x, axis=1))


def trim_silence(samples: np.ndarray, sample_rate: int, pad_s: float = TRIM_PAD_S) -> np.ndarray:
    """The clip without long leading / trailing silence (the learner
    starting late or forgetting to stop), keeping `pad_s` around the speech.
    Unchanged when nothing stands out."""
    frame = max(1, int(sample_rate * 0.02))
    rms = _frame_rms(samples, frame)
    if rms.size == 0 or float(rms.max()) <= 0:
        return samples
    loud = np.flatnonzero(rms >= max(float(rms.max()) * 0.05, 1e-4))
    if loud.size == 0:
        return samples
    pad = int(pad_s * sample_rate)
    start = max(0, loud[0] * frame - pad)
    end = min(len(samples), (loud[-1] + 1) * frame + pad)
    return samples[start:end]


def audio_windows(samples: np.ndarray, sample_rate: int, window_s: float = WINDOW_S) -> list[np.ndarray]:
    """Consecutive windows of at most `window_s`, each cut at the quietest
    20 ms in the last CUT_SEARCH_S before the limit (so a word is not split
    in two -- which would come back as a false repetition); a tail under
    1 s joins the previous window rather than going out alone."""
    size = max(1, int(window_s * sample_rate))
    n = len(samples)
    frame = max(1, int(sample_rate * 0.02))
    search = int(CUT_SEARCH_S * sample_rate)
    cuts = [0]
    while n - cuts[-1] > size:
        lo, hi = cuts[-1] + max(size - search, frame), cuts[-1] + size
        rms = _frame_rms(samples[lo:hi], frame)
        # The latest quietest frame: on flat audio, cut right at the limit.
        quiet = len(rms) - 1 - int(np.argmin(rms[::-1])) if rms.size else 0
        cut = min(hi, lo + (quiet + 1) * frame) if rms.size else hi
        if n - cut < sample_rate:
            break
        cuts.append(cut)
    return [samples[a:b] for a, b in zip(cuts, cuts[1:] + [n])]


# --- reply -----------------------------------------------------------------

def parse_reply(content: str) -> tuple[str | None, str]:
    """("English", "Hello there.") from "language English<asr_text>Hello there.".
    Language None (no speech) comes back as (None, "")."""
    head, tag, text = (content or "").partition(ASR_TAG)
    if not tag:
        raise ValueError("reply has no <asr_text> marker")
    language = head.strip()
    if language.lower().startswith("language"):
        language = language[len("language"):].strip()
    if not language or language.lower() == "none":
        return None, text.strip()
    return language, text.strip()


def raw_word_confidences(logprobs: Sequence[dict] | None, transcript: str) -> list[float] | None:
    """Per whitespace-separated word of `transcript`: exp(min logprob) of
    the tokens that make it up. None when logprobs are missing or do not
    spell the same transcript."""
    if not logprobs:
        return None
    pieces = [(str(t.get("token", "")), t.get("logprob")) for t in logprobs]
    if any(not isinstance(lp, (int, float)) for _tok, lp in pieces):
        return None
    text = "".join(tok for tok, _lp in pieces)
    tag_at = text.find(ASR_TAG)
    if tag_at < 0:
        return None
    start = tag_at + len(ASR_TAG)
    body = text[start:]
    lead = len(body) - len(body.lstrip())
    if body.strip() != transcript.strip():
        return None
    spans, pos = [], 0
    for tok, lp in pieces:
        spans.append((pos - start - lead, pos - start - lead + len(tok), float(lp)))
        pos += len(tok)
    out = []
    for m in re.finditer(r"\S+", body.strip()):
        lps = [lp for a, b, lp in spans if a < m.end() and b > m.start()]
        if not lps:
            return None
        out.append(round(math.exp(min(lps)), 4))
    return out


# --- Ollama ----------------------------------------------------------------

def base_url(host: str) -> str:
    """OLLAMA_HOST as a URL: "127.0.0.1:11434", "0.0.0.0" or "localhost"
    work as they do for the ollama client and server."""
    host = (host or "").strip().rstrip("/") or "127.0.0.1:11434"
    if "://" not in host:
        host = "http://" + host
    parts = urlsplit(host)
    name = parts.hostname or "127.0.0.1"
    if name == "0.0.0.0":
        name = "127.0.0.1"
    port = parts.port or (443 if parts.scheme == "https" else 11434)
    return f"{parts.scheme}://{name}:{port}{parts.path}"


async def chat(settings: "Settings", wav: bytes, num_predict: int = NUM_PREDICT, top_logprobs: int = 0,
               language: str | None = None) -> dict:
    """One /api/chat call with the audio; returns the response JSON. With
    `language` the reply is prefilled "language <language><asr_text>" (the
    model's own forced-language format: the legacy template stops right
    after the last assistant message), and that prefix is put back in front
    of the content and the logprobs so the reply parses like a free one."""
    import httpx

    prefill = f"language {language}{ASR_TAG}" if language else ""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": VERBATIM_PROMPT},
        {"role": "user", "content": "", "images": [base64.b64encode(wav).decode("ascii")]},
    ]
    if prefill:
        messages.append({"role": "assistant", "content": prefill})
    payload: dict[str, Any] = {
        "model": settings.en_ollama_asr_model,
        "stream": False,
        "keep_alive": KEEP_ALIVE,
        "messages": messages,
        "options": {"temperature": 0, "seed": 42, "num_predict": num_predict},
        # Ignored by servers older than 0.12.11; then confidence is unknown.
        "logprobs": True,
    }
    if top_logprobs:
        payload["top_logprobs"] = top_logprobs
    url = f"{base_url(settings.ollama_host)}/api/chat"
    try:
        async with httpx.AsyncClient(timeout=settings.en_ollama_assessment_timeout_s) as client:
            response = await client.post(url, json=payload)
    except httpx.TimeoutException as e:
        raise AsrError(
            "unavailable",
            f"Ollama tại {settings.ollama_host} không trả lời trong {settings.en_ollama_assessment_timeout_s:g}s "
            f"({type(e).__name__}); tăng EN_OLLAMA_ASSESSMENT_TIMEOUT_S nếu model đang nạp.",
        ) from e
    except httpx.HTTPError as e:
        raise AsrError(
            "unavailable", f"Không kết nối được Ollama tại {settings.ollama_host}: {str(e) or type(e).__name__}"
        ) from e
    if response.status_code == 404:
        raise AsrError(
            "unavailable",
            f"Model '{settings.en_ollama_asr_model}' chưa được tải. Chạy: ollama pull {settings.en_ollama_asr_model}",
        )
    if response.status_code >= 400:
        body = response.text[:300]
        status = "unsupported" if "does not support" in body else "unavailable"
        raise AsrError(status, f"Ollama HTTP {response.status_code}: {body}")
    try:
        data = response.json()
    except ValueError as e:
        raise AsrError("invalid_output", f"Ollama trả về không phải JSON: {e}") from e
    if prefill and isinstance(data, dict):
        message = data.get("message") or {}
        data["message"] = {**message, "content": prefill + (message.get("content") or "")}
        if isinstance(data.get("logprobs"), list):
            data["logprobs"] = [{"token": prefill, "logprob": 0.0, "top_logprobs": []}] + data["logprobs"]
    return data


def _content(data: dict) -> str:
    return ((data.get("message") or {}).get("content") or "").strip()


async def _decode(settings: "Settings", wav: bytes, language: str | None = None) -> tuple[dict, str | None, str]:
    """(response JSON, language, text) for one window."""
    data = await chat(settings, wav, top_logprobs=TOP_LOGPROBS, language=language)
    content = _content(data)
    try:
        heard_language, text = parse_reply(content)
    except ValueError as e:
        raise AsrError(
            "invalid_output",
            f"Không có <asr_text> trong câu trả lời ({content[:120]!r}). Bản Ollama có thể quá cũ "
            "để đưa audio vào model này: cần bản hỗ trợ qwen3a (nên dùng 0.40.x mới nhất).",
        ) from e
    return data, heard_language, text


async def transcribe(samples: np.ndarray, sample_rate: int, settings: "Settings",
                     language: str | None = None) -> dict:
    """{"language", "transcript", "word_confidence" (or None), "stream" (the
    tokens after <asr_text> with their top-20 alternatives, windows joined
    by a space; None without logprobs), "calls", "latency_ms"}. Leading /
    trailing silence is trimmed first; long clips go window by window.
    `language` is the caller's target language as the model names it
    ("English"): a window heard as another one is decoded again forced to
    it (one more call). None = keep whatever language it detects."""
    t0 = time.perf_counter()
    forced = (language or "").strip()
    languages, texts, logprobs, streams = [], [], [], []
    have_logprobs = True
    calls = 0
    windows = audio_windows(trim_silence(samples, sample_rate), sample_rate)
    for window in windows:
        wav = wav_bytes(window, sample_rate)
        data, language, text = await _decode(settings, wav)
        calls += 1
        if forced and language and language.lower() != forced.lower():
            logger.info("Qwen3-ASR heard %s (%r); decoding again as %s", language, text[:60], forced)
            data, language, text = await _decode(settings, wav, forced)
            calls += 1
        languages.append(language)
        texts.append(text)
        if text:
            streams.append(stream_after_tag(data))
        window_lp = data.get("logprobs")
        if isinstance(window_lp, list) and window_lp:
            logprobs.append((window_lp, text))
        else:
            have_logprobs = False
    language = next((lang for lang in languages if lang), None)
    transcript = " ".join(t for t in texts if t).strip()
    stream: list[dict] | None = None
    if streams and all(streams):
        stream = list(streams[0])
        for more in streams[1:]:
            stream += [{"token": " ", "logprob": 0.0, "top_logprobs": []}] + list(more)
    confidences: list[float] | None = None
    if have_logprobs and logprobs:
        confidences = []
        for window_lp, text in logprobs:
            part = raw_word_confidences(window_lp, text)
            if part is None:
                confidences = None
                break
            confidences.extend(part)
    return {
        "language": language,
        "transcript": transcript,
        "word_confidence": confidences,
        "stream": stream,
        "calls": calls,
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
    }


# --- pronunciation evidence ------------------------------------------------

def letters(token: str) -> str:
    """Lowercase a-z of a token, accents folded ("naï" -> "nai")."""
    folded = unicodedata.normalize("NFKD", token.lower())
    return _LETTERS.sub("", "".join(ch for ch in folded if not unicodedata.combining(ch)))


def word_evidence(top: Sequence[dict], target: str,
                  same_word: Callable[[str, str], bool] | None = None,
                  ignore: Sequence[str] = (),
                  extra_own: Sequence[str] = (),
                  is_word: Callable[[str], bool] | None = None) -> dict:
    """Pronunciation evidence for `target` (lowercase a-z) from the
    distribution over the token where it starts.

    A candidate is the target when it spells the target, is in `extra_own`
    (the token the model wrote for this very word, a joined "cannot"), is a
    homophone (`same_word`: "meat" for "meet"), or is a fragment the target
    starts with ("gl" for glass) -- but a shorter REAL word (`is_word`: "a"
    for "an", "walk" for "walked") is a rival, as is any other lettered
    token ("grass", "gr", "glasses"). Fillers, punctuation and candidates
    spelling a word in `ignore` (target words skipped here, the next target
    word) are left out. Not in the top list = at most the list's floor.
    No usable candidate -> score None."""
    none = {"score": None, "p_target": None, "rival": None, "p_rival": None, "margin": None}
    raw = [(letters(str(t.get("token", ""))), float(t["logprob"])) for t in top
           if isinstance(t.get("logprob"), (int, float))]
    own_set = set(extra_own) | {target}

    def own(lt: str) -> bool:
        if lt in own_set or bool(same_word and same_word(lt, target)):
            return True
        if not target.startswith(lt):
            return False
        # A real word starting the target is a rival ("a" for "an", "walk"
        # for "walked") -- unless the rest is a word too: "any" + "time" is
        # how "anytime" may be written, not a different thing said.
        return not (is_word and is_word(lt)) or bool(is_word and is_word(target[len(lt):]))

    def ignored(lt: str) -> bool:
        # The word itself, or a fragment of it ("gl" for a skipped "glass")
        # -- not another real word it happens to start with ("a", "apple").
        return any(w == lt or (w.startswith(lt) and not (is_word and is_word(lt))) for w in ignore)

    def kept(lt: str) -> bool:
        return not lt or own(lt) or not ignored(lt)

    raw = [(lt, lp) for lt, lp in raw if kept(lt)]
    lps = [(lt, lp) for lt, lp in raw if lt and lt not in FILLERS]
    mine = [lp for lt, lp in lps if own(lt)]
    rivals = [(lp, lt) for lt, lp in lps if not own(lt)]
    if not raw or (not mine and not rivals):
        return none
    # Anything not listed is at most as likely as the list's last entry.
    floor = min(lp for _lt, lp in raw)
    target_lp = max(mine) if mine else floor
    if rivals:
        rival_lp, rival = max(rivals)
        margin = target_lp - rival_lp if mine else min(0.0, floor - rival_lp)
    else:
        rival_lp, rival = floor, None
        # No rival listed: at least target - floor; nothing below = clean.
        margin = target_lp - floor if floor < target_lp else MARGIN_FULL
    return {
        "score": round(100.0 * max(0.0, min(1.0, margin / MARGIN_FULL)), 2),
        "p_target": round(math.exp(target_lp), 4) if mine else None,
        "rival": rival,
        "p_rival": round(math.exp(rival_lp), 4) if rivals else None,
        "margin": round(margin, 3),
    }


def stream_after_tag(data: dict) -> list[dict]:
    """Generated tokens (with top_logprobs) after <asr_text>; a token that
    straddles the tag keeps only what follows it."""
    toks = [t for t in (data.get("logprobs") or []) if isinstance(t, dict)]
    text = ""
    for k, t in enumerate(toks):
        text += str(t.get("token", ""))
        if ASR_TAG in text:
            rest = text[text.index(ASR_TAG) + len(ASR_TAG):]
            return ([dict(t, token=rest)] if rest else []) + toks[k + 1:]
    return []


def stream_positions(stream: Sequence[dict], transcript: str) -> tuple[list[int], Callable[[int], int | None]] | None:
    """(character start of each whitespace word of `transcript`, a function
    giving the stream index of the token that holds a character), so a
    word's letters -- also "one" inside "twenty-one" -- can be found among
    the generated tokens. None if the stream does not spell the transcript."""
    texts = [str(t.get("token", "")) for t in stream]
    text = "".join(texts)
    lead = len(text) - len(text.lstrip())
    body = text.strip()
    if not body or body != transcript.strip():
        return None
    bounds, pos = [], -lead
    for tok in texts:
        bounds.append((pos, pos + len(tok)))
        pos += len(tok)

    def token_at(char: int) -> int | None:
        return next((k for k, (a, b) in enumerate(bounds) if a <= char < b), None)

    return [m.start() for m in re.finditer(r"\S+", body)], token_at
