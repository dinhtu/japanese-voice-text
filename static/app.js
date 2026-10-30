/**
 * Read-aloud practice: take audio from the mic or an uploaded file, send a WAV
 * to the API, show the score.
 *
 * The preset sentences are rendered server-side (see templates/index.html), so
 * this file only deals with the target text, the audio, the request, and
 * painting the result.
 */

/** Origin the API lives on. Empty data-api-base (the default) = same origin. */
const API_BASE = (document.body.dataset.apiBase || "").replace(/\/$/, "");
const API_URL = document.body.dataset.evaluateUrl
  || `${API_BASE}/api/pronunciation/evaluate`;
const PITCH_API_URL = document.body.dataset.pitchUrl
  || `${API_BASE}/api/pronunciation/pitch-accent`;
const COACH_API_URL = document.body.dataset.coachUrl
  || `${API_BASE}/api/pronunciation/coach`;
const TEXT_GUIDE_API_URL = document.body.dataset.textGuideUrl
  || `${API_BASE}/api/pronunciation/text-guide`;
/** English only: Qwen3-ASR + Ollama scoring (en-reading-v2). */
const API_V2_URL = document.body.dataset.evaluateV2Url || "";
const TTS_LANG = document.body.dataset.ttsLang || "ja-JP";
/** "Nghe mẫu": server-side proxy to the Qwen3-TTS service (app/api/tts_routes.py). */
const SPEAK_API_URL = document.body.dataset.speakUrl || `${API_BASE}/api/tts`;
const TARGET_LANG = document.body.dataset.targetLang || "ja";
/** Sample rate the ASR model runs at. */
const TARGET_SAMPLE_RATE = 16_000;
/** Stop on our own so a forgotten recording cannot exceed the upload limit. */
const MAX_DURATION_SECONDS = 30;
/** Uploads are re-encoded to 16 kHz mono (32 KB/s), so this stays well under
 *  the server's MAX_AUDIO_MB and keeps inference short. */
const MAX_UPLOAD_SECONDS = 120;
/** Rejected before decoding, so a huge file never reaches the AudioContext. */
const MAX_UPLOAD_BYTES = 50 * 1024 * 1024;
/** Supported audio upload extensions. */
const ALLOWED_UPLOAD_EXTENSIONS = [
  ".wav", ".mp3", ".ogg", ".opus", ".flac", ".m4a", ".aac",
  ".webm", ".weba", ".wma", ".aiff", ".aif", ".mp4"
];

const $ = (id) => document.getElementById(id);

const el = {
  chips: $("chips"),
  targetText: $("target-text"),
  targetReading: $("target-reading"),
  targetMeaning: $("target-meaning"),
  speak: $("speak"),
  speakIcon: $("speak-icon"),
  speakLabel: $("speak-label"),
  customForm: $("custom-form"),
  customInput: $("custom-input"),
  customApply: $("custom-apply"),
  suggest: $("search-suggest"),
  searchEmpty: $("search-empty"),
  targetResult: $("target-result"),
  recorder: $("recorder"),
  dial: $("dial"),
  level: $("level"),
  mic: $("mic"),
  micIcon: $("mic-icon"),
  status: $("status"),
  timer: $("timer"),
  timerBar: $("timer-bar"),
  timerMax: $("timer-max"),
  elapsed: $("elapsed"),
  pickFile: $("pick-file"),
  file: $("file"),
  uploadHint: $("upload-hint"),
  playback: $("playback"),
  audio: $("audio"),
  reset: $("reset"),
  error: $("error"),
  errorText: $("error-text"),
  result: $("result"),
  ringValue: $("ring-value"),
  score: $("score"),
  pill: $("pill"),
  pillIcon: $("pill-icon"),
  pillText: $("pill-text"),
  feedback: $("feedback"),
  mCer: $("m-cer"),
  mDistance: $("m-distance"),
  mDuration: $("m-duration"),
  aspects: $("aspects"),
  aspectsList: $("aspects-list"),
  aspectsMethod: $("aspects-method"),
  rTarget: $("r-target"),
  rHeard: $("r-heard"),
  syll: $("syll"),
  syllGrid: $("syll-grid"),
  diffs: $("diffs"),
  diffsLabel: $("diffs-label"),
  diffsList: $("diffs-list"),
  pitchBtn: $("pitch-btn"),
  pitch: $("pitch"),
  pitchTitle: $("pitch-title"),
  pitchLegend: $("pitch-legend"),
  pitchStatus: $("pitch-status"),
  pitchChart: $("pitch-chart"),
  coachBtn: $("coach-btn"),
  coachLang: $("coach-lang"),
  coachPanel: $("coach-panel"),
  coachStatus: $("coach-status"),
  coachAssessmentBlock: $("coach-assessment-block"),
  coachAssessment: $("coach-assessment"),
  coachSuggestionBlock: $("coach-suggestion-block"),
  coachSuggestion: $("coach-suggestion"),
  guideBtn: $("guide-btn"),
  guidePanel: $("guide-panel"),
  guideLang: $("guide-lang"),
  guideStatus: $("guide-status"),
  guideContent: $("guide-content"),
  guideRefresh: $("guide-refresh"),
};

const STATUS_LABEL = {
  idle: "Nhấn vào micro để ghi âm, hoặc tải lên file có sẵn",
  requesting: "Đang xin quyền micro…",
  recording: "Đang ghi âm — nhấn lại để dừng",
  processing: "Đang xử lý bản ghi…",
  evaluating: "Đang chấm điểm…",
};

const LEVEL_BADGE = {
  excellent: { label: "Xuất sắc", color: "#047857", bg: "#ecfdf5", border: "#a7f3d0" },
  good: { label: "Tốt", color: "#1d4ed8", bg: "#eff6ff", border: "#bfdbfe" },
  fair: { label: "Khá", color: "#b45309", bg: "#fffbeb", border: "#fde68a" },
  poor: { label: "Cần luyện thêm", color: "#be123c", bg: "#fff1f2", border: "#fecdd3" },
  mismatch: { label: "Không khớp", color: "#be123c", bg: "#fff1f2", border: "#fecdd3" },
};

const RING_COLOR = {
  excellent: "var(--score-excellent)",
  good: "var(--score-good)",
  fair: "var(--score-fair)",
  poor: "var(--score-poor)",
  mismatch: "var(--score-poor)",
};

const formatDuration = (seconds) =>
  `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, "0")}`;

/* ------------------------------------------------------------------ WAV */

/**
 * Decode a container the browser understands (webm/ogg from MediaRecorder,
 * wav from an upload) into an AudioBuffer.
 */
async function decodeAudio(blob) {
  const decodeContext = new AudioContext();
  try {
    return await decodeContext.decodeAudioData(await blob.arrayBuffer());
  } finally {
    decodeContext.close().catch(() => undefined);
  }
}

/**
 * Downmix a decoded buffer to mono PCM at TARGET_SAMPLE_RATE.
 *
 * The API accepts .wav only and the model runs at 16 kHz, so every source —
 * recorded or uploaded — goes through this one offline pass.
 */
async function toMonoPcm(decoded) {
  const frameCount = Math.max(1, Math.ceil((decoded.duration || 0) * TARGET_SAMPLE_RATE));
  const offline = new OfflineAudioContext(1, frameCount, TARGET_SAMPLE_RATE);
  const source = offline.createBufferSource();
  source.buffer = decoded;
  source.connect(offline.destination);
  source.start();

  return (await offline.startRendering()).getChannelData(0);
}

/** Write 16-bit PCM samples into a RIFF/WAVE container. */
function encodeWav(samples, sampleRate) {
  const bytesPerSample = 2;
  const dataSize = samples.length * bytesPerSample;
  const view = new DataView(new ArrayBuffer(44 + dataSize));

  const writeString = (offset, value) => {
    for (let i = 0; i < value.length; i += 1) view.setUint8(offset + i, value.charCodeAt(i));
  };

  writeString(0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true); // PCM chunk size
  view.setUint16(20, 1, true); // audio format: PCM
  view.setUint16(22, 1, true); // channels: mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * bytesPerSample, true); // byte rate
  view.setUint16(32, bytesPerSample, true); // block align
  view.setUint16(34, 8 * bytesPerSample, true); // bits per sample
  writeString(36, "data");
  view.setUint32(40, dataSize, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i += 1) {
    // Clamp before scaling so loud passages clip instead of wrapping around.
    const sample = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
    offset += bytesPerSample;
  }

  return new Blob([view], { type: "audio/wav" });
}

/* --------------------------------------------------------------- State */

/** Empty until the learner searches a sentence (or picks a sample). */
let target = {
  text: "",
};
let status = "idle";
let audioUrl = null;
/** The exact WAV last sent to /evaluate, kept so the on-demand /coach
 *  request (see requestCoach()) can reuse it instead of re-recording. */
let lastWav = null;

let recorder = null;
let stream = null;
let audioContext = null;
let analyser = null;
let frameId = null;
let timerId = null;
let startedAt = 0;

function setStatus(next) {
  status = next;
  const recording = next === "recording";
  const busy = next === "requesting" || next === "processing" || next === "evaluating";

  el.status.textContent = STATUS_LABEL[next];
  el.dial.classList.toggle("is-recording", recording);
  el.dial.classList.toggle("is-busy", busy);
  el.mic.disabled = busy;
  el.mic.setAttribute("aria-pressed", String(recording));
  el.mic.setAttribute("aria-label", recording ? "Dừng ghi âm" : "Bắt đầu ghi âm");
  el.micIcon.firstElementChild.setAttribute(
    "href",
    busy ? "#i-loader" : recording ? "#i-stop" : "#i-mic",
  );
  el.timer.hidden = !recording;

  for (const chip of el.chips.children) chip.disabled = recording || busy;
  el.customInput.disabled = recording || busy;
  el.customApply.disabled = recording || busy;
  el.pickFile.disabled = recording || busy;
  el.reset.disabled = busy;
}

/** Point the practice at another sentence, whichever source picked it. */
function setTarget({ text, reading = "", meaning = "", chip = null }) {
  target = { text };
  el.targetText.textContent = text;
  el.targetReading.textContent = reading;
  el.targetReading.hidden = !reading;
  el.targetMeaning.textContent = meaning;
  el.targetMeaning.removeAttribute("title");
  el.targetMeaning.hidden = !meaning;
  if (el.targetResult) el.targetResult.hidden = false;
  if (el.searchEmpty) el.searchEmpty.hidden = true;

  for (const other of el.chips.children) other.classList.toggle("chip--active", other === chip);

  clearOutput();
  clearPlayback();
  resetPitch();
  resetTextGuide();
  if (speakState !== "idle") stopSpeak();
  if (document.body.dataset.readingUrl && !reading) {
    fetch(`${document.body.dataset.readingUrl}?text=${encodeURIComponent(text)}`)
      .then((response) => response.ok ? response.json() : null)
      .then((data) => {
        if (target.text === text && data?.reading) {
          el.targetReading.textContent = data.reading;
          el.targetReading.hidden = false;
        }
      })
      .catch(() => {});
  }
}

function showError(message) {
  el.errorText.textContent = message;
  el.error.hidden = false;
  el.result.hidden = true;
}

function clearOutput() {
  el.error.hidden = true;
  el.result.hidden = true;
  const v2 = $("result-v2");
  if (v2) v2.hidden = true;
  resetCoach();
}

function clearPlayback() {
  if (audioUrl) URL.revokeObjectURL(audioUrl);
  audioUrl = null;
  el.audio.removeAttribute("src");
  el.playback.hidden = true;
  lastWav = null;
}

/* ------------------------------------------------------------ Recording */

function releaseResources() {
  if (frameId !== null) cancelAnimationFrame(frameId);
  frameId = null;

  if (timerId !== null) clearInterval(timerId);
  timerId = null;

  stream?.getTracks().forEach((track) => track.stop());
  stream = null;

  audioContext?.close().catch(() => undefined);
  audioContext = null;
  analyser = null;

  el.level.style.transform = "scale(1)";
}

/** Drive the ring around the mic button from the real input level. */
function trackLevel() {
  const samples = new Uint8Array(analyser.fftSize);
  let level = 0;

  const tick = () => {
    analyser.getByteTimeDomainData(samples);

    // RMS around the 128 midpoint of the unsigned byte waveform.
    let sum = 0;
    for (let i = 0; i < samples.length; i += 1) {
      const deviation = (samples[i] - 128) / 128;
      sum += deviation * deviation;
    }
    const scaled = Math.min(1, Math.sqrt(sum / samples.length) * 3.2);

    // Ease toward the new value so the ring pulses instead of flickering.
    level += (scaled - level) * 0.35;
    el.level.style.transform = `scale(${1 + level * 0.45})`;

    frameId = requestAnimationFrame(tick);
  };
  frameId = requestAnimationFrame(tick);
}

function stopRecording() {
  if (recorder && recorder.state !== "inactive") recorder.stop();
}

/** Recording/upload need a sentence to score against. */
function requireTarget() {
  if (target.text) return true;
  showError("Hãy nhập câu cần đọc ở bước 1 và bấm Tìm trước khi ghi âm.");
  el.customInput.focus();
  return false;
}

async function startRecording() {
  if (status === "recording" || status === "requesting") return;
  if (!requireTarget()) return;

  if (!navigator.mediaDevices?.getUserMedia) {
    showError("Trình duyệt không hỗ trợ ghi âm. Hãy dùng Chrome hoặc Edge (qua HTTPS hoặc localhost).");
    return;
  }

  setStatus("requesting");
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
    });
  } catch {
    setStatus("idle");
    showError("Không truy cập được micro. Hãy cho phép quyền micro trong trình duyệt.");
    return;
  }

  audioContext = new AudioContext();
  analyser = audioContext.createAnalyser();
  analyser.fftSize = 1024;
  audioContext.createMediaStreamSource(stream).connect(analyser);

  const chunks = [];
  recorder = new MediaRecorder(stream);
  recorder.ondataavailable = (event) => {
    if (event.data.size > 0) chunks.push(event.data);
  };

  recorder.onstop = async () => {
    releaseResources();
    setStatus("processing");

    let wav;
    try {
      const recorded = new Blob(chunks, { type: recorder.mimeType || "audio/webm" });
      if (recorded.size === 0) throw new Error("empty recording");
      wav = encodeWav(await toMonoPcm(await decodeAudio(recorded)), TARGET_SAMPLE_RATE);
    } catch {
      setStatus("idle");
      showError("Không xử lý được bản ghi âm. Hãy thử ghi lại.");
      return;
    }

    showPlayback(wav);
    await Promise.all([evaluate(wav), comparePitch(), TARGET_LANG === "en" ? null : requestCoach()]);
  };

  startedAt = Date.now();
  el.elapsed.textContent = "● 0:00";
  el.timerBar.style.width = "0%";
  recorder.start();
  setStatus("recording");
  trackLevel();

  timerId = setInterval(() => {
    const seconds = (Date.now() - startedAt) / 1000;
    el.elapsed.textContent = `● ${formatDuration(seconds)}`;
    el.timerBar.style.width = `${Math.min(1, seconds / MAX_DURATION_SECONDS) * 100}%`;
    if (seconds >= MAX_DURATION_SECONDS) stopRecording();
  }, 100);
}

/* --------------------------------------------------------------- Upload */

/** Show the WAV we are about to send, so the user can listen back to it. */
function showPlayback(wav) {
  clearPlayback();
  audioUrl = URL.createObjectURL(wav);
  el.audio.src = audioUrl;
  el.playback.hidden = false;
  lastWav = wav;
}

/** Re-encode a picked/dropped file and score it like a fresh recording. */
async function submitFile(file) {
  if (!file || status !== "idle") return;
  if (!requireTarget()) return;

  const ext = "." + (file.name.split(".").pop() || "").toLowerCase();
  // A dropped file bypasses the input's accept filter, so check it here too.
  if (file.name.includes(".") && !ALLOWED_UPLOAD_EXTENSIONS.includes(ext) && !file.type.startsWith("audio/")) {
    showError(`Định dạng file "${file.name}" không được hỗ trợ. Hãy chọn file âm thanh (WAV, MP3, FLAC, OGG, M4A, WebM...).`);
    return;
  }

  if (file.size > MAX_UPLOAD_BYTES) {
    showError(`File quá lớn (tối đa ${MAX_UPLOAD_BYTES / (1024 * 1024)}MB).`);
    return;
  }

  clearOutput();
  setStatus("processing");

  let wav;
  try {
    const decoded = await decodeAudio(file);
    if (decoded.duration > MAX_UPLOAD_SECONDS) {
      setStatus("idle");
      showError(
        `File dài ${formatDuration(decoded.duration)} — tối đa ${formatDuration(MAX_UPLOAD_SECONDS)}.`,
      );
      return;
    }
    wav = encodeWav(await toMonoPcm(decoded), TARGET_SAMPLE_RATE);
  } catch {
    setStatus("idle");
    showError("Không đọc được file âm thanh này. Hãy thử một file khác.");
    return;
  }

  showPlayback(wav);
  await Promise.all([evaluate(wav), comparePitch(), TARGET_LANG === "en" ? null : requestCoach()]);
}

/* ------------------------------------------------------------ Evaluate */

async function evaluate(wav) {
  clearOutput();
  setStatus("evaluating");
  // Never leave the previous take's curve / checks up during (or after a
  // failed) retake.
  if (TARGET_LANG === "en" && englishTake) {
    englishTake = null;
    if (englishReference) renderPitchChart();
  }

  // English: the AI comment comes back inside the evaluate response (`coach`).
  if (TARGET_LANG === "en") coachPending();
  try {
    // English page: "Cách chấm" switches between /evaluate and /evaluate-v2;
    // both answer in the same shape, v2 adds its details in v2_assessment.
    const v2 = TARGET_LANG === "en" && Boolean(API_V2_URL) && apiMode() === "v2";
    const result = v2
      ? await postEvaluate(API_V2_URL, wav, { feedback: "true" })
      : await postEvaluate(API_URL, wav, TARGET_LANG === "en" ? { coach: "true" } : {});
    renderResult(result);
    const noteV1 = $("result-note-v1");
    const noteV2 = $("result-note-v2");
    if (noteV1 && noteV2) {
      noteV1.hidden = v2;
      noteV2.hidden = !v2;
    }
    if (TARGET_LANG === "en") showCoachFrom(result);
    if (v2 && result.v2_assessment) renderV2(result.v2_assessment, { showFeedback: false });
    // F0 measured from the WAV, one point per target mora — not the
    // dictionary H/L of whatever kana ASR printed.
    learnerPitch = result.measured_pitch ?? [];
    if (result.recognized_pitch_patterns && result.recognized_pitch_patterns.length) {
      referencePatterns = result.recognized_pitch_patterns;
    }
    if (TARGET_LANG === "en") englishTake = result.intonation || null;
    if (document.body.dataset.targetLang === "zh" && referencePattern) renderChinesePitch();
    else if (referencePattern || englishTake) renderPitchChart();
  } catch (error) {
    showError(error instanceof Error ? error.message : "Đã có lỗi xảy ra.");
    resetCoach();
  } finally {
    setStatus("idle");
  }
}

/** POST the take to one evaluate endpoint; the parsed JSON, or an Error with the API's detail. */
async function postEvaluate(url, wav, extra = {}) {
  const formData = new FormData();
  formData.append("text", target.text);
  formData.append("audio", new File([wav], "recording.wav", { type: "audio/wav" }));
  // English reports (improvements / Ollama feedback) follow the coach language.
  if (TARGET_LANG === "en" && el.coachLang) formData.append("lang", el.coachLang.value);
  for (const [key, value] of Object.entries(extra)) formData.append(key, value);
  const response = await fetch(url, { method: "POST", body: formData });
  if (!response.ok) {
    const detail = await response
      .json()
      .then((body) => body.detail)
      .catch(() => undefined);
    throw new Error(detail || `Yêu cầu thất bại (HTTP ${response.status})`);
  }
  return response.json();
}

/* ------------------------------------------------- Which API to score with */

const API_MODE_KEY = "pronunciation-en-api-mode";

function apiMode() {
  const checked = document.querySelector('input[name="api-mode"]:checked');
  return checked ? checked.value : "v1";
}

function initApiMode() {
  const radios = document.querySelectorAll('input[name="api-mode"]');
  if (!radios.length) return;
  let saved = null;
  try {
    saved = localStorage.getItem(API_MODE_KEY);
  } catch {
    saved = null; // storage blocked: keep the default
  }
  radios.forEach((radio) => {
    if (saved && radio.value === saved) radio.checked = true;
    radio.addEventListener("change", () => {
      try {
        localStorage.setItem(API_MODE_KEY, radio.value);
      } catch {
        /* not remembered, still used for this take */
      }
    });
  });
}

initApiMode();

function aspectTone(value) {
  if (value >= 90) return "excellent";
  if (value >= 75) return "good";
  if (value >= 60) return "fair";
  return "poor";
}

function fluencyMethodHint(result) {
  const parts = [];
  if (result.vad_method === "silero") parts.push("VAD: Silero");
  else if (result.vad_method === "energy") parts.push("VAD: energy (chưa có Silero)");
  else parts.push("VAD: không chạy");
  if (result.pause_count != null) parts.push(`${result.pause_count} ngắt`);
  if (result.speech_ratio != null) {
    parts.push(`${Math.round(Number(result.speech_ratio) * 100)}% tiếng`);
  }
  if (result.aspect_method) parts.push(result.aspect_method);
  return parts.join(" · ");
}

function renderAspects(result) {
  if (el.aspects && result.aspect_method) {
    el.aspects.dataset.method = result.aspect_method;
  }
  if (el.aspectsMethod) {
    el.aspectsMethod.hidden = false;
    el.aspectsMethod.textContent = fluencyMethodHint(result);
  }
  const rows = [
    [document.body.dataset.targetLang === "zh" ? "Khớp âm tiết" : "Phát âm", result.pronunciation_score, true],
    ["Trôi chảy", result.fluency_score, true],
    ["Nhịp", result.rhythm_score, result.rhythm_measured !== false],
    ["Ngữ điệu", result.intonation_score, result.intonation_measured],
  ];
  el.aspectsList.replaceChildren(
    ...rows.map(([label, value, measured]) => {
      const row = document.createElement("div");
      row.className = "aspect";
      const title = document.createElement("span");
      title.className = "aspect__name";
      title.textContent = label;
      const track = document.createElement("div");
      track.className = "aspect__track";
      const fill = document.createElement("i");
      fill.className = "aspect__fill";
      const num = document.createElement("span");
      num.className = "aspect__num";
      if (value == null || !measured) {
        fill.style.width = "0%";
        num.textContent = "—";
        num.classList.add("aspect__num--na");
      } else {
        const n = Math.min(100, Math.max(0, Number(value)));
        fill.style.width = `${n}%`;
        fill.dataset.tone = aspectTone(n);
        num.textContent = n.toFixed(1);
      }
      track.appendChild(fill);
      row.append(title, track, num);
      return row;
    }),
  );
}

function renderResult(result) {
  const badge = LEVEL_BADGE[result.feedback.level] ?? LEVEL_BADGE.mismatch;
  const ringScore = Number(result.score);
  const color = RING_COLOR[aspectTone(ringScore)] ?? RING_COLOR.mismatch;

  const circumference = 2 * Math.PI * 52;
  const filled = (Math.min(100, Math.max(0, ringScore)) / 100) * circumference;
  el.ringValue.style.stroke = color;
  el.ringValue.style.strokeDasharray = `${filled} ${circumference}`;
  el.score.style.color = color;
  el.score.textContent = Number.isInteger(ringScore)
    ? String(ringScore)
    : ringScore.toFixed(1);

  el.pill.style.color = badge.color;
  el.pill.style.background = badge.bg;
  el.pill.style.borderColor = badge.border;
  el.pillText.textContent = badge.label;
  el.pillIcon.firstElementChild.setAttribute(
    "href",
    result.errors.length === 0 ? "#i-check-circle" : "#i-warning",
  );

  el.feedback.textContent = result.feedback.message;
  el.mCer.textContent = result.cer.toFixed(3);
  el.mDistance.textContent = `${result.distance} ${document.body.dataset.targetLang === "zh" ? "âm tiết" : "ký tự"}`;
  el.mDuration.textContent = `${result.audio_duration.toFixed(2)}s`;
  renderAspects(result);

  // Mark the target characters the model did not hear as expected.
  const wrong = new Set(
    result.errors.filter((error) => error.type !== "ins").map((error) => error.position),
  );
  const targetUnits = document.body.dataset.targetLang === "zh" && result.target_reading
    ? result.target_reading.split(" ") : [...result.target_hiragana];
  el.rTarget.replaceChildren(
    ...targetUnits.map((char, index) => {
      const node = document.createElement(wrong.has(index) ? "mark" : "span");
      node.textContent = document.body.dataset.targetLang === "zh" ? `${char} ` : char;
      return node;
    }),
  );
  el.rHeard.textContent = document.body.dataset.targetLang === "zh"
    ? `${result.recognized_hiragana || "—"} (${result.recognized_reading || "—"})`
    : result.recognized_hiragana || "—";

  // Per-mora "dung/sai" grid - real character-level edit-distance data
  // (see app/services/mora_diff.py) regrouped onto the target's morae, not
  // a fabricated confidence score.
  const moraStatus = result.mora_status || [];
  el.syll.hidden = moraStatus.length === 0;
  if (moraStatus.length > 0) {
    el.syllGrid.replaceChildren(
      ...moraStatus.map((m) => {
        const cell = document.createElement("span");
        cell.className = m.ok ? "syll__cell" : "syll__cell syll__cell--bad";
        if (Array.isArray(m.syllables) && m.syllables.length) {
          cell.classList.add("syll__cell--word");
          cell.innerHTML = englishWordCellHtml(m);
        } else {
          cell.textContent = m.mora;
        }
        return cell;
      }),
    );
  }

  if (TARGET_LANG === "en") renderProsody(result.intonation || null);

  el.diffs.hidden = result.errors.length === 0;
  if (result.errors.length > 0) {
    el.diffsLabel.textContent = `Chi tiết ${result.errors.length} điểm lệch`;
    const items = result.errors.slice(0, 12).map((error) => {
      const li = document.createElement("li");
      li.innerHTML =
        '<span class="jp"></span><span class="to">→</span>' +
        '<span class="jp heard"></span><span class="pos"></span>';
      li.children[0].textContent = error.target || "∅";
      li.children[2].textContent = error.recognized || "∅";
      li.children[3].textContent = `#${error.position}`;
      return li;
    });
    if (result.errors.length > 12) {
      const more = document.createElement("li");
      more.className = "more";
      more.textContent = `+${result.errors.length - 12} nữa`;
      items.push(more);
    }
    el.diffsList.replaceChildren(...items);
  }

  if (TARGET_LANG === "en") renderComparison(result);
  el.result.hidden = false;
}

/* ------------------------------------------- Score reports, side by side */

const CMP_STATUS = {
  gated: "Không chấm (đọc sai nhiều từ)",
  insufficient_evidence: "Thiếu dữ liệu",
  not_assessed: "Không áp dụng",
  unavailable: "Không chạy được",
  disabled: "Đã tắt",
  invalid_output: "Trả lời không hợp lệ",
  unsupported: "Ollama không nhận audio",
  no_speech: "Không nghe thấy lời",
  language_mismatch: "Không phải tiếng Anh",
  asr_unavailable: "Ollama không chạy được",
  no_voice: "Không có giọng",
  skipped: "Bỏ qua",
  content_mismatch: "Đọc sai nhiều từ",
  complete: "Đủ tiêu chí",
  partial_assessment: "Thiếu một số tiêu chí",
  scored: "Đã chấm",
};

const CMP_WORD = {
  correct: "Đúng",
  mispronounced: "Gần đúng",
  substituted: "Đọc sai",
  omitted: "Bỏ sót",
};

const CMP_CRITERIA = [
  ["content_accuracy", "Đọc đúng nội dung"],
  ["phoneme_accuracy", "Phát âm"],
  ["word_stress", "Trọng âm từ"],
  ["intonation", "Ngữ điệu"],
  ["fluency", "Trôi chảy"],
  ["rhythm_and_linking", "Nhịp, nối âm"],
];

const fmtScore = (value) => (value == null ? null : Number(value).toFixed(1));

/** One table cell: coloured number, or why there is none. */
function cmpCell(value, { status = null, sub = null, subLabel = "" } = {}) {
  if (value == null) {
    const why = status ? CMP_STATUS[status] || status : "—";
    return `<td><span class="cmp__na">${escapeHtml(why)}</span></td>`;
  }
  const n = Math.min(100, Math.max(0, Number(value)));
  const extra = sub == null
    ? ""
    : `<span class="cmp__sub">${escapeHtml(subLabel)} ${escapeHtml(fmtScore(sub))}</span>`;
  return `<td><span class="cmp__num" data-tone="${aspectTone(n)}">${n.toFixed(1)}</span>${extra}</td>`;
}

function renderComparison(result) {
  const box = $("compare");
  if (!box) return;
  const reading = result.reading_assessment;
  const gopt = result.gopt_assessment;
  const ollama = result.ollama_assessment;
  box.hidden = !(reading || gopt || ollama);
  if (box.hidden) return;

  const rc = (key) => reading?.criteria?.[key] || {};
  const oc = (key) => ollama?.criteria?.[key] || {};
  // Qwen3-ASR report: criteria present = it ran; otherwise its status says why.
  const ollamaStatus = ollama?.criteria ? null : ollama?.status || "unavailable";
  const goptStatus = gopt?.heads ? null : gopt?.status || "unavailable";
  const head = (key) => gopt?.heads?.[key] ?? null;
  const adj = (key) => (gopt?.adjusted_heads ? gopt.adjusted_heads[key] : null);
  const goptCell = (key) => (head(key) == null
    ? cmpCell(null, { status: goptStatus })
    : cmpCell(head(key), { sub: adj(key), subLabel: "đ/c" }));
  // Legacy bars per criterion (null = the old page has no such bar).
  const legacy = {
    content_accuracy: result.score,
    phoneme_accuracy: result.pronunciation_score,
    word_stress: null,
    intonation: result.intonation_measured ? result.intonation_score : null,
    fluency: result.fluency_score,
    rhythm_and_linking: result.rhythm_measured !== false ? result.rhythm_score : null,
  };
  const goptKey = {
    content_accuracy: "completeness",
    phoneme_accuracy: "accuracy",
    intonation: "prosodic",
    fluency: "fluency",
  };
  const rows = [
    `<tr class="cmp__total"><th scope="row">Điểm tổng</th>
      ${cmpCell(result.score)}
      ${cmpCell(reading?.overall_score, { status: reading?.overall_score == null ? reading?.status : null })}
      ${gopt?.average == null ? cmpCell(null, { status: goptStatus }) : cmpCell(gopt.average, { sub: gopt.adjusted_average, subLabel: "đ/c" })}
      ${cmpCell(ollama?.overall_score, { status: ollama?.overall_score == null ? ollamaStatus || ollama?.status : null })}
    </tr>`,
    ...CMP_CRITERIA.map(([key, label]) => {
      const weight = rc(key).weight;
      return `<tr><th scope="row">${escapeHtml(label)}${weight ? ` <span class="cmp__w">${Math.round(weight * 100)}%</span>` : ""}</th>
        ${cmpCell(legacy[key])}
        ${cmpCell(rc(key).score, { status: rc(key).status })}
        ${goptKey[key] ? goptCell(goptKey[key]) : cmpCell(null)}
        ${cmpCell(oc(key).score, { status: ollamaStatus || oc(key).status })}
      </tr>`;
    }),
  ];
  $("compare-table").innerHTML = `
    <thead><tr><th scope="col">Tiêu chí</th><th scope="col">Hiện tại</th>
      <th scope="col">Reading</th><th scope="col">GOPT</th><th scope="col">Qwen3-ASR</th></tr></thead>
    <tbody>${rows.join("")}</tbody>`;

  $("compare-reading").innerHTML = readingCardHtml(reading);
  $("compare-gopt").innerHTML = goptCardHtml(gopt);
  $("compare-ollama").innerHTML = ollamaCardHtml(ollama);
}

function cmpCardHead(title, status, meta = "") {
  const tone = ["scored", "complete"].includes(status) ? "ok"
    : ["partial_assessment"].includes(status) ? "warn" : "bad";
  return `<div class="cmp-card__head"><h3>${escapeHtml(title)}</h3>
    <span class="cmp-card__status" data-tone="${tone}">${escapeHtml(CMP_STATUS[status] || status || "—")}</span></div>
    ${meta ? `<p class="cmp-card__meta">${meta}</p>` : ""}`;
}

function readingCardHtml(r) {
  if (!r) return cmpCardHead("Reading (en-reading-v1)", "unavailable");
  const gate = r.content_gate || {};
  const audio = r.audio_analysis || {};
  const facts = [
    `Đọc đúng ${Math.round((gate.word_coverage || 0) * 100)}% số từ`,
    gate.passed ? "qua ngưỡng nội dung" : "dưới ngưỡng nội dung → không chấm nhịp/ngữ điệu",
    audio.correct_words_per_minute != null ? `${audio.correct_words_per_minute} từ đúng/phút` : null,
    audio.pause_count != null ? `${audio.pause_count} lần ngắt` : null,
  ].filter(Boolean).map(escapeHtml).join(" · ");
  const words = (r.word_results || []).map((w) => {
    const tip = [CMP_WORD[w.status] || w.status, w.recognized ? `nghe: ${w.recognized}` : "",
      w.phoneme_score != null ? `rõ: ${w.phoneme_score}` : "", w.stress_ok === false ? "sai trọng âm" : ""]
      .filter(Boolean).join(" · ");
    return `<span class="cmp-word" data-status="${escapeHtml(w.status)}" title="${escapeHtml(tip)}">${escapeHtml(w.word)}${
      w.status === "substituted" && w.recognized ? `<small>${escapeHtml(w.recognized)}</small>` : ""}${
      w.stress_ok === false ? "<small>trọng âm</small>" : ""}</span>`;
  }).join("");
  const extra = [...(r.inserted || []).map((e) => `+${e.text}`), ...(r.repeated || []).map((e) => `↻${e.text}`)];
  const tips = (r.improvements || []).map((i) =>
    `<li data-priority="${escapeHtml(i.priority)}">${escapeHtml(i.message)}</li>`).join("");
  return `${cmpCardHead("Reading (en-reading-v1)", r.status, facts)}
    <div class="cmp-words">${words}</div>
    ${extra.length ? `<p class="cmp-card__meta">Từ thừa (+) / lặp (↻): ${escapeHtml(extra.join(", "))}</p>` : ""}
    ${tips ? `<ul class="cmp-tips">${tips}</ul>` : `<p class="cmp-card__meta">Không có gợi ý sửa.</p>`}`;
}

function goptCardHtml(g) {
  if (!g) return cmpCardHead("GOPT", "unavailable");
  const meta = g.heads
    ? escapeHtml([
      g.feature_source === "forced_alignment" ? "khung âm theo forced alignment" : "khung âm chia đều",
      `${g.phone_count} âm`,
      g.truncated ? "quá 50 âm, phần cuối bị cắt" : null,
    ].filter(Boolean).join(" · "))
    : "";
  const heads = g.heads
    ? `<table class="cmp-mini"><thead><tr><th></th><th>Thô</th><th>Điều chỉnh</th></tr></thead><tbody>${
      ["accuracy", "completeness", "fluency", "prosodic"].map((k) => `<tr><th>${k}${
        g.head_reliable && g.head_reliable[k] === false ? " ⚠" : ""}</th><td>${fmtScore(g.heads[k])}</td><td>${
        fmtScore(g.adjusted_heads?.[k]) ?? "—"}</td></tr>`).join("")
    }<tr><th>total</th><td>${fmtScore(g.heads.total)}</td><td></td></tr></tbody></table>`
    : "";
  const words = (g.words || []).map((w) => `<span class="cmp-word" data-status="${w.read_ok ? "correct" : "omitted"}"
      title="${escapeHtml(`${w.phones.join(" ")} · thô: ${fmtScore(w.raw?.accuracy) ?? "—"}`)}">${escapeHtml(w.word)}<small>${
      fmtScore(w.accuracy) ?? "—"}</small></span>`).join("");
  return `${cmpCardHead("GOPT", g.status, meta)}${heads}
    ${words ? `<p class="cmp-card__meta">Điểm từng từ (accuracy; từ không đọc = 0):</p><div class="cmp-words">${words}</div>` : ""}
    ${g.status === "unavailable" ? `<p class="cmp-card__meta">GOPT không chạy (thiếu checkpoint hoặc bị tắt).</p>` : ""}`;
}

function ollamaCardHtml(o) {
  const title = "Qwen3-ASR (Ollama)";
  if (!o) return cmpCardHead(title, "unavailable");
  const meta = escapeHtml([
    o.model,
    o.language ? `ngôn ngữ: ${o.language}${o.language_mismatch ? " (không phải tiếng Anh)" : ""}` : null,
    o.latency_ms != null ? `${(o.latency_ms / 1000).toFixed(1)}s` : null,
  ].filter(Boolean).join(" · "));
  if (!o.criteria) {
    return `${cmpCardHead(title, o.status, meta)}
      ${o.error ? `<p class="cmp-card__meta cmp-card__error">${escapeHtml(o.error)}</p>` : ""}`;
  }
  const words = (o.word_results || []).map((w) => {
    const tip = [CMP_WORD[w.status] || w.status, w.recognized ? `nghe: ${w.recognized}` : "",
      w.phoneme_score != null ? `tin cậy: ${w.phoneme_score}` : ""].filter(Boolean).join(" · ");
    return `<span class="cmp-word" data-status="${escapeHtml(w.status)}" title="${escapeHtml(tip)}">${escapeHtml(w.word)}${
      w.status === "substituted" && w.recognized ? `<small>${escapeHtml(w.recognized)}</small>` : ""}</span>`;
  }).join("");
  const a = o.agreement;
  const list = (items) => (items && items.length ? escapeHtml(items.join(", ")) : "—");
  const agree = a
    ? `<ul class="cmp-tips">
        <li>Cùng kết luận ở ${Math.round(a.same_verdict_share * 100)}% số từ</li>
        <li><b>Cả hai đều nghe sai</b> (lỗi gần như chắc chắn): ${list(a.both_wrong)}</li>
        <li>Chỉ wav2vec2 nghe sai: ${list(a.wrong_only_by_wav2vec2)}</li>
        <li>Chỉ Qwen3-ASR nghe sai: ${list(a.wrong_only_by_qwen3_asr)}</li>
      </ul>`
    : "";
  const delta = o.comparison?.delta_vs_reading;
  const source = o.confidence_source === "qwen3_asr_logprobs"
    ? "Phát âm: độ tin cậy từng từ của Qwen3-ASR"
    : "Phát âm: ước lượng forced alignment (Ollama không trả logprobs)";
  return `${cmpCardHead(title, o.status, meta)}
    <p class="cmp-card__summary">“${escapeHtml(o.transcript || "")}”</p>
    <p class="cmp-card__meta">${escapeHtml(source)}${delta != null
      ? ` · lệch so với Reading: ${delta > 0 ? "+" : ""}${escapeHtml(delta.toFixed(1))} điểm` : ""}</p>
    <div class="cmp-words">${words}</div>
    ${agree ? `<p class="cmp-card__meta">Đối chiếu wav2vec2 ↔ Qwen3-ASR:</p>${agree}` : ""}`;
}

/* --------------------------------------------- /evaluate-v2 (Qwen3-ASR) */

const V2_CRITERIA = [
  ["content_accuracy", "Đọc đúng nội dung"],
  ["pronunciation", "Phát âm"],
  ["word_stress", "Trọng âm từ"],
  ["intonation", "Nhịp điệu, ngữ điệu"],
  ["fluency", "Trôi chảy"],
];

const V2_ERROR = {
  omission: "Bỏ sót",
  substitution: "Đọc sai thành",
  mispronunciation: "Phát âm dễ nghe thành",
  word_stress: "Sai trọng âm",
  insertion: "Từ thừa",
  repetition: "Lặp từ",
  filler: "Từ đệm",
};

function v2BarsHtml(criteria) {
  return V2_CRITERIA.map(([key, label]) => {
    const c = criteria?.[key] || {};
    const has = c.score != null;
    const n = has ? Math.min(100, Math.max(0, Number(c.score))) : 0;
    const weight = c.weight ? ` <span class="cmp__w">${Math.round(c.weight * 100)}%</span>` : "";
    return `<div class="aspect">
      <span class="aspect__name">${escapeHtml(label)}${weight}</span>
      <div class="aspect__track"><i class="aspect__fill" style="width:${n}%"${has ? ` data-tone="${aspectTone(n)}"` : ""}></i></div>
      <span class="aspect__num${has ? "" : " aspect__num--na"}" title="${escapeHtml(has ? "" : CMP_STATUS[c.status] || c.status || "")}">${has ? n.toFixed(1) : "—"}</span>
    </div>`;
  }).join("");
}

function renderV2(report, { showFeedback = true } = {}) {
  const card = $("result-v2");
  if (!card) return;
  const overall = report.overall_score;
  const tone = overall == null ? null : aspectTone(Number(overall));
  const asr = report.asr || {};
  const meta = [
    asr.model,
    asr.latency_ms != null ? `ASR ${(asr.latency_ms / 1000).toFixed(1)}s` : null,
    report.alignment?.method ? "căn thời gian: wav2vec2" : "không có căn thời gian",
    report.detected_language ? `ngôn ngữ: ${report.detected_language}` : null,
  ].filter(Boolean).join(" · ");
  const words = (report.word_results || []).map((w) => {
    const p = w.pronunciation;
    const tip = [CMP_WORD[w.status] || w.status, w.heard ? `nghe: ${w.heard}` : "",
      p && p.score != null ? `phát âm: ${p.score}` : "", p && p.rival ? `đối thủ: ${p.rival}` : ""]
      .filter(Boolean).join(" · ");
    const flagged = (w.issues || []).includes("mispronunciation") ? "mispronounced" : w.status;
    return `<span class="cmp-word" data-status="${escapeHtml(flagged)}" title="${escapeHtml(tip)}">${escapeHtml(w.text)}${
      p && p.score != null ? `<small>${escapeHtml(Number(p.score).toFixed(0))}</small>` : ""}</span>`;
  }).join("");
  const errors = (report.errors || []).map((e) => `<li>${escapeHtml(V2_ERROR[e.type] || e.type)}: <b>${escapeHtml(e.word)}</b>${
    e.heard && ["substitution", "mispronunciation"].includes(e.type) ? ` → ${escapeHtml(e.heard)}` : ""}${
    e.expected ? ` (đúng: ${escapeHtml(e.expected)})` : ""}</li>`).join("");
  const tips = (report.improvements || []).map((i) =>
    `<li data-priority="${escapeHtml(i.priority)}">${escapeHtml(i.message)}</li>`).join("");
  const fb = showFeedback ? report.feedback : null;
  let feedback = "";
  if (fb && fb.status === "generated") {
    feedback = `<div class="coach__panel">
      ${fb.assessment ? `<div class="coach__block"><p class="coach__block-label">Nhận xét (${escapeHtml(fb.model || "Ollama")})</p><p class="coach__block-text">${escapeHtml(fb.assessment)}</p></div>` : ""}
      ${fb.suggestion ? `<div class="coach__block"><p class="coach__block-label">Gợi ý luyện tập</p><p class="coach__block-text">${escapeHtml(fb.suggestion)}</p></div>` : ""}
    </div>`;
  } else if (fb) {
    feedback = `<p class="cmp-card__meta cmp-card__error">Nhận xét AI chưa có: ${escapeHtml(fb.error || fb.status)}</p>`;
  }
  const asrError = report.status === "asr_unavailable" && asr.error
    ? `<p class="cmp-card__meta cmp-card__error">${escapeHtml(asr.error)}</p>` : "";

  $("result-v2-body").innerHTML = `
    <div class="v2-top">
      <div class="v2-score"${tone ? ` data-tone="${tone}"` : ""}>
        <span class="v2-score__num">${overall == null ? "—" : escapeHtml(Number(overall).toFixed(1))}</span>
        <span class="v2-score__max">tổng / 100</span>
      </div>
      <div class="v2-summary">
        ${cmpCardHead("en-reading-v2", report.status)}
        <p class="cmp-card__meta">${escapeHtml(meta)}</p>
        ${report.recognized_text != null ? `<p class="cmp-card__summary">Nghe được: “${escapeHtml(report.recognized_text)}”</p>` : ""}
        ${asrError}
      </div>
    </div>
    <div class="aspects__list v2-bars">${v2BarsHtml(report.criteria)}</div>
    ${words ? `<p class="label v2-label">Từng từ (số = điểm phát âm)</p><div class="cmp-words">${words}</div>` : ""}
    ${errors ? `<p class="label v2-label">Lỗi phát hiện</p><ul class="cmp-tips">${errors}</ul>` : ""}
    ${tips ? `<p class="label v2-label">Gợi ý</p><ul class="cmp-tips">${tips}</ul>` : ""}
    ${feedback}`;
  card.hidden = false;
}

/* -------------------------------------------------------------- Coaching */

function resetCoach() {
  el.coachPanel.hidden = true;
  el.coachStatus.hidden = true;
  el.coachStatus.classList.remove("is-error");
  el.coachStatus.textContent = "";
  el.coachAssessmentBlock.hidden = true;
  el.coachAssessment.textContent = "";
  el.coachSuggestionBlock.hidden = true;
  el.coachSuggestion.textContent = "";
  el.coachBtn.disabled = false;
}

/** English: the comment is being written as part of the evaluate request. */
function coachPending() {
  el.coachPanel.hidden = false;
  el.coachAssessmentBlock.hidden = true;
  el.coachSuggestionBlock.hidden = true;
  el.coachStatus.hidden = false;
  el.coachStatus.classList.remove("is-error");
  el.coachStatus.textContent = "Đang phân tích và viết nhận xét…";
  el.coachBtn.disabled = true;
}

/** English: show the `coach` comment (or `coach_error`) the evaluate
 *  response carried. The button can still regenerate it through /coach. */
function showCoachFrom(result) {
  el.coachBtn.disabled = false;
  const coach = result && result.coach;
  if (coach) {
    el.coachPanel.hidden = false;
    el.coachAssessment.textContent = coach.assessment || "";
    el.coachAssessmentBlock.hidden = !coach.assessment;
    el.coachSuggestion.textContent = coach.suggestion || "";
    el.coachSuggestionBlock.hidden = !coach.suggestion;
    el.coachStatus.hidden = true;
  } else if (result && result.coach_error) {
    el.coachPanel.hidden = false;
    el.coachAssessmentBlock.hidden = true;
    el.coachSuggestionBlock.hidden = true;
    el.coachStatus.hidden = false;
    el.coachStatus.textContent = result.coach_error;
    el.coachStatus.classList.add("is-error");
  } else {
    resetCoach();
  }
}

/** Ask the locally-run Ollama model (see app/services/coaching.py) for a
 *  natural-language comment on the take just scored, in `el.coachLang`'s
 *  language. Fired automatically alongside evaluate()/comparePitch() for
 *  every take on the non-English pages (see the two Promise.all([...]) call
 *  sites; English gets `coach` inside the evaluate response), and also
 *  re-triggered by the coach button (e.g. to regenerate after switching
 *  language) -- the `disabled` guard below keeps a manual click a no-op
 *  while a request from either source is already in flight. */
async function requestCoach() {
  if (!lastWav || el.coachBtn.disabled) return;

  el.coachPanel.hidden = false;
  el.coachAssessmentBlock.hidden = true;
  el.coachSuggestionBlock.hidden = true;
  el.coachStatus.hidden = false;
  el.coachStatus.classList.remove("is-error");
  el.coachStatus.textContent = "Đang phân tích và viết nhận xét…";
  el.coachBtn.disabled = true;

  const formData = new FormData();
  formData.append("text", target.text);
  formData.append("audio", new File([lastWav], "recording.wav", { type: "audio/wav" }));
  formData.append("lang", el.coachLang.value);

  try {
    const response = await fetch(COACH_API_URL, { method: "POST", body: formData });
    if (!response.ok) {
      const detail = await response
        .json()
        .then((body) => body.detail)
        .catch(() => undefined);
      throw new Error(detail || `Yêu cầu thất bại (HTTP ${response.status})`);
    }
    const result = await response.json();
    el.coachAssessment.textContent = result.assessment;
    el.coachAssessmentBlock.hidden = !result.assessment;
    el.coachSuggestion.textContent = result.suggestion;
    el.coachSuggestionBlock.hidden = !result.suggestion;
    el.coachStatus.hidden = true;
  } catch (error) {
    el.coachStatus.textContent =
      error instanceof Error ? error.message : "Không tạo được nhận xét. Hãy thử lại.";
    el.coachStatus.classList.add("is-error");
  } finally {
    el.coachBtn.disabled = false;
  }
}

/* ---------------------------------------------------------- Pitch accent */

/** Text the reference pattern on screen belongs to, so re-opening the
 *  panel for the same sentence does not re-fetch it. Reset by resetPitch(). */
let pitchLoadedFor = null;
/** Reference H/L pattern for `pitchLoadedFor`, from /pitch-accent. */
let referencePattern = null;
/** All accepted pitch patterns (primary + alternatives) for `pitchLoadedFor`. */
let referencePatterns = null;
/** Measured F0 per target mora from /evaluate's measured_pitch.
 *  null = no take scored yet; [] = scored but no usable F0. */
let learnerPitch = null;

function resetPitch() {
  pitchLoadedFor = null;
  referencePattern = null;
  referencePatterns = null;
  learnerPitch = null;
  englishReference = null;
  englishTake = null;
  el.pitch.hidden = true;
  el.pitchBtn?.setAttribute("aria-expanded", "false");
  el.pitchChart.hidden = true;
  el.pitchChart.innerHTML = "";
}

function setPitchStatus(message, isError = false) {
  el.pitchStatus.textContent = message;
  el.pitchStatus.classList.toggle("is-error", isError);
  el.pitchStatus.hidden = false;
  el.pitchChart.hidden = true;
}

/** Mora-center dots for the OJAD step (H top rail, L bottom rail). */
function patternPoints(pattern, width, yHigh, yLow) {
  return pattern.map((mora, index) => ({
    x: ((index + 0.5) / pattern.length) * width,
    y: mora.pitch === "H" ? yHigh : yLow,
  }));
}

/** OJAD-style step: flat on each mora, vertical only when H/L changes,
 *  gap at accent-phrase boundaries (jp-pitch-accent-analyzer `ax.step`). */
function patternStepPath(pattern, width, yHigh, yLow) {
  const n = pattern.length;
  if (!n) return "";
  const col = width / n;
  let d = "";
  let prevY = null;
  let prevPhrase = null;
  for (let i = 0; i < n; i += 1) {
    const y = pattern[i].pitch === "H" ? yHigh : yLow;
    const left = i * col;
    const right = (i + 1) * col;
    const phrase = pattern[i].phrase;
    const newPhrase = prevPhrase !== null && phrase !== prevPhrase;
    if (prevY === null || newPhrase) {
      d += `${d ? " " : ""}M ${left.toFixed(1)},${y.toFixed(1)} H ${right.toFixed(1)}`;
    } else if (y === prevY) {
      d += ` H ${right.toFixed(1)}`;
    } else {
      d += ` H ${left.toFixed(1)} V ${y.toFixed(1)} H ${right.toFixed(1)}`;
    }
    prevY = y;
    prevPhrase = phrase;
  }
  return d;
}

function patternPolyline(points) {
  return points.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ");
}

function patternDots(pattern, points, variant) {
  return points
    .map((p, index) => {
      const classes = ["pitch__dot"];
      if (variant === "you") classes.push("pitch__dot--you");
      if (pattern[index].pitch === "H") classes.push("pitch__dot--h");
      return `<circle class="${classes.join(" ")}" cx="${p.x.toFixed(1)}" cy="${p.y.toFixed(1)}" r="3.5" />`;
    })
    .join("");
}

function patternLabels(pattern, variant) {
  const cls = variant === "you" ? "pitch__label pitch__label--you" : "pitch__label";
  const labels = pattern.map((mora) => `<span class="${cls}">${mora.mora}</span>`).join("");
  const rowCls = variant === "you" ? "pitch__labels pitch__labels--you" : "pitch__labels";
  return `<div class="${rowCls}" style="grid-template-columns: repeat(${pattern.length}, 1fr)">${labels}</div>`;
}

function semitoneToY(semitone, yHigh, yLow) {
  const mid = (yHigh + yLow) / 2;
  const half = (yLow - yHigh) / 2;
  const t = Math.max(-6, Math.min(6, Number(semitone))) / 6;
  return mid - t * half;
}

function f0Segments(points) {
  const segments = [];
  let current = [];
  for (const point of points) {
    if (point == null) {
      if (current.length) segments.push(current);
      current = [];
    } else {
      current.push(point);
    }
  }
  if (current.length) segments.push(current);
  return segments;
}

function phraseMeans(learner, reference) {
  const sums = {};
  const counts = {};
  for (let i = 0; i < learner.length; i += 1) {
    const point = learner[i];
    if (!point || !point.voiced || point.semitone == null) continue;
    const phrase = (reference[i] && reference[i].phrase) || 0;
    sums[phrase] = (sums[phrase] || 0) + Number(point.semitone);
    counts[phrase] = (counts[phrase] || 0) + 1;
  }
  const means = {};
  Object.keys(sums).forEach((phrase) => {
    means[phrase] = counts[phrase] >= 2 ? sums[phrase] / counts[phrase] : 0;
  });
  return means;
}

function directionOk(point, index, reference, means) {
  if (!point || !point.voiced || point.semitone == null) return null;
  const phrase = (reference[index] && reference[index].phrase) || 0;
  const mean = means[phrase];
  if (mean == null) return null;
  if (point.expected === "H") return point.semitone > mean;
  if (point.expected === "L") return point.semitone < mean;
  return null;
}

/** Reference H/L (dictionary) + learner F0 measured from the WAV. */
function renderPitchChart() {
  if (TARGET_LANG === "en" && (englishTake || englishReference)) {
    renderEnglishIntonation();
    return;
  }
  if (!referencePattern || referencePattern.length === 0) {
    setPitchStatus("Không phân tích được cao độ cho câu này.", true);
    return;
  }

  const learnerReady = learnerPitch !== null;
  const voicedCount = (learnerPitch || []).filter((p) => p && p.voiced && p.semitone != null).length;
  const hasLearner = learnerReady && voicedCount > 0;
  el.pitchTitle.textContent = hasLearner ? "Cao độ: mẫu và F0 của bạn" : "Cao độ mẫu";
  el.pitchLegend.innerHTML = `
    <span class="pitch__legend-item"><i class="pitch__swatch pitch__swatch--step"></i>Mẫu bậc</span>
    <span class="pitch__legend-item"><i class="pitch__swatch pitch__swatch--ref"></i>Mẫu (H/L)</span>
    ${hasLearner ? '<span class="pitch__legend-item"><i class="pitch__swatch pitch__swatch--you"></i>Bạn (F0 đo)</span>' : ""}
  `;

  const n = referencePattern.length;
  const columnWidth = 40;
  const width = n * columnWidth;
  const yHigh = 22;
  const yLow = 72;
  const yMid = (yHigh + yLow) / 2;

  const refPoints = patternPoints(referencePattern, width, yHigh, yLow);
  const refDots = patternDots(referencePattern, refPoints, "ref");

  let learnerSvg = "";
  let hint = "";
  if (hasLearner) {
    const localMeans = phraseMeans(learnerPitch, referencePattern);
    const f0Points = referencePattern.map((mora, index) => {
      const measured = learnerPitch[index];
      if (!measured || !measured.voiced || measured.semitone == null) return null;
      return {
        x: ((index + 0.5) / n) * width,
        y: semitoneToY(measured.semitone, yHigh, yLow),
        ok: directionOk(measured, index, referencePattern, localMeans),
        mora: mora.mora,
      };
    });
    const lines = f0Segments(f0Points)
      .filter((seg) => seg.length >= 1)
      .map((seg) =>
        seg.length === 1
          ? ""
          : `<polyline class="pitch__line pitch__line--you" points="${patternPolyline(seg)}" />`,
      )
      .join("");
    const dots = f0Points
      .filter((p) => p)
      .map((p) => {
        const tone = p.ok === false ? "pitch__dot--miss" : p.ok ? "pitch__dot--hit" : "pitch__dot--you";
        return `<circle class="pitch__dot ${tone}" cx="${p.x.toFixed(1)}" cy="${p.y.toFixed(1)}" r="3.5" />`;
      })
      .join("");
    learnerSvg = `${lines}${dots}`;
    let methodText = "Nét xanh bậc vuông là mẫu H/L (OpenJTalk).";
    if (TARGET_LANG === "ko") {
      methodText = "Nét xanh bậc vuông là cao độ mẫu Accentual Phrase (H/L).";
    } else if (TARGET_LANG === "zh") {
      methodText = "Nét xanh bậc vuông là cao độ mẫu thanh điệu.";
    } else if (TARGET_LANG === "en") {
      methodText = "Nét xanh bậc vuông là trọng âm từ mẫu (H/L).";
    }
    hint = `<p class="pitch__hint">${methodText} Nét liền chấm là cùng mẫu nối âm. Nét đứt là F0 đo từ bản ghi.</p>`;
  } else if (learnerReady) {
    hint = `<p class="pitch__hint">Không đo được F0 có thanh trong bản ghi này.</p>`;
  } else {
    hint = `<p class="pitch__hint">Nét xanh bậc vuông là cao độ mẫu (H cao, L thấp, đổi bậc theo ngữ điệu).</p>`;
  }

  let variantsHtml = "";
  if (referencePatterns && referencePatterns.length > 1) {
    const alts = referencePatterns.slice(1);
    const altItems = alts
      .map((pat, idx) => {
        const vPoints = patternPoints(pat, width, yHigh, yLow);
        const vDots = patternDots(pat, vPoints, "ref");
        return `
          <div class="pitch__variant" style="padding: 8px 10px; background: rgba(0,0,0,0.02); border: 1px solid rgba(0,0,0,0.07); border-radius: 8px; margin-top: 8px;">
            <div style="font-size: 0.78rem; font-weight: 600; color: var(--text-muted, #64748b); margin-bottom: 4px;">Cách đọc ${idx + 2}:</div>
            <div class="pitch__scroll">
              <div class="pitch__plot" style="width: max(100%, ${width}px)">
                <svg class="pitch__svg" viewBox="0 0 ${width} 94" preserveAspectRatio="none">
                  <line class="pitch__guide" x1="0" y1="${yLow}" x2="${width}" y2="${yLow}" />
                  <line class="pitch__guide" x1="0" y1="${yMid}" x2="${width}" y2="${yMid}" />
                  <path class="pitch__line pitch__line--step" d="${patternStepPath(pat, width, yHigh, yLow)}" />
                  <polyline class="pitch__line" points="${patternPolyline(vPoints)}" />
                  ${vDots}
                </svg>
                ${patternLabels(pat, "ref")}
              </div>
            </div>
          </div>
        `;
      })
      .join("");

    variantsHtml = `
      <details class="pitch__accordion" style="margin-top: 12px; border-top: 1px dashed rgba(0,0,0,0.15); padding-top: 10px;">
        <summary style="cursor: pointer; font-size: 0.82rem; font-weight: 600; color: var(--text-muted, #64748b); user-select: none;">
          ▶ Các cách nhấn trọng âm chuẩn khác (${alts.length} biến thể)
        </summary>
        <div class="pitch__variants-list" style="margin-top: 6px;">
          ${altItems}
        </div>
      </details>
    `;
  }

  el.pitchChart.innerHTML = `
    <div class="pitch__scroll">
      <div class="pitch__plot" style="width: max(100%, ${width}px)">
        <svg class="pitch__svg" viewBox="0 0 ${width} 94" preserveAspectRatio="none">
          <line class="pitch__guide" x1="0" y1="${yLow}" x2="${width}" y2="${yLow}" />
          <line class="pitch__guide" x1="0" y1="${yMid}" x2="${width}" y2="${yMid}" />
          <path class="pitch__line pitch__line--step" d="${patternStepPath(referencePattern, width, yHigh, yLow)}" />
          <polyline class="pitch__line" points="${patternPolyline(refPoints)}" />
          ${refDots}
          ${learnerSvg}
        </svg>
        ${patternLabels(referencePattern, "ref")}
      </div>
    </div>
    ${hint}
    ${variantsHtml}
  `;
  el.pitchChart.hidden = false;
  el.pitchStatus.hidden = true;
}

async function loadPitchAccent(text) {
  setPitchStatus("\u0110ang ph\u00e2n t\u00edch cao \u0111\u1ed9\u2026");

  try {
    const response = await fetch(`${PITCH_API_URL}?text=${encodeURIComponent(text)}`);
    if (!response.ok) {
      const detail = await response
        .json()
        .then((body) => body.detail)
        .catch(() => undefined);
      throw new Error(detail || `Y\u00eau c\u1ea7u th\u1ea5t b\u1ea1i (HTTP ${response.status})`);
    }
    const result = await response.json();
    if (text !== target.text) return;
    referencePattern = result.pattern;
    referencePatterns = result.patterns || (result.pattern ? [result.pattern] : []);
    englishReference = result.intonation || null;
    pitchLoadedFor = text;
    if (document.body.dataset.targetLang === "zh") renderChinesePitch();
    else renderPitchChart();
  } catch (error) {
    if (text !== target.text) return;
    referencePattern = null;
    referencePatterns = null;
    pitchLoadedFor = null;
    setPitchStatus(
      error instanceof Error ? error.message : "Kh\u00f4ng l\u1ea5y \u0111\u01b0\u1ee3c cao \u0111\u1ed9 m\u1eabu.",
      true,
    );
  }
}

/** Show the pitch panel and load the reference H/L pattern. The learner
 *  F0 curve comes from /evaluate's measured_pitch after scoring. */
function comparePitch() {
  el.pitch.hidden = false;
  el.pitchBtn?.setAttribute("aria-expanded", "true");
  if (pitchLoadedFor !== target.text) return loadPitchAccent(target.text);
  if (document.body.dataset.targetLang === "zh") renderChinesePitch();
  else renderPitchChart();
  return Promise.resolve();
}

el.coachBtn.addEventListener("click", requestCoach);

/* -------------------------------------------------------- Text reading guide */

let textGuideLoadedFor = null;
let textGuideLoadedLang = null;
/** Bumped per request so a slow answer for an old search is dropped. */
let textGuideRequestId = 0;

/* AI guides are a few KB each — too big for a cookie (≈4 KB, and sent with
 * every request) — so they are cached in localStorage, per practice
 * language, keyed by sentence + guide language. Least-recently-used first
 * out. Storage can be missing or full (private mode, quota), so every
 * access is guarded and the page still works without it. */
// v4: entries carry the AI `meaning`. Older keys (v1 guide-only, v2/v3 saved
// while guides could be cut at the first quote / at 900 tokens) are dropped.
const GUIDE_CACHE_KEY = `pv_guide_cache_v4_${(TTS_LANG.split("-")[0] || "ja").toLowerCase()}`;
try {
  // Drop caches written by older versions of this page.
  for (const key of Object.keys(localStorage)) {
    if (key.startsWith("pv_guide_cache_") && !key.startsWith("pv_guide_cache_v4_")) localStorage.removeItem(key);
  }
} catch { /* storage unavailable */ }
const GUIDE_CACHE_MAX = 50;

function guideCacheId(text, lang) {
  return `${lang}\u0000${text}`;
}

function readGuideCache() {
  try {
    const data = JSON.parse(localStorage.getItem(GUIDE_CACHE_KEY) || "[]");
    return Array.isArray(data) ? data.filter((item) => item && item.id && typeof item.guide === "string") : [];
  } catch {
    return [];
  }
}

function writeGuideCache(entries) {
  let list = entries.slice(0, GUIDE_CACHE_MAX);
  while (list.length) {
    try {
      localStorage.setItem(GUIDE_CACHE_KEY, JSON.stringify(list));
      return;
    } catch {
      // Quota exceeded: drop the older half and retry.
      list = list.slice(0, Math.floor(list.length / 2));
    }
  }
  try { localStorage.removeItem(GUIDE_CACHE_KEY); } catch { /* storage unavailable */ }
}

function getCachedGuide(text, lang) {
  const id = guideCacheId(text, lang);
  const entries = readGuideCache();
  const hit = entries.find((item) => item.id === id);
  if (hit) writeGuideCache([hit, ...entries.filter((item) => item.id !== id)]);
  return hit || null;
}

function saveCachedGuide(text, lang, guide, meaning) {
  const id = guideCacheId(text, lang);
  writeGuideCache([{ id, guide, meaning, savedAt: Date.now() }, ...readGuideCache().filter((item) => item.id !== id)]);
}

/** Meaning of the searched text, shown under it. The AI one (in the guide
 *  language) replaces a preset sentence's built-in meaning. */
function showMeaning(meaning) {
  if (!meaning) return;
  el.targetMeaning.textContent = meaning;
  el.targetMeaning.title = "Nghĩa do AI dịch";
  el.targetMeaning.hidden = false;
}

function showGuide(guide, savedAt = null, meaning = "") {
  showMeaning(meaning);
  el.guideContent.textContent = guide;
  el.guideContent.hidden = false;
  el.guideStatus.hidden = true;
  if (el.guideRefresh) {
    el.guideRefresh.hidden = false;
    el.guideRefresh.disabled = false;
    el.guideRefresh.title = savedAt
      ? `Đã lưu lúc ${new Date(savedAt).toLocaleString("vi-VN")} — bấm để AI tạo lại`
      : "Bấm để AI tạo lại hướng dẫn";
  }
}

function resetTextGuide() {
  textGuideRequestId += 1;
  textGuideLoadedFor = null;
  textGuideLoadedLang = null;
  if (el.guidePanel) el.guidePanel.hidden = true;
  if (el.guideBtn) el.guideBtn.setAttribute("aria-expanded", "false");
  if (el.guideStatus) {
    el.guideStatus.hidden = true;
    el.guideStatus.textContent = "";
  }
  if (el.guideContent) {
    el.guideContent.hidden = true;
    el.guideContent.textContent = "";
  }
  if (el.guideRefresh) el.guideRefresh.hidden = true;
}

/** Show the AI guide for the current sentence: from the local cache when
 *  saved earlier, else from the API (then saved). `force` skips the cache. */
async function requestTextGuide({ force = false } = {}) {
  if (!el.guidePanel || !target.text) return;

  const selectedLang = el.guideLang ? el.guideLang.value : "vi";
  const requestedText = target.text;
  const requestId = ++textGuideRequestId;

  el.guidePanel.hidden = false;
  el.guideBtn?.setAttribute("aria-expanded", "true");
  el.guideStatus.classList.remove("is-error");

  if (!force) {
    const cached = getCachedGuide(requestedText, selectedLang);
    if (cached) {
      showGuide(cached.guide, cached.savedAt, cached.meaning || "");
      textGuideLoadedFor = requestedText;
      textGuideLoadedLang = selectedLang;
      return;
    }
  }

  el.guideContent.hidden = true;
  el.guideStatus.hidden = false;
  el.guideStatus.textContent = "Đang tạo hướng dẫn đọc từ AI…";
  if (el.guideRefresh) el.guideRefresh.disabled = true;

  try {
    const response = await fetch(`${TEXT_GUIDE_API_URL}?text=${encodeURIComponent(requestedText)}&lang=${encodeURIComponent(selectedLang)}`);
    if (requestId !== textGuideRequestId) return;
    if (!response.ok) {
      const detail = await response
        .json()
        .then((body) => body.detail)
        .catch(() => undefined);
      throw new Error(detail || `Yêu cầu thất bại (HTTP ${response.status})`);
    }
    const result = await response.json();
    if (requestId !== textGuideRequestId) return;
    const meaning = (result.meaning || "").trim();
    saveCachedGuide(requestedText, selectedLang, result.guide, meaning);
    showGuide(result.guide, Date.now(), meaning);
    textGuideLoadedFor = requestedText;
    textGuideLoadedLang = selectedLang;
  } catch (error) {
    if (requestId !== textGuideRequestId) return;
    el.guideStatus.textContent =
      error instanceof Error ? error.message : "Không tạo được hướng dẫn đọc. Hãy thử lại.";
    el.guideStatus.classList.add("is-error");
    if (el.guideRefresh) {
      el.guideRefresh.hidden = false;
      el.guideRefresh.disabled = false;
    }
  }
}

el.guideRefresh?.addEventListener("click", () => requestTextGuide({ force: true }));

if (el.guideBtn) {
  el.guideBtn.addEventListener("click", () => {
    const opening = el.guidePanel.hidden;
    if (opening) {
      if (textGuideLoadedFor !== target.text || textGuideLoadedLang !== (el.guideLang?.value || "vi")) {
        requestTextGuide();
      } else {
        el.guidePanel.hidden = false;
        el.guideBtn.setAttribute("aria-expanded", "true");
      }
    } else {
      el.guidePanel.hidden = true;
      el.guideBtn.setAttribute("aria-expanded", "false");
    }
  });
}

if (el.guideLang) {
  el.guideLang.addEventListener("change", () => {
    if (!el.guidePanel.hidden) {
      requestTextGuide();
    }
  });
}

el.pitchBtn?.addEventListener("click", () => {
  const opening = el.pitch.hidden;
  el.pitch.hidden = !opening;
  el.pitchBtn.setAttribute("aria-expanded", String(opening));
  if (opening && pitchLoadedFor !== target.text) loadPitchAccent(target.text);
  else if (opening && document.body.dataset.targetLang === "zh") renderChinesePitch();
  else if (opening) renderPitchChart();
});

function renderChinesePitch() {
  const points = (learnerPitch || []).filter((p) => p.voiced && p.semitone != null);
  const pattern = referencePattern || [];
  if (!pattern.length) {
    setPitchStatus("Không lấy được cao độ mẫu tiếng Trung.", true);
    return;
  }
  const width = Math.max(320, pattern.length * 64);
  const reference = pattern.map((item, index) => item.contour.map((level, step) => {
    const x = (index + (step + 0.15) / (item.contour.length - 1 + 0.3)) * width / pattern.length;
    return `${x.toFixed(1)},${(82 - level * 13).toFixed(1)}`;
  }).join(" "));
  const values = points.map((p) => Number(p.semitone));
  const low = values.length ? Math.min(...values) : 0;
  const high = values.length ? Math.max(...values) : 1;
  const range = Math.max(1, high - low);
  const all = learnerPitch || [];
  const coords = pattern.map((_, i) => all[i] && all[i].voiced && all[i].semitone != null
    ? `${((i + 0.5) * width / pattern.length).toFixed(1)},${(76 - 52 * (Number(all[i].semitone) - low) / range).toFixed(1)}`
    : null);
  const segments = [];
  let current = [];
  for (const point of coords) {
    if (point) current.push(point);
    else if (current.length) { segments.push(current); current = []; }
  }
  if (current.length) segments.push(current);
  el.pitchTitle.textContent = points.length ? "Cao độ: mẫu và F0 của bạn" : "Cao độ mẫu tiếng Trung";
  el.pitchLegend.innerHTML = `<span class="pitch__legend-item"><i class="pitch__swatch pitch__swatch--ref"></i>Mẫu thanh điệu</span>${points.length ? '<span class="pitch__legend-item"><i class="pitch__swatch pitch__swatch--you"></i>Bạn (F0 đo)</span>' : ""}`;
  el.pitchChart.innerHTML = `<div class="pitch__scroll"><div class="pitch__plot" style="width:max(100%,${width}px)"><svg class="pitch__svg" viewBox="0 0 ${width} 94" preserveAspectRatio="none">${reference.map((line) => `<polyline class="pitch__line" points="${line}" />`).join("")}${segments.map((s) => `<polyline class="pitch__line pitch__line--you" points="${s.join(" ")}" />`).join("")}</svg><div class="pitch__labels" style="grid-template-columns:repeat(${pattern.length},1fr)">${pattern.map((p) => `<span class="pitch__label">${p.mora}<sup>${p.surface_tone}</sup></span>`).join("")}</div></div></div><p class="pitch__hint">Đường mẫu dùng thang Chao 1–5; F0 xuất hiện sau khi ghi âm.</p>`;
  el.pitchChart.hidden = false;
  el.pitchStatus.hidden = true;
}

/* --------------------------------------------------- English intonation */

/* English is a stress language: the chart is one column per syllable, a
 * reference contour that peaks on accented syllables (sentence stress, ◉ =
 * nucleus) and falls/rises at the end, and the learner's F0 per syllable.
 * Data comes from `intonation` on /pitch-accent (reference) and /evaluate
 * (reference + learner + checks); see app/schemas/english_intonation.py. */

/** /pitch-accent `intonation` for the current sentence. */
let englishReference = null;
/** /evaluate `intonation` for the last take of the current sentence. */
let englishTake = null;
/** Chart toggle: "both" | "ref" | "you". Kept across takes. */
let englishView = "both";

const EN_LEFT = 44;
const EN_TOP = 30;
const EN_BOTTOM = 198;
const EN_HEIGHT = 214;
const EN_RANGE_ST = 7;

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function enY(semitone) {
  const t = Math.max(-EN_RANGE_ST, Math.min(EN_RANGE_ST, Number(semitone))) / EN_RANGE_ST;
  return (EN_TOP + EN_BOTTOM) / 2 - (t * (EN_BOTTOM - EN_TOP)) / 2;
}

/** Catmull-Rom through the points as cubic Béziers ("M ..." path data). */
function smoothPath(points, move = true) {
  if (!points.length) return "";
  let d = `${move ? "M" : "L"} ${points[0].x.toFixed(1)},${points[0].y.toFixed(1)}`;
  for (let i = 0; i < points.length - 1; i += 1) {
    const p0 = points[i - 1] || points[i];
    const p1 = points[i];
    const p2 = points[i + 1];
    const p3 = points[i + 2] || p2;
    const c1x = p1.x + (p2.x - p0.x) / 6;
    const c1y = p1.y + (p2.y - p0.y) / 6;
    const c2x = p2.x - (p3.x - p1.x) / 6;
    const c2y = p2.y - (p3.y - p1.y) / 6;
    d += ` C ${c1x.toFixed(1)},${c1y.toFixed(1)} ${c2x.toFixed(1)},${c2y.toFixed(1)} ${p2.x.toFixed(1)},${p2.y.toFixed(1)}`;
  }
  return d;
}

const isAccent = (role) => role === "stress" || role === "nucleus";

/** One column per syllable, a gap column at each phrase break. */
function englishLayout(data, available) {
  const refs = new Map(data.reference.map((r) => [`${r.word}:${r.syllable}`, r]));
  const count = data.words.reduce((n, w) => n + w.syllables.length, 0) || 1;
  const gap = 36;
  const breaks = Math.max(0, data.phrase_ends.length - 1);
  // Wide enough for the longest syllable label (capitals, bold) and for a
  // one-syllable word's IPA; otherwise fill the panel. Too many columns for
  // the screen scroll sideways instead of overlapping.
  let widest = 0;
  for (const word of data.words) {
    for (const syl of word.syllables) widest = Math.max(widest, syl.text.length * 9 + 10);
    if (word.syllables.length === 1 && word.ipa) widest = Math.max(widest, (word.ipa.length + 2) * 7);
  }
  const col = Math.max(44, widest, (available - EN_LEFT - 12 - breaks * gap) / count);
  const sylls = [];
  const gaps = [];
  const columns = [`${EN_LEFT}px`];
  let x = EN_LEFT;
  let phrase = data.words.length ? data.words[0].phrase : 0;
  for (const word of data.words) {
    if (word.phrase !== phrase) {
      gaps.push({ x: x + gap / 2, afterWord: word.index - 1, column: columns.length + 1 });
      columns.push(`${gap}px`);
      x += gap;
      phrase = word.phrase;
    }
    word.syllables.forEach((syl, i) => {
      const ref = refs.get(`${word.index}:${i}`);
      sylls.push({
        word: word.index,
        syllable: i,
        phrase: word.phrase,
        text: syl.text,
        role: ref ? ref.role : "weak",
        ref,
        x0: x,
        x1: x + col,
        xc: x + col / 2,
        column: columns.length + 1,
      });
      columns.push(`${col}px`);
      x += col;
    });
  }
  return { sylls, gaps, width: x + 12, col, columns: columns.join(" ") };
}

function syllableLabel(syl) {
  return isAccent(syl.role) ? syl.text.toUpperCase() : syl.text;
}

/** "my NAME is AN·na ‖ I WORK as a TEA·cher ↘" */
function englishSentenceHtml(data, layout) {
  const byWord = new Map();
  for (const syl of layout.sylls) {
    if (!byWord.has(syl.word)) byWord.set(syl.word, []);
    byWord.get(syl.word).push(syl);
  }
  const parts = [];
  data.words.forEach((word, i) => {
    const sylls = byWord.get(word.index) || [];
    const html = sylls
      .map((syl) => {
        const text = escapeHtml(syllableLabel(syl));
        if (syl.role === "nucleus") return `<span class="eni-sent__acc eni-sent__nuc">${text}</span>`;
        if (syl.role === "stress") return `<span class="eni-sent__acc">${text}</span>`;
        return text;
      })
      .join('<span class="eni-sent__dot">·</span>');
    parts.push(`<span class="eni-sent__word">${html}</span>`);
    const next = data.words[i + 1];
    if (!next || next.phrase !== word.phrase) {
      const ending = data.phrase_ends[word.phrase];
      if (ending && ending !== "continue") parts.push(`<span class="eni-sent__arrow">${ENDING_ARROW[ending]}</span>`);
      if (next) parts.push('<span class="eni-sent__break">‖</span>');
    }
  });
  return `<p class="eni-sent">${parts.join(" ")}</p>`;
}

const ENDING_ARROW = { fall: "↘", rise: "↗", either: "↘↗" };
const ENDING_VERB = { fall: "Xuống giọng", rise: "Lên giọng", either: "Lên/xuống giọng" };

/** Whole word with its stressed syllable in capitals: "AN·na", "NAME". */
function stressedWordLabel(word) {
  return word.syllables
    .map((s, i) => (i === word.stress_index ? s.text.toUpperCase() : s.text.toLowerCase()))
    .join("·");
}

function lastWordOfPhrase(data, phrase) {
  const own = data.words.filter((w) => w.phrase === phrase);
  return own[own.length - 1];
}

/** Every sentence end (older payloads only have `final`). */
function englishEnds(take) {
  if (take.ends && take.ends.length) return take.ends;
  return take.final ? [take.final] : [];
}

function listWords(words, max = 3) {
  const shown = words.slice(0, max).join(" · ");
  return words.length > max ? `${shown} +${words.length - max}` : shown;
}

function englishChipsHtml(take) {
  const chips = [];
  const missing = new Set(take.missing || []);
  if (missing.size) {
    chips.push(["bad", `Bỏ sót: ${escapeHtml(listWords([...missing].map((i) => take.words[i]?.text || "")))}`]);
  }
  if (take.peaks.length) {
    // Only peaks that could be measured count either way.
    const known = take.peaks.filter((c) => c.ok !== null);
    const hit = known.filter((c) => c.ok).length;
    const unknown = take.peaks.length - known.length;
    if (!known.length) {
      chips.push(["na", "Đỉnh nhấn: không đo được"]);
    } else {
      const extra = unknown ? ` (${unknown} không đo được)` : "";
      chips.push([hit === known.length ? "ok" : "bad", `Đỉnh nhấn khớp: ${hit}/${known.length}${extra}`]);
    }
  }
  for (const b of take.breaks) {
    const word = escapeHtml(take.words[b.after_word]?.text || "");
    const state = b.ok === null ? "không đo được" : b.ok ? "rõ" : "chưa rõ";
    chips.push([b.ok === null ? "na" : b.ok ? "ok" : "bad", `Ngắt sau “${word}”: ${state}`]);
  }
  const ends = englishEnds(take);
  for (const end of ends) {
    const verb = ENDING_VERB[end.expected] || ENDING_VERB.fall;
    const where = ends.length > 1
      ? `sau “${escapeHtml(lastWordOfPhrase(take, end.phrase)?.text || "")}”`
      : "cuối câu";
    const state = end.ok === null ? "không đo được" : end.ok ? "đạt" : "chưa đạt";
    chips.push([end.ok === null ? "na" : end.ok ? "ok" : "bad", `${verb} ${where}: ${state}`]);
  }
  return `<div class="eni-chips">${chips
    .map(([tone, text]) => `<span class="eni-chip eni-chip--${tone}"><i></i>${text}</span>`)
    .join("")}</div>`;
}

/** Up to three short, concrete tips from the checks (not an LLM). */
function englishCardsHtml(take) {
  const cards = [];
  const missing = new Set(take.missing || []);
  if (missing.size) {
    cards.push([
      "Bỏ sót từ",
      `Bản ghi không có ${listWords([...missing].map((i) => `“${take.words[i]?.text || ""}”`), 4)}. ` +
        "Đọc đủ cả câu để chấm được ngữ điệu.",
    ]);
  }
  const missed = take.peaks.filter((c) => c.ok === false && !missing.has(c.word));
  const plainMissed = missed.filter((c) => !take.words[c.word]?.nucleus);
  if (plainMissed.length) {
    cards.push([
      listWords(plainMissed.map((c) => stressedWordLabel(take.words[c.word]))),
      "Giọng gần như phẳng ở đây. Nâng cao và kéo dài âm tiết nhấn, đọc lướt các từ nhỏ xung quanh.",
    ]);
  }
  for (const end of englishEnds(take)) {
    const nucleus = take.words.find((w) => w.phrase === end.phrase && w.nucleus)
      || lastWordOfPhrase(take, end.phrase);
    if (!nucleus || missing.has(nucleus.index)) continue;
    const label = stressedWordLabel(nucleus);
    const rise = end.expected === "rise";
    const nucleusMissed = missed.some((c) => c.word === nucleus.index);
    if (end.ok === false) {
      cards.push([
        label,
        rise
          ? "Câu hỏi Yes/No: hạ giọng ở từ nhấn chính rồi kéo giọng lên rõ ở cuối câu."
          : end.expected === "either"
            ? "Câu hỏi đuôi: kết thúc rõ ràng, hoặc hạ hẳn (xác nhận) hoặc lên hẳn (hỏi thật)."
            : "Sau từ nhấn chính, hạ giọng xuống hẳn ở cuối câu thay vì giữ ngang hoặc đi lên.",
      ]);
    } else if (nucleusMissed && end.ok) {
      const tail = nucleus.syllables.slice(nucleus.stress_index + 1).map((s) => s.text).join("·");
      const stressed = nucleus.syllables[nucleus.stress_index]?.text.toUpperCase() || nucleus.text;
      cards.push([
        label,
        `Đã ${rise ? "lên" : "xuống"} giọng ở cuối, nhưng cần một đỉnh rõ ở ${stressed}` +
          (tail ? ` trước khi ${rise ? "lên" : "rơi xuống"} “${tail}”.` : "."),
      ]);
    } else if (nucleusMissed) {
      cards.push([label, "Nhấn rõ từ nhấn chính: giọng cao và dài hơn các từ xung quanh."]);
    }
  }
  for (const b of take.breaks.filter((c) => c.ok === false)) {
    const word = take.words[b.after_word];
    if (!word) continue;
    const lastSyl = word.syllables[word.syllables.length - 1].text;
    cards.push([
      `Sau “${word.text}”`,
      `Dừng ngắn, hạ giọng ở “${lastSyl}”, rồi bắt đầu cụm sau ở mức trung.`,
    ]);
  }
  for (const ws of take.word_stress.filter((c) => c.ok === false && c.heard !== null)) {
    const word = take.words[ws.word];
    if (!word) continue;
    const label = (idx) =>
      word.syllables.map((s, i) => (i === idx ? s.text.toUpperCase() : s.text.toLowerCase())).join("·");
    cards.push([
      `Trọng âm “${word.text}”`,
      `Nhấn ${label(ws.expected)}, bạn đang nhấn ${label(ws.heard)}.`,
    ]);
  }
  if (!cards.length) {
    cards.push(["Rất tốt", "Đỉnh nhấn, chỗ ngắt và cuối câu đều đúng mẫu."]);
  }
  return `<div class="eni-cards">${cards
    .slice(0, 3)
    .map(([title, text]) => `<div class="eni-card"><p class="eni-card__title">${escapeHtml(title)}</p><p class="eni-card__text">${escapeHtml(text)}</p></div>`)
    .join("")}</div>`;
}

function englishChartSvg(take, layout) {
  const { sylls, gaps, width, col } = layout;
  const yHigh = enY(4);
  const yMid = enY(0);
  const yLow = enY(-4);
  const svg = [];

  // Accented columns, guides, phrase breaks.
  for (const syl of sylls) {
    if (isAccent(syl.role)) {
      svg.push(`<rect class="eni-col" x="${(syl.x0 + 6).toFixed(1)}" y="8" width="${(col - 12).toFixed(1)}" height="${EN_HEIGHT - 14}" rx="10" />`);
    }
  }
  for (const [y, label] of [[yHigh, "Cao"], [yMid, "Trung"], [yLow, "Thấp"]]) {
    svg.push(`<line class="eni-guide" x1="${EN_LEFT}" y1="${y.toFixed(1)}" x2="${width}" y2="${y.toFixed(1)}" />`);
    svg.push(`<text class="eni-axis" x="${EN_LEFT - 8}" y="${(y + 4).toFixed(1)}" text-anchor="end">${label}</text>`);
  }
  for (const g of gaps) {
    svg.push(`<line class="eni-break" x1="${g.x}" y1="14" x2="${g.x}" y2="${EN_HEIGHT - 4}" />`);
    svg.push(`<text class="eni-break__label" x="${g.x}" y="10" text-anchor="middle">ngắt cụm</text>`);
  }

  // Reference contour + acceptable band, one path per phrase.
  const phrases = [...new Set(sylls.map((s) => s.phrase))];
  const refSvg = [];
  for (const phrase of phrases) {
    const own = sylls.filter((s) => s.phrase === phrase && s.ref);
    const mid = [];
    const hi = [];
    const lo = [];
    for (const syl of own) {
      mid.push({ x: syl.xc, y: enY(syl.ref.semitone) });
      hi.push({ x: syl.xc, y: enY(syl.ref.hi) });
      lo.push({ x: syl.xc, y: enY(syl.ref.lo) });
      if (Math.abs(syl.ref.end - syl.ref.semitone) > 0.01) {
        const xe = syl.x1 - col * 0.12;
        mid.push({ x: xe, y: enY(syl.ref.end) });
        hi.push({ x: xe, y: enY(syl.ref.end + 2.5) });
        lo.push({ x: xe, y: enY(syl.ref.end - 2.5) });
      }
    }
    if (!mid.length) continue;
    refSvg.push(`<path class="eni-band" d="${smoothPath(hi)} ${smoothPath([...lo].reverse(), false)} Z" />`);
    refSvg.push(`<path class="eni-ref" d="${smoothPath(mid)}" />`);
  }
  for (const syl of sylls) {
    if (!syl.ref || !isAccent(syl.role)) continue;
    const r = syl.role === "nucleus" ? 8 : 6;
    refSvg.push(`<circle class="eni-ref__dot" cx="${syl.xc.toFixed(1)}" cy="${enY(syl.ref.semitone).toFixed(1)}" r="${r}" />`);
  }
  svg.push(`<g class="eni-layer-ref">${refSvg.join("")}</g>`);

  // Learner F0 per syllable (dashed), split at unvoiced syllables / breaks.
  if (take && take.learner) {
    const measured = new Map(take.learner.map((m) => [`${m.word}:${m.syllable}`, m]));
    const lastSyl = sylls[sylls.length - 1];
    const segments = [];
    let current = [];
    let phrase = null;
    for (const syl of sylls) {
      const m = measured.get(`${syl.word}:${syl.syllable}`);
      if (!m || m.semitone === null || syl.phrase !== phrase) {
        if (current.length) segments.push(current);
        current = [];
        phrase = syl.phrase;
        if (!m || m.semitone === null) continue;
      }
      current.push({ x: syl.xc, y: enY(m.semitone), dot: true });
      if (syl === lastSyl && m.end_semitone !== null && Math.abs(m.end_semitone - m.semitone) > 0.5) {
        current.push({ x: syl.x1 - col * 0.12, y: enY(m.end_semitone), dot: true });
      }
    }
    if (current.length) segments.push(current);
    const youSvg = [];
    for (const seg of segments) {
      if (seg.length > 1) youSvg.push(`<path class="eni-you" d="${smoothPath(seg)}" />`);
      for (const p of seg) youSvg.push(`<circle class="eni-you__dot" cx="${p.x.toFixed(1)}" cy="${p.y.toFixed(1)}" r="4.5" />`);
    }
    const missing = new Set(take.missing || []);
    for (const word of missing) {
      const syl = sylls.find((s) => s.word === word);
      if (syl) youSvg.push(`<text class="eni-note eni-note--bad" x="${syl.xc.toFixed(1)}" y="${(EN_TOP - 10).toFixed(1)}" text-anchor="middle">bỏ sót</text>`);
    }
    let noteRow = 0;
    for (const check of take.peaks) {
      if (check.ok || missing.has(check.word)) continue;
      const syl = sylls.find((s) => s.word === check.word && s.syllable === check.syllable);
      if (!syl || !syl.ref) continue;
      const text = check.ok === false ? "thiếu đỉnh" : "không đo được";
      const cls = check.ok === false ? "eni-note eni-note--bad" : "eni-note";
      // Alternate two rows so neighbouring notes in narrow columns don't overlap.
      const y = enY(syl.ref.semitone) + (noteRow++ % 2 ? 50 : 36);
      youSvg.push(`<text class="${cls}" x="${syl.xc.toFixed(1)}" y="${y.toFixed(1)}" text-anchor="middle">${text}</text>`);
    }
    for (const end of englishEnds(take)) {
      if (end.ok === null) continue;
      const own = sylls.filter((s) => s.phrase === end.phrase);
      const last = own[own.length - 1];
      if (!last) continue;
      const verb = end.expected === "rise" ? "lên" : end.expected === "either" ? "kết" : "rơi";
      const text = end.ok ? `${verb} ✓` : `chưa ${verb}`;
      const cls = end.ok ? "eni-note eni-note--ok" : "eni-note eni-note--bad";
      const y = end.expected === "rise" ? EN_TOP - 8 : EN_HEIGHT - 2;
      youSvg.push(`<text class="${cls}" x="${(last.x1 - 4).toFixed(1)}" y="${y}" text-anchor="end">${text}</text>`);
    }
    svg.push(`<g class="eni-layer-you">${youSvg.join("")}</g>`);
  }
  return `<svg class="eni-svg" width="${width}" height="${EN_HEIGHT}" viewBox="0 0 ${width} ${EN_HEIGHT}" role="img" aria-label="Biểu đồ ngữ điệu câu">${svg.join("")}</svg>`;
}

/** Syllable, IPA-per-word and sentence-stress rows under the chart. */
function englishRowsHtml(data, layout) {
  const missing = new Set(data.missing || []);
  const cells = [];
  cells.push('<span class="eni-row__label" style="grid-row:2;grid-column:1">Từ</span>');
  cells.push('<span class="eni-row__label" style="grid-row:3;grid-column:1">Câu</span>');
  for (const syl of layout.sylls) {
    let cls = isAccent(syl.role) ? "eni-syl eni-syl--acc" : "eni-syl";
    if (missing.has(syl.word)) cls += " eni-syl--missing";
    cells.push(`<span class="${cls}" style="grid-row:1;grid-column:${syl.column}">${escapeHtml(syllableLabel(syl))}</span>`);
    const mark = syl.role === "nucleus" ? "eni-mark--nuc" : syl.role === "stress" ? "eni-mark--acc" : "";
    cells.push(`<span class="eni-mark ${mark}" style="grid-row:3;grid-column:${syl.column}"></span>`);
  }
  for (const word of data.words) {
    const own = layout.sylls.filter((s) => s.word === word.index);
    if (!own.length) continue;
    const span = own[own.length - 1].column - own[0].column + 1;
    const cls = own.length > 1 ? "eni-ipa eni-ipa--multi" : "eni-ipa";
    const ipa = word.ipa ? `/${escapeHtml(word.ipa)}/` : "—";
    cells.push(`<span class="${cls}" style="grid-row:2;grid-column:${own[0].column} / span ${span}">${ipa}</span>`);
  }
  return `<div class="eni-rows" style="grid-template-columns:${layout.columns}">${cells.join("")}</div>`;
}

const ENGLISH_LEGEND = `
  <div class="eni-legend">
    <span><i class="eni-sw eni-sw--ref"></i>Mẫu ngữ điệu</span>
    <span><i class="eni-sw eni-sw--band"></i>Khoảng chấp nhận</span>
    <span><i class="eni-sw eni-sw--you"></i>Bạn (F0 đo)</span>
    <span><i class="eni-mark eni-mark--nuc"></i>Nhấn chính (nucleus)</span>
    <span class="eni-legend__end">ˈ trọng âm từ (từ điển) · ● trọng âm câu</span>
  </div>`;

/** "Từng từ" cell: TEA·cher, /ˈtiː.tʃɚ/, and whether the stress landed on
 *  the dictionary syllable (multi-syllable words only). */
function englishWordCellHtml(m) {
  const multi = m.syllables.length > 1;
  const marked = multi || (m.ipa || "").startsWith("ˈ");
  const word = m.syllables
    .map((s, i) => (marked && i === m.stress_index
      ? `<b class="syll__stressed">${escapeHtml(s.toUpperCase())}</b>`
      : escapeHtml(s)))
    .join('<span class="syll__dot">·</span>');
  const ipa = m.ipa ? `<span class="syll__ipa">/${escapeHtml(m.ipa)}/</span>` : "";
  let stress = "";
  if (!m.ok) {
    stress = ""; // not said (or misread): no verdict on its stress
  } else if (multi && m.stress_ok === true) {
    stress = '<span class="syll__stress syll__stress--ok">✓ trọng âm</span>';
  } else if (multi && m.stress_ok === false) {
    const heard = m.syllables[m.heard_stress_index] || "";
    stress = `<span class="syll__stress syll__stress--bad">✗ bạn nhấn ${escapeHtml(heard.toUpperCase())}</span>`;
  } else if (multi) {
    stress = '<span class="syll__stress">trọng âm: —</span>';
  }
  return `<span class="syll__word">${word}</span>${ipa}${stress}`;
}

/* "Trọng âm & ngữ điệu" in the result card: the take's own checks (peaks,
 * word_stress, breaks, ends, missing) laid out word by word, model above,
 * learner below, so the learner sees where it went off. Nothing is
 * re-scored here; every verdict comes from /evaluate `intonation`. */

const prosodyEl = {
  root: $("prosody"),
  summary: $("prosody-summary"),
  track: $("prosody-track"),
  list: $("prosody-list"),
};

/** "ROB·ber·y" with syllable `index` in capitals (lower-case otherwise). */
function syllablesWith(word, index) {
  return word.syllables
    .map((s, i) => (i === index ? s.text.toUpperCase() : s.text.toLowerCase()))
    .join("·");
}

/** Same as syllablesWith, as HTML with the capital syllable in <b>. */
function syllablesHtml(word, index) {
  return word.syllables
    .map((s, i) => (i === index
      ? `<b class="pros__stress">${escapeHtml(s.text.toUpperCase())}</b>`
      : escapeHtml(s.text.toLowerCase())))
    .join('<span class="syll__dot">·</span>');
}

/** Learner's pitch movement at a sentence end, from `delta` (end minus nucleus, st). */
function heardEnding(end) {
  if (end.delta == null) return end.creaky ? "↘" : "?";
  if (end.creaky || end.delta <= -1) return "↘";
  if (end.delta >= 1) return "↗";
  return "→";
}

const HEARD_ENDING = { "↘": "xuống giọng", "↗": "lên giọng", "→": "giữ ngang", "?": "không đo được" };

/** Every check of the take, keyed the way the track and the list need them. */
function prosodyChecks(take) {
  const peaks = new Map();
  for (const c of take.peaks || []) if (!peaks.has(c.word)) peaks.set(c.word, c);
  return {
    missing: new Set(take.missing || []),
    peaks,
    stress: new Map((take.word_stress || []).map((c) => [c.word, c])),
    breaks: new Map((take.breaks || []).map((c) => [c.after_word, c])),
    ends: new Map(englishEnds(take).map((c) => [c.phrase, c])),
  };
}

/** One word column: marks, model, learner, verdict. */
function prosodyWordHtml(word, checks) {
  const multi = word.syllables.length > 1;
  const plain = escapeHtml(word.syllables[0]?.text || word.text);
  const model = multi ? syllablesHtml(word, word.stress_index) : plain;
  const mark = word.nucleus ? "◉" : word.sentence_stress ? "●" : "";
  const peak = checks.peaks.get(word.index);
  const ws = multi ? checks.stress.get(word.index) : null;
  const tags = [];
  // Unjudged: no stress mark on the learner row, it was not measured.
  let you = multi ? syllablesHtml(word, -1) : plain;
  let state = "plain";

  if (checks.missing.has(word.index)) {
    you = "—";
    state = "bad";
    tags.push(["bad", "bỏ sót"]);
  } else {
    if (ws && ws.ok === false && ws.heard != null) {
      you = syllablesHtml(word, ws.heard);
      tags.push(["bad", "sai trọng âm"]);
    } else if (ws && ws.ok === true) {
      you = syllablesHtml(word, ws.heard ?? word.stress_index);
    } else if (ws) {
      tags.push(["na", "trọng âm ?"]);
    }
    if (peak && peak.ok === false) tags.push(["bad", "thiếu nhấn"]);
    else if (peak && peak.ok === null) tags.push(["na", "nhấn ?"]);
    const judged = (ws && ws.ok !== null) || (peak && peak.ok !== null);
    if (tags.some(([tone]) => tone === "bad")) state = "bad";
    else if (judged) state = "ok";
  }
  if (state === "ok") tags.push(["ok", "✓"]);
  const youMark = !mark || checks.missing.has(word.index)
    ? ""
    : peak && peak.ok === true ? "●" : peak && peak.ok === false ? "○" : "";
  return `<div class="pros__word" data-state="${state}">
    <span class="pros__mark">${mark}</span>
    <span class="pros__model">${model}</span>
    <span class="pros__you">${you}</span>
    <span class="pros__mark pros__mark--you" data-ok="${peak ? String(peak.ok) : ""}">${youMark}</span>
    <span class="pros__tags">${tags.map(([tone, text]) => `<i class="pros__tag pros__tag--${tone}">${text}</i>`).join("")}</span>
  </div>`;
}

/** The ‖ between two phrases, with the break check under it. */
function prosodyBreakHtml(check) {
  const state = !check || check.ok === null ? "na" : check.ok ? "ok" : "bad";
  const text = check ? { na: "?", ok: "✓ ngắt", bad: "chưa ngắt" }[state] : "";
  return `<div class="pros__sep" data-state="${state}">
    <span class="pros__mark"></span>
    <span class="pros__model">‖</span>
    <span class="pros__you">${!check ? "" : state === "bad" ? "→" : "‖"}</span>
    <span class="pros__mark"></span>
    <span class="pros__tags">${text ? `<i class="pros__tag pros__tag--${state}">${text}</i>` : ""}</span>
  </div>`;
}

/** The phrase-end arrow: expected above, what the take did below. */
function prosodyEndHtml(expected, check) {
  const state = !check || check.ok === null ? "na" : check.ok ? "ok" : "bad";
  const heard = check ? heardEnding(check) : "";
  const text = check ? { na: "?", ok: "✓", bad: HEARD_ENDING[heard] }[state] : "";
  return `<div class="pros__sep pros__sep--end" data-state="${state}">
    <span class="pros__mark"></span>
    <span class="pros__model">${ENDING_ARROW[expected] || ""}</span>
    <span class="pros__you">${state === "ok" ? ENDING_ARROW[expected] || "" : heard}</span>
    <span class="pros__mark"></span>
    <span class="pros__tags">${text ? `<i class="pros__tag pros__tag--${state}">${escapeHtml(text)}</i>` : ""}</span>
  </div>`;
}

function prosodyTrackHtml(take, checks) {
  const cells = ['<div class="pros__rowlabels"><span></span><span>Mẫu</span><span>Bạn</span></div>'];
  take.words.forEach((word, i) => {
    cells.push(prosodyWordHtml(word, checks));
    const next = take.words[i + 1];
    if (next && next.phrase === word.phrase) return;
    const ending = take.phrase_ends[word.phrase];
    if (ending && ending !== "continue") cells.push(prosodyEndHtml(ending, checks.ends.get(word.phrase)));
    if (next) cells.push(prosodyBreakHtml(checks.breaks.get(word.index)));
  });
  return cells.join("");
}

/** x/y over the checks that could be measured; null when none could. */
function tally(items) {
  const known = items.filter((c) => c && c.ok !== null);
  return known.length ? { ok: known.filter((c) => c.ok).length, of: known.length } : null;
}

function prosodySummaryHtml(take, checks) {
  const chips = [];
  const chip = (label, t) => {
    if (!t) return chips.push(["na", `${label}: không đo được`]);
    return chips.push([t.ok === t.of ? "ok" : "bad", `${label}: ${t.ok}/${t.of}`]);
  };
  const multiStress = (take.word_stress || []).filter((c) => !checks.missing.has(c.word));
  if (multiStress.length) chip("Trọng âm từ", tally(multiStress));
  else chips.push(["na", "Trọng âm từ: câu không có từ nhiều âm tiết"]);
  const peaks = (take.peaks || []).filter((c) => !checks.missing.has(c.word));
  if (peaks.length) chip("Nhấn câu", tally(peaks));
  if ((take.breaks || []).length) chip("Ngắt cụm", tally(take.breaks));
  const ends = englishEnds(take);
  if (ends.length) chip(ends.length > 1 ? "Lên/xuống cuối cụm" : "Lên/xuống cuối câu", tally(ends));
  if (checks.missing.size) chips.push(["bad", `Bỏ sót ${checks.missing.size} từ`]);
  return chips.map(([tone, text]) => `<span class="eni-chip eni-chip--${tone}"><i></i>${escapeHtml(text)}</span>`).join("");
}

/** Every deviation, in sentence order, as "where — what to do". */
function prosodyIssues(take, checks) {
  const items = [];
  const unmeasured = [];
  if (checks.missing.size) {
    items.push(["Bỏ sót", `${listWords([...checks.missing].map((i) => `“${take.words[i]?.text || ""}”`), 5)} — đọc đủ cả câu thì mới so được ngữ điệu.`]);
  }
  take.words.forEach((word, i) => {
    if (checks.missing.has(word.index)) return;
    const ws = word.syllables.length > 1 ? checks.stress.get(word.index) : null;
    const peak = checks.peaks.get(word.index);
    if (ws && ws.ok === false && ws.heard != null) {
      items.push([
        `Trọng âm “${word.text}”`,
        `Nhấn ${syllablesWith(word, ws.expected)}, bạn đang nhấn ${syllablesWith(word, ws.heard)}.`,
      ]);
    } else if (ws && ws.ok === null) {
      unmeasured.push(`trọng âm “${word.text}”`);
    }
    if (peak && peak.ok === false) {
      const where = word.syllables.length > 1 ? syllablesWith(word, peak.syllable) : word.text.toUpperCase();
      items.push([
        `Nhấn câu “${word.text}”`,
        `Giọng gần như phẳng ở ${where}. ${word.nucleus ? "Đây là từ nhấn chính: n" : "N"}âng cao và kéo dài âm tiết này hơn các từ xung quanh.`,
      ]);
    } else if (peak && peak.ok === null) {
      unmeasured.push(`nhấn “${word.text}”`);
    }
    const next = take.words[i + 1];
    if (next && next.phrase === word.phrase) return;
    const end = checks.ends.get(word.phrase);
    const expected = take.phrase_ends[word.phrase];
    if (end && end.ok === false) {
      const want = ENDING_VERB[end.expected] || ENDING_VERB[expected] || ENDING_VERB.fall;
      items.push([
        `Cuối ${next ? "cụm" : "câu"} “${word.text}”`,
        `${want} ${ENDING_ARROW[end.expected] || ""}, bạn đang ${HEARD_ENDING[heardEnding(end)]} ${heardEnding(end)}.`,
      ]);
    } else if (end && end.ok === null) {
      unmeasured.push(`cuối ${next ? "cụm" : "câu"}`);
    }
    const br = next ? checks.breaks.get(word.index) : null;
    if (br && br.ok === false) {
      const pause = br.pause != null ? ` (bạn dừng ${br.pause.toFixed(2)}s)` : "";
      items.push([`Sau “${word.text}”`, `Chưa ngắt cụm${pause}: dừng ngắn, hạ giọng ở cuối cụm rồi mới đọc tiếp.`]);
    } else if (br && br.ok === null) {
      unmeasured.push(`ngắt sau “${word.text}”`);
    }
  });
  return { items, unmeasured };
}

function renderProsody(take) {
  if (!prosodyEl.root) return;
  const words = take && Array.isArray(take.words) ? take.words : [];
  prosodyEl.root.hidden = !words.length;
  if (!words.length) return;
  const checks = prosodyChecks(take);
  const measured = Boolean(take.learner && take.learner.some((m) => m.semitone !== null));
  prosodyEl.summary.innerHTML = measured
    ? prosodySummaryHtml(take, checks)
    : '<span class="eni-chip eni-chip--na"><i></i>Không đo được cao độ trong bản ghi, chỉ hiện mẫu</span>';
  prosodyEl.track.innerHTML = prosodyTrackHtml(take, measured ? checks : prosodyChecks({ words }));
  const { items, unmeasured } = measured ? prosodyIssues(take, checks) : { items: [], unmeasured: [] };
  const rows = items.map(([where, text]) =>
    `<li class="pros__issue"><b>${escapeHtml(where)}</b><span>${escapeHtml(text)}</span></li>`);
  if (measured && !items.length) {
    rows.push('<li class="pros__issue pros__issue--ok"><b>Khớp mẫu</b><span>Trọng âm, chỗ nhấn, chỗ ngắt và lên/xuống giọng cuối câu đều đúng.</span></li>');
  }
  if (unmeasured.length) {
    rows.push(`<li class="pros__issue pros__issue--na"><b>Không đo được</b><span>${escapeHtml(listWords(unmeasured, 6))} (giọng quá nhỏ hoặc không có cao độ).</span></li>`);
  }
  prosodyEl.list.innerHTML = rows.join("");
  prosodyEl.list.hidden = !rows.length;
}

function renderEnglishIntonation() {
  const take = englishTake;
  const data = take || englishReference;
  if (!data || !data.words || !data.words.length) {
    setPitchStatus("Không phân tích được ngữ điệu cho câu này.", true);
    return;
  }
  const hasLearner = Boolean(take && take.learner && take.learner.some((m) => m.semitone !== null));
  if (!hasLearner && englishView === "you") englishView = "both";
  el.pitchTitle.textContent = hasLearner ? "Ngữ điệu câu · mẫu và giọng của bạn" : "Ngữ điệu câu · mẫu";
  el.pitchLegend.innerHTML = hasLearner
    ? `<div class="eni-toggle" role="group" aria-label="Hiển thị">${[["ref", "Mẫu"], ["you", "Bạn"], ["both", "Cả hai"]]
        .map(([key, label]) => `<button type="button" data-en-view="${key}" aria-pressed="${englishView === key}">${label}</button>`)
        .join("")}</div>`
    : "";

  // The chart box itself when it is on screen; the panel minus its padding
  // otherwise (first render happens while the chart is still hidden).
  const available = Math.max(
    300,
    el.pitchChart.clientWidth || (el.pitch.clientWidth ? el.pitch.clientWidth - 34 : 680),
  );
  const layout = englishLayout(data, available);
  const measuredNote = take && !hasLearner
    ? '<p class="pitch__hint">Không đo được cao độ (F0) trong bản ghi này — chỉ hiện đường mẫu.</p>'
    : "";
  const roughNote = take && take.aligned === false
    ? '<p class="pitch__hint">Không dóng được thời gian từng âm tiết, đường của bạn chỉ là ước lượng.</p>'
    : "";
  el.pitchChart.innerHTML = `
    <div class="eni" data-view="${hasLearner ? englishView : "ref"}">
      ${englishSentenceHtml(data, layout)}
      ${take && take.peaks ? englishChipsHtml(take) : ""}
      <div class="pitch__scroll">
        <div class="eni-plot" style="width:${layout.width}px">
          ${englishChartSvg(hasLearner ? take : null, layout)}
          ${englishRowsHtml(data, layout)}
        </div>
      </div>
      ${measuredNote}${roughNote}
      ${hasLearner ? englishCardsHtml(take) : ""}
      ${ENGLISH_LEGEND}
    </div>`;
  el.pitchChart.hidden = false;
  el.pitchStatus.hidden = true;
}

el.pitchLegend.addEventListener("click", (event) => {
  const button = event.target.closest("[data-en-view]");
  if (!button) return;
  englishView = button.dataset.enView;
  const root = el.pitchChart.querySelector(".eni");
  if (root) root.dataset.view = englishView;
  for (const other of el.pitchLegend.querySelectorAll("[data-en-view]")) {
    other.setAttribute("aria-pressed", String(other === button));
  }
});

/* --------------------------------------------------------------- Wiring */

/* ------------------------------------------------------ Search + history */

/** One history per practice language, so /en suggestions never show on /. */
const HISTORY_COOKIE = `pv_search_history_${(TTS_LANG.split("-")[0] || "ja").toLowerCase()}`;
const HISTORY_MAX_ITEMS = 10;
const HISTORY_SHOWN = 8;
/** Cookies cap out near 4 KB; CJK text triples in size once URL-encoded. */
const HISTORY_MAX_BYTES = 3500;

function readHistory() {
  try {
    const raw = document.cookie
      .split("; ")
      .find((part) => part.startsWith(`${HISTORY_COOKIE}=`));
    if (!raw) return [];
    const list = JSON.parse(decodeURIComponent(raw.slice(HISTORY_COOKIE.length + 1)));
    return Array.isArray(list) ? list.filter((item) => typeof item === "string" && item) : [];
  } catch {
    return [];
  }
}

function writeHistory(list) {
  let items = list.slice(0, HISTORY_MAX_ITEMS);
  let value = encodeURIComponent(JSON.stringify(items));
  while (items.length && value.length > HISTORY_MAX_BYTES) {
    items = items.slice(0, -1);
    value = encodeURIComponent(JSON.stringify(items));
  }
  const secure = location.protocol === "https:" ? "; Secure" : "";
  document.cookie = `${HISTORY_COOKIE}=${value}; Max-Age=31536000; Path=/; SameSite=Lax${secure}`;
}

function addHistory(text) {
  writeHistory([text, ...readHistory().filter((item) => item !== text)]);
}

function removeHistory(text) {
  writeHistory(readHistory().filter((item) => item !== text));
}

let suggestItems = [];
let suggestIndex = -1;

function hideSuggest() {
  if (!el.suggest) return;
  el.suggest.hidden = true;
  el.suggest.replaceChildren();
  el.customInput.setAttribute("aria-expanded", "false");
  el.customInput.removeAttribute("aria-activedescendant");
  suggestItems = [];
  suggestIndex = -1;
}

/** `text` with the first case-insensitive hit of `query` wrapped in <mark>. */
function highlighted(text, query) {
  const span = document.createElement("span");
  span.className = "search__option-text";
  const lower = text.toLowerCase();
  const at = query && lower.length === text.length ? lower.indexOf(query) : -1;
  if (at < 0) {
    span.textContent = text;
    return span;
  }
  const mark = document.createElement("mark");
  mark.textContent = text.slice(at, at + query.length);
  span.append(text.slice(0, at), mark, text.slice(at + query.length));
  return span;
}

function renderSuggest() {
  if (!el.suggest || el.customInput.disabled) return;
  const query = el.customInput.value.replace(/\s+/g, " ").trim().toLowerCase();
  suggestItems = readHistory()
    .filter((item) => !query || item.toLowerCase().includes(query))
    .filter((item) => item.toLowerCase() !== query)
    .slice(0, HISTORY_SHOWN);
  suggestIndex = -1;
  el.customInput.removeAttribute("aria-activedescendant");

  if (!suggestItems.length) {
    hideSuggest();
    return;
  }

  const head = document.createElement("li");
  head.className = "search__suggest-head";
  head.setAttribute("role", "presentation");
  head.textContent = "Tìm kiếm gần đây";

  const options = suggestItems.map((item, index) => {
    const li = document.createElement("li");
    li.className = "search__option";
    li.id = `search-option-${index}`;
    li.setAttribute("role", "option");
    li.setAttribute("aria-selected", "false");
    li.dataset.index = String(index);

    const icon = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    icon.setAttribute("class", "icon icon--sm search__option-icon");
    icon.setAttribute("aria-hidden", "true");
    const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
    use.setAttribute("href", "#i-history");
    icon.append(use);

    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "search__option-remove";
    remove.title = "Xóa khỏi lịch sử";
    remove.setAttribute("aria-label", `Xóa "${item}" khỏi lịch sử`);
    remove.dataset.remove = String(index);
    remove.textContent = "×";

    li.append(icon, highlighted(item, query), remove);
    return li;
  });

  el.suggest.replaceChildren(head, ...options);
  el.suggest.hidden = false;
  el.customInput.setAttribute("aria-expanded", "true");
}

function moveSuggest(step) {
  if (!suggestItems.length) return;
  suggestIndex = (suggestIndex + step + suggestItems.length) % suggestItems.length;
  for (const option of el.suggest.querySelectorAll(".search__option")) {
    const active = Number(option.dataset.index) === suggestIndex;
    option.classList.toggle("is-active", active);
    option.setAttribute("aria-selected", String(active));
    if (active) option.scrollIntoView({ block: "nearest" });
  }
  el.customInput.setAttribute("aria-activedescendant", `search-option-${suggestIndex}`);
}

/** Show `text` as the sentence to read, then load its AI guide and pitch. */
function runSearch(text, chip = null) {
  hideSuggest();
  const owner = chip || [...el.chips.children].find((item) => item.dataset.text === text) || null;
  addHistory(text);
  if (text === target.text) return;

  setTarget({
    text,
    reading: owner?.dataset.reading || "",
    meaning: owner?.dataset.meaning || "",
    chip: owner,
  });
  requestTextGuide();
  comparePitch();
}

el.chips.addEventListener("click", (event) => {
  const chip = event.target.closest(".chip");
  if (!chip || chip.disabled) return;
  el.customInput.value = chip.dataset.text;
  runSearch(chip.dataset.text, chip);
});

el.customForm.addEventListener("submit", (event) => {
  event.preventDefault();
  // The API scores one sentence, so collapse any pasted newlines/whitespace.
  const text = el.customInput.value.replace(/\s+/g, " ").trim();
  if (!text) {
    el.customInput.focus();
    return;
  }
  el.customInput.value = text;
  runSearch(text);
});

el.customInput.addEventListener("input", renderSuggest);
el.customInput.addEventListener("focus", renderSuggest);
el.customInput.addEventListener("blur", hideSuggest);

el.customInput.addEventListener("keydown", (event) => {
  if (el.suggest?.hidden !== false) {
    if (event.key === "ArrowDown") {
      renderSuggest();
      if (suggestItems.length) {
        event.preventDefault();
        moveSuggest(1);
      }
    }
    return;
  }
  if (event.key === "ArrowDown" || event.key === "ArrowUp") {
    event.preventDefault();
    moveSuggest(event.key === "ArrowDown" ? 1 : -1);
  } else if (event.key === "Enter" && suggestIndex >= 0) {
    event.preventDefault();
    el.customInput.value = suggestItems[suggestIndex];
    el.customForm.requestSubmit();
  } else if (event.key === "Escape") {
    event.preventDefault();
    hideSuggest();
  }
});

if (el.suggest) {
  // Keep focus in the input, so a click does not blur (and close) the list first.
  el.suggest.addEventListener("mousedown", (event) => event.preventDefault());
  el.suggest.addEventListener("click", (event) => {
    const remove = event.target.closest("[data-remove]");
    if (remove) {
      removeHistory(suggestItems[Number(remove.dataset.remove)]);
      renderSuggest();
      return;
    }
    const option = event.target.closest(".search__option");
    if (!option) return;
    el.customInput.value = suggestItems[Number(option.dataset.index)];
    el.customForm.requestSubmit();
  });
}

function pickTtsVoice(lang) {
  const voices = window.speechSynthesis.getVoices();
  const prefix = lang.slice(0, 2).toLowerCase();
  return (
    voices.find((v) => v.lang === lang) ||
    voices.find((v) => v.lang.toLowerCase().startsWith(prefix)) ||
    null
  );
}

/* ------------------------------------------------------ Reference audio */

/** Generated audio URL per sentence for this page visit, so a second click
 *  replays instead of generating again. */
const speakUrls = new Map();
const speakPlayer = new Audio();
speakPlayer.preload = "auto";
let speakState = "idle"; // idle | loading | playing
let speakRequestId = 0;

function setSpeakState(next) {
  speakState = next;
  el.speak.classList.toggle("is-busy", next === "loading");
  el.speak.setAttribute("aria-busy", String(next === "loading"));
  el.speakIcon?.firstElementChild?.setAttribute(
    "href",
    next === "loading" ? "#i-loader" : next === "playing" ? "#i-stop" : "#i-volume",
  );
  if (el.speakLabel) {
    el.speakLabel.textContent = next === "loading" ? "Đang tạo…" : next === "playing" ? "Dừng" : "Nghe mẫu";
  }
}

function stopSpeak() {
  speakRequestId += 1;
  speakPlayer.pause();
  window.speechSynthesis?.cancel();
  setSpeakState("idle");
}

/** Fallback when the TTS service is not configured or fails. */
function speakWithBrowser(text) {
  if (!window.speechSynthesis) return;
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = TTS_LANG;
  utterance.rate = TTS_LANG.startsWith("en") ? 0.95 : 0.85;
  const voice = pickTtsVoice(TTS_LANG);
  if (voice) utterance.voice = voice;
  utterance.onend = () => setSpeakState("idle");
  utterance.onerror = () => setSpeakState("idle");
  window.speechSynthesis.cancel();
  setSpeakState("playing");
  window.speechSynthesis.speak(utterance);
}

async function fetchSpeakUrl(text) {
  const formData = new FormData();
  formData.append("text", text);
  formData.append("lang", (TTS_LANG.split("-")[0] || "ja").toLowerCase());
  const response = await fetch(SPEAK_API_URL, { method: "POST", body: formData });
  if (!response.ok) {
    const detail = await response.json().then((body) => body.detail).catch(() => undefined);
    throw new Error(typeof detail === "string" ? detail : `Yêu cầu thất bại (HTTP ${response.status})`);
  }
  const result = await response.json();
  if (!result.audio_url) throw new Error("Không nhận được audio mẫu.");
  return result.audio_url;
}

async function playSpeakUrl(url) {
  speakPlayer.src = url;
  setSpeakState("playing");
  await speakPlayer.play();
}

async function speakTarget() {
  const text = target.text;
  if (!text) return;
  const requestId = ++speakRequestId;
  window.speechSynthesis?.cancel();

  let url = speakUrls.get(text);
  try {
    if (!url) {
      setSpeakState("loading");
      url = await fetchSpeakUrl(text);
      if (requestId !== speakRequestId) return;
      speakUrls.set(text, url);
    }
    try {
      await playSpeakUrl(url);
    } catch (playError) {
      // A replayed file may have been removed from the service: regenerate once.
      if (requestId !== speakRequestId || !speakUrls.has(text) || playError?.name === "NotAllowedError") throw playError;
      speakUrls.delete(text);
      setSpeakState("loading");
      url = await fetchSpeakUrl(text);
      if (requestId !== speakRequestId) return;
      speakUrls.set(text, url);
      await playSpeakUrl(url);
    }
    el.speak.title = "Nghe giọng đọc mẫu (AI tạo audio)";
  } catch (error) {
    if (requestId !== speakRequestId) return;
    setSpeakState("idle");
    const message = error instanceof Error ? error.message : "Không tạo được audio mẫu.";
    console.warn("[Nghe mẫu] AI audio unavailable, using browser voice:", message);
    el.speak.title = `${message} — đang dùng giọng đọc của trình duyệt.`;
    speakWithBrowser(text);
  }
}

speakPlayer.addEventListener("ended", () => setSpeakState("idle"));
speakPlayer.addEventListener("pause", () => {
  if (speakState === "playing") setSpeakState("idle");
});

el.speak.addEventListener("click", () => {
  if (speakState === "loading") return;
  if (speakState === "playing") {
    stopSpeak();
    return;
  }
  speakTarget();
});


el.mic.addEventListener("click", () => {
  if (status === "recording") stopRecording();
  else startRecording();
});

el.pickFile.addEventListener("click", () => el.file.click());

el.file.addEventListener("change", () => {
  // One recording at a time — the input has no `multiple`, so take the first.
  const [file] = el.file.files;
  // Clear first, so picking the same file twice still fires a change event.
  el.file.value = "";
  submitFile(file);
});

// Drag & drop anywhere on the recorder card.
for (const type of ["dragenter", "dragover"]) {
  el.recorder.addEventListener(type, (event) => {
    if (!event.dataTransfer?.types.includes("Files")) return;
    event.preventDefault();
    el.recorder.classList.add("is-dropping");
  });
}

for (const type of ["dragleave", "dragend"]) {
  el.recorder.addEventListener(type, (event) => {
    if (event.target === el.recorder) el.recorder.classList.remove("is-dropping");
  });
}

el.recorder.addEventListener("drop", (event) => {
  event.preventDefault();
  el.recorder.classList.remove("is-dropping");
  submitFile(event.dataTransfer?.files?.[0]);
});

el.reset.addEventListener("click", () => {
  clearOutput();
  clearPlayback();
  learnerPitch = null;
  englishTake = null;
  if (document.body.dataset.targetLang === "zh" && referencePattern) renderChinesePitch();
  else if (referencePattern) renderPitchChart();
});

// Release the mic if the user navigates away mid-recording.
window.addEventListener("pagehide", releaseResources);

el.timerMax.textContent = `tối đa ${formatDuration(MAX_DURATION_SECONDS)}`;
el.uploadHint.textContent =
  `Hỗ trợ WAV, MP3, FLAC, OGG, M4A, WebM..., tối đa ${formatDuration(MAX_UPLOAD_SECONDS)} — kéo thả vào đây cũng được`;
setStatus("idle");
