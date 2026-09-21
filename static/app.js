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
const API_URL = `${API_BASE}/api/pronunciation/evaluate`;
const PITCH_API_URL = `${API_BASE}/api/pronunciation/pitch-accent`;
const PITCH_CONTOUR_API_URL = `${API_BASE}/api/pronunciation/pitch-contour`;
const COACH_API_URL = `${API_BASE}/api/pronunciation/coach`;
/** Sample rate the ASR model runs at. */
const TARGET_SAMPLE_RATE = 16_000;
/** Stop on our own so a forgotten recording cannot exceed the upload limit. */
const MAX_DURATION_SECONDS = 30;
/** Uploads are re-encoded to 16 kHz mono (32 KB/s), so this stays well under
 *  the server's MAX_AUDIO_MB and keeps inference short. */
const MAX_UPLOAD_SECONDS = 120;
/** Rejected before decoding, so a huge file never reaches the AudioContext. */
const MAX_UPLOAD_BYTES = 50 * 1024 * 1024;
/** Uploads are .wav only for now, to match what the API accepts. */
const UPLOAD_EXTENSION = ".wav";

const $ = (id) => document.getElementById(id);

const el = {
  chips: $("chips"),
  targetText: $("target-text"),
  targetReading: $("target-reading"),
  targetMeaning: $("target-meaning"),
  speak: $("speak"),
  customForm: $("custom-form"),
  customInput: $("custom-input"),
  customApply: $("custom-apply"),
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
  coachPanel: $("coach-panel"),
  coachStatus: $("coach-status"),
  coachComment: $("coach-comment"),
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

let target = {
  text: el.targetText.textContent.trim(),
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
  el.targetMeaning.hidden = !meaning;

  for (const other of el.chips.children) other.classList.toggle("chip--active", other === chip);

  clearOutput();
  clearPlayback();
  resetPitch();
}

function showError(message) {
  el.errorText.textContent = message;
  el.error.hidden = false;
  el.result.hidden = true;
}

function clearOutput() {
  el.error.hidden = true;
  el.result.hidden = true;
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

async function startRecording() {
  if (status === "recording" || status === "requesting") return;

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
    await Promise.all([evaluate(wav), comparePitch(wav)]);
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

  // A dropped file bypasses the input's accept filter, so check it here too.
  if (!file.name.toLowerCase().endsWith(UPLOAD_EXTENSION)) {
    showError(`Chỉ nhận file ${UPLOAD_EXTENSION} — file bạn chọn là "${file.name}".`);
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
    showError("Không đọc được file WAV này. Hãy thử một file khác.");
    return;
  }

  showPlayback(wav);
  await Promise.all([evaluate(wav), comparePitch(wav)]);
}

/* ------------------------------------------------------------ Evaluate */

async function evaluate(wav) {
  clearOutput();
  setStatus("evaluating");

  const formData = new FormData();
  formData.append("text", target.text);
  formData.append("audio", new File([wav], "recording.wav", { type: "audio/wav" }));

  try {
    const response = await fetch(API_URL, { method: "POST", body: formData });
    if (!response.ok) {
      const detail = await response
        .json()
        .then((body) => body.detail)
        .catch(() => undefined);
      throw new Error(detail || `Yêu cầu thất bại (HTTP ${response.status})`);
    }
    renderResult(await response.json());
  } catch (error) {
    showError(error instanceof Error ? error.message : "Đã có lỗi xảy ra.");
  } finally {
    setStatus("idle");
  }
}

function renderResult(result) {
  const badge = LEVEL_BADGE[result.feedback.level] ?? LEVEL_BADGE.mismatch;
  const color = RING_COLOR[result.feedback.level] ?? RING_COLOR.mismatch;

  const circumference = 2 * Math.PI * 52;
  const filled = (Math.min(100, Math.max(0, result.score)) / 100) * circumference;
  el.ringValue.style.stroke = color;
  el.ringValue.style.strokeDasharray = `${filled} ${circumference}`;
  el.score.style.color = color;
  el.score.textContent = result.score;

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
  el.mDistance.textContent = `${result.distance} ký tự`;
  el.mDuration.textContent = `${result.audio_duration.toFixed(2)}s`;

  // Mark the target characters the model did not hear as expected.
  const wrong = new Set(
    result.errors.filter((error) => error.type !== "ins").map((error) => error.position),
  );
  el.rTarget.replaceChildren(
    ...[...result.target_hiragana].map((char, index) => {
      const node = document.createElement(wrong.has(index) ? "mark" : "span");
      node.textContent = char;
      return node;
    }),
  );
  el.rHeard.textContent = result.recognized_hiragana || "—";

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
        cell.textContent = m.mora;
        return cell;
      }),
    );
  }

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

  el.result.hidden = false;
}

/* -------------------------------------------------------------- Coaching */

function resetCoach() {
  el.coachPanel.hidden = true;
  el.coachStatus.hidden = true;
  el.coachStatus.classList.remove("is-error");
  el.coachStatus.textContent = "";
  el.coachComment.hidden = true;
  el.coachComment.textContent = "";
  el.coachBtn.disabled = false;
}

/** Ask the locally-run Ollama model (see app/services/coaching.py) for a
 *  natural-language Vietnamese comment on the take just scored. On-demand
 *  rather than automatic -- generation takes a few seconds and this app's
 *  own score/diff/pitch feedback is already shown instantly above. */
async function requestCoach() {
  if (!lastWav || el.coachBtn.disabled) return;

  el.coachPanel.hidden = false;
  el.coachComment.hidden = true;
  el.coachStatus.hidden = false;
  el.coachStatus.classList.remove("is-error");
  el.coachStatus.textContent = "Đang phân tích và viết nhận xét…";
  el.coachBtn.disabled = true;

  const formData = new FormData();
  formData.append("text", target.text);
  formData.append("audio", new File([lastWav], "recording.wav", { type: "audio/wav" }));

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
    el.coachComment.textContent = result.comment;
    el.coachComment.hidden = false;
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
/** Learner's own pitch curve for the take just recorded/uploaded, from
 *  /pitch-contour — time-normalized, not mora-aligned (see renderPitchChart). */
let learnerContour = null;

function resetPitch() {
  pitchLoadedFor = null;
  referencePattern = null;
  learnerContour = null;
  el.pitch.hidden = true;
  el.pitchBtn.setAttribute("aria-expanded", "false");
  el.pitchChart.hidden = true;
  el.pitchChart.innerHTML = "";
}

function setPitchStatus(message, isError = false) {
  el.pitchStatus.textContent = message;
  el.pitchStatus.classList.toggle("is-error", isError);
  el.pitchStatus.hidden = false;
  el.pitchChart.hidden = true;
}

/** Semitone span (± this many semitones) mapped onto the same vertical
 *  space as the reference chart's H/L levels. */
const PITCH_SEMITONE_RANGE = 7;

/** One evenly-spaced High/Low point per mora, drawn as a single polyline
 *  between two guide levels — the same shape shown on OJAD-style pitch
 *  accent references. If the learner's own recording has been analyzed,
 *  /pitch-contour bucketed it into the same number of morae, so its curve
 *  is drawn through the *exact same x positions* as the reference dots —
 *  see the docstring on extract_pitch_per_mora for what that alignment
 *  does and does not guarantee. */
function renderPitchChart() {
  if (!referencePattern || referencePattern.length === 0) {
    setPitchStatus("Không phân tích được cao độ cho câu này.", true);
    return;
  }

  const hasLearner = Boolean(learnerContour && learnerContour.some((point) => point.voiced));
  el.pitchTitle.textContent = hasLearner ? "Cao độ: mẫu và bạn" : "Cao độ mẫu";
  el.pitchLegend.innerHTML = `
    <span class="pitch__legend-item"><i class="pitch__swatch pitch__swatch--ref"></i>Mẫu</span>
    ${hasLearner ? '<span class="pitch__legend-item"><i class="pitch__swatch pitch__swatch--you"></i>Bạn</span>' : ""}
  `;

  const columnWidth = 40;
  const width = referencePattern.length * columnWidth;
  const yHigh = 22;
  const yLow = 72;

  const refPoints = referencePattern.map((mora, index) => ({
    x: (index + 0.5) * columnWidth,
    y: mora.pitch === "H" ? yHigh : yLow,
  }));
  const refPolyline = refPoints.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ");
  const refDots = refPoints
    .map((p, index) => {
      const isHigh = referencePattern[index].pitch === "H";
      return `<circle class="pitch__dot${isHigh ? " pitch__dot--h" : ""}" cx="${p.x.toFixed(1)}" cy="${p.y.toFixed(1)}" r="3.5" />`;
    })
    .join("");
  const labels = referencePattern
    .map((mora) => `<span class="pitch__label">${mora.mora}</span>`)
    .join("");

  let learnerSvg = "";
  let hint = "";
  if (hasLearner) {
    const yMid = (yHigh + yLow) / 2;
    const yScale = (yLow - yHigh) / 2 / PITCH_SEMITONE_RANGE;
    const toY = (semitone) => {
      const clamped = Math.max(-PITCH_SEMITONE_RANGE, Math.min(PITCH_SEMITONE_RANGE, semitone));
      return yMid - clamped * yScale;
    };

    // One learner point per reference point (same array length, same x) —
    // break into separate polylines wherever a mora had no voiced audio,
    // instead of drawing a straight (misleading) line across the gap.
    const segments = [];
    let current = [];
    learnerContour.forEach((point, index) => {
      if (point.voiced && point.semitone !== null) {
        current.push(`${refPoints[index].x.toFixed(1)},${toY(point.semitone).toFixed(1)}`);
      } else if (current.length) {
        segments.push(current);
        current = [];
      }
    });
    if (current.length) segments.push(current);

    learnerSvg = segments
      .filter((segment) => segment.length > 1)
      .map((segment) => `<polyline class="pitch__line pitch__line--learner" points="${segment.join(" ")}" />`)
      .join("");
    hint = `<p class="pitch__hint">So khớp theo thứ tự mora, không phải căn chỉnh thời gian chính xác từng âm.</p>`;
  }

  el.pitchChart.innerHTML = `
    <svg class="pitch__svg" viewBox="0 0 ${width} 94" preserveAspectRatio="none">
      <line class="pitch__guide" x1="0" y1="${yLow}" x2="${width}" y2="${yLow}" />
      <polyline class="pitch__line" points="${refPolyline}" />
      ${refDots}
      ${learnerSvg}
    </svg>
    <div class="pitch__labels" style="grid-template-columns: repeat(${referencePattern.length}, 1fr)">
      ${labels}
    </div>
    ${hint}
  `;
  el.pitchChart.hidden = false;
  el.pitchStatus.hidden = true;
}

async function loadPitchAccent(text) {
  setPitchStatus("Đang phân tích cao độ…");

  try {
    const response = await fetch(`${PITCH_API_URL}?text=${encodeURIComponent(text)}`);
    if (!response.ok) {
      const detail = await response
        .json()
        .then((body) => body.detail)
        .catch(() => undefined);
      throw new Error(detail || `Yêu cầu thất bại (HTTP ${response.status})`);
    }
    const result = await response.json();
    referencePattern = result.pattern;
    pitchLoadedFor = text;
    renderPitchChart();
  } catch (error) {
    referencePattern = null;
    pitchLoadedFor = null;
    setPitchStatus(
      error instanceof Error ? error.message : "Không lấy được cao độ mẫu.",
      true,
    );
  }
}

/** Analyze the take just recorded/uploaded and overlay it on the chart.
 *  Failure here is non-fatal — the reference pattern alone is still useful
 *  — so it degrades quietly instead of showing an error banner. */
async function loadPitchContour(wav) {
  try {
    const formData = new FormData();
    formData.append("text", target.text);
    formData.append("audio", new File([wav], "recording.wav", { type: "audio/wav" }));
    const response = await fetch(PITCH_CONTOUR_API_URL, { method: "POST", body: formData });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const result = await response.json();
    learnerContour = result.points;
  } catch (error) {
    learnerContour = null;
    console.warn("Pitch contour extraction failed:", error);
  }
  if (referencePattern) renderPitchChart();
}

/** Show the pitch panel and load whatever it is missing: the reference
 *  pattern (once per target sentence) and the learner's contour (every
 *  take). Called right after a recording/upload is sent for scoring. */
async function comparePitch(wav) {
  el.pitch.hidden = false;
  el.pitchBtn.setAttribute("aria-expanded", "true");
  const tasks = [loadPitchContour(wav)];
  if (pitchLoadedFor !== target.text) tasks.unshift(loadPitchAccent(target.text));
  await Promise.all(tasks);
}

el.coachBtn.addEventListener("click", requestCoach);

el.pitchBtn.addEventListener("click", () => {
  const opening = el.pitch.hidden;
  el.pitch.hidden = !opening;
  el.pitchBtn.setAttribute("aria-expanded", String(opening));
  if (opening && pitchLoadedFor !== target.text) loadPitchAccent(target.text);
});

/* --------------------------------------------------------------- Wiring */

el.chips.addEventListener("click", (event) => {
  const chip = event.target.closest(".chip");
  if (!chip || chip.classList.contains("chip--active")) return;

  el.customInput.value = "";
  setTarget({
    text: chip.dataset.text,
    reading: chip.dataset.reading,
    meaning: chip.dataset.meaning,
    chip,
  });
});

el.customForm.addEventListener("submit", (event) => {
  event.preventDefault();
  // Collapse the newlines a textarea allows; the API scores one sentence.
  const text = el.customInput.value.replace(/\s+/g, " ").trim();
  if (!text) {
    el.customInput.focus();
    return;
  }
  // No preset chip owns this text, so there is no reading or meaning to show.
  setTarget({ text });
});

// Enter submits, Shift+Enter keeps the newline.
el.customInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    el.customForm.requestSubmit();
  }
});

el.speak.addEventListener("click", () => {
  if (!window.speechSynthesis) return;
  const utterance = new SpeechSynthesisUtterance(target.text);
  utterance.lang = "ja-JP";
  utterance.rate = 0.85;
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(utterance);
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
  learnerContour = null;
  if (referencePattern) renderPitchChart();
});

// Release the mic if the user navigates away mid-recording.
window.addEventListener("pagehide", releaseResources);

el.timerMax.textContent = `tối đa ${formatDuration(MAX_DURATION_SECONDS)}`;
el.uploadHint.textContent =
  `Chỉ nhận file ${UPLOAD_EXTENSION}, tối đa ${formatDuration(MAX_UPLOAD_SECONDS)} — kéo thả vào đây cũng được`;
setStatus("idle");
