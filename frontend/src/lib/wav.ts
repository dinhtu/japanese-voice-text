/**
 * Browser-side WAV encoding.
 *
 * MediaRecorder only produces compressed containers (webm/ogg), but the API
 * accepts .wav only. So we decode the recording with the Web Audio API,
 * downmix to mono, resample to the model's 16kHz, and write a RIFF header
 * ourselves — no external dependency needed.
 */

/** Sample rate the ASR model runs at. */
export const TARGET_SAMPLE_RATE = 16_000;

/**
 * Decode a recorded blob into mono PCM at {@link TARGET_SAMPLE_RATE}.
 */
async function toMonoPcm(blob: Blob): Promise<Float32Array> {
  const arrayBuffer = await blob.arrayBuffer();

  const decodeContext = new AudioContext();
  let decoded: AudioBuffer;
  try {
    decoded = await decodeContext.decodeAudioData(arrayBuffer);
  } finally {
    void decodeContext.close();
  }

  // Downmix + resample in one offline render pass.
  const frameCount = Math.max(
    1,
    Math.ceil((decoded.duration || 0) * TARGET_SAMPLE_RATE),
  );
  const offline = new OfflineAudioContext(1, frameCount, TARGET_SAMPLE_RATE);
  const source = offline.createBufferSource();
  source.buffer = decoded;
  source.connect(offline.destination);
  source.start();

  const rendered = await offline.startRendering();
  return rendered.getChannelData(0);
}

/**
 * Write 16-bit PCM samples into a RIFF/WAVE container.
 */
export function encodeWav(samples: Float32Array, sampleRate: number): Blob {
  const bytesPerSample = 2;
  const buffer = new ArrayBuffer(44 + samples.length * bytesPerSample);
  const view = new DataView(buffer);

  const writeString = (offset: number, value: string) => {
    for (let i = 0; i < value.length; i += 1) {
      view.setUint8(offset + i, value.charCodeAt(i));
    }
  };

  const dataSize = samples.length * bytesPerSample;

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

/**
 * Convert a recorded blob (any format the browser can decode) into a
 * 16kHz mono WAV file ready to upload.
 */
export async function blobToWav(blob: Blob): Promise<Blob> {
  const samples = await toMonoPcm(blob);
  return encodeWav(samples, TARGET_SAMPLE_RATE);
}
