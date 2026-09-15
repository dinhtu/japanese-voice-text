"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { blobToWav } from "@/lib/wav";

export type RecorderStatus = "idle" | "requesting" | "recording" | "processing";

const MAX_DURATION_SECONDS = 30;

interface UseAudioRecorderOptions {
  /** Called with the encoded 16kHz mono WAV once recording stops. */
  onComplete: (wav: Blob, durationSeconds: number) => void;
  onError: (message: string) => void;
}

/**
 * Microphone recording with a live input level and a WAV result.
 *
 * Stops automatically at {@link MAX_DURATION_SECONDS} so a forgotten
 * recording cannot grow past the API's upload limit.
 */
export function useAudioRecorder({ onComplete, onError }: UseAudioRecorderOptions) {
  const [status, setStatus] = useState<RecorderStatus>("idle");
  const [elapsed, setElapsed] = useState(0);
  /** Smoothed input level, 0-1, for the mic button animation. */
  const [level, setLevel] = useState(0);

  const recorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const frameRef = useRef<number | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const startedAtRef = useRef(0);

  const releaseResources = useCallback(() => {
    if (frameRef.current !== null) cancelAnimationFrame(frameRef.current);
    frameRef.current = null;

    if (timerRef.current !== null) clearInterval(timerRef.current);
    timerRef.current = null;

    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;

    void audioContextRef.current?.close().catch(() => undefined);
    audioContextRef.current = null;
    analyserRef.current = null;

    setLevel(0);
  }, []);

  // Make sure the mic is released if the user navigates away mid-recording.
  useEffect(() => releaseResources, [releaseResources]);

  const trackLevel = useCallback(() => {
    const analyser = analyserRef.current;
    if (!analyser) return;

    const samples = new Uint8Array(analyser.fftSize);
    const tick = () => {
      analyser.getByteTimeDomainData(samples);

      // RMS around the 128 midpoint of the unsigned byte waveform.
      let sum = 0;
      for (let i = 0; i < samples.length; i += 1) {
        const deviation = (samples[i] - 128) / 128;
        sum += deviation * deviation;
      }
      const rms = Math.sqrt(sum / samples.length);

      setLevel((previous) => {
        const scaled = Math.min(1, rms * 3.2);
        // Ease toward the new value so the ring pulses instead of flickering.
        return previous + (scaled - previous) * 0.35;
      });

      frameRef.current = requestAnimationFrame(tick);
    };
    frameRef.current = requestAnimationFrame(tick);
  }, []);

  const stop = useCallback(() => {
    const recorder = recorderRef.current;
    if (!recorder || recorder.state === "inactive") return;
    recorder.stop();
  }, []);

  const start = useCallback(async () => {
    if (status === "recording" || status === "requesting") return;

    if (typeof navigator === "undefined" || !navigator.mediaDevices?.getUserMedia) {
      onError("Trình duyệt không hỗ trợ ghi âm. Hãy dùng Chrome hoặc Edge (qua HTTPS hoặc localhost).");
      return;
    }

    setStatus("requesting");
    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
      });
    } catch {
      setStatus("idle");
      onError("Không truy cập được micro. Hãy cho phép quyền micro trong trình duyệt.");
      return;
    }

    streamRef.current = stream;

    const audioContext = new AudioContext();
    const analyser = audioContext.createAnalyser();
    analyser.fftSize = 1024;
    audioContext.createMediaStreamSource(stream).connect(analyser);
    audioContextRef.current = audioContext;
    analyserRef.current = analyser;

    const recorder = new MediaRecorder(stream);
    recorderRef.current = recorder;

    const chunks: Blob[] = [];
    recorder.ondataavailable = (event) => {
      if (event.data.size > 0) chunks.push(event.data);
    };

    recorder.onstop = async () => {
      const seconds = (Date.now() - startedAtRef.current) / 1000;
      releaseResources();
      setStatus("processing");

      try {
        const recorded = new Blob(chunks, { type: recorder.mimeType || "audio/webm" });
        if (recorded.size === 0) throw new Error("empty recording");
        onComplete(await blobToWav(recorded), seconds);
      } catch {
        onError("Không xử lý được bản ghi âm. Hãy thử ghi lại.");
      } finally {
        setStatus("idle");
      }
    };

    startedAtRef.current = Date.now();
    setElapsed(0);
    recorder.start();
    setStatus("recording");
    trackLevel();

    timerRef.current = setInterval(() => {
      const seconds = (Date.now() - startedAtRef.current) / 1000;
      setElapsed(seconds);
      if (seconds >= MAX_DURATION_SECONDS) stop();
    }, 100);
  }, [onComplete, onError, releaseResources, status, stop, trackLevel]);

  const toggle = useCallback(() => {
    if (status === "recording") stop();
    else void start();
  }, [start, status, stop]);

  return {
    status,
    elapsed,
    level,
    isRecording: status === "recording",
    isBusy: status === "requesting" || status === "processing",
    maxDuration: MAX_DURATION_SECONDS,
    start,
    stop,
    toggle,
  };
}
