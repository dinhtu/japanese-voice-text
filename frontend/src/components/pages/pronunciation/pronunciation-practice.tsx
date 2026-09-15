"use client";

import { useCallback, useEffect, useState } from "react";
import { AlertCircle, Languages } from "lucide-react";

import { useAudioRecorder } from "@/hooks/use-audio-recorder";
import { evaluatePronunciation } from "@/lib/api/pronunciation";
import { DEFAULT_PRACTICE_TEXT, type PracticeText } from "@/constants/practice-texts";
import type { EvaluateResponse } from "@/types/pronunciation";

import { RecorderPanel } from "./recorder-panel";
import { ResultCard } from "./result-card";
import { TargetTextCard } from "./target-text-card";

/**
 * Read-aloud practice screen: pick a sentence, record it, get a score.
 */
export function PronunciationPractice() {
  const [target, setTarget] = useState<PracticeText>(DEFAULT_PRACTICE_TEXT);
  const [result, setResult] = useState<EvaluateResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);

  const revokeAudioUrl = useCallback(() => {
    setAudioUrl((previous) => {
      if (previous) URL.revokeObjectURL(previous);
      return null;
    });
  }, []);

  useEffect(() => revokeAudioUrl, [revokeAudioUrl]);

  const handleRecordingComplete = useCallback(
    async (wav: Blob) => {
      revokeAudioUrl();
      setAudioUrl(URL.createObjectURL(wav));
      setResult(null);
      setError(null);
      setIsEvaluating(true);

      try {
        setResult(await evaluatePronunciation(target.text, wav));
      } catch (caught) {
        setError(caught instanceof Error ? caught.message : "Đã có lỗi xảy ra.");
      } finally {
        setIsEvaluating(false);
      }
    },
    [revokeAudioUrl, target.text],
  );

  const handleRecorderError = useCallback((message: string) => {
    setError(message);
    setResult(null);
  }, []);

  const recorder = useAudioRecorder({
    onComplete: handleRecordingComplete,
    onError: handleRecorderError,
  });

  const handleSelect = useCallback(
    (next: PracticeText) => {
      setTarget(next);
      setResult(null);
      setError(null);
      revokeAudioUrl();
    },
    [revokeAudioUrl],
  );

  const handleReset = useCallback(() => {
    setResult(null);
    setError(null);
    revokeAudioUrl();
  }, [revokeAudioUrl]);

  return (
    <main className="mx-auto w-full max-w-3xl px-4 py-8 sm:px-6 sm:py-12">
      <header className="mb-8 text-center sm:mb-10">
        <span className="inline-flex items-center gap-2 rounded-full border border-brand-200 bg-surface/80 px-3 py-1 text-xs font-medium text-brand-700 shadow-sm">
          <Languages className="size-3.5" />
          hiragana ASR · wav2vec2
        </span>
        <h1 className="mt-4 text-2xl font-bold tracking-tight text-ink sm:text-3xl">
          Luyện phát âm tiếng Nhật
        </h1>
        <p className="mx-auto mt-2 max-w-md text-sm leading-relaxed text-ink-muted">
          Đọc to câu bên dưới, hệ thống sẽ nhận dạng giọng nói và chấm điểm mức
          khớp với câu mục tiêu.
        </p>
      </header>

      <div className="space-y-4 sm:space-y-5">
        <TargetTextCard
          selected={target}
          onSelect={handleSelect}
          disabled={recorder.isRecording || recorder.isBusy || isEvaluating}
        />

        <RecorderPanel
          status={recorder.status}
          isRecording={recorder.isRecording}
          isBusy={recorder.isBusy}
          isEvaluating={isEvaluating}
          elapsed={recorder.elapsed}
          maxDuration={recorder.maxDuration}
          level={recorder.level}
          audioUrl={audioUrl}
          onToggle={recorder.toggle}
          onReset={handleReset}
        />

        {error && (
          <div
            role="alert"
            className="animate-rise flex items-start gap-2.5 rounded-2xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-800"
          >
            <AlertCircle className="mt-0.5 size-4 shrink-0" />
            <p className="leading-relaxed">{error}</p>
          </div>
        )}

        {result && <ResultCard result={result} />}
      </div>

      <footer className="mt-10 text-center text-xs text-ink-muted/70">
        Ghi âm chỉ được gửi tới API cục bộ để chấm điểm, không lưu lại trên máy chủ.
      </footer>
    </main>
  );
}
