"use client";

import { Loader2, Mic, RotateCcw, Square } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { RecorderStatus } from "@/hooks/use-audio-recorder";
import { cn, formatDuration } from "@/lib/utils";

interface RecorderPanelProps {
  status: RecorderStatus;
  isRecording: boolean;
  isBusy: boolean;
  isEvaluating: boolean;
  elapsed: number;
  maxDuration: number;
  /** Live mic level, 0-1. */
  level: number;
  /** Object URL of the last recording, if any. */
  audioUrl: string | null;
  onToggle: () => void;
  onReset: () => void;
}

const STATUS_LABEL: Record<RecorderStatus, string> = {
  idle: "Nhấn vào micro để bắt đầu ghi âm",
  requesting: "Đang xin quyền micro…",
  recording: "Đang ghi âm — nhấn lại để dừng",
  processing: "Đang xử lý bản ghi…",
};

/**
 * The mic button: one click records, another stops and produces the WAV.
 */
export function RecorderPanel({
  status,
  isRecording,
  isBusy,
  isEvaluating,
  elapsed,
  maxDuration,
  level,
  audioUrl,
  onToggle,
  onReset,
}: RecorderPanelProps) {
  const disabled = isBusy || isEvaluating;
  const progress = Math.min(1, elapsed / maxDuration);

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <span className="inline-flex size-6 items-center justify-center rounded-lg bg-brand-100 text-[11px] font-bold text-brand-700">
            2
          </span>
          Đọc to và ghi âm
        </CardTitle>
      </CardHeader>

      <CardContent className="flex flex-col items-center pt-2">
        <div className="relative flex size-32 items-center justify-center sm:size-36">
          {isRecording && (
            <>
              <span className="absolute inset-0 rounded-full bg-brand-400/30 animate-pulse-ring" />
              {/* Ring that swells with the actual mic level. */}
              <span
                className="absolute inset-0 rounded-full bg-brand-500/20 transition-transform duration-75"
                style={{ transform: `scale(${1 + level * 0.45})` }}
              />
            </>
          )}

          <button
            type="button"
            onClick={onToggle}
            disabled={disabled}
            aria-label={isRecording ? "Dừng ghi âm" : "Bắt đầu ghi âm"}
            aria-pressed={isRecording}
            className={cn(
              "relative z-10 flex size-24 items-center justify-center rounded-full text-white shadow-lg transition-all sm:size-28",
              "disabled:cursor-not-allowed disabled:opacity-60",
              isRecording
                ? "bg-rose-600 shadow-rose-600/30 hover:bg-rose-700"
                : "bg-brand-600 shadow-brand-600/30 hover:bg-brand-700 hover:shadow-xl active:scale-95",
            )}
          >
            {isBusy || isEvaluating ? (
              <Loader2 className="size-9 animate-spin" />
            ) : isRecording ? (
              <Square className="size-8 fill-current" />
            ) : (
              <Mic className="size-10" />
            )}
          </button>
        </div>

        <p
          className="mt-5 text-center text-sm text-ink-muted"
          role="status"
          aria-live="polite"
        >
          {isEvaluating ? "Đang chấm điểm…" : STATUS_LABEL[status]}
        </p>

        {isRecording && (
          <div className="mt-4 w-full max-w-xs">
            <div className="flex items-center justify-between text-xs font-medium tabular-nums text-ink-muted">
              <span className="text-rose-600">● {formatDuration(elapsed)}</span>
              <span>tối đa {formatDuration(maxDuration)}</span>
            </div>
            <div className="mt-1.5 h-1 overflow-hidden rounded-full bg-line">
              <div
                className="h-full rounded-full bg-rose-500 transition-[width] duration-100"
                style={{ width: `${progress * 100}%` }}
              />
            </div>
          </div>
        )}

        {audioUrl && !isRecording && (
          <div className="mt-5 flex w-full max-w-sm flex-col items-center gap-3">
            <audio controls src={audioUrl} className="h-10 w-full" />
            <Button variant="ghost" size="sm" onClick={onReset} disabled={disabled}>
              <RotateCcw />
              Ghi lại từ đầu
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
