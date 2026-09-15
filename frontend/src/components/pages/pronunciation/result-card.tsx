"use client";

import { AlertTriangle, CheckCircle2, Clock, Gauge, Waves } from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import type { EvaluateResponse, FeedbackLevel } from "@/types/pronunciation";

import { ScoreRing } from "./score-ring";

const LEVEL_BADGE: Record<FeedbackLevel, { label: string; className: string }> = {
  excellent: { label: "Xuất sắc", className: "bg-emerald-50 text-emerald-700 border-emerald-200" },
  good: { label: "Tốt", className: "bg-brand-50 text-brand-700 border-brand-200" },
  fair: { label: "Khá", className: "bg-amber-50 text-amber-700 border-amber-200" },
  poor: { label: "Cần luyện thêm", className: "bg-rose-50 text-rose-700 border-rose-200" },
  mismatch: { label: "Không khớp", className: "bg-rose-50 text-rose-700 border-rose-200" },
};

/** Positions in the target reading that the model did not hear correctly. */
function wrongPositions(result: EvaluateResponse): Set<number> {
  const positions = new Set<number>();
  for (const error of result.errors) {
    if (error.type !== "ins") positions.add(error.position);
  }
  return positions;
}

function KanaDiff({ reading, wrong }: { reading: string; wrong: Set<number> }) {
  return (
    <p className="font-jp text-xl leading-loose tracking-wide sm:text-2xl">
      {[...reading].map((char, index) => (
        <span
          key={`${char}-${index}`}
          className={cn(
            "rounded px-px",
            wrong.has(index) ? "bg-rose-100 text-rose-700" : "text-ink",
          )}
        >
          {char}
        </span>
      ))}
    </p>
  );
}

function Metric({
  icon: Icon,
  label,
  value,
}: {
  icon: typeof Gauge;
  label: string;
  value: string;
}) {
  return (
    <div className="rounded-xl border border-line/70 bg-canvas/60 px-3 py-2.5">
      <p className="flex items-center gap-1.5 text-[11px] font-medium uppercase tracking-wider text-ink-muted">
        <Icon className="size-3.5" />
        {label}
      </p>
      <p className="mt-1 text-sm font-semibold tabular-nums text-ink">{value}</p>
    </div>
  );
}

export function ResultCard({ result }: { result: EvaluateResponse }) {
  const badge = LEVEL_BADGE[result.feedback.level];
  const wrong = wrongPositions(result);
  const isPerfect = result.errors.length === 0;

  return (
    <Card className="animate-rise">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <span className="inline-flex size-6 items-center justify-center rounded-lg bg-brand-100 text-[11px] font-bold text-brand-700">
            3
          </span>
          Kết quả đánh giá
        </CardTitle>
      </CardHeader>

      <CardContent className="pt-4">
        <div className="flex flex-col items-center gap-5 sm:flex-row sm:gap-7">
          <ScoreRing score={result.score} level={result.feedback.level} />

          <div className="min-w-0 flex-1 text-center sm:text-left">
            <span
              className={cn(
                "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-semibold",
                badge.className,
              )}
            >
              {isPerfect ? (
                <CheckCircle2 className="size-3.5" />
              ) : (
                <AlertTriangle className="size-3.5" />
              )}
              {badge.label}
            </span>
            <p className="mt-3 text-sm leading-relaxed text-ink-muted">
              {result.feedback.message}
            </p>
            <div className="mt-4 grid grid-cols-2 gap-2 sm:grid-cols-3">
              <Metric icon={Gauge} label="CER" value={result.cer.toFixed(3)} />
              <Metric icon={Waves} label="Sai lệch" value={`${result.distance} ký tự`} />
              <Metric
                icon={Clock}
                label="Thời lượng"
                value={`${result.audio_duration.toFixed(2)}s`}
              />
            </div>
          </div>
        </div>

        <div className="mt-6 grid gap-4 border-t border-line/70 pt-5 sm:grid-cols-2">
          <div>
            <p className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-ink-muted">
              Mục tiêu (cách đọc)
            </p>
            <KanaDiff reading={result.target_hiragana} wrong={wrong} />
          </div>
          <div>
            <p className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-ink-muted">
              Bạn đã đọc
            </p>
            <p className="font-jp text-xl leading-loose tracking-wide text-brand-700 sm:text-2xl">
              {result.recognized_hiragana || "—"}
            </p>
          </div>
        </div>

        {result.errors.length > 0 && (
          <div className="mt-5 border-t border-line/70 pt-5">
            <p className="mb-3 text-[11px] font-semibold uppercase tracking-wider text-ink-muted">
              Chi tiết {result.errors.length} điểm lệch
            </p>
            <ul className="flex flex-wrap gap-2">
              {result.errors.slice(0, 12).map((error, index) => (
                <li
                  key={`${error.position}-${index}`}
                  className="flex items-center gap-1.5 rounded-lg border border-line bg-canvas/60 px-2.5 py-1.5 text-sm"
                >
                  <span className="font-jp font-medium text-ink">{error.target || "∅"}</span>
                  <span className="text-ink-muted">→</span>
                  <span className="font-jp font-medium text-rose-600">
                    {error.recognized || "∅"}
                  </span>
                  <span className="text-[10px] tabular-nums text-ink-muted">
                    #{error.position}
                  </span>
                </li>
              ))}
              {result.errors.length > 12 && (
                <li className="self-center text-xs text-ink-muted">
                  +{result.errors.length - 12} nữa
                </li>
              )}
            </ul>
          </div>
        )}

        <p className="mt-5 rounded-xl bg-brand-50/70 px-3.5 py-2.5 text-xs leading-relaxed text-brand-900/70">
          Điểm số phản ánh mức khớp giữa kết quả nhận dạng và câu mục tiêu, chưa phải đánh
          giá ngữ âm thật sự (trọng âm, ngữ điệu). Tiếng ồn hoặc tốc độ đọc bất thường cũng
          có thể làm giảm điểm.
        </p>
      </CardContent>
    </Card>
  );
}
