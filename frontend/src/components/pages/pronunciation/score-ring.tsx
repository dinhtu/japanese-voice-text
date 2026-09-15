"use client";

import type { FeedbackLevel } from "@/types/pronunciation";

const LEVEL_COLOR: Record<FeedbackLevel, string> = {
  excellent: "var(--color-score-excellent)",
  good: "var(--color-score-good)",
  fair: "var(--color-score-fair)",
  poor: "var(--color-score-poor)",
  mismatch: "var(--color-score-poor)",
};

interface ScoreRingProps {
  score: number;
  level: FeedbackLevel;
}

/** Circular score gauge, 0-100. */
export function ScoreRing({ score, level }: ScoreRingProps) {
  const radius = 52;
  const circumference = 2 * Math.PI * radius;
  const filled = (Math.min(100, Math.max(0, score)) / 100) * circumference;
  const color = LEVEL_COLOR[level];

  return (
    <div className="relative size-32 shrink-0 sm:size-36">
      <svg viewBox="0 0 120 120" className="size-full -rotate-90">
        <circle
          cx="60"
          cy="60"
          r={radius}
          fill="none"
          stroke="var(--color-line)"
          strokeWidth="10"
        />
        <circle
          cx="60"
          cy="60"
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth="10"
          strokeLinecap="round"
          strokeDasharray={`${filled} ${circumference}`}
          style={{ transition: "stroke-dasharray 0.8s cubic-bezier(0.22, 1, 0.36, 1)" }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span
          className="text-4xl font-bold tabular-nums leading-none sm:text-5xl"
          style={{ color }}
        >
          {score}
        </span>
        <span className="mt-1 text-[11px] font-medium uppercase tracking-wider text-ink-muted">
          / 100
        </span>
      </div>
    </div>
  );
}
