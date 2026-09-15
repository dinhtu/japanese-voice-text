"use client";

import { Check, Volume2 } from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { PRACTICE_TEXTS, type PracticeText } from "@/constants/practice-texts";
import { cn } from "@/lib/utils";

interface TargetTextCardProps {
  selected: PracticeText;
  onSelect: (text: PracticeText) => void;
  disabled?: boolean;
}

/**
 * Shows the sentence to read aloud, plus a picker for the other
 * hard-coded practice sentences.
 */
export function TargetTextCard({ selected, onSelect, disabled }: TargetTextCardProps) {
  const speak = () => {
    if (typeof window === "undefined" || !window.speechSynthesis) return;
    const utterance = new SpeechSynthesisUtterance(selected.text);
    utterance.lang = "ja-JP";
    utterance.rate = 0.85;
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(utterance);
  };

  return (
    <Card>
      <CardHeader className="flex items-center justify-between gap-3">
        <CardTitle className="flex items-center gap-2">
          <span className="inline-flex size-6 items-center justify-center rounded-lg bg-brand-100 text-[11px] font-bold text-brand-700">
            1
          </span>
          Câu cần đọc
        </CardTitle>
        <button
          type="button"
          onClick={speak}
          className="inline-flex items-center gap-1.5 rounded-lg px-2 py-1 text-xs font-medium text-brand-700 transition-colors hover:bg-brand-50"
          title="Nghe mẫu (giọng đọc của trình duyệt)"
        >
          <Volume2 className="size-3.5" />
          Nghe mẫu
        </button>
      </CardHeader>

      <CardContent className="pt-4">
        <p className="font-jp text-2xl leading-relaxed tracking-wide text-ink sm:text-3xl sm:leading-relaxed">
          {selected.text}
        </p>
        <p className="mt-3 text-sm text-ink-muted">{selected.reading}</p>
        <p className="mt-1 text-sm text-ink-muted/80">{selected.meaning}</p>

        <div className="mt-5 border-t border-line/70 pt-4">
          <p className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-ink-muted">
            Chọn câu khác
          </p>
          <div className="flex flex-wrap gap-2">
            {PRACTICE_TEXTS.map((item) => {
              const isActive = item.id === selected.id;
              return (
                <button
                  key={item.id}
                  type="button"
                  disabled={disabled}
                  onClick={() => onSelect(item)}
                  className={cn(
                    "inline-flex max-w-full items-center gap-1.5 rounded-full border px-3 py-1.5 text-xs transition-colors disabled:cursor-not-allowed disabled:opacity-50",
                    isActive
                      ? "border-brand-600 bg-brand-600 text-white"
                      : "border-line bg-surface text-ink-muted hover:border-brand-300 hover:text-brand-700",
                  )}
                >
                  {isActive && <Check className="size-3" />}
                  <span className="font-jp truncate">{item.text}</span>
                </button>
              );
            })}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
