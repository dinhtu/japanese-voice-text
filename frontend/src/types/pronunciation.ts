/**
 * Mirrors the FastAPI response schema in
 * `app/interface/schemas/pronunciation.py`.
 */

export type FeedbackLevel =
  | "excellent"
  | "good"
  | "fair"
  | "poor"
  | "mismatch";

export interface Feedback {
  level: FeedbackLevel;
  message: string;
}

export type PronunciationErrorType = "sub" | "del" | "ins";

export interface PronunciationErrorItem {
  type: PronunciationErrorType;
  /** Expected character (empty for an insertion). */
  target: string;
  /** What the model heard (empty for a deletion). */
  recognized: string;
  /** Index in the normalized target reading. */
  position: number;
}

export interface EvaluateResponse {
  success: boolean;
  target_text: string;
  target_hiragana: string;
  recognized_text: string;
  recognized_hiragana: string;
  score: number;
  cer: number;
  distance: number;
  audio_duration: number;
  inference_ms: number;
  feedback: Feedback;
  errors: PronunciationErrorItem[];
}
