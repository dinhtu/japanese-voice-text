/**
 * Target sentences the learner reads aloud.
 *
 * Hard-coded for the MVP — this is what an admin will eventually set.
 */

export interface PracticeText {
  id: string;
  /** Japanese text as shown to the learner. */
  text: string;
  /** Reading hint displayed under the text. */
  reading: string;
  /** Vietnamese meaning, to give the sentence context. */
  meaning: string;
}

export const PRACTICE_TEXTS: PracticeText[] = [
  {
    id: "ganbarimasho",
    text: "げんきょうもいちにちがんばりましょう",
    reading: "genkyou mo ichinichi ganbarimashou",
    meaning: "Hôm nay cũng cố gắng cả ngày nhé!",
  },
  {
    id: "hajimemashite",
    text: "はじめまして、よろしくおねがいします",
    reading: "hajimemashite, yoroshiku onegaishimasu",
    meaning: "Rất vui được gặp bạn, mong được giúp đỡ.",
  },
  {
    id: "kyou-no-tenki",
    text: "今日はとてもいい天気ですね",
    reading: "kyou wa totemo ii tenki desu ne",
    meaning: "Hôm nay thời tiết đẹp quá nhỉ.",
  },
  {
    id: "arigatou",
    text: "ご協力ありがとうございました",
    reading: "gokyouryoku arigatou gozaimashita",
    meaning: "Cảm ơn bạn đã hợp tác.",
  },
];

export const DEFAULT_PRACTICE_TEXT = PRACTICE_TEXTS[0];
