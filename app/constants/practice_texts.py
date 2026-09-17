"""Target sentences the learner reads aloud.

Hard-coded for the MVP — this is what an admin will eventually set.
Rendered into the page server-side, so there is no second copy in JS.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PracticeText:
    id: str
    #: Japanese text as shown to the learner.
    text: str
    #: Reading hint displayed under the text.
    reading: str
    #: Vietnamese meaning, to give the sentence context.
    meaning: str


PRACTICE_TEXTS: list[PracticeText] = [
    PracticeText(
        id="ganbarimasho",
        text="げんきょうもいちにちがんばりましょう",
        reading="genkyou mo ichinichi ganbarimashou",
        meaning="Hôm nay cũng cố gắng cả ngày nhé!",
    ),
    PracticeText(
        id="hajimemashite",
        text="はじめまして、よろしくおねがいします",
        reading="hajimemashite, yoroshiku onegaishimasu",
        meaning="Rất vui được gặp bạn, mong được giúp đỡ.",
    ),
    PracticeText(
        id="kyou-no-tenki",
        text="今日はとてもいい天気ですね",
        reading="kyou wa totemo ii tenki desu ne",
        meaning="Hôm nay thời tiết đẹp quá nhỉ.",
    ),
    PracticeText(
        id="arigatou",
        text="ご協力ありがとうございました",
        reading="gokyouryoku arigatou gozaimashita",
        meaning="Cảm ơn bạn đã hợp tác.",
    ),
]

DEFAULT_PRACTICE_TEXT = PRACTICE_TEXTS[0]
