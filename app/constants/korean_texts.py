"""Korean practice sentences for /ko."""
from dataclasses import dataclass


@dataclass(frozen=True)
class PracticeText:
    id: str
    text: str
    reading: str
    meaning: str


PRACTICE_TEXTS = [
    PracticeText("hello", "안녕하세요, 만나서 반가워요.", "안녕하세요, 만나서 반가워요", "Xin chào, rất vui được gặp bạn."),
    PracticeText("thanks", "도와주셔서 정말 감사합니다.", "도와주셔서 정말 감사합니다", "Thật sự cảm ơn bạn rất nhiều vì đã giúp đỡ."),
    PracticeText("weather", "오늘 날씨가 정말 맑고 좋아요.", "오늘 날씨가 정말 맑고 좋아요", "Hôm nay thời tiết thật trong lành và đẹp."),
    PracticeText("study", "저는 한국어를 열심히 공부하고 있어요.", "저는 한국어를 열심히 공부하고 있어요", "Tôi đang chăm chỉ học tiếng Hàn."),
]

DEFAULT_PRACTICE_TEXT = PRACTICE_TEXTS[0]
