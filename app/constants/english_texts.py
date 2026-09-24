"""English practice sentences for the /en page."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PracticeText:
    id: str
    text: str
    reading: str
    meaning: str


PRACTICE_TEXTS: list[PracticeText] = [
    PracticeText(
        id="nice-to-meet",
        text="Nice to meet you. How are you today?",
        reading="naɪs tə miːt ju",
        meaning="Rất vui được gặp bạn. Hôm nay bạn thế nào?",
    ),
    PracticeText(
        id="weather",
        text="It's a beautiful day. Let's go for a walk.",
        reading="ɪts ə ˈbjuːtɪfl deɪ",
        meaning="Hôm nay trời đẹp. Chúng ta đi dạo nhé.",
    ),
    PracticeText(
        id="thank-you",
        text="Thank you so much for your help.",
        reading="θæŋk ju soʊ mʌtʃ",
        meaning="Cảm ơn bạn rất nhiều vì đã giúp đỡ.",
    ),
    PracticeText(
        id="introduce",
        text="My name is Anna. I work as a teacher.",
        reading="maɪ neɪm ɪz ˈænə",
        meaning="Tôi tên Anna. Tôi làm giáo viên.",
    ),
]

DEFAULT_PRACTICE_TEXT = PRACTICE_TEXTS[0]
