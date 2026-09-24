"""Mandarin practice sentences for /zh."""
from dataclasses import dataclass


@dataclass(frozen=True)
class PracticeText:
    id: str
    text: str
    reading: str
    meaning: str

PRACTICE_TEXTS = [
    PracticeText("hello", "你好，很高兴认识你。", "nǐ hǎo, hěn gāo xìng rèn shi nǐ", "Xin chào, rất vui được gặp bạn."),
    PracticeText("thanks", "谢谢你的帮助。", "xiè xie nǐ de bāng zhù", "Cảm ơn bạn đã giúp đỡ."),
    PracticeText("today", "今天天气很好。", "jīn tiān tiān qì hěn hǎo", "Hôm nay thời tiết rất đẹp."),
    PracticeText("name", "我叫安娜，我是老师。", "wǒ jiào ān nà, wǒ shì lǎo shī", "Tôi tên Anna, tôi là giáo viên."),
]
DEFAULT_PRACTICE_TEXT = PRACTICE_TEXTS[0]
