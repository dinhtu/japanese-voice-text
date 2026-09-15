# Related Work Survey (2024--2026)

Lightweight Japanese ASR with phoneme output for real-time voice dialogue に関する類似研究調査。

## Documents

| File | Content |
|------|---------|
| [01_japanese_asr_models.md](01_japanese_asr_models.md) | Japanese ASR models and datasets |
| [02_ctc_advances.md](02_ctc_advances.md) | CTC loss improvements and decoding |
| [03_phoneme_recognition.md](03_phoneme_recognition.md) | Phoneme-level recognition research |
| [04_voice_dialogue.md](04_voice_dialogue.md) | Real-time voice dialogue and AVSR |
| [05_edge_deployment.md](05_edge_deployment.md) | Edge / on-device deployment |
| [06_implications.md](06_implications.md) | Implications for this project |

## Survey Date

2026-02-20 (last updated)

## Key Conclusion

最終出力: ひらがな CTC (語彙的補正ゼロ、LLM が直接理解)
中間出力: 音素 InterCTC (低レベル正則化、vsr-test 共有)
→ Apple Diverse Modeling Units (Interspeech 2024) の日本語適用
