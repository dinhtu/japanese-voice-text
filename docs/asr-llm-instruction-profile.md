# ASR Profile For LLM Instruction (Compressed)

Source: `docs/report-h100-medium-training.md` (H100 medium training, ep5)

## Reliability

- Domain shift is significant:
  - JSUT KER 7.47%
  - JVS KER 15.68%
  - ReazonSpeech KER 21.65%
- Noise/wild speech is much less reliable than clean studio speech.

## Typical Kana Errors

- Long-vowel mark `ー` is unstable: often confused with `ん / い / あ / え`.
- Vowel swaps are common: `い <-> え`.
- Small-kana related errors appear frequently.

## Typical Phoneme Errors

- Voiceless/regular vowel confusions: `U <-> u`, `I <-> i`.
- Frequent substitutions: `u -> i`, `d -> t`.
- Palatalized drop: `my/by/gy -> m/b/g`.

## Practical Limits

- Weak regions: long vowels, small kana, glides/palatalized sounds, noisy speech, loanwords.
- Low-confidence spans correlate strongly with recognition errors.

## Instruction Hint For LLM

- Treat low-confidence tokens as uncertain observations, not facts.
- Prefer short clarification questions when confidence is low.
- Use conservative intent inference; avoid over-committing to specific entities.
