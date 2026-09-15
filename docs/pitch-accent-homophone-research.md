# ピッチアクセントによる同音異義語解消 — 関連研究サーベイ

## 背景

日本語ASRにおいて、ひらがな出力のCTCモデルは同音異義語を区別できない（橋/箸/端 → すべて「はし」）。一方で、Whisperなどのモデルはピッチアクセントの音響的違いを内部的に捉えている可能性がある。本サーベイでは、ピッチアクセント情報を活用した同音異義語解消に関連する研究を整理する。

---

## 1. SSLモデルはピッチ/韻律情報を保持しているか

### A Layer-wise Analysis of Mandarin and English Suprasegmentals in SSL Speech Models
- **著者:** Anton de la Fuente, Dan Jurafsky
- **年/会議:** 2024 / arXiv
- **URL:** https://arxiv.org/abs/2408.13678
- **要点:**
  - wav2vec2のプロービング分析で、**中間層（ネットワークの中央1/3）にレキシカルトーン・ストレス情報が最も強く表現**されることを確認
  - fine-tuningにより後半層で**語彙的に対立する韻律特徴**（トーン、ストレス）の表現が強化される
  - 日本語wav2vec2のピッチアクセント情報にも直接適用可能な知見

### Prosody Labeling with Phoneme-BERT and Speech Foundation Models
- **著者:** Tomoki Koriyama
- **年/会議:** 2025 / SSW13
- **URL:** https://arxiv.org/abs/2507.03912
- **要点:**
  - HuBERT/wav2vec2/WavLM/Whisperのencoderで日本語CSJの韻律ラベルを予測
  - アクセントラベル **89.8%**、高低アクセント **93.2%**、ブレークインデックス **94.3%** の精度
  - **明示的なF0抽出なしでもSSLモデルは豊かな韻律情報を保持**していることを実証

### What Can an Accent Identifier Learn? Probing Phonetic and Prosodic Information in a Wav2vec2-based Accent Identification Model
- **著者:** Mu Yang et al.
- **年/会議:** 2023 / Interspeech
- **URL:** https://arxiv.org/abs/2306.06524
- **要点:**
  - アクセント識別fine-tuningにより、wav2vec2の**上位2層**で音素・韻律表現がリッチになることを確認
  - wav2vec2の表現がアクセントに関連する韻律情報をエンコード可能であることを示す

---

## 2. ピッチアクセント + ASR マルチタスク学習

### Pitch Accent Detection improves Pretrained Automatic Speech Recognition
- **著者:** David Sasu, Natalie Schluter
- **年/会議:** 2025 / Interspeech
- **URL:** https://arxiv.org/abs/2508.04814
- **要点:**
  - wav2vec2ベースのASR + ピッチアクセント検出のjoint学習
  - ピッチアクセント検出のF1スコアが先行研究比 **41%改善**
  - ASRのWERが **28.3%改善**（LibriSpeech、限定リソース条件）
  - 英語での評価だが、手法は日本語ピッチアクセントに直接適用可能

### Building Tailored Speech Recognizers for Japanese Speaking Assessment
- **年/会議:** 2025 / arXiv
- **URL:** https://arxiv.org/abs/2509.20655
- **要点:**
  - 日本語CTC + ピッチアクセント推定 + トークンタイプ推定の**3並列マルチタスク**
  - カタカナモーラにアクセント記号を付与した **243トークン**で出力（例: アクセント核のあるモーラに `'` を付与）
  - **F0分類器（10カテゴリ）** の補助タスクがテキスト予測性能を向上
  - CTCの条件付き独立性を逆に活かし、発話誤り検出（ピッチアクセント誤り含む）に適用

---

## 3. CTC + 漢字出力の改善手法

### Relaxing the Conditional Independence Assumption of CTC-based ASR by Conditioning on Intermediate Predictions
- **著者:** Jumon Nozaki, Tatsuya Komatsu
- **年/会議:** 2021 / Interspeech
- **URL:** https://arxiv.org/abs/2104.02724
- **要点:**
  - **Self-Conditioned CTC**: 中間層のCTC予測を後段レイヤーの入力として条件付け
  - CTCの条件付き独立性の仮定を緩和し、**WER 20%以上改善**（WSJ）
  - autoregressiveモデル比 **30倍高速**なデコーディングを維持

### Alternate Intermediate Conditioning with Syllable-level and Character-level Targets for Japanese ASR
- **著者:** Yusuke Fujita, Tatsuya Komatsu, Yusuke Kida
- **年/会議:** 2022 / SLT (IEEE Spoken Language Technology Workshop)
- **URL:** https://arxiv.org/abs/2204.00175
- **要点:**
  - **日本語ASRにおける漢字CTCの核心的課題に直接対処**
  - Self-Conditioned CTCで**文字レベル（漢字）と音節レベル（かな）の中間予測を交互に条件付け**
  - 上位層が下位層の中間予測で条件付けされることで、文字と音節の相互依存を学習
  - CSJ（日本語自発音声コーパス）で従来のマルチタスク・Self-Conditioned CTC手法を**上回る性能**
  - **本プロジェクトのInterCTCアーキテクチャ（中間層phoneme + 最終層kana）と構造的に近い**

---

## 4. Whisper + ピッチアクセント

### Transcript-Prompted Whisper with Dictionary-Enhanced Decoding for Japanese Speech Annotation
- **著者:** Rui Hu, Xiaolong Lin, Jiawang Liu, Shixi Huang, Zhenpeng Zhan (Baidu)
- **年/会議:** 2025 / Interspeech
- **URL:** https://arxiv.org/html/2506.07646v1
- **要点:**
  - Whisperをfine-tuneし、音声から**フレーズレベルの表記 + 音素 + 韻律（ピッチアクセント）ラベル**を同時出力
  - 出力にカタカナ読み、**ピッチマーカー（上昇/下降）**、アクセント句境界、ポーズを含む
  - 辞書拡張デコーディング: MeCab/UniDicでセグメント化し、最短編集距離で辞書読みを選択
  - naive fine-tuningではWhisperの意味知識が失われ、同音異義語の区別能力が低下する問題にも対処

### AkitoP/whisper-large-v3-japanese-phone_accent（実用モデル）
- **著者:** AkitoP（コミュニティ）
- **年:** 2024-2025
- **URL:** https://huggingface.co/AkitoP/whisper-large-v3-japense-phone_accent
- **要点:**
  - Whisper-large-v3-turboを**カタカナ + ピッチアクセント注釈**出力にfine-tune
  - pyopenjtalk由来のラベルで学習（Galgame-Speech + JSUT-5000）
  - JSUT-5000テストで **CER ~4%**（pyopenjtalkテキストベースライン7%を上回る）
  - 音読み/訓読みの誤分類が主要エラー → 同音異義語問題と直接関連

---

## 5. 日本語ASRにおける同音異義語・表記揺れの問題

### Lenient Evaluation of Japanese Speech Recognition: Modeling Naturally Occurring Spelling Inconsistency
- **著者:** Shigeki Karita, Richard Sproat, Haruko Ishikawa
- **年/会議:** 2023 / ACL CAWL Workshop
- **URL:** https://aclanthology.org/2023.cawl-1.8/
- **要点:**
  - 日本語は「正書法がない」— ほとんどの語が複数表記可能（漢字/ひらがな/カタカナ）
  - 妥当な再表記の**ラティス**を構築し、寛容な評価を実現
  - 提案した表記バリエーションの **95.4%** が人間評価で妥当と判定
  - 寛容評価で **CER 2.4-3.1%** 絶対改善 → 従来「誤り」とされていたものの多くが妥当な別表記

### Pronunciation Ambiguities in Japanese Kanji
- **著者:** Wen Zhang
- **年/会議:** 2023 / ACL CAWL Workshop
- **URL:** https://aclanthology.org/2023.cawl-1.7/
- **要点:**
  - 日本語テキストのトークンの **10%以上** が同形異音語（複数読みを持つ漢字）
  - 単一漢字の同形異音語曖昧性解消の初のアノテーションデータセットを提供
  - ASR/TTSにおける漢字読み曖昧性問題の規模を定量化

### Disambiguating Homophones and Homographs Simultaneously: A Regrouping Method for Japanese
- **著者:** Yo Sato
- **年/会議:** 2024 / LREC-COLING
- **URL:** https://aclanthology.org/2024.lrec-main.442.pdf
- **要点:**
  - 同音異義語と同形異音語を同時に解消する「リグルーピング手法」を提案
  - BERT-base-Japanese + CSJコーパスで評価

---

## 6. LLMによるASR誤り訂正（後処理アプローチ）

### Benchmarking Japanese Speech Recognition on ASR-LLM Setups with Multi-Pass Augmented Generative Error Correction
- **著者:** Yuka Ko, Sheng Li, Chao-Han Huck Yang, Tatsuya Kawahara
- **年/会議:** 2024 / arXiv
- **URL:** https://arxiv.org/abs/2408.16180
- **要点:**
  - 日本語ASRの**GER（生成的誤り訂正）** ベンチマーク
  - 複数ASRシステムの仮説をLLMで統合・修正するマルチパスアプローチ
  - 医療テキストで同音異義語のrecallが **27.6% → 85.0%** に改善（合成データ + N-best仮説）
  - 漢字の同音異義語が多い日本語で特に有効

---

## 7. トーン言語向けの韻律活用ASR（応用可能な手法）

### CantoASR: Prosody-Aware ASR-LALM Collaboration for Low-Resource Cantonese
- **著者:** Dazhong Chen et al.
- **年/会議:** 2025 / arXiv
- **URL:** https://arxiv.org/abs/2511.04139
- **要点:**
  - **韻律キュー（F0、傾き、持続時間）** をLALMベースのASR誤り訂正に統合
  - LoRA fine-tuned Whisper（トーン識別）+ instruction-tuned Qwen-Audio（韻律対応訂正）
  - 広東語（6声調）対象だが、日本語ピッチアクセントへの手法転用が可能

### SITA: Learning Speaker-Invariant and Tone-Aware Speech Representations for Low-Resource Tonal Languages
- **年/会議:** 2026 / arXiv
- **URL:** https://arxiv.org/abs/2601.09050
- **要点:**
  - wav2vecスタイルのencoderに**tone-repulsive loss**を導入し、トーン崩壊を防止
  - 補助CTC + 蒸留でword+toneターゲットを学習
  - tone-repulsive lossの概念は日本語ピッチアクセントカテゴリに適応可能

---

## 8. 日本語CTC/ハイブリッドASRシステム

### Hybrid CTC/Attention Architecture for End-to-End Speech Recognition
- **著者:** Shinji Watanabe, Takaaki Hori et al.
- **年/会議:** 2017 / IEEE JSTSP
- **URL:** https://www.merl.com/publications/docs/TR2017-190.pdf
- **要点:**
  - ハイブリッドCTC/Attention アーキテクチャの基礎論文
  - 日本語自発音声に適用、**約3260文字語彙**（漢字+ひらがな+カタカナ）
  - 形態素解析器・発音辞書・言語モデルなしで従来のDNN/HMMに匹敵する性能

### NVIDIA Parakeet-TDT-CTC-0.6B-ja（実用モデル）
- **開発:** NVIDIA NeMo
- **年:** 2025
- **URL:** https://huggingface.co/nvidia/parakeet-tdt_ctc-0.6b-ja
- **要点:**
  - FastConformer TDT-CTCハイブリッド（約0.6Bパラメータ）
  - **ReazonSpeech v2.0（35,000時間以上）** で学習
  - SentencePieceトークナイザー **3072トークン**（サブワードレベル）
  - CTCベース日本語ASRの現時点での実用最高峰

---

## まとめ：研究ギャップと機会

### 確認された事実
| 知見 | 根拠 |
|------|------|
| wav2vec2/Whisperのencoderはピッチアクセント情報を保持 | de la Fuente 2024, Koriyama 2025 |
| ピッチアクセントとASRのjoint学習でWER改善 | Sasu 2025, arXiv:2509.20655 |
| Self-Conditioned CTCでかな⇔漢字の条件付けが有効 | Fujita 2022 |
| LLM後処理で同音異義語recall大幅改善 | Ko 2024 |

### 未踏の領域
**「ピッチアクセント情報をCTC内で活用して日本語の同音異義語を直接解消する」研究は存在しない。** 要素技術はすべて揃っており、以下の組み合わせが研究機会として残されている：

1. InterCTC（phoneme中間層）にピッチアクセント補助タスクを追加
2. Self-Conditioned CTCで漢字出力 + ピッチアクセント条件付け
3. CTC encoder のF0/ピッチ表現を明示的に強化するtone-aware loss

本プロジェクトのInterCTCアーキテクチャ（中間層phoneme + 最終層kana）は、Fujita et al. (2022) の交互条件付けと構造的に近く、ピッチアクセント補助タスクの追加は自然な拡張となる。
