# Limitations and Future Work

## 1. Dataset Limitations

### Sentiment
- **Size.** ~9,400 cleaned instances is small by modern standards. Three-class
  macro-F1 scores in the 0.55–0.72 range are typical for low-resource social-media
  sentiment at this scale; they should not be compared to English benchmarks.
- **Domain.** All data is Twitter/X; performance on other Amharic domains
  (reviews, news, customer support) is unknown.
- **Annotation quality.** We inherit AfriSenti's label quality; no re-annotation
  was performed.

### Summarization
- **500-pair evaluation set.** Confidence intervals are wide (ROUGE-L 95 % CI
  width ≈ 0.02–0.04). Differences between systems that fall within overlapping
  CIs should not be over-interpreted.
- **Single reference.** XL-Sum provides one summary per article. ROUGE and
  BLEU under-reward valid alternative phrasings. Multi-reference evaluation
  would be more informative but requires additional annotation.
- **BBC genre.** Results may not generalise to non-news domains or to
  informal/spoken Amharic.

## 2. Model Limitations

### Architecture size
All models are ≤ 560 M parameters. Larger models (BLOOM-3B, mT5-base,
XLM-RoBERTa-large) may shift the relative rankings, particularly for
summarization where model capacity matters more.

### mBERT summarization
mBERT is not a generative model. We construct a BERT2BERT
`EncoderDecoderModel` whose cross-attention is randomly initialised. This
places mBERT at a structural disadvantage for summarization; its scores
reflect the cost of absent generative pre-training rather than the quality
of the encoder representations.

### mT5 precision
mT5-small produces NaN losses under fp16 on NVIDIA T4 (no bf16 support).
Training in fp32 is slower and uses more VRAM than the other two models;
this is a hardware limitation, not a methodological choice.

### LoRA rank
We use a single LoRA rank (r = 8). Rank sensitivity analysis (r ∈ {2, 4, 8,
16, 32}) would clarify whether the architecture comparison is robust to this
hyperparameter.

## 3. Tokenisation

All three tokenizers over-segment Ge'ez script:

| Tokenizer | Approximate Amharic fertility (tokens/word) |
|---|---|
| mBERT WordPiece | 3–5 |
| mT5 SentencePiece | 4–6 |
| BLOOMZ BPE | 5–7 |

This means (a) the effective context window is 3–7× shorter for Amharic than
for English, and (b) evaluation metrics (ROUGE, BLEU) operate on units of
different granularity across languages. We mitigate (b) by computing
ROUGE/BLEU on whitespace-tokenized Ge'ez text (not on subword tokens), but
(a) remains an inherent disadvantage of current multilingual tokenizers.

## 4. Bias Evaluation

- **Lexicon validity.** The gendered/professional swap lists and cultural
  group terms were compiled by the authors without native-speaker review.
  Swapping nouns/pronouns without adjusting Amharic verb agreement (which
  marks gender) can produce ungrammatical sentences, adding noise to
  counterfactual metrics.
- **Probe coverage.** We probe three dimensions (gender/profession,
  position, culture) but omit others (age, disability, socio-economic
  status, urban/rural). The bias landscape is richer than our probes capture.
- **Composite metric.** "Bias robustness" is a heuristic average of three
  rates; it is not a validated psychometric or fairness metric.

## 5. Energy Measurement

- NVML measures GPU board power, not full-system power (CPU, RAM, storage,
  cooling, networking).
- CodeCarbon models CPU and RAM energy using TDP-based estimates, which can
  differ substantially from actual consumption.
- Neither method accounts for data-centre power usage effectiveness (PUE),
  typically 1.1–1.6× measured IT power.
- Cloud GPU instances may use different hardware revisions than reported;
  energy is hardware-dependent.

## 6. Reproducibility Caveats

- **Non-determinism.** Despite seeding and `torch.use_deterministic_algorithms`,
  some CUDA kernels (e.g. `atomicAdd` in flash attention) remain
  non-deterministic. Bit-for-bit reproducibility is not guaranteed across
  hardware generations.
- **HF dataset versioning.** If Hugging Face updates the AfriSenti or XL-Sum
  dataset, the downloaded data may differ. We record the HF revision SHA in
  `manifest.json`; future users should verify it matches.

## 7. Future Directions

1. **Larger models** (mT5-base, BLOOM-3B, XLM-R-large) and instruction-tuned
   variants (Flan-T5, BLOOMZ-7B1).
2. **Multi-seed runs** with mean ± std reporting (add seeds to `experiment.seeds`).
3. **Cross-lingual transfer:** evaluate Amharic models zero-shot on related
   Semitic/Ethiopic languages (Tigrinya, Oromo).
4. **Human evaluation** of summarization quality by native speakers.
5. **Native-speaker validation** of the bias lexicon and cultural probe templates.
6. **Rank sensitivity analysis** for LoRA (r ∈ {2, 4, 8, 16, 32}).
7. **Other Amharic tasks:** NER, machine translation, question answering.

