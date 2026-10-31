# Methodology

## 1. Research Design

This study is a **controlled, single-factor comparison**: three transformer
architecture families (encoder-only, encoder-decoder, decoder-only) are
evaluated on two Amharic NLP tasks under matched training budgets, LoRA
configurations, and preprocessing pipelines. The independent variable is the
architecture; all other factors (data, hyperparameters, evaluation protocol)
are held constant or adapted minimally where architectural constraints require
it (documented as deviations below).

## 2. Data Collection and Preprocessing

### 2.1 Sentiment

We use the **AfriSenti** Amharic subset (`shmuhammad/AfriSenti-twitter-sentiment`,
config `amh`), a peer-reviewed benchmark (Muhammad et al., EMNLP 2023). We pool
the official train/validation/test splits and re-split 80/20 (stratified) to
maximise training data while maintaining a clean held-out evaluation set. The
official splits are preserved as a column (`orig_split`) for future analysis.

### 2.2 Summarization

We use the **XL-Sum** Amharic subset (`csebuetnlp/xlsum`, config `amharic`).
The "amharic-eval-500" evaluation set is a seeded 500-pair sample from the
official test split. The training set uses the official train + validation
splits with any article/summary overlap against eval-500 removed.

**Deviation from spec:** The spec suggests constructing a dataset from
`dagn/expanded-amharic-news-dataset` (article + headline). We use XL-Sum
instead because it provides professionally written abstractive summaries
(not just headlines), is peer-reviewed, and is already split into
train/validation/test. This is documented in `data/README.md`.

### 2.3 Preprocessing Pipeline

Identical for all models (spec §2):

1. **Unicode NFC normalisation.** Ensures composed Ge'ez characters.
2. **URL removal** (`https?://...`, `www.…`).
3. **@mention removal** (sentiment task).
4. **HTML entity decoding** (`&amp;` → `&`).
5. **Whitespace standardisation** (zero-width spaces, NBSP, multiple spaces).
6. **Deduplication** on a normalised key (lowercased, no punctuation/digits/spaces).
7. **Malformation filter:** < 5 characters or < 30 % Ethiopic characters.
8. **Conflicting-label filter** (sentiment): texts that map to different labels
   under the normalised key are dropped entirely.
9. **Cross-split leakage verification** (assert zero overlap on the normalised key).

## 3. Model Selection and Configuration

| Arch. | Model | HF ID | Params | Tokenizer | Why selected |
|---|---|---|---|---|---|
| Encoder-only | mBERT-Amharic | `Davlan/bert-base-multilingual-cased-finetuned-amharic` | ~110 M | WordPiece | Strongest public multilingual BERT with Amharic-specific continued pre-training. |
| Enc-Dec | mT5-small | `google/mt5-small` | ~300 M | SentencePiece | Canonical multilingual seq2seq; 101 languages including Amharic. |
| Dec-only | BLOOMZ-560M | `bigscience/bloomz-560m` | ~560 M | Byte-level BPE | Multilingual autoregressive; 46+ languages, instruction-tuned. |

**Deviation:** The spec names "mBERT-Amharic (~110 M)". Since the vanilla
`bert-base-multilingual-cased` was not pre-trained on Amharic, we use
Davlan's community checkpoint, which replaces the vocabulary and continues
pre-training on CC-100 Amharic. Parameter count is identical (~110 M).

### 3.1 LoRA Configuration

Applied identically to every condition except `probe` (head-only baseline):

- **Rank (r):** 8
- **Alpha (α):** 16 (effective scaling α/r = 2)
- **Dropout:** 0.1
- **Target modules:** attention projections (architecture-specific names)
- **Modules to save:** task head (always trained in full)
- **Trainable %:** < 1 % of total parameters

### 3.2 Summarization with mBERT

BERT is not a generative model. We construct a **BERT2BERT
`EncoderDecoderModel`** warm-started from the same Amharic-adapted checkpoint.
Cross-attention layers are randomly initialised and trained in full (they are
not LoRA targets because LoRA on random weights would be meaningless). This is
an architectural disadvantage documented in the results.

## 4. Training Protocol

| Hyper-parameter | Sentiment | Summarization |
|---|---|---|
| Learning rate | 2 × 10⁻⁴ (LoRA) / 1 × 10⁻³ (probe) | 5 × 10⁻⁴ |
| Effective batch size | 32 | 32 |
| Epochs | 5 | 3 |
| Warmup | 10 % | 5 % |
| Weight decay | 0.01 | 0.01 |
| Precision | fp16 (mBERT, BLOOMZ) / fp32 (mT5) | same |
| Gradient clipping | 1.0 | 1.0 |
| Optimiser | AdamW | AdamW |
| Schedule | linear with warmup | linear with warmup |

mT5 is trained in fp32 because it exhibits NaN loss under fp16 on T4 (no bf16
support). This is a known issue documented in the Hugging Face model card.

## 5. Evaluation

### 5.1 Sentiment Metrics

- **Accuracy** and **macro-F1** over 3 classes.
- **Per-class precision, recall, F1** and confusion matrix.
- **Expected Calibration Error (ECE):** top-label, 15 equal-width bins (Guo et al., 2017).
- **Brier score** as a proper scoring rule complement.

### 5.2 Summarization Metrics

- **Amharic-aware ROUGE-1/2/L F1:** whitespace tokenization with Ethiopic
  punctuation stripping (the reference `rouge_score` package deletes Ge'ez).
- **sacreBLEU** (`tokenize="none"` on the same pre-tokenized text).
- **BERTScore F1** (`xlm-roberta-large`, layer 17, no baseline rescaling).

### 5.3 Statistical Significance

- **McNemar's test** for paired classifier accuracy (exact binomial if < 25
  discordant pairs, else χ² with continuity correction).
- **Paired percentile bootstrap** (1,000 resamples, seed = 42) for Δ macro-F1
  and Δ corpus-level NLG metrics.
- **Wilcoxon signed-rank test** on per-document ROUGE-L / BERTScore.
- **Holm–Bonferroni correction** across the family of pairwise comparisons.

### 5.4 Efficiency

- **Latency:** mean of 100 (sentiment) / 50 (summarization) single-sample
  forward passes after 10 / 5 warmup passes; CUDA synchronisation enforced.
- **Peak VRAM:** `torch.cuda.max_memory_allocated()`.
- **GPU energy:** NVML board power sampled every 100 ms, integrated via the
  trapezoidal rule. Measures the GPU only.
- **Total energy + CO₂:** CodeCarbon (CPU/RAM modelled, not measured).

## 6. Bias Analysis

Three complementary probes (all operating on the **sentiment classifier**):

1. **Counterfactual probe.** Swap gendered/professional Amharic terms (see
   `src/bias/lexicon.py`) in validation texts; measure flip rate (% of
   predictions that change) and total-variation distance. Combines natural
   swaps (terms found in text) and template swaps (terms prepended to text).

2. **Positional probe.** Concatenate the evaluation text before vs. after a
   randomly drawn neutral filler text; measure flip rate and accuracy gap.

3. **Cultural probe.** Fill neutral templates with ethnicity, religion, and
   region group terms; measure max−min P(negative) across groups.

**Bias robustness** = 1 − mean(counterfactual flip rate, positional flip rate,
cultural disparity). This is a heuristic composite defined by this study;
components are always reported separately.

## 7. Bias Mitigation

Four additive conditions (sentiment only):

| Condition | Parameter | Data | Loss |
|---|---|---|---|
| `probe` | head only | original | CE |
| `lora` | LoRA r=8 | original | CE |
| `lora_cda` | LoRA | + CDA (~30 %) | CE |
| `lora_cda_rw` | LoRA | + CDA | weighted CE (λ = 2 for bias-term examples) |

CDA (counterfactual data augmentation): every training text containing a
swappable term gets a gender/profession-swapped copy with the same label.
If natural swaps provide < 30 % expansion, template-composited examples
(bias-term phrase + random training text) fill the remainder.

## 8. Deviations from the Original Specification

| Spec item | Deviation | Justification |
|---|---|---|
| Summarization dataset: "dagn/expanded-amharic-news-dataset" | XL-Sum Amharic | Abstractive summaries (not headlines); peer-reviewed; properly split. |
| mBERT: "mBERT-Amharic" | `Davlan/…-finetuned-amharic` | Vanilla mBERT has no Amharic; this is the closest public match. |
| mBERT summarization | BERT2BERT `EncoderDecoderModel` | BERT is not a generative model; this is the standard HF adaptation. |
| mT5 fp16 | fp32 | NaN loss on T4; documented upstream limitation. |
| Training loop | Hand-written (not HF `Trainer`) | Needed for per-example loss weighting and precise energy boundaries. |

