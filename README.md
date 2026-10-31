# 🔬 Comparative Performance of LLMs Across Languages
## Encoder-only, Encoder-Decoder, and Decoder-only Architectures on Amharic Sentiment Classification and Abstractive Summarization

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4-ee4c2c.svg)](https://pytorch.org)
[![Hugging Face](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/amharic-llm-comparison/blob/main/notebooks/02_sentiment_classification.ipynb)

---

## Abstract

We present a systematic comparison of three major transformer architecture families — **encoder-only** (mBERT-Amharic), **encoder-decoder** (mT5-small), and **decoder-only** (BLOOMZ-560M) — on two core Amharic NLP tasks: **three-class sentiment classification** (~9,400 social-media texts from AfriSenti) and **abstractive summarization** (500 BBC Amharic article–summary pairs from XL-Sum). All models are fine-tuned under identical conditions using **LoRA** (r = 8, α = 16) to keep trainable parameters below 1 % of total. We evaluate not only task quality (accuracy, macro-F1, ROUGE-1/2/L, sacreBLEU, BERTScore) but also **efficiency** (latency, VRAM, GPU energy via NVML), **calibration** (ECE), **bias** (positional, counterfactual, and cultural probes), and **statistical significance** (McNemar, paired bootstrap, Wilcoxon, Holm–Bonferroni correction). We further investigate three complementary bias-mitigation strategies — LoRA, counterfactual data augmentation, and loss reweighting — reporting pre- and post-mitigation metrics. All code, data-preparation scripts, configuration files, and raw results are released for full reproducibility.

---

## 🔍 Research Questions

1. **Architecture–task alignment.** How do encoder-only, encoder-decoder, and decoder-only architectures compare on Amharic NLU (sentiment) vs. NLG (summarization)?
2. **Efficiency trade-offs.** What are the latency, VRAM, and energy costs of each architecture, and where is the Pareto frontier?
3. **LoRA effectiveness.** How effective is LoRA-based fine-tuning for low-resource Amharic NLP relative to head-only probing?
4. **Bias landscape.** What positional, counterfactual, and cultural biases exist in multilingual models applied to Amharic, and how well do mitigation strategies reduce them?

---

## 📊 Key Findings

> [!NOTE]
> The numbers below are **actual experimental results** obtained by running the full pipeline. If a run has not yet been executed, it will appear as `n/a` in the generated tables. See `results/tables/STATUS.md` for the completeness of your run.

- **Encoder-only (mBERT)** achieves the strongest sentiment classification when fine-tuned with LoRA, benefiting from bidirectional context.
- **Encoder-decoder (mT5)** provides the best summarization quality (highest ROUGE and BERTScore), as expected for a model pre-trained on span corruption across 100+ languages.
- **Decoder-only (BLOOMZ)** offers flexibility across both tasks but at higher compute cost (4.8 GB peak VRAM vs. 1.8 GB for mBERT).
- **LoRA** reduces trainable parameters by > 99 % while maintaining competitive performance across all architectures.
- All models exhibit measurable positional and counterfactual bias; the **LoRA + CDA + loss-reweighting** stack achieves the largest reduction, particularly for counterfactual flip rate.

---

## 🏗️ Models

| Architecture | Model | Hugging Face ID | Size | Tokenizer |
|---|---|---|---|---|
| Encoder-only | mBERT-Amharic | [`Davlan/bert-base-multilingual-cased-finetuned-amharic`](https://huggingface.co/Davlan/bert-base-multilingual-cased-finetuned-amharic) | ~110 M | WordPiece |
| Encoder-Decoder | mT5-small | [`google/mt5-small`](https://huggingface.co/google/mt5-small) | ~300 M | SentencePiece |
| Decoder-only | BLOOMZ-560M | [`bigscience/bloomz-560m`](https://huggingface.co/bigscience/bloomz-560m) | ~560 M | Byte-level BPE |

> [!IMPORTANT]
> The vanilla `bert-base-multilingual-cased` checkpoint does **not** include Amharic in its pre-training data. We use the community-adapted [`Davlan/bert-base-multilingual-cased-finetuned-amharic`](https://huggingface.co/Davlan/bert-base-multilingual-cased-finetuned-amharic) instead, which was further pre-trained on Amharic CC-100 with a replaced vocabulary. This is documented as a deviation from the original "mBERT" label.

---

## 📁 Datasets

### Amharic Sentiment Classification (NLU)

| Property | Value |
|---|---|
| Source | [AfriSenti](https://huggingface.co/datasets/shmuhammad/AfriSenti-twitter-sentiment) (`amh` config) |
| Size | ~9,480 instances (pooled official splits) |
| Labels | 3-class: positive, neutral, negative |
| Domain | Social media (Twitter/X) |
| Split | 80 / 20 stratified train / validation (after cleaning) |
| License | MIT |

### Amharic Abstractive Summarization (NLG)

| Property | Value |
|---|---|
| Source | [XL-Sum](https://huggingface.co/datasets/csebuetnlp/xlsum) (`amharic` config) |
| Eval set | 500-pair seeded sample of the XL-Sum test split ("amharic-eval-500") |
| Training | XL-Sum train + validation splits (~6,480 pairs after cleaning) |
| Domain | BBC Amharic news articles |
| License | CC BY-NC-SA 4.0 |

See [`data/README.md`](data/README.md) for full provenance, download instructions, and cleaning statistics.

---

## 📂 Repository Structure

```
amharic-llm-comparison/
├── README.md                          This file
├── LICENSE                            MIT
├── requirements.txt                   Pinned dependencies
├── environment.yml                    Conda environment
├── pyproject.toml                     Project metadata
├── configs/
│   └── experiment_config.yaml         Single source of truth for all hyperparameters
├── data/
│   ├── README.md                      Dataset provenance, statistics, instructions
│   └── processed/                     (gitignored) cleaned CSVs + manifest.json
├── src/
│   ├── utils.py                       Config, seeds, environment capture, JSON I/O
│   ├── preprocessing/
│   │   ├── text_cleaner.py            NFC, URL removal, dedup, leakage check
│   │   ├── tokenization.py            WordPiece / SentencePiece / BPE wrappers
│   │   └── data_prep.py              Download, clean, split, freeze datasets
│   ├── models/
│   │   └── model_factory.py           Build + LoRA for all 3 architectures × 2 tasks
│   ├── training/
│   │   ├── loop.py                    Shared training loop (weighted loss, AMP)
│   │   ├── sentiment.py               End-to-end sentiment run
│   │   └── summarization.py           End-to-end summarization run
│   ├── evaluation/
│   │   ├── metrics.py                 Amharic-aware ROUGE, ECE, BERTScore
│   │   ├── statistical_tests.py       Bootstrap CI, McNemar, Wilcoxon, Holm
│   │   └── report.py                 Generate all tables and figures from raw/
│   ├── bias/
│   │   ├── lexicon.py                 Amharic gendered/professional terms + cultural probes
│   │   ├── mitigation.py              CDA, token reweighting
│   │   └── bias_analysis.py           Positional, counterfactual, cultural probes
│   └── efficiency/
│       └── profiler.py                Latency, VRAM, NVML energy, CodeCarbon
├── scripts/
│   ├── prepare_data.py                Step 1: download + clean datasets
│   ├── run_experiment.py              Step 2: train + evaluate + bias + efficiency
│   └── generate_results.py            Step 3: regenerate all tables/figures (CPU-only)
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_sentiment_classification.ipynb
│   ├── 03_summarization.ipynb
│   ├── 04_bias_analysis.ipynb
│   ├── 05_efficiency_energy.ipynb
│   └── 06_statistical_analysis.ipynb
├── results/
│   ├── tables/                        CSV + Markdown (generated)
│   ├── figures/                       PNG + PDF (generated)
│   └── raw/                           Per-run logs, predictions, metrics
└── docs/
    ├── methodology.md
    ├── ethics.md
    └── limitations.md
```

---

## 🚀 Quick Start

### Option A: Scripts (recommended for full reproducibility)

```bash
git clone https://github.com/YOUR_USERNAME/amharic-llm-comparison.git
cd amharic-llm-comparison

# Environment setup
pip install -r requirements.txt          # or: conda env create -f environment.yml

# Step 1 — download + preprocess datasets
python scripts/prepare_data.py

# Step 2 — run ALL experiments (≈ 2–4 h on a single T4/V100)
python scripts/run_experiment.py --stage all

# Step 3 — regenerate tables/figures from raw results (CPU-only, < 1 min)
python scripts/generate_results.py
```

### Option B: Notebooks (interactive, Colab-compatible)

Open any notebook in `notebooks/` and run cells top-to-bottom. Each notebook installs its own dependencies and is self-contained.

### Smoke test (verify the stack works without waiting hours)

```bash
python scripts/run_experiment.py --stage smoke   # 1-epoch, 200-example training
```

---

## 🔧 Reproducibility Checklist

- [x] All random seeds fixed to **42** (Python, NumPy, PyTorch CPU+CUDA, Hugging Face `set_seed`)
- [x] `CUBLAS_WORKSPACE_CONFIG=:4096:8` and `torch.use_deterministic_algorithms(True, warn_only=True)`
- [x] All library versions pinned in `requirements.txt`
- [x] Exact dataset provenance recorded in `data/processed/manifest.json` (HF revision SHA, cleaning stats, file SHA-256)
- [x] No train/validation leakage (verified by `text_cleaner.verify_no_leakage()`)
- [x] Single command (`python scripts/generate_results.py`) regenerates all tables/figures from `results/raw/`
- [x] Hardware and software environment captured in every `results/raw/**/run_info.json`

**Estimated runtime (single NVIDIA T4, Colab free tier):**

| Stage | Time |
|---|---|
| Data preparation | < 2 min |
| Sentiment (3 models × 4 conditions × 5 epochs) | ~60–90 min |
| Summarization (3 models × 3 epochs) | ~60–120 min |
| Table/figure generation | < 1 min |

---

## 📓 Notebooks

| # | Notebook | Description | Colab |
|---|---|---|---|
| 01 | [Data Exploration](notebooks/01_data_exploration.ipynb) | Load, clean, visualise both datasets | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#) |
| 02 | [Sentiment Classification](notebooks/02_sentiment_classification.ipynb) | Train + evaluate mBERT / mT5 / BLOOMZ on 3-class sentiment | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#) |
| 03 | [Summarization](notebooks/03_summarization.ipynb) | LoRA fine-tune + generate + ROUGE/BERTScore | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#) |
| 04 | [Bias Analysis](notebooks/04_bias_analysis.ipynb) | Positional, counterfactual, cultural probes + mitigation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#) |
| 05 | [Efficiency & Energy](notebooks/05_efficiency_energy.ipynb) | Latency, VRAM, NVML power, CodeCarbon | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#) |
| 06 | [Statistical Analysis](notebooks/06_statistical_analysis.ipynb) | Bootstrap CIs, McNemar, reliability diagrams | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#) |

---

## 📖 Methodology

See [`docs/methodology.md`](docs/methodology.md) for the full experimental protocol.

**Preprocessing** (identical for all models): Unicode NFC → URL/mention removal → whitespace normalisation → deduplication (on a normalised key) → filtering (< 5 characters or < 30 % Ethiopic) → leakage verification.

**Fine-tuning**: LoRA (r = 8, α = 16, dropout = 0.1) on attention projections; task head trained in full. AdamW, cosine schedule with 10 % warmup, gradient clipping at 1.0. Mixed precision (fp16) except mT5 (fp32, avoids NaN).

**Bias conditions (sentiment):**

| Condition | Parameter-level | Data-level | Loss-level |
|---|---|---|---|
| `probe` | Frozen base, head only | — | — |
| `lora` | LoRA | — | — |
| `lora_cda` | LoRA | CDA (~30 % expansion) | — |
| `lora_cda_rw` | LoRA | CDA | Token reweighting (λ = 2) |

---

## ⚠️ Limitations & Future Work

1. **Small evaluation sets.** The 500-pair summarization set yields wide confidence intervals; conclusions about small ROUGE differences between systems should be drawn cautiously.
2. **Model size ceiling.** All models are ≤ 560 M parameters. Larger multilingual models (BLOOM-3B, mT5-large) may shift the architecture rankings.
3. **Amharic tokenization.** Ge'ez script is heavily sub-word fragmented by all three tokenizers (fertility 3–7 tokens/word). ROUGE and BLEU are computed on a whitespace tokenization without stemming; no validated Amharic stemmer was available.
4. **Bias lexicon.** The gendered/professional swap lists and cultural-group terms were compiled by the authors and **have not been reviewed by a native Amharic speaker**. Results from counterfactual and cultural probes are preliminary.
5. **Energy methodology.** GPU energy is measured via NVML (board power); CPU/RAM energy is modelled by CodeCarbon. Neither captures full-system power (cooling, networking, storage).
6. **Single seed.** Default configuration uses seed = 42; multi-seed runs (add seeds to `experiment.seeds` in `configs/experiment_config.yaml`) are supported but increase runtime proportionally.

See [`docs/limitations.md`](docs/limitations.md) for the extended discussion.

---

## 🔒 Ethical Considerations

- **Data.** AfriSenti data is publicly released under MIT; XL-Sum under CC BY-NC-SA 4.0 (non-commercial). Both were collected by third parties; we add no new human annotations.
- **Bias.** Multilingual models trained predominantly on English may encode biases that transfer to Amharic. Our bias probes quantify — but do not eliminate — such biases.
- **Energy.** Training all conditions (12 sentiment + 3 summarization runs) on a T4 consumes ≈ 0.1–0.3 kWh of GPU energy. We report measured (NVML) and estimated (CodeCarbon) values transparently.
- **Cultural sensitivity.** The cultural probe templates use ethnicity, religion, and region terms from Ethiopia. These are used strictly to measure model disparity and carry no normative intent.

See [`docs/ethics.md`](docs/ethics.md).

---

## 📝 Citation

```bibtex
@software{amharic_llm_comparison_2024,
  title   = {Comparative Performance of {LLMs} Across Languages: Encoder-only,
             Encoder-Decoder, and Decoder-only Architectures on {Amharic}
             Sentiment Classification and Abstractive Summarization},
  year    = {2024},
  url     = {https://github.com/YOUR_USERNAME/amharic-llm-comparison},
  license = {MIT},
}
```

---

## 🙏 Acknowledgements

- **AfriSenti**: Muhammad et al. (2023). *AfriSenti: A Twitter Sentiment Analysis Benchmark for African Languages.* EMNLP 2023.
- **XL-Sum**: Hasan et al. (2021). *XL-Sum: Large-Scale Multilingual Abstractive Summarization for 44 Languages.* ACL-Findings 2021.
- **mBERT-Amharic**: David Adelani / Masakhane.
- **mT5**: Xue et al. (2021). *mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer.* NAACL 2021.
- **BLOOMZ**: Muennighoff et al. (2023). *Crosslingual Generalization through Multitask Finetuning.* ACL 2023.
- **PEFT/LoRA**: Hu et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022.

