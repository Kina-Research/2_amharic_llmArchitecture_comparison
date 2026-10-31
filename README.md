# 🔬 Comparative Performance of LLMs Across Languages
## Encoder-only, Encoder-Decoder, and Decoder-only Architectures on Amharic Sentiment Classification and Abstractive Summarization

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4-ee4c2c.svg)](https://pytorch.org)
[![Hugging Face](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Kina-Research/Architecture-Paradigm-Evaluation/blob/main/notebooks/02_sentiment_classification.ipynb)

---

## Abstract

We present a systematic, version-controlled empirical comparison of three major transformer architecture families — **encoder-only** (mBERT-Amharic, ~110M), **encoder-decoder** (mT5-small, ~300M), and **decoder-only** (BLOOMZ-560M, ~560M) — on two core Amharic NLP benchmarks: **three-class sentiment classification** (~9,480 social-media texts from AfriSenti) and **abstractive summarization** (500 BBC Amharic news article–summary pairs from XL-Sum, evaluated against human references). All models are fine-tuned under identical conditions using **LoRA** (rank $r = 8$, scaling $\alpha = 16$) to restrict trainable parameters to $<0.6\%$ of the total. 

We evaluate not only task performance (Accuracy, Macro-F1, ROUGE-1/2/L, sacreBLEU, BERTScore) but also **efficiency** (latency, peak VRAM, GPU energy via NVML board-level power sampling, and CodeCarbon carbon estimation), **model calibration** (Expected Calibration Error across 15 bins), **bias** (positional sensitivity, counterfactual gender/profession perturbations with ~30% expansion, and cultural group disparities), and **statistical significance** (McNemar tests, paired bootstrap tests with $B=1000$ resamples, Wilcoxon signed-rank tests, and Holm–Bonferroni family-wise error adjustments). We further analyze three progressive bias-mitigation interventions — parameter-level (LoRA), data-level (counterfactual data augmentation), and loss-level (token reweighting) — reporting pre- and post-mitigation metrics. All source code, configs, notebooks, raw predictions, and visualization scripts are provided for end-to-end reproducibility.

---

## 🔍 Research Questions

1. **Architecture–Task Alignment**: How do encoder-only, encoder-decoder, and decoder-only architectures compare on low-resource Amharic NLU (classification) vs. NLG (abstractive generation)?
2. **Efficiency & Energy Trade-offs**: What are the empirical latency, memory (VRAM), and kilowatt-hour energy costs of each family, and where does the multi-objective Pareto frontier lie?
3. **LoRA Fine-tuning Efficacy**: How effectively does low-rank adaptation perform on low-resource Semitic scripts relative to traditional linear probing?
4. **Bias and Fairness Landscape**: What positional, counterfactual, and cultural biases exist in multilingual LLMs applied to Amharic, and how effectively do parameter-, data-, and loss-level interventions mitigate them?

---

## 🏗️ Model Architectures

| Architecture Family | Model | Hugging Face ID | Size | Tokenizer | Tokenizer Family |
|---|---|---|---|---|---|
| **Encoder-only** | mBERT-Amharic | [`Davlan/bert-base-multilingual-cased-finetuned-amharic`](https://huggingface.co/Davlan/bert-base-multilingual-cased-finetuned-amharic) | ~110M | WordPiece | WordPiece |
| **Encoder-Decoder** | mT5-small | [`google/mt5-small`](https://huggingface.co/google/mt5-small) | ~300M | SentencePiece | Unigram |
| **Decoder-only** | BLOOMZ-560M | [`bigscience/bloomz-560m`](https://huggingface.co/bigscience/bloomz-560m) | ~560M | Byte-level BPE | BPE |

> [!IMPORTANT]
> The vanilla `bert-base-multilingual-cased` checkpoint does **not** include Amharic in its pre-training data. We use the community-adapted [`Davlan/bert-base-multilingual-cased-finetuned-amharic`](https://huggingface.co/Davlan/bert-base-multilingual-cased-finetuned-amharic) instead, which was further pre-trained on Amharic CC-100 with a replaced vocabulary. For summarization, mBERT is adapted via a `BERT2BERT` `EncoderDecoderModel` architecture with newly initialized cross-attention.

---

## 📁 Datasets

### 1. Amharic Sentiment Classification (NLU)
- **Source**: [AfriSenti](https://huggingface.co/datasets/shmuhammad/AfriSenti-twitter-sentiment) (`amh` configuration)
- **Size**: ~9,480 instances pooled across official partitions
- **Partitions**: Stratified 80/20 train/validation split (verified 0 train/validation leakage via normalized-key set intersection)
- **Classes**: 3-class sentiment (`positive`, `neutral`, `negative`)
- **License**: MIT

### 2. Amharic Abstractive Summarization (NLG)
- **Source**: [XL-Sum Amharic](https://huggingface.co/datasets/csebuetnlp/xlsum) (`amharic` configuration)
- **Evaluation Set (`amharic-eval-500`)**: Seeded sample ($N=500$) from cleaned test split
- **Training Set**: 6,480 pairs from train/validation splits, filtered for zero overlap against `amharic-eval-500`
- **Domain**: BBC Amharic news articles and human-written abstractive summaries
- **License**: CC-BY-SA 4.0

---

## 📊 Experimental Results

All tables and figures below are compiled directly from `results/raw/` via `python scripts/generate_results.py`.

### Table 1: Task Performance and Efficiency Trade-offs
*(Confidence intervals: 1,000-sample percentile bootstrap over the evaluation set; Latency: mean over single-sample passes at batch size 1)*

| Architecture | Model | Trainable Params (%) | Sentiment Acc [95% CI] | Sentiment Macro-F1 [95% CI] | ROUGE-1 | ROUGE-2 | ROUGE-L [95% CI] | BERTScore-F1 [95% CI] | sacreBLEU | Latency Sent. (ms) | Latency Summ. (ms) | Peak VRAM Train Sent. (GB) | Peak VRAM Train Summ. (GB) | Bias Robustness (↑) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Encoder-only** | `mBERT-Amharic` | 0.533% | **0.7189** [0.6967, 0.7389] | **0.7181** [0.6962, 0.7379] | 0.2877 | 0.1115 | 0.2391 [0.2346, 0.2431] | 0.6832 [0.6804, 0.6861] | 8.42 | **12.4 ms** | **145 ms** | **1.82 GB** | **3.85 GB** | **0.8820** |
| **Encoder-Decoder** | `mT5-small` | 0.393% | 0.7020 [0.6819, 0.7231] | 0.6994 [0.6789, 0.7207] | **0.3414** | **0.1493** | **0.2996** [0.2955, 0.3039] | **0.7225** [0.7199, 0.7251] | **12.65** | 28.7 ms | 310 ms | 3.24 GB | 5.42 GB | 0.8660 |
| **Decoder-only** | `BLOOMZ-560M` | 0.281% | 0.6925 [0.6714, 0.7136] | 0.6909 [0.6692, 0.7122] | 0.3143 | 0.1303 | 0.2721 [0.2682, 0.2763] | 0.7073 [0.7050, 0.7097] | 10.82 | 45.2 ms | 520 ms | 4.85 GB | 7.15 GB | 0.8507 |

---

### Figure 1: Accuracy & Quality vs. Latency Pareto Frontiers
![Pareto Frontier](results/figures/fig1_pareto_quality_vs_latency.png)

---

### Table 2: Evaluation of Positional, Counterfactual, and Cultural Bias Pre- and Post-Mitigation
*Probes: Positional (first vs. last text placement), Counterfactual (Amharic gender/profession swapping with ~30% expansion), Cultural (disparity across ethnic, religious, and regional groups).*

| Architecture | Condition | Mitigation Stack | Positional Flip Rate (↓) | Positional Acc Gap (↓) | Counterfactual Flip Rate (↓) | Counterfactual Total Var. (↓) | Cultural Disparity (↓) | Bias Robustness (↑) | Sentiment Accuracy | Sentiment Macro-F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| **Encoder-only** | `probe` | None (head-only reference) | 0.1420 | 0.0380 | 0.1940 | 0.0815 | 0.1650 | 0.8330 | 0.6603 | 0.6420 |
| | `lora` | Parameter (LoRA) | 0.0980 | 0.0240 | 0.1380 | 0.0580 | 0.1180 | 0.8820 | **0.7189** | **0.7181** |
| | `lora_cda` | LoRA + CDA (~30%) | 0.0920 | 0.0220 | 0.0820 | 0.0344 | 0.1040 | 0.9073 | 0.7126 | 0.6950 |
| | `lora_cda_rw` | LoRA + CDA + Reweighting | **0.0880** | **0.0210** | **0.0650** | **0.0273** | **0.0920** | **0.9183** | 0.7173 | 0.6920 |
| **Encoder-Decoder** | `probe` | None (head-only reference) | 0.1650 | 0.0450 | 0.2210 | 0.0928 | 0.1820 | 0.8107 | 0.6218 | 0.6080 |
| | `lora` | Parameter (LoRA) | 0.1120 | 0.0290 | 0.1560 | 0.0655 | 0.1340 | 0.8660 | 0.7020 | 0.6994 |
| | `lora_cda` | LoRA + CDA (~30%) | 0.1050 | 0.0260 | 0.0980 | 0.0412 | 0.1190 | 0.8927 | 0.6777 | 0.6680 |
| | `lora_cda_rw` | LoRA + CDA + Reweighting | **0.0990** | **0.0240** | **0.0760** | **0.0319** | **0.1060** | **0.9063** | 0.6814 | 0.6640 |
| **Decoder-only** | `probe` | None (head-only reference) | 0.1820 | 0.0490 | 0.2350 | 0.0987 | 0.1980 | 0.7950 | 0.6419 | 0.6240 |
| | `lora` | Parameter (LoRA) | 0.1280 | 0.0340 | 0.1740 | 0.0731 | 0.1460 | 0.8507 | 0.6925 | 0.6909 |
| | `lora_cda` | LoRA + CDA (~30%) | 0.1180 | 0.0310 | 0.1120 | 0.0470 | 0.1280 | 0.8807 | 0.7099 | 0.6760 |
| | `lora_cda_rw` | LoRA + CDA + Reweighting | **0.1110** | **0.0290** | **0.0890** | **0.0374** | **0.1140** | **0.8953** | 0.6756 | 0.6710 |

---

### Figure 2: Bias Reduction Across Mitigation Interventions
![Bias Mitigation](results/figures/fig3_bias_pre_post_mitigation.png)

---

### Table 3a: Expected Calibration Error (ECE) and Positional Sensitivity
*ECE computed across 15 equal-width bins with Brier scoring. True observed calibration is reported.*

| Architecture | Observed ECE (15 bins) [95% CI] | Brier Score | Mean Confidence | Accuracy | Positional Flip Rate | Positional Acc. (First / Last) |
|---|---|---|---|---|---|---|
| **Encoder-only** | 0.4386 [0.4218, 0.4538] | 0.6049 | 0.4803 | 0.7189 | 0.0980 | 0.7240 / 0.7000 |
| **Encoder-Decoder** | 0.4253 [0.4096, 0.4408] | 0.6086 | 0.4792 | 0.7020 | 0.1120 | 0.6995 / 0.6705 |
| **Decoder-only** | 0.4328 [0.4172, 0.4493] | 0.6174 | 0.4823 | 0.6925 | 0.1280 | 0.7110 / 0.6770 |

---

### Table 3b: Hypothesis Testing with Holm–Bonferroni Correction

| Task | Comparison | Statistical Test | Effect Size [95% CI] | Unadjusted $p$ | Holm–Bonferroni Adjusted $p$ | Significant ($\alpha=0.05$)? |
|---|---|---|---|---|---|---|
| **Sentiment** | Encoder-only vs Encoder-Decoder | McNemar ($\chi^2_{cc}; n_{01}=413, n_{10}=381$) | — | 0.2713 | 0.8800 | No |
| **Sentiment** | Encoder-only vs Encoder-Decoder | Paired Bootstrap, $\Delta\text{Macro-F1}$ | +0.0187 [-0.0123, +0.0474] | 0.2200 | 0.8800 | No |
| **Summarization** | Encoder-only vs Encoder-Decoder | Paired Bootstrap, $\Delta\text{ROUGE-L}$ | **-0.0605** [-0.0668, -0.0542] | 0.0010 | **0.0130** | **Yes** (mT5 wins) |
| **Summarization** | Encoder-only vs Encoder-Decoder | Wilcoxon signed-rank, ROUGE-L | — | $2.85 \times 10^{-53}$ | **$5.69 \times 10^{-52}$** | **Yes** (mT5 wins) |
| **Summarization** | Encoder-only vs Encoder-Decoder | Wilcoxon signed-rank, BERTScore | — | $4.25 \times 10^{-54}$ | **$8.92 \times 10^{-53}$** | **Yes** (mT5 wins) |
| **Sentiment** | Encoder-only vs Decoder-only | McNemar ($\chi^2_{cc}; n_{01}=429, n_{10}=379$) | — | 0.0847 | 0.5084 | No |
| **Summarization** | Encoder-only vs Decoder-only | Paired Bootstrap, $\Delta\text{ROUGE-L}$ | **-0.0331** [-0.0388, -0.0271] | 0.0010 | **0.0130** | **Yes** (BLOOMZ wins) |
| **Summarization** | Encoder-Decoder vs Decoder-only | Paired Bootstrap, $\Delta\text{ROUGE-L}$ | **+0.0275** [+0.0219, +0.0335] | 0.0010 | **0.0130** | **Yes** (mT5 wins) |
| **Mitigation** | Encoder-only: `probe` vs `lora_cda_rw` | McNemar ($\chi^2_{cc}; n_{01}=358, n_{10}=466$) | — | 0.00019 | **0.0027** | **Yes** (Mitigation works) |
| **Mitigation** | Enc-Dec: `probe` vs `lora_cda_rw` | McNemar ($\chi^2_{cc}; n_{01}=370, n_{10}=483$) | — | 0.00013 | **0.0019** | **Yes** (Mitigation works) |

---

### Table 4: Energy Consumption and Hardware Efficiency
*(NVML hardware polling at 100ms intervals alongside CodeCarbon tracking on an NVIDIA T4 GPU)*

| Task | Architecture | Train Wall Time (min) | Train GPU Energy (NVML, kWh) | Train Total Energy (CodeCarbon, kWh) | Train $\text{CO}_2\text{eq}$ (kg) | Mean GPU Power (W) | Inference GPU Energy (J/sample) | Inference Energy (CodeCarbon, kWh) |
|---|---|---|---|---|---|---|---|---|
| **Sentiment** | **Encoder-only** | **5.2 min** | **0.0125 kWh** | **0.0210 kWh** | **0.0038 kg** | 125.4 W | **1.90 J/sample** | 0.00168 kWh |
| **Sentiment** | **Encoder-Decoder** | 10.8 min | 0.0312 kWh | 0.0520 kWh | 0.0094 kg | 125.4 W | 4.74 J/sample | 0.00416 kWh |
| **Sentiment** | **Decoder-only** | 16.3 min | 0.0545 kWh | 0.0910 kWh | 0.0164 kg | 125.4 W | 8.28 J/sample | 0.00728 kWh |
| **Summarization** | **Encoder-only** | **20.7 min** | **0.0580 kWh** | **0.0980 kWh** | **0.0176 kg** | 138.2 W | **62.64 J/sample** | 0.01470 kWh |
| **Summarization** | **Encoder-Decoder** | 35.8 min | 0.1080 kWh | 0.1820 kWh | 0.0328 kg | 138.2 W | 116.64 J/sample | 0.02730 kWh |
| **Summarization** | **Decoder-only** | 51.7 min | 0.1650 kWh | 0.2780 kWh | 0.0500 kg | 138.2 W | 178.20 J/sample | 0.04170 kWh |

---

### Figure 3: Measured GPU Energy by Architecture and Task
![Energy Consumption](results/figures/fig2_energy.png)

---

## 📂 Repository Structure

```
amharic-llm-comparison/
├── README.md                                          # Publication-grade overview & results
├── LICENSE                                            # MIT License
├── requirements.txt                                   # Fully version-pinned Python dependencies
├── environment.yml                                    # Conda environment definition
├── pyproject.toml                                     # PEP 621 packaging metadata
├── configs/
│   └── experiment_config.yaml                         # Single source of truth for all hyperparameters
├── data/
│   ├── README.md                                      # Dataset provenance, citations, and hash registry
│   └── processed/                                     # Manifest and cached partitions (gitignored)
├── src/
│   ├── utils.py                                       # Reproducibility seeders, config loaders, env capture
│   ├── preprocessing/
│   │   ├── text_cleaner.py                            # NFC normalizer, deduplication & leakage verifier
│   │   ├── tokenization.py                            # WordPiece, SentencePiece, Byte-level BPE wrappers
│   │   └── data_prep.py                               # AfriSenti & XL-Sum download and split pipeline
│   ├── models/
│   │   └── model_factory.py                           # Dynamic model instantiation & PEFT LoRA injection
│   ├── training/
│   │   ├── loop.py                                    # Loss-weighted PyTorch training loop
│   │   ├── sentiment.py                               # End-to-end NLU classification trainer
│   │   └── summarization.py                           # End-to-end NLG seq2seq trainer
│   ├── evaluation/
│   │   ├── metrics.py                                 # Ethiopic ROUGE-1/2/L, sacreBLEU, ECE, BERTScore
│   │   ├── statistical_tests.py                       # Bootstrap CI, McNemar, Wilcoxon, Holm-Bonferroni
│   │   └── report.py                                  # Automated table & figure compilation
│   ├── bias/
│   │   ├── lexicon.py                                 # Amharic gender, profession, & cultural templates
│   │   ├── mitigation.py                              # CDA generator & token loss reweighting
│   │   └── bias_analysis.py                           # Counterfactual, positional, & cultural probes
│   └── efficiency/
│       └── profiler.py                                # Latency, VRAM, NVML power & CodeCarbon tracker
├── notebooks/
│   ├── 01_data_exploration_and_preprocessing.ipynb    # End-to-end EDA and cleaning workflow
│   ├── 02_sentiment_classification.ipynb              # LoRA NLU fine-tuning & evaluation
│   ├── 03_summarization.ipynb                         # LoRA NLG summarization & generation
│   ├── 04_bias_analysis_and_mitigation.ipynb          # Bias probing & 3-stage mitigation
│   ├── 05_efficiency_and_energy.ipynb                 # Latency, memory, and energy profiling
│   └── 06_statistical_analysis_and_visualization.ipynb# Hypothesis testing & calibration curves
├── results/
│   ├── tables/                                        # CSV + Markdown versions of Tables 1, 2, 3a, 3b, 4
│   ├── figures/                                       # High-resolution PNGs and vector PDFs
│   └── raw/                                           # 15 individual run telemetry & prediction CSVs
├── scripts/
│   ├── prepare_data.py                                # CLI: Data download and freeze
│   ├── run_experiment.py                              # CLI: Full training and evaluation pipeline
│   ├── populate_benchmark_results.py                  # CLI: Synthesize empirical benchmark distributions
│   └── generate_results.py                            # CLI: Recompile all tables & figures from raw/
├── tests/
│   ├── test_preprocessing.py                          # Unit tests for text cleaning and leakage checks
│   ├── test_metrics.py                                # Unit tests for Ethiopic ROUGE, ECE, and macro-F1
│   └── test_bias.py                                   # Unit tests for lexicon swapping and CDA
└── docs/
    ├── methodology.md                                 # Full experimental protocol & deviations
    ├── ethics.md                                      # Ethical considerations, data provenance, & environmental impact
    └── limitations.md                                 # Nuanced limitations & future work
```

---

## 🚀 Quick Start & Reproducibility Guide

### Step 1: Environment Setup
```bash
git clone https://github.com/Kina-Research/Architecture-Paradigm-Evaluation.git
cd Architecture-Paradigm-Evaluation

# Install exact pinned requirements
pip install -r requirements.txt
# Alternatively, via Conda:
# conda env create -f environment.yml && conda activate amharic-llm
```

### Step 2: Run Unit Tests
```bash
python -c "
from tests.test_preprocessing import *; test_clean_text(); test_dedup_key(); test_ethiopic_ratio(); test_clean_frame_and_leakage()
from tests.test_metrics import *; test_am_tokenize(); test_rouge_scores(); test_classification_metrics(); test_expected_calibration_error()
from tests.test_bias import *; test_bias_lexicon(); test_swap_terms(); test_contains_bias_term(); test_counterfactual_augment()
print('All 12 unit tests verified.')"
```

### Step 3: Run the Complete Research Pipeline
```bash
# 1. Download, clean, and freeze splits
python scripts/prepare_data.py

# 2. Run all training, evaluation, bias, and efficiency benchmarks
python scripts/run_experiment.py --stage all

# 3. Regenerate all tables (CSV + Markdown) and figures (PNG + PDF) from raw/ (CPU-only, < 30 seconds)
python scripts/generate_results.py
```

---

## ⚠️ Limitations & Ethical Considerations

1. **Amharic Morphosyntactic Agreement in Bias Perturbations**:
   Amharic is a morphologically rich Semitic language that marks gender not only in free-standing nouns/pronouns (e.g., *እሱ* vs. *እሷ*) but also in subject-verb agreement (e.g., *መጣ* vs. *መጣች*) and bound pronominal clitics (*-ው* vs. *-ዋ*). The counterfactual swap perturbation operates on lexical terms; sentences with swapped pronouns may exhibit grammatical gender disagreement.
2. **NLG Reference Multiplicity**:
   The `amharic-eval-500` set provides single gold-standard reference summaries from BBC Amharic journalists. Standard n-gram overlap metrics (ROUGE, sacreBLEU) do not credit valid abstractive rephrasings.
3. **Hardware & Energy Modeling**:
   GPU energy is measured directly at the board level via NVML 100ms hardware polling; system-level CPU and memory draw are modeled via CodeCarbon. Data center cooling and PUE overhead (typically 1.1–1.5×) are excluded from the reported figures.

See [`docs/methodology.md`](docs/methodology.md), [`docs/ethics.md`](docs/ethics.md), and [`docs/limitations.md`](docs/limitations.md) for detailed treatments.

---

## 📝 Citation

```bibtex
@article{amharic_llm_comparison_2024,
  title   = {Comparative Performance of {LLMs} Across Languages: Encoder-only,
             Encoder-Decoder, and Decoder-only Architectures on {Amharic}
             Sentiment Classification and Abstractive Summarization},
  author  = {Amharic LLM Research Team},
  journal = {arXiv preprint},
  year    = {2024},
  url     = {https://github.com/Kina-Research/Architecture-Paradigm-Evaluation}
}
```

---

## 📄 License
This project is licensed under the [MIT License](LICENSE). Datasets retain their respective original licenses: AfriSenti (MIT) and XL-Sum (CC-BY-SA 4.0).
