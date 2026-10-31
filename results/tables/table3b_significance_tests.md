### Table 3b. Pairwise significance tests (Holm–Bonferroni adjusted)

| Task | Comparison | Test | Effect | p | p (Holm) |
|---|---|---|---|---|---|
| Sentiment | Encoder-only vs Encoder-Decoder | McNemar (chi2_cc); n01=413, n10=381 | — | 0.2713 | 0.88 |
| Sentiment | Encoder-only vs Encoder-Decoder | Paired bootstrap, Δ macro-F1 | +0.0187 [-0.0123, +0.0474] | 0.22 | 0.88 |
| Summarization | Encoder-only vs Encoder-Decoder | Paired bootstrap, Δ rougeL | -0.0605 [-0.0668, -0.0542] | 0.001 | 0.013 |
| Summarization | Encoder-only vs Encoder-Decoder | Wilcoxon signed-rank, rougeL | — | 2.847e-53 | 5.694e-52 |
| Summarization | Encoder-only vs Encoder-Decoder | Paired bootstrap, Δ bertscore_f1 | -0.0393 [-0.0431, -0.0355] | 0.001 | 0.013 |
| Summarization | Encoder-only vs Encoder-Decoder | Wilcoxon signed-rank, bertscore_f1 | — | 4.247e-54 | 8.92e-53 |
| Sentiment | Encoder-only vs Decoder-only | McNemar (chi2_cc); n01=429, n10=379 | — | 0.08474 | 0.5084 |
| Sentiment | Encoder-only vs Decoder-only | Paired bootstrap, Δ macro-F1 | +0.0272 [-0.0030, +0.0577] | 0.088 | 0.5084 |
| Summarization | Encoder-only vs Decoder-only | Paired bootstrap, Δ rougeL | -0.0331 [-0.0388, -0.0271] | 0.001 | 0.013 |
| Summarization | Encoder-only vs Decoder-only | Wilcoxon signed-rank, rougeL | — | 2.844e-23 | 5.119e-22 |
| Summarization | Encoder-only vs Decoder-only | Paired bootstrap, Δ bertscore_f1 | -0.0241 [-0.0279, -0.0204] | 0.001 | 0.013 |
| Summarization | Encoder-only vs Decoder-only | Wilcoxon signed-rank, bertscore_f1 | — | 5.449e-28 | 1.035e-26 |
| Sentiment | Encoder-Decoder vs Decoder-only | McNemar (chi2_cc); n01=411, n10=393 | — | 0.5488 | 1 |
| Sentiment | Encoder-Decoder vs Decoder-only | Paired bootstrap, Δ macro-F1 | +0.0085 [-0.0235, +0.0393] | 0.596 | 1 |
| Summarization | Encoder-Decoder vs Decoder-only | Paired bootstrap, Δ rougeL | +0.0275 [+0.0219, +0.0335] | 0.001 | 0.013 |
| Summarization | Encoder-Decoder vs Decoder-only | Wilcoxon signed-rank, rougeL | — | 1.427e-16 | 2.426e-15 |
| Summarization | Encoder-Decoder vs Decoder-only | Paired bootstrap, Δ bertscore_f1 | +0.0152 [+0.0117, +0.0187] | 0.001 | 0.013 |
| Summarization | Encoder-Decoder vs Decoder-only | Wilcoxon signed-rank, bertscore_f1 | — | 5.039e-15 | 8.062e-14 |
| Sentiment (mitigation) | Encoder-only: probe vs LoRA+CDA+RW | McNemar (chi2_cc); n01=358, n10=466 | — | 0.0001934 | 0.002707 |
| Sentiment (mitigation) | Encoder-Decoder: probe vs LoRA+CDA+RW | McNemar (chi2_cc); n01=370, n10=483 | — | 0.0001257 | 0.001885 |
| Sentiment (mitigation) | Decoder-only: probe vs LoRA+CDA+RW | McNemar (chi2_cc); n01=391, n10=455 | — | 0.03031 | 0.2122 |


**NLG evaluation caveats.**
1. Single reference per article (XL-Sum one-sentence BBC summaries); ROUGE/BLEU under-reward valid paraphrases.
2. ROUGE is computed on an Ethiopic-aware whitespace tokenization without stemming (the reference
   `rouge_score` package deletes Ge'ez characters). Scores are not directly comparable to papers using
   other tokenizers (e.g. XL-Sum's multilingual ROUGE).
3. sacreBLEU uses `tokenize="none"` on the same pre-tokenized text; signature is therefore non-standard.
4. BERTScore uses xlm-roberta-large (layer 17, no baseline rescaling); absolute values are compressed
   toward ~0.8–0.9 and only differences between systems are meaningful.
5. The mBERT summarizer is a BERT2BERT model whose cross-attention is newly initialised; it is not an
   off-the-shelf generator and its scores reflect that.
6. Evaluation set = 500 pairs → CIs are wide; differences inside overlapping CIs should not be over-read.
