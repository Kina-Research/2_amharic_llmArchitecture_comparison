# Ethical Considerations

## 1. Data Provenance and Consent

### AfriSenti (Sentiment)
AfriSenti (Muhammad et al., EMNLP 2023) collected publicly posted tweets from
Twitter/X under the platform's then-current terms of service for academic
research. The dataset was annotated by native speakers and released under the
MIT licence. We add no new annotations and do not attempt to de-anonymise any
users. Tweets containing personal identifiable information (PII) may remain
in the dataset; we do not filter for PII beyond the original curators' process.

### XL-Sum (Summarization)
XL-Sum (Hasan et al., ACL-Findings 2021) consists of BBC news articles and
their professionally written summaries. Copyright belongs to the BBC; the
dataset is released under CC BY-NC-SA 4.0, restricting use to **non-commercial
research**. We comply with this restriction.

## 2. Amharic Language Considerations

Amharic is the official working language of Ethiopia, spoken by ~30–50 million
people. It is classified as "low-resource" in NLP because pre-training corpora,
annotated benchmarks, and NLP tools are sparse relative to high-resource
languages. This has several ethical implications:

- **Representation.** Models pre-trained predominantly on English and
  European languages may encode cultural assumptions that do not transfer to
  Ethiopian contexts. Our cultural bias probes quantify — but cannot
  eliminate — such misalignments.

- **Tokenisation equity.** All three tokenizers fragment Ge'ez script into
  3–7 subword tokens per word (vs. ~1.3 for English). This wastes context
  window capacity and may degrade performance. We report tokenizer fertility
  to make this cost visible.

- **Evaluation validity.** No validated Amharic stemmer exists; our ROUGE
  implementation uses unstemmed whitespace tokenization. Scores are not
  directly comparable to papers using different tokenizers.

## 3. Bias

### What we measure
We probe three dimensions of bias:
1. **Counterfactual (gender/profession):** Does swapping "he" ↔ "she" or
   "doctor" ↔ "nurse" change the model's prediction?
2. **Positional:** Does the location of the informative text in the input
   affect the prediction?
3. **Cultural:** Does mentioning different Ethiopian ethnic, religious, or
   regional groups change the predicted sentiment?

### What we do NOT claim
- Our bias lexicon (gender pairs, professional pairs, cultural group terms)
  was compiled by the authors and **has not been validated by a native
  Amharic-speaking linguist**. Results from these probes are preliminary.
- Bias probes measure model behaviour under synthetic perturbations; they do
  not guarantee fairness in deployment.
- Mitigation (LoRA, CDA, reweighting) reduces measured bias metrics but
  cannot certify bias-free operation.

## 4. Environmental Impact

Training all conditions (12 sentiment + 3 summarization runs) on an NVIDIA T4
consumes an estimated 0.1–0.3 kWh of GPU energy. We report both measured (NVML)
and modelled (CodeCarbon) energy and carbon-equivalent emissions. The Ethiopian
electricity grid is predominantly hydroelectric, so the CO₂ intensity is lower
than in most countries; however, experiments run on cloud infrastructure may be
powered by different grids.

## 5. Intended Use and Misuse

This repository is intended for **academic research** into multilingual NLP
architecture comparison and bias evaluation. It is **not** intended for:

- Production sentiment analysis or summarization systems.
- Automated decision-making affecting individuals.
- Surveillance or content moderation targeting specific communities.

The cultural bias probes use ethnicity, religion, and region terms. These are
used strictly to measure model disparity and carry no normative intent. Any use
of these lists to target, profile, or discriminate against communities would
be a misuse of this work.

## 6. Recommendations for Downstream Users

1. **Do not deploy these models without further evaluation** on your target
   domain, population, and use case.
2. **Engage native Amharic speakers** to validate bias lexicons, cultural
   probe templates, and summarization quality before drawing conclusions.
3. **Report energy consumption** in any publication that fine-tunes or runs
   these models.

