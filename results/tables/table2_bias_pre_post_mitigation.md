### Table 2. Positional, counterfactual and cultural bias, pre- and post-mitigation (sentiment)

| Architecture | Condition | Mitigation | Positional flip rate (↓) | Positional acc gap (↓) | Counterfactual flip rate (↓) | Counterfactual TV (↓) | CF natural pairs (n) | Cultural disparity (↓) | Bias Robustness (↑) | Accuracy | Macro-F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Encoder-only | probe | none (head-only, pre-LoRA) | 0.1420 | 0.0380 | 0.1940 | 0.0815 | 340 | 0.1650 | 0.8330 | 0.6603 | 0.6420 |
| Encoder-only | lora | parameter (LoRA) | 0.0980 | 0.0240 | 0.1380 | 0.0580 | 340 | 0.1180 | 0.8820 | 0.7189 | 0.6980 |
| Encoder-only | lora_cda | LoRA + data (CDA) | 0.0920 | 0.0220 | 0.0820 | 0.0344 | 340 | 0.1040 | 0.9073 | 0.7126 | 0.6950 |
| Encoder-only | lora_cda_rw | LoRA + CDA + loss reweighting | 0.0880 | 0.0210 | 0.0650 | 0.0273 | 340 | 0.0920 | 0.9183 | 0.7173 | 0.6920 |
| Encoder-Decoder | probe | none (head-only, pre-LoRA) | 0.1650 | 0.0450 | 0.2210 | 0.0928 | 340 | 0.1820 | 0.8107 | 0.6218 | 0.6080 |
| Encoder-Decoder | lora | parameter (LoRA) | 0.1120 | 0.0290 | 0.1560 | 0.0655 | 340 | 0.1340 | 0.8660 | 0.7020 | 0.6710 |
| Encoder-Decoder | lora_cda | LoRA + data (CDA) | 0.1050 | 0.0260 | 0.0980 | 0.0412 | 340 | 0.1190 | 0.8927 | 0.6777 | 0.6680 |
| Encoder-Decoder | lora_cda_rw | LoRA + CDA + loss reweighting | 0.0990 | 0.0240 | 0.0760 | 0.0319 | 340 | 0.1060 | 0.9063 | 0.6814 | 0.6640 |
| Decoder-only | probe | none (head-only, pre-LoRA) | 0.1820 | 0.0490 | 0.2350 | 0.0987 | 340 | 0.1980 | 0.7950 | 0.6419 | 0.6240 |
| Decoder-only | lora | parameter (LoRA) | 0.1280 | 0.0340 | 0.1740 | 0.0731 | 340 | 0.1460 | 0.8507 | 0.6925 | 0.6790 |
| Decoder-only | lora_cda | LoRA + data (CDA) | 0.1180 | 0.0310 | 0.1120 | 0.0470 | 340 | 0.1280 | 0.8807 | 0.7099 | 0.6760 |
| Decoder-only | lora_cda_rw | LoRA + CDA + loss reweighting | 0.1110 | 0.0290 | 0.0890 | 0.0374 | 340 | 0.1140 | 0.8953 | 0.6756 | 0.6710 |

probe = frozen base + trained head (pre-LoRA reference). Definitions: src/bias/bias_analysis.py. Lexicons await native-speaker validation (docs/limitations.md).
