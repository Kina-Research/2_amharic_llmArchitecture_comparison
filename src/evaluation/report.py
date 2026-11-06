"""Build every table and figure from results/raw/ (spec §5).

Never invents numbers: any missing run is shown as "n/a" and listed in
results/tables/STATUS.md. Entry point: scripts/generate_results.py.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.evaluation import statistical_tests as st
from src.evaluation.metrics import expected_calibration_error
from src.utils import ARCH_LABEL, MODEL_KEYS, load_config, load_json

NA = "n/a"


# ------------------------------------------------------------------ loading
def collect(results_dir: Path) -> dict:
    """runs[(task, model, condition)] -> list of per-seed dicts (sorted by seed)."""
    runs: dict = {}
    for mfile in sorted((results_dir / "raw").glob("*/*/*/seed*/metrics.json")):
        d = mfile.parent
        task, model, cond = d.parts[-4], d.parts[-3], d.parts[-2]
        rec = {"dir": d, "seed": int(d.name[4:]), "metrics": load_json(mfile)}
        for name in ("efficiency", "bias", "run_info"):
            f = d / f"{name}.json"
            rec[name] = load_json(f) if f.exists() else None
        rec["pred"] = pd.read_csv(d / "predictions.csv") if (d / "predictions.csv").exists() else None
        runs.setdefault((task, model, cond), []).append(rec)
    for v in runs.values():
        v.sort(key=lambda r: r["seed"])
    return runs


def _get(runs, task, model, cond="lora"):
    return runs.get((task, model, cond), [])


def _fmt(x, nd=4):
    return NA if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def _ci(c, nd=4):
    return f"{c['point']:.{nd}f} [{c['lo']:.{nd}f}, {c['hi']:.{nd}f}]"


def _seed_mean(rs, fn):
    vals = [fn(r) for r in rs]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    return float(np.mean(vals)), (float(np.std(vals, ddof=1)) if len(vals) > 1 else None)


def _mstd(rs, fn, nd=4):
    m, s = _seed_mean(rs, fn)
    if m is None:
        return NA
    return f"{m:.{nd}f}" + (f" ± {s:.{nd}f}" if s is not None else "")


# ------------------------------------------------------------------ writing
def to_markdown(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.values) + " |")
    return "\n".join(lines) + "\n"


def write_table(df: pd.DataFrame, out: Path, name: str, title: str, notes: str = "") -> None:
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{name}.csv", index=False, encoding="utf-8")
    (out / f"{name}.md").write_text(f"### {title}\n\n{to_markdown(df)}\n{notes}", encoding="utf-8")


# ------------------------------------------------------------------ bootstrap helpers
def _cls_cis(pred: pd.DataFrame, n_boot: int, seed: int) -> dict:
    y, p = pred["gold"].values, pred["pred"].values
    acc = st.bootstrap_ci(lambda i: float((y[i] == p[i]).mean()), len(y), n_boot, seed=seed)
    f1 = st.bootstrap_ci(lambda i: float(f1_score(y[i], p[i], average="macro", labels=[0, 1, 2],
                                                   zero_division=0)), len(y), n_boot, seed=seed)
    probs = pred[[c for c in pred.columns if c.startswith("p_")]].values
    ece = st.bootstrap_ci(lambda i: expected_calibration_error(probs[i], y[i])["ece"], len(y), n_boot, seed=seed)
    return {"acc": acc, "f1": f1, "ece": ece}


def _nlg_ci(pred: pd.DataFrame, col: str, n_boot: int, seed: int):
    if pred is None or col not in pred:
        return None
    v = pred[col].values
    return st.bootstrap_ci(lambda i: float(v[i].mean()), len(v), n_boot, seed=seed)


# ------------------------------------------------------------------ tables
def table_performance(runs, cfg) -> pd.DataFrame:
    nb, seed = cfg["evaluation"]["bootstrap_samples"], cfg["experiment"]["seeds"][0]
    rows = []
    for m in MODEL_KEYS:
        s, z = _get(runs, "sentiment", m), _get(runs, "summarization", m)
        cis = _cls_cis(s[0]["pred"], nb, seed) if s else None
        rl = _nlg_ci(z[0]["pred"], "rougeL", nb, seed) if z else None
        bs = _nlg_ci(z[0]["pred"], "bertscore_f1", nb, seed) if z else None
        eff = lambda r, st_, k: (r["efficiency"] or {}).get(st_, {}).get(k)  # noqa: E731
        rows.append({
            "Architecture": ARCH_LABEL[m], "Model": cfg["models"][m]["hf_id"],
            "Trainable params (sent., %)": _mstd(s, lambda r: r["run_info"]["params"]["trainable_pct"], 3) if s else NA,
            "Sentiment Acc [95% CI]": _ci(cis["acc"]) if cis else NA,
            "Sentiment Macro-F1 [95% CI]": _ci(cis["f1"]) if cis else NA,
            "Seeds (sent.)": len(s),
            "ROUGE-1": _mstd(z, lambda r: r["metrics"]["rouge1"]) if z else NA,
            "ROUGE-2": _mstd(z, lambda r: r["metrics"]["rouge2"]) if z else NA,
            "ROUGE-L [95% CI]": _ci(rl) if rl else NA,
            "BERTScore-F1 [95% CI]": _ci(bs) if bs else NA,
            "sacreBLEU": _mstd(z, lambda r: r["metrics"]["sacrebleu"], 2) if z else NA,
            "Latency sent. (ms/sample, bs=1)": _mstd(s, lambda r: eff(r, "latency_bs1", "latency_ms_mean"), 1) if s else NA,
            "Latency summ. (ms/sample, bs=1)": _mstd(z, lambda r: eff(r, "latency_bs1", "latency_ms_mean"), 0) if z else NA,
            "Throughput sent. (samples/s, batched)": _mstd(s, lambda r: eff(r, "inference_val", "throughput_batched_samples_per_s"), 1) if s else NA,
            "Peak VRAM train sent. (GB)": _mstd(s, lambda r: eff(r, "train", "peak_vram_gb"), 2) if s else NA,
            "Peak VRAM train summ. (GB)": _mstd(z, lambda r: eff(r, "train", "peak_vram_gb"), 2) if z else NA,
            "Peak VRAM infer sent. (GB)": _mstd(s, lambda r: eff(r, "inference_val", "peak_vram_gb"), 2) if s else NA,
            "Bias Robustness (↑)": _mstd(s, lambda r: (r["bias"] or {}).get("bias_robustness")) if s else NA,
        })
    return pd.DataFrame(rows)


def table_bias(runs, cfg) -> pd.DataFrame:
    rows = []
    for m in MODEL_KEYS:
        for c in cfg["bias"]["conditions"]:
            rs = [r for r in _get(runs, "sentiment", m, c) if r["bias"]]
            g = lambda path: (lambda r: _dig(r["bias"], path))  # noqa: E731
            rows.append({
                "Architecture": ARCH_LABEL[m], "Condition": c,
                "Mitigation": {"probe": "none (head-only, pre-LoRA)", "lora": "parameter (LoRA)",
                               "lora_cda": "LoRA + data (CDA)", "lora_cda_rw": "LoRA + CDA + loss reweighting"}[c],
                "Positional flip rate (↓)": _mstd(rs, g("positional.flip_rate")) if rs else NA,
                "Positional acc gap (↓)": _mstd(rs, g("positional.acc_gap")) if rs else NA,
                "Counterfactual flip rate (↓)": _mstd(rs, g("counterfactual.flip_rate")) if rs else NA,
                "Counterfactual TV (↓)": _mstd(rs, g("counterfactual.tv")) if rs else NA,
                "CF natural pairs (n)": _dig(rs[0]["bias"], "counterfactual.natural.n") if rs else NA,
                "Cultural disparity (↓)": _mstd(rs, g("cultural.disparity")) if rs else NA,
                "Bias Robustness (↑)": _mstd(rs, g("bias_robustness")) if rs else NA,
                "Accuracy": _mstd(rs, lambda r: r["metrics"]["accuracy"]) if rs else NA,
                "Macro-F1": _mstd(rs, lambda r: r["metrics"]["macro_f1"]) if rs else NA,
            })
    return pd.DataFrame(rows)


def _dig(d, path):
    for k in path.split("."):
        if d is None:
            return None
        d = d.get(k)
    return d


def table_stats(runs, cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    nb, seed = cfg["evaluation"]["bootstrap_samples"], cfg["experiment"]["seeds"][0]
    per_model = []
    for m in MODEL_KEYS:
        s = _get(runs, "sentiment", m)
        if not s:
            per_model.append({"Architecture": ARCH_LABEL[m], "ECE (15 bins) [95% CI]": NA, "Brier": NA,
                              "Mean confidence": NA, "Accuracy": NA, "Positional flip rate": NA,
                              "Positional acc (first / last)": NA})
            continue
        r0 = s[0]
        cis = _cls_cis(r0["pred"], nb, seed)
        cal, pos = r0["metrics"]["calibration"], (r0["bias"] or {}).get("positional", {})
        per_model.append({
            "Architecture": ARCH_LABEL[m], "ECE (15 bins) [95% CI]": _ci(cis["ece"]),
            "Brier": _fmt(cal["brier"]), "Mean confidence": _fmt(cal["mean_confidence"]),
            "Accuracy": _fmt(r0["metrics"]["accuracy"]),
            "Positional flip rate": _fmt(pos.get("flip_rate")),
            "Positional acc (first / last)": f"{_fmt(pos.get('acc_text_first'))} / {_fmt(pos.get('acc_text_last'))}"
            if pos else NA,
        })

    tests, raw_p = [], {}
    for a, b in st.pairwise(list(MODEL_KEYS)):
        ra, rb = _get(runs, "sentiment", a), _get(runs, "sentiment", b)
        if ra and rb:
            pa, pb = ra[0]["pred"], rb[0]["pred"]
            if not (pa["id"].values == pb["id"].values).all():
                raise ValueError("prediction files not aligned")
            y = pa["gold"].values
            mc = st.mcnemar(y, pa["pred"].values, pb["pred"].values)
            f1 = lambda idx, w: float(f1_score(y[idx], (pa if w == "a" else pb)["pred"].values[idx],  # noqa: E731
                                               average="macro", labels=[0, 1, 2], zero_division=0))
            pbt = st.paired_bootstrap(f1, len(y), nb, seed)
            key = f"sent:{a}-{b}"
            raw_p[key + ":mcnemar"], raw_p[key + ":f1"] = mc["p_value"], pbt["p_value"]
            tests.append({"key": key, "Task": "Sentiment", "Comparison": f"{ARCH_LABEL[a]} vs {ARCH_LABEL[b]}",
                          "Test": f"McNemar ({mc['method']}); n01={mc['n01']}, n10={mc['n10']}",
                          "Effect": "—", "p": mc["p_value"], "pkey": key + ":mcnemar"})
            tests.append({"key": key, "Task": "Sentiment", "Comparison": f"{ARCH_LABEL[a]} vs {ARCH_LABEL[b]}",
                          "Test": "Paired bootstrap, Δ macro-F1",
                          "Effect": f"{pbt['delta']:+.4f} [{pbt['lo']:+.4f}, {pbt['hi']:+.4f}]",
                          "p": pbt["p_value"], "pkey": key + ":f1"})
        za, zb = _get(runs, "summarization", a), _get(runs, "summarization", b)
        if za and zb:
            for col in ("rougeL", "bertscore_f1"):
                if col not in za[0]["pred"] or col not in zb[0]["pred"]:
                    continue
                va, vb = za[0]["pred"][col].values, zb[0]["pred"][col].values
                pbt = st.paired_bootstrap(lambda idx, w: float((va if w == "a" else vb)[idx].mean()), len(va), nb, seed)
                wx = st.wilcoxon(va, vb)
                key = f"summ:{a}-{b}:{col}"
                raw_p[key + ":boot"], raw_p[key + ":wil"] = pbt["p_value"], wx["p_value"]
                tests.append({"key": key, "Task": "Summarization", "Comparison": f"{ARCH_LABEL[a]} vs {ARCH_LABEL[b]}",
                              "Test": f"Paired bootstrap, Δ {col}",
                              "Effect": f"{pbt['delta']:+.4f} [{pbt['lo']:+.4f}, {pbt['hi']:+.4f}]",
                              "p": pbt["p_value"], "pkey": key + ":boot"})
                tests.append({"key": key, "Task": "Summarization", "Comparison": f"{ARCH_LABEL[a]} vs {ARCH_LABEL[b]}",
                              "Test": f"Wilcoxon signed-rank, {col}", "Effect": "—",
                              "p": wx["p_value"], "pkey": key + ":wil"})
    # Mitigation effect within each architecture (probe -> lora_cda_rw)
    for m in MODEL_KEYS:
        r0, r1 = _get(runs, "sentiment", m, "probe"), _get(runs, "sentiment", m, "lora_cda_rw")
        if r0 and r1:
            y = r0[0]["pred"]["gold"].values
            mc = st.mcnemar(y, r0[0]["pred"]["pred"].values, r1[0]["pred"]["pred"].values)
            key = f"mitig:{m}"
            raw_p[key] = mc["p_value"]
            tests.append({"key": key, "Task": "Sentiment (mitigation)", "Comparison": f"{ARCH_LABEL[m]}: probe vs LoRA+CDA+RW",
                          "Test": f"McNemar ({mc['method']}); n01={mc['n01']}, n10={mc['n10']}", "Effect": "—",
                          "p": mc["p_value"], "pkey": key})
    adj = st.holm(raw_p) if raw_p else {}
    tdf = pd.DataFrame(tests)
    if len(tdf):
        tdf["p (Holm)"] = tdf["pkey"].map(adj).map(lambda v: f"{v:.4g}")
        tdf["p"] = tdf["p"].map(lambda v: f"{v:.4g}")
        tdf = tdf.drop(columns=["key", "pkey"])
    return pd.DataFrame(per_model), tdf


def table_energy(runs, cfg) -> pd.DataFrame:
    rows = []
    for task in ("sentiment", "summarization"):
        for m in MODEL_KEYS:
            rs = _get(runs, task, m)
            istage = "inference_val" if task == "sentiment" else "inference_eval"
            e = lambda st_, k: (lambda r: ((r["efficiency"] or {}).get(st_) or {}).get(k))  # noqa: E731

            def j_per_sample(r):
                inf = (r["efficiency"] or {}).get(istage) or {}
                n = inf.get("n") or (len(r["pred"]) if r["pred"] is not None else None)
                return inf["nvml_gpu_kwh"] * 3.6e6 / n if inf.get("nvml_gpu_kwh") and n else None

            rows.append({
                "Task": task, "Architecture": ARCH_LABEL[m],
                "Train wall time (min)": _mstd(rs, lambda r: (e("train", "wall_s")(r) or 0) / 60, 1) if rs else NA,
                "Train GPU energy, NVML (kWh)": _mstd(rs, e("train", "nvml_gpu_kwh"), 4) if rs else NA,
                "Train total energy, CodeCarbon (kWh)": _mstd(rs, e("train", "codecarbon_kwh"), 4) if rs else NA,
                "Train CO2eq, CodeCarbon (kg)": _mstd(rs, e("train", "codecarbon_kg_co2eq"), 4) if rs else NA,
                "Mean GPU power train (W)": _mstd(rs, e("train", "nvml_mean_power_w"), 1) if rs else NA,
                "Inference GPU energy (J/sample, batched)": _mstd(rs, j_per_sample, 2) if rs else NA,
                "Inference energy, CodeCarbon (kWh)": _mstd(rs, e(istage, "codecarbon_kwh"), 5) if rs else NA,
            })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ figures
def _save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")


def pareto_front(xs, ys):
    """Indices on the frontier minimising x and maximising y."""
    order = np.argsort(xs)
    front, best = [], -np.inf
    for i in order:
        if ys[i] > best:
            front.append(i)
            best = ys[i]
    return front


def figures(runs, cfg, out: Path) -> list[str]:
    os.environ.setdefault("MPLCONFIGDIR", str(out / ".mplcache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    made, colors = [], {"mbert": "#1f77b4", "mt5": "#2ca02c", "bloomz": "#d62728"}
    nb, seed = cfg["evaluation"]["bootstrap_samples"], cfg["experiment"]["seeds"][0]

    # Fig 1: Pareto frontier (accuracy/F1 vs latency; ROUGE-L vs latency)
    panels = [("sentiment", "Macro-F1 (sentiment)", lambda r: _cls_cis(r["pred"], nb, seed)["f1"]),
              ("summarization", "ROUGE-L (summarization)", lambda r: _nlg_ci(r["pred"], "rougeL", nb, seed))]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    any_pts = False
    for ax, (task, ylab, cifn) in zip(axes, panels):
        pts = []
        for m in MODEL_KEYS:
            rs = _get(runs, task, m)
            if not rs or not rs[0]["efficiency"]:
                continue
            lat = rs[0]["efficiency"]["latency_bs1"]["latency_ms_mean"]
            ci = cifn(rs[0])
            pts.append((m, lat, ci))
            ax.errorbar(lat, ci["point"], yerr=[[ci["point"] - ci["lo"]], [ci["hi"] - ci["point"]]],
                        fmt="o", ms=8, capsize=4, color=colors[m], label=ARCH_LABEL[m])
            ax.annotate(cfg["models"][m]["hf_id"].split("/")[-1], (lat, ci["point"]),
                        textcoords="offset points", xytext=(6, 6), fontsize=8)
        if pts:
            any_pts = True
            xs, ys = np.array([p[1] for p in pts]), np.array([p[2]["point"] for p in pts])
            f = pareto_front(xs, ys)
            ax.plot(xs[f], ys[f], "--", color="gray", lw=1, label="Pareto frontier")
        ax.set_xscale("log")
        ax.set_xlabel("Latency (ms/sample, batch size 1, log scale)")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Quality vs. latency (95% bootstrap CIs)")
    if any_pts:
        _save(fig, out, "fig1_pareto_quality_vs_latency")
        made.append("fig1_pareto_quality_vs_latency")
    plt.close(fig)

    # Fig 2: energy
    edf = []
    for task in ("sentiment", "summarization"):
        for m in MODEL_KEYS:
            rs = _get(runs, task, m)
            if rs and rs[0]["efficiency"] and rs[0]["efficiency"]["train"].get("nvml_gpu_kwh") is not None:
                istage = "inference_val" if task == "sentiment" else "inference_eval"
                edf.append((task, m, rs[0]["efficiency"]["train"]["nvml_gpu_kwh"],
                            rs[0]["efficiency"][istage].get("nvml_gpu_kwh") or 0.0))
    if edf:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, task in zip(axes, ("sentiment", "summarization")):
            d = [x for x in edf if x[0] == task]
            ax.bar([ARCH_LABEL[x[1]] for x in d], [x[2] * 1000 for x in d], color=[colors[x[1]] for x in d],
                   label="training")
            ax.bar([ARCH_LABEL[x[1]] for x in d], [x[3] * 1000 for x in d], bottom=[x[2] * 1000 for x in d],
                   color="lightgray", label="inference (eval set)")
            ax.set_title(task)
            ax.set_ylabel("GPU energy, NVML-measured (Wh)")
            ax.legend(fontsize=8)
        fig.suptitle("Measured GPU energy per architecture")
        _save(fig, out, "fig2_energy")
        made.append("fig2_energy")
        plt.close(fig)

    # Fig 3: bias pre/post mitigation
    conds = cfg["bias"]["conditions"]
    metrics = [("positional.flip_rate", "Positional flip"), ("counterfactual.flip_rate", "Counterfactual flip"),
               ("cultural.disparity", "Cultural disparity")]
    if any(_get(runs, "sentiment", m, c) for m in MODEL_KEYS for c in conds):
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
        w = 0.8 / len(conds)
        for ax, (path, title) in zip(axes, metrics):
            for j, c in enumerate(conds):
                vals = [(_dig((_get(runs, "sentiment", m, c) or [{"bias": None}])[0]["bias"], path) or np.nan)
                        for m in MODEL_KEYS]
                ax.bar(np.arange(3) + j * w, vals, w, label=c)
            ax.set_xticks(np.arange(3) + w * (len(conds) - 1) / 2, [ARCH_LABEL[m] for m in MODEL_KEYS], fontsize=8)
            ax.set_title(title + " (↓)")
        axes[0].legend(fontsize=8)
        _save(fig, out, "fig3_bias_pre_post_mitigation")
        made.append("fig3_bias_pre_post_mitigation")
        plt.close(fig)

    # Fig 4: reliability diagrams; Fig 5: confusion matrices
    avail = [m for m in MODEL_KEYS if _get(runs, "sentiment", m)]
    if avail:
        fig, axes = plt.subplots(1, len(avail), figsize=(4.2 * len(avail), 4), squeeze=False)
        for ax, m in zip(axes[0], avail):
            cal = _get(runs, "sentiment", m)[0]["metrics"]["calibration"]
            b = cal["bins"]
            ax.bar([x["conf"] for x in b], [x["acc"] for x in b], width=1 / 15, alpha=0.7, edgecolor="k")
            ax.plot([0, 1], [0, 1], "k--", lw=1)
            ax.set_title(f"{ARCH_LABEL[m]}  ECE={cal['ece']:.4f}")
            ax.set_xlabel("confidence")
            ax.set_ylabel("accuracy")
        _save(fig, out, "fig4_reliability_diagrams")
        made.append("fig4_reliability_diagrams")
        plt.close(fig)

        labels = cfg["datasets"]["sentiment"]["labels"]
        fig, axes = plt.subplots(1, len(avail), figsize=(4.2 * len(avail), 3.8), squeeze=False)
        for ax, m in zip(axes[0], avail):
            cm = np.array(_get(runs, "sentiment", m)[0]["metrics"]["confusion_matrix"])
            ax.imshow(cm, cmap="Blues")
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=9)
            ax.set_xticks(range(len(labels)), labels, fontsize=8)
            ax.set_yticks(range(len(labels)), labels, fontsize=8)
            ax.set_title(ARCH_LABEL[m])
            ax.set_xlabel("predicted")
            ax.set_ylabel("gold")
        _save(fig, out, "fig5_confusion_matrices")
        made.append("fig5_confusion_matrices")
        plt.close(fig)
    return made


NLG_CAVEATS = """
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
"""


def build_all(results_dir: str | Path = "results", config: str | None = None) -> dict:
    cfg = load_config(config)
    results_dir = Path(results_dir)
    runs = collect(results_dir)
    tdir, fdir = results_dir / "tables", results_dir / "figures"
    seeds_note = f"\nRuns found: {len(runs)} (task, model, condition) groups.\n"
    write_table(table_performance(runs, cfg), tdir, "table1_performance_efficiency",
                "Table 1. Task performance and efficiency trade-offs (LoRA r=8, α=16)",
                "CIs: 1,000-sample percentile bootstrap over the evaluation set. Latency: mean over 100 "
                "(sentiment) / 50 (summarization) single-sample forward passes on the GPU in env.json."
                + seeds_note)
    write_table(table_bias(runs, cfg), tdir, "table2_bias_pre_post_mitigation",
                "Table 2. Positional, counterfactual and cultural bias, pre- and post-mitigation (sentiment)",
                "probe = frozen base + trained head (pre-LoRA reference). Definitions: src/bias/bias_analysis.py. "
                "Lexicons await native-speaker validation (docs/limitations.md).\n")
    per_model, tests = table_stats(runs, cfg)
    write_table(per_model, tdir, "table3a_calibration_positional",
                "Table 3a. Calibration (ECE) and positional sensitivity", "ECE is reported as observed.\n")
    write_table(tests, tdir, "table3b_significance_tests",
                "Table 3b. Pairwise significance tests (Holm–Bonferroni adjusted)", NLG_CAVEATS)
    write_table(table_energy(runs, cfg), tdir, "table4_energy",
                "Table 4. Energy consumption", "NVML = measured GPU board power (100 ms sampling). "
                "CodeCarbon = modelled CPU+GPU+RAM. See docs/methodology.md §Energy.\n")
    made = figures(runs, cfg, fdir)

    expected = ([("sentiment", m, c) for m in MODEL_KEYS for c in cfg["bias"]["conditions"]]
                + [("summarization", m, "lora") for m in MODEL_KEYS])
    missing = [k for k in expected if k not in runs]
    status = ["# Results status", "", f"Found {len(runs)}/{len(expected)} expected run groups.", ""]
    status += ["Missing (shown as n/a in tables):"] + [f"- {t}/{m}/{c}" for t, m, c in missing] if missing else ["All runs present."]
    status += ["", "Figures generated: " + (", ".join(made) or "none")]
    # Real runs (src/training/*) always record the software/hardware environment in
    # run_info.json; scripts/populate_benchmark_results.py does not. Flag the latter.
    unverified = sorted(f"{t}/{m}/{c}/seed{r['seed']}" for (t, m, c), rs in runs.items() for r in rs
                        if not (r["run_info"] or {}).get("environment"))
    if unverified:
        status += ["", f"WARNING: {len(unverified)} run(s) have no captured environment in run_info.json and are",
                   "probably SIMULATED placeholders (scripts/populate_benchmark_results.py), not experiments:"]
        status += [f"- {u}" for u in unverified]
    (tdir / "STATUS.md").write_text("\n".join(status) + "\n", encoding="utf-8")
    return {"runs": len(runs), "missing": missing, "figures": made, "unverified_runs": len(unverified)}

