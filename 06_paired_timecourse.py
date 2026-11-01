#!/usr/bin/env python3
"""
06_paired_timecourse.py
=======================
Within-patient (paired) post-transplant time-course analysis of GSE24223 whole
blood: baseline (Pre-Tx) vs each post-transplant week, matched by donor.

Why a separate script
---------------------
The healthy-vs-ESRD comparison (scripts 01-04) is a between-person comparison.
This script uses each patient as their own control, which removes between-
person differences (age, sex, comorbidity) and, because all samples come from
one processing batch, avoids the spike-in batch confound found by script 05.

Pipeline
--------
1. Select samples: cell subtype matching --cell-regex (case-SENSITIVE, default
   '^Whole Blood$', which excludes the separately processed 'Whole blood'
   batch), healthy controls excluded.
2. Spike-in QC: report poly-A spike-in level per time point; drop samples that
   deviate from the selection median by more than --max-spikein-dev log2.
3. Drop AFFX control probes, log2 check/transform, quantile normalisation
   (all selected samples together), drop lowest-variance 20% probes, impute.
4. For each week: paired t-test (one-sample t-test on within-donor differences),
   log2FC = mean(post - pre), Benjamini-Hochberg FDR.
5. Gene annotation from the GPL table.

Outputs
-------
* ./data/paired/<Week>_vs_<baseline>.csv   all probes, per contrast
* ./data/paired/paired_summary.csv         counts of significant probes per contrast
* ./data/paired/paired_top_genes.csv       top genes per contrast
* ./data/paired/paired_sample_table.csv    samples used, with spike-in level
* ./figures/Figure5_Paired_Volcano.png     one volcano per contrast (300 DPI)
* ./figures/FigureS4_Paired_PCA.png        sample PCA coloured by time point

NOTE: large post-transplant changes in whole blood largely reflect
induction-therapy lymphocyte depletion (changed cell composition), not only
changes in per-cell expression. Interpret accordingly.
"""

import argparse
import logging
import re
import sys
from pathlib import Path

import GEOparse
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from scipy import stats  # noqa: E402
from scipy.stats import rankdata  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("06_paired")

UP_COLOR, DOWN_COLOR, NS_COLOR = "#d62728", "#1f77b4", "#b0b0b0"
POLYA_REGEX = r"(?i)(?:lys|phe|dap|thr|trpn)"
SYMBOL_CANDIDATES = ["Gene Symbol", "GENE_SYMBOL", "Gene_Symbol", "Symbol", "SYMBOL", "ILMN_Gene", "gene_symbol"]
NAME_CANDIDATES = ["Gene Title", "GENE_NAME", "Gene_Title", "Definition", "Description", "gene_name", "Gene Name"]


def sanitize(key):
    return re.sub(r"[^0-9a-zA-Z]+", "_", key.strip().lower()).strip("_")


def week_number(label):
    m = re.search(r"\d+", str(label))
    return int(m.group()) if m else 0


def load_series(accession, data_dir, soft_file):
    if soft_file:
        return GEOparse.get_GEO(filepath=str(soft_file), silent=True)
    return GEOparse.get_GEO(geo=accession, destdir=str(data_dir), silent=True)


def sample_table(gse):
    rows = {}
    for gsm_id, gsm in gse.gsms.items():
        chars = {}
        for entry in gsm.metadata.get("characteristics_ch1", []) or []:
            if ":" in entry:
                k, v = entry.split(":", 1)
                chars[sanitize(k)] = v.strip()
        rows[gsm_id] = {"time_point": chars.get("time_point", "NA"),
                        "cell_subtype": chars.get("cell_subtype", "NA"),
                        "donor": chars.get("donor_number", "NA")}
    return pd.DataFrame.from_dict(rows, orient="index")


def expression_matrix(gse, sample_ids):
    series = {}
    for sid in sample_ids:
        tbl = gse.gsms[sid].table
        if tbl is None or tbl.empty or "VALUE" not in tbl.columns:
            log.warning("Sample %s has no VALUE column; skipped.", sid)
            continue
        series[sid] = pd.to_numeric(tbl.set_index("ID_REF")["VALUE"], errors="coerce").groupby(level=0).mean()
    expr = pd.concat(series, axis=1)
    expr.index = expr.index.astype(str)
    return expr


def looks_log2(expr):
    v = expr.to_numpy(float).ravel()
    v = v[np.isfinite(v)]
    q = np.quantile(v, [0.0, 0.25, 0.5, 0.75, 0.99, 1.0])
    return not ((q[4] > 100) or ((q[5] - q[0] > 50) and (q[1] > 0)))


def quantile_normalize(expr):
    arr = expr.to_numpy(float)
    nan_mask = np.isnan(arr)
    filled = np.where(nan_mask, np.nanmedian(arr, axis=1, keepdims=True), arr)
    rank_means = np.sort(filled, axis=0).mean(axis=1)
    grid = np.arange(filled.shape[0])
    out = np.empty_like(filled)
    for j in range(filled.shape[1]):
        out[:, j] = np.interp(rankdata(filled[:, j], method="average") - 1, grid, rank_means)
    out[nan_mask] = np.nan
    return pd.DataFrame(out, index=expr.index, columns=expr.columns)


def find_col(columns, candidates):
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def annotation(gse):
    """probe -> (symbol, name) from the first platform table; empty frame if unavailable."""
    try:
        gpl = next(iter(gse.gpls.values()))
        tbl = gpl.table.copy()
        id_col = find_col(tbl.columns, ["ID", "ID_REF"]) or tbl.columns[0]
        tbl[id_col] = tbl[id_col].astype(str)
        tbl = tbl.drop_duplicates(id_col).set_index(id_col)
        sym, nm = find_col(tbl.columns, SYMBOL_CANDIDATES), find_col(tbl.columns, NAME_CANDIDATES)

        def first(v):
            if pd.isna(v):
                return np.nan
            v = re.split(r"\s*///\s*", str(v).strip())[0].strip()
            return np.nan if v.lower() in {"", "---", "na", "nan"} else v
        out = pd.DataFrame(index=tbl.index)
        out["gene_symbol"] = tbl[sym].map(first) if sym else np.nan
        out["gene_name"] = tbl[nm].map(first) if nm else np.nan
        return out
    except Exception as exc:  # annotation is optional
        log.warning("Could not read platform annotation: %s", exc)
        return pd.DataFrame(columns=["gene_symbol", "gene_name"])


def paired_test(pre, post, alpha, lfc_thr):
    diff = post.to_numpy(float) - pre.to_numpy(float)
    with np.errstate(all="ignore"):
        t, p = stats.ttest_1samp(diff, 0.0, axis=1, nan_policy="omit")
    p = np.where(np.isnan(p), 1.0, p)
    t = np.where(np.isnan(t), 0.0, t)
    _, adj, _, _ = multipletests(p, method="fdr_bh")
    res = pd.DataFrame({"probe_id": pre.index, "mean_pre": pre.mean(axis=1).to_numpy(),
                        "mean_post": post.mean(axis=1).to_numpy(),
                        "log2FC": np.nanmean(diff, axis=1), "t_stat": t, "p_value": p, "adj_p_value": adj})
    sig = (res["adj_p_value"] < alpha) & (res["log2FC"].abs() > lfc_thr)
    res["direction"] = np.where(~sig, "NS", np.where(res["log2FC"] > 0, "Up", "Down"))
    return res


def volcano_grid(results, alpha, lfc_thr, out_path):
    n = len(results)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    sns.set_theme(style="ticks", context="paper", font_scale=1.2)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 4.8 * nrow), squeeze=False)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    for ax, (name, (res, n_pairs)) in zip(axes.ravel(), results.items()):
        y = -np.log10(res["adj_p_value"].clip(lower=1e-300))
        for d, c, s, a in (("NS", NS_COLOR, 6, 0.4), ("Down", DOWN_COLOR, 10, 0.8), ("Up", UP_COLOR, 10, 0.8)):
            m = res["direction"] == d
            ax.scatter(res.loc[m, "log2FC"], y[m], s=s, c=c, alpha=a, edgecolors="none", rasterized=True)
        ax.axvline(lfc_thr, ls="--", lw=0.8, c="k")
        ax.axvline(-lfc_thr, ls="--", lw=0.8, c="k")
        ax.axhline(-np.log10(alpha), ls="--", lw=0.8, c="k")
        ax.set_title(f"{name} (n={n_pairs} pairs)\nUp {int((res.direction == 'Up').sum())} | "
                     f"Down {int((res.direction == 'Down').sum())}")
        ax.set_xlabel("log$_2$ fold change (post vs baseline)")
        ax.set_ylabel("-log$_{10}$ adj. p")
    sns.despine()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out_path)


def pca_figure(expr, meta, out_path):
    X = expr.T.to_numpy(float)
    pca = PCA(n_components=2, random_state=0)
    sc = pca.fit_transform(X - X.mean(axis=0))
    df = pd.DataFrame(sc, index=expr.columns, columns=["PC1", "PC2"]).join(meta[["time_point", "donor"]])
    order = sorted(df["time_point"].unique(), key=lambda t: (0 if t.lower().startswith("pre") else 1, week_number(t)))
    sns.set_theme(style="ticks", context="paper", font_scale=1.3)
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    sns.scatterplot(data=df, x="PC1", y="PC2", hue="time_point", hue_order=order, palette="viridis",
                    s=90, edgecolor="k", linewidth=0.5, ax=ax)
    ev = pca.explained_variance_ratio_ * 100
    ax.set_xlabel(f"PC1 ({ev[0]:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({ev[1]:.1f}% variance)")
    ax.set_title("Whole-blood samples by time point")
    ax.legend(title="Time point", frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    sns.despine()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out_path)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--accession", default="GSE24223")
    p.add_argument("--soft-file", default=None)
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--fig-dir", default="./figures")
    p.add_argument("--cell-regex", default=r"^Whole Blood$",
                   help="Case-sensitive regex on the 'cell subtype' field (default excludes 'Whole blood').")
    p.add_argument("--baseline", default="Pre-Tx")
    p.add_argument("--min-pairs", type=int, default=4)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--lfc-threshold", type=float, default=1.0)
    p.add_argument("--max-spikein-dev", type=float, default=2.0,
                   help="Drop samples whose poly-A spike-in level deviates from the median by more than this (log2).")
    p.add_argument("--top-n", type=int, default=20)
    return p.parse_args()


def main():
    args = parse_args()
    data_dir, fig_dir = Path(args.data_dir), Path(args.fig_dir)
    out_dir = data_dir / "paired"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    log.info("[1/6] Loading series and selecting samples")
    gse = load_series(args.accession, data_dir, args.soft_file)
    meta = sample_table(gse)
    is_patient = ~meta["time_point"].str.contains("control", case=False)
    sel = meta[meta["cell_subtype"].str.contains(args.cell_regex, regex=True) & is_patient].copy()
    if sel.empty:
        raise ValueError(f"No samples match --cell-regex {args.cell_regex!r}. Cell subtypes present: "
                         f"{sorted(meta['cell_subtype'].unique())}")
    log.info("Selected %d patient samples; per time point:\n%s", len(sel), sel["time_point"].value_counts().to_string())

    log.info("[2/6] Building matrix and spike-in QC")
    expr = expression_matrix(gse, sel.index)
    sel = sel.loc[expr.columns]
    aff = expr[expr.index.str.startswith("AFFX")]
    polya = aff[aff.index.str.contains(POLYA_REGEX)]
    sel["polyA_spikein_mean"] = polya.mean(axis=0) if len(polya) else np.nan
    if len(polya):
        med = sel["polyA_spikein_mean"].median()
        bad = (sel["polyA_spikein_mean"] - med).abs() > args.max_spikein_dev
        log.info("Spike-in level per time point (mean log2):\n%s",
                 sel.groupby("time_point")["polyA_spikein_mean"].mean().round(2).to_string())
        if bad.any():
            log.warning("Dropping %d sample(s) with deviating spike-in level (batch outliers): %s",
                        int(bad.sum()), list(sel.index[bad]))
            sel, expr = sel[~bad], expr.loc[:, ~bad.reindex(expr.columns).fillna(False).to_numpy()]
    sel.to_csv(out_dir / "paired_sample_table.csv", index_label="sample_id")

    log.info("[3/6] Normalising")
    expr = expr[~expr.index.str.startswith("AFFX")].dropna(how="all")
    if looks_log2(expr):
        log.info("Values already look log2-scale.")
    else:
        expr = np.log2(expr - min(0.0, float(np.nanmin(expr.to_numpy()))) + 1.0)
        log.info("Applied log2(x + 1).")
    expr = quantile_normalize(expr)
    var = expr.var(axis=1)
    expr = expr.drop(index=var.fillna(-1).nsmallest(int(0.2 * len(var))).index)
    expr = expr.apply(lambda c: c.fillna(expr.median(axis=1)))
    log.info("Matrix after filtering: %d probes x %d samples", *expr.shape)
    pca_figure(expr, sel, fig_dir / "FigureS4_Paired_PCA.png")

    log.info("[4/6] Paired tests per week vs %s", args.baseline)
    donor_expr = {}
    for tp, grp in sel.groupby("time_point"):
        per_donor = {}
        for donor, sub in grp.groupby("donor"):
            per_donor[donor] = expr[sub.index].mean(axis=1)
        donor_expr[tp] = pd.DataFrame(per_donor)
    if args.baseline not in donor_expr:
        raise ValueError(f"Baseline '{args.baseline}' not among time points: {sorted(donor_expr)}")
    pre_all = donor_expr[args.baseline]
    weeks = sorted([t for t in donor_expr if t != args.baseline], key=week_number)
    annot = annotation(gse)
    results, summary, tops = {}, [], []
    for tp in weeks:
        common = pre_all.columns.intersection(donor_expr[tp].columns)
        if len(common) < args.min_pairs:
            log.warning("%s: only %d matched donors (< %d); skipped.", tp, len(common), args.min_pairs)
            continue
        res = paired_test(pre_all[common], donor_expr[tp][common], args.alpha, args.lfc_threshold)
        res = res.merge(annot, left_on="probe_id", right_index=True, how="left").sort_values("adj_p_value")
        name = f"{tp.replace(' ', '')}_vs_{args.baseline}"
        res.to_csv(out_dir / f"{name}.csv", index=False)
        results[f"{tp} vs {args.baseline}"] = (res, len(common))
        n_up, n_dn = int((res.direction == "Up").sum()), int((res.direction == "Down").sum())
        log.info("%s: %d pairs, significant = %d (Up %d, Down %d)", tp, len(common), n_up + n_dn, n_up, n_dn)
        summary.append({"contrast": f"{tp} vs {args.baseline}", "n_pairs": len(common),
                        "n_significant": n_up + n_dn, "n_up": n_up, "n_down": n_dn})
        tops.append(res.head(args.top_n).assign(contrast=f"{tp} vs {args.baseline}"))
    if not results:
        raise ValueError("No contrast had enough matched donors.")

    log.info("[5/6] Figures")
    volcano_grid(results, args.alpha, args.lfc_threshold, fig_dir / "Figure5_Paired_Volcano.png")

    log.info("[6/6] Summary tables")
    pd.DataFrame(summary).to_csv(out_dir / "paired_summary.csv", index=False)
    pd.concat(tops).to_csv(out_dir / "paired_top_genes.csv", index=False)
    log.info("Saved tables in %s", out_dir)
    log.info("Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log.error("Pipeline step failed: %s", exc)
        sys.exit(1)
