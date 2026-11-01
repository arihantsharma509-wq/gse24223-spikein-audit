#!/usr/bin/env python3
"""
02_differential_expression.py
=============================
Probe-wise differential expression between Control and Disease cohorts.

* Welch's two-sample t-test per probe (scipy.stats.ttest_ind, equal_var=False)
* Benjamini-Hochberg FDR (statsmodels multipletests)
* log2FC = mean(Disease) - mean(Control)   (data are on the log2 scale)
* Significant: adj. p < 0.05 and |log2FC| > 1.0 (configurable)
* Volcano plot -> ./figures/Figure1_VolcanoPlot.png (300 DPI)
* Sample PCA (QC for batch / confounding) -> ./figures/FigureS2_PCA.png
  and ./data/pca_coordinates.csv
* Table -> ./data/differentially_expressed_genes.csv
  (all-probe statistics -> ./data/all_probe_statistics.csv)
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("02_de")

UP_COLOR, DOWN_COLOR, NS_COLOR = "#d62728", "#1f77b4", "#b0b0b0"


def load_inputs(data_dir):
    expr_path = data_dir / "cleaned_expression.csv"
    meta_path = data_dir / "sample_metadata.csv"
    for p in (expr_path, meta_path):
        if not p.exists():
            raise FileNotFoundError(f"{p} not found. Run 01_ingest_normalize.py first.")
    expr = pd.read_csv(expr_path, index_col=0)
    meta = pd.read_csv(meta_path, index_col=0)
    common = expr.columns.intersection(meta.index)
    if len(common) < len(expr.columns):
        log.warning("%d expression columns lack metadata; ignored.", len(expr.columns) - len(common))
    expr, meta = expr[common], meta.loc[common]
    return expr, meta


def run_welch(expr, meta, alpha, lfc_thr):
    ctrl = meta.index[meta["group"] == "Control"]
    dis = meta.index[meta["group"] == "Disease"]
    if len(ctrl) < 2 or len(dis) < 2:
        raise ValueError(f"Need >= 2 samples per cohort (Control={len(ctrl)}, Disease={len(dis)}).")
    log.info("Control n=%d, Disease n=%d, probes=%d", len(ctrl), len(dis), len(expr))

    a, b = expr[dis].to_numpy(float), expr[ctrl].to_numpy(float)
    with np.errstate(all="ignore"):
        t_stat, p_val = stats.ttest_ind(a, b, axis=1, equal_var=False, nan_policy="omit")
    p_val = np.where(np.isnan(p_val), 1.0, p_val)  # zero-variance probes -> not significant
    t_stat = np.where(np.isnan(t_stat), 0.0, t_stat)

    _, adj_p, _, _ = multipletests(p_val, method="fdr_bh")
    res = pd.DataFrame({
        "probe_id": expr.index,
        "mean_control": np.nanmean(b, axis=1),
        "mean_disease": np.nanmean(a, axis=1),
        "log2FC": np.nanmean(a, axis=1) - np.nanmean(b, axis=1),
        "t_stat": t_stat,
        "p_value": p_val,
        "adj_p_value": adj_p,
    })
    res["significant"] = (res["adj_p_value"] < alpha) & (res["log2FC"].abs() > lfc_thr)
    res["direction"] = np.where(~res["significant"], "NS",
                                np.where(res["log2FC"] > 0, "Up", "Down"))
    return res


def pca_plot(expr, meta, out_png, out_csv, shape_col=None, annotate=True):
    """PCA of samples on all probes (centred). If cohorts separate on PC1/PC2 for
    technical reasons (batch, cell mix), DE results are suspect."""
    X = expr.T.to_numpy(float)
    n_comp = min(5, X.shape[0] - 1)
    pca = PCA(n_components=n_comp, random_state=0)
    scores = pca.fit_transform(X - X.mean(axis=0))
    cols = [f"PC{i + 1}" for i in range(n_comp)]
    df = pd.DataFrame(scores, index=expr.columns, columns=cols)
    df["group"] = meta.loc[df.index, "group"].values
    if shape_col:
        if shape_col not in meta.columns:
            log.warning("PCA shape column '%s' not found; ignoring. Available: %s", shape_col,
                        [c for c in meta.columns if c.startswith("char_")])
            shape_col = None
        else:
            df[shape_col] = meta.loc[df.index, shape_col].astype(str).values
    df.to_csv(out_csv, index_label="sample_id")

    ev = pca.explained_variance_ratio_ * 100
    g = df["group"] == "Disease"
    for c in cols[:3]:
        if g.sum() > 1 and (~g).sum() > 1:
            p = stats.ttest_ind(df.loc[g, c], df.loc[~g, c], equal_var=False).pvalue
            log.info("%s (%.1f%% var): Disease vs Control Welch p = %.3g", c, ev[cols.index(c)], p)

    sns.set_theme(style="ticks", context="paper", font_scale=1.4)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    sns.scatterplot(data=df, x="PC1", y="PC2", hue="group", style=shape_col,
                    palette={"Control": DOWN_COLOR, "Disease": UP_COLOR}, s=110,
                    edgecolor="k", linewidth=0.6, ax=ax)
    if annotate:
        for sid, r in df.iterrows():
            ax.annotate(sid, (r["PC1"], r["PC2"]), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel(f"PC1 ({ev[0]:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({ev[1]:.1f}% variance)")
    ax.set_title("Sample PCA (all probes)")
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    sns.despine()
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out_png)


def volcano_plot(res, alpha, lfc_thr, out_path, label_top=0):
    df = res.copy()
    df["neg_log10_adj_p"] = -np.log10(df["adj_p_value"].clip(lower=1e-300))
    palette = {"Up": UP_COLOR, "Down": DOWN_COLOR, "NS": NS_COLOR}
    counts = df["direction"].value_counts()

    sns.set_theme(style="ticks", context="paper", font_scale=1.4)
    fig, ax = plt.subplots(figsize=(8, 7))
    order = ["NS", "Down", "Up"]  # draw significant points last (on top)
    for d in order:
        sub = df[df["direction"] == d]
        sns.scatterplot(data=sub, x="log2FC", y="neg_log10_adj_p", color=palette[d],
                        s=14 if d == "NS" else 26, alpha=0.5 if d == "NS" else 0.85,
                        edgecolor="none", ax=ax,
                        label=f"{'Upregulated' if d == 'Up' else 'Downregulated' if d == 'Down' else 'Not significant'} "
                              f"(n={counts.get(d, 0)})")
    ax.axvline(lfc_thr, ls="--", lw=0.9, color="k")
    ax.axvline(-lfc_thr, ls="--", lw=0.9, color="k")
    ax.axhline(-np.log10(alpha), ls="--", lw=0.9, color="k")

    if label_top > 0:
        for _, r in df[df["significant"]].nsmallest(label_top, "adj_p_value").iterrows():
            ax.annotate(r["probe_id"], (r["log2FC"], r["neg_log10_adj_p"]), fontsize=8,
                        xytext=(3, 3), textcoords="offset points")

    ax.set_xlabel("log$_2$ fold change (Disease vs Control)")
    ax.set_ylabel("-log$_{10}$ (BH-adjusted p-value)")
    ax.set_title("Differential expression: Disease vs Control PBMC")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=1)
    sns.despine()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out_path)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--fig-dir", default="./figures")
    p.add_argument("--alpha", type=float, default=0.05, help="Adjusted p-value cutoff.")
    p.add_argument("--lfc-threshold", type=float, default=1.0, help="|log2FC| cutoff.")
    p.add_argument("--fallback-top-n", type=int, default=0,
                   help="If no probe passes the thresholds, export the N probes with the smallest "
                        "adjusted p-value instead (flagged relaxed_selection=True). Default: off.")
    p.add_argument("--pca-shape-col", default=None,
                   help="Metadata column to shape-code in the PCA, e.g. char_donor_number or char_cell_subtype.")
    p.add_argument("--no-pca-labels", action="store_true", help="Do not print sample IDs on the PCA plot.")
    p.add_argument("--label-top", type=int, default=0, help="Label the top-N significant probes on the volcano plot.")
    return p.parse_args()


def main():
    args = parse_args()
    data_dir, fig_dir = Path(args.data_dir), Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    log.info("[1/4] Loading cleaned data")
    expr, meta = load_inputs(data_dir)

    log.info("PCA quality-control plot")
    pca_plot(expr, meta, fig_dir / "FigureS2_PCA.png", data_dir / "pca_coordinates.csv",
             args.pca_shape_col, not args.no_pca_labels)

    log.info("[2/4] Welch t-tests + BH correction")
    res = run_welch(expr, meta, args.alpha, args.lfc_threshold)
    res.sort_values("adj_p_value").to_csv(data_dir / "all_probe_statistics.csv", index=False)

    n_sig = int(res["significant"].sum())
    log.info("Significant probes (adj.p<%.3g, |log2FC|>%.2f): %d (Up=%d, Down=%d)",
             args.alpha, args.lfc_threshold, n_sig,
             int((res["direction"] == "Up").sum()), int((res["direction"] == "Down").sum()))

    deg = res[res["significant"]].copy()
    deg["relaxed_selection"] = False
    if n_sig == 0:
        log.warning("No probes passed the thresholds. Consider --lfc-threshold 0.58 (1.5-fold) or "
                    "--fallback-top-n.")
        if args.fallback_top_n > 0:
            deg = res.nsmallest(args.fallback_top_n, "adj_p_value").copy()
            deg["relaxed_selection"] = True
            log.warning("Fallback: exporting top %d probes by adjusted p-value (NOT thresholded).", len(deg))

    log.info("[3/4] Volcano plot")
    volcano_plot(res, args.alpha, args.lfc_threshold, fig_dir / "Figure1_VolcanoPlot.png", args.label_top)

    log.info("[4/4] Exporting DEG table")
    deg = deg.sort_values("adj_p_value").drop(columns=["significant"])
    out = data_dir / "differentially_expressed_genes.csv"
    deg.to_csv(out, index=False)
    log.info("Saved %s (%d rows)", out, len(deg))
    log.info("Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log.error("Pipeline step failed: %s", exc)
        sys.exit(1)
