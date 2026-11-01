#!/usr/bin/env python3
"""
03_ml_feature_selection.py
==========================
Random-forest classification of Disease vs Control with SHAP-based biomarker
selection.

Two modes
---------
Default (as originally specified)
    Features = the differentially expressed probes from script 02, chosen on
    ALL samples. Stratified 5-fold CV on those probes.
    CAVEAT: the probe selection saw every sample's label, so CV metrics (AUC)
    are optimistic.

--nested-top-k K  (recommended for small cohorts)
    Leakage-free: inside EACH training fold, rank all probes by Welch t-test on
    the training samples only, keep the top K, fit the forest, then score the
    held-out samples. AUC/accuracy are therefore honest estimates of how the
    whole procedure would generalise. Features not selected in a fold get SHAP
    value 0 for that fold's test samples; the signature also reports
    `selection_frequency` (fraction of folds in which a probe was selected).

Outputs
-------
* ./figures/Figure2_ROC_Curve.png   mean ROC with +/-1 SD shading
* ./figures/Figure3_SHAP_Summary.png SHAP dot plot (top probes)
* ./data/final_biomarker_signature.csv  top 5-8 probes + CV metrics
* ./data/classification_metrics_per_fold.csv
"""

import argparse
import gc
import logging
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
import shap  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,  # noqa: E402
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold  # noqa: E402

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("03_ml")
warnings.filterwarnings("ignore", message="Precision loss occurred")


def load_inputs(data_dir, nested=False):
    expr_path = data_dir / "cleaned_expression.csv"
    meta_path = data_dir / "sample_metadata.csv"
    needed = [expr_path, meta_path] + ([] if nested else [data_dir / "differentially_expressed_genes.csv"])
    for p in needed:
        if not p.exists():
            raise FileNotFoundError(f"{p} not found. Run scripts 01 and 02 first.")
    expr = pd.read_csv(expr_path, index_col=0)
    meta = pd.read_csv(meta_path, index_col=0)
    expr.index = expr.index.astype(str)
    common = expr.columns.intersection(meta.index)
    y = (meta.loc[common, "group"] == "Disease").astype(int)
    if nested:
        return expr[common].T, y

    deg = pd.read_csv(data_dir / "differentially_expressed_genes.csv")
    available = set(expr.index)
    probes = [p for p in deg["probe_id"].astype(str) if p in available]
    if len(probes) < 5:
        raise ValueError(f"Only {len(probes)} differentially expressed probes available (need >= 5). "
                         "Relax thresholds in script 02 (--lfc-threshold / --fallback-top-n), "
                         "or use --nested-top-k to select probes inside cross-validation.")
    return expr.loc[probes, common].T, y


def class1_shap_values(explainer, X_test):
    """(n_samples, n_features) SHAP values for the Disease class, robust to the
    output shape differences between shap versions."""
    sv = explainer.shap_values(X_test, check_additivity=False)
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    return sv


def cross_validate(X, y, n_splits, n_estimators, seed, nested_k=0, compute_shap=True, n_jobs=-1):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    mean_fpr = np.linspace(0, 1, 101)
    tprs, fold_rows = [], []
    shap_all = np.zeros(X.shape, dtype=float)
    sel_count = np.zeros(X.shape[1], dtype=int)
    oof_prob = np.zeros(len(y))
    all_idx = np.arange(X.shape[1])

    for fold, (tr, te) in enumerate(skf.split(X, y), start=1):
        if nested_k > 0:
            xtr, ytr = X.iloc[tr].to_numpy(float), y.iloc[tr].to_numpy()
            with np.errstate(all="ignore"):
                _, p = stats.ttest_ind(xtr[ytr == 1], xtr[ytr == 0], axis=0, equal_var=False)
            idx = np.argsort(np.nan_to_num(p, nan=1.0))[:nested_k]
        else:
            idx = all_idx
        sel_count[idx] += 1

        Xtr, Xte = X.iloc[tr, idx], X.iloc[te, idx]
        rf = RandomForestClassifier(n_estimators=n_estimators, class_weight="balanced",
                                    random_state=seed, n_jobs=n_jobs)
        rf.fit(Xtr, y.iloc[tr])
        prob = rf.predict_proba(Xte)[:, 1]
        pred = (prob >= 0.5).astype(int)
        oof_prob[te] = prob

        auc = roc_auc_score(y.iloc[te], prob)
        fpr, tpr, _ = roc_curve(y.iloc[te], prob)
        interp = np.interp(mean_fpr, fpr, tpr)
        interp[0] = 0.0
        tprs.append(interp)
        fold_rows.append({
            "fold": fold, "n_train": len(tr), "n_test": len(te), "n_features": len(idx), "roc_auc": auc,
            "accuracy": accuracy_score(y.iloc[te], pred),
            "balanced_accuracy": balanced_accuracy_score(y.iloc[te], pred),
            "precision": precision_score(y.iloc[te], pred, zero_division=0),
            "recall": recall_score(y.iloc[te], pred, zero_division=0),
            "f1": f1_score(y.iloc[te], pred, zero_division=0),
        })
        if compute_shap:
            log.info("Fold %d/%d: AUC=%.3f acc=%.3f (test n=%d, features=%d)", fold, n_splits, auc,
                     fold_rows[-1]["accuracy"], len(te), len(idx))
            shap_all[np.ix_(te, idx)] = class1_shap_values(shap.TreeExplainer(rf), Xte)

    return pd.DataFrame(fold_rows), mean_fpr, np.array(tprs), shap_all, oof_prob, sel_count


def permutation_test(X, y, n_splits, nested_k, trees, seed, n_perm, observed_auc):
    """Null distribution of the pooled out-of-fold AUC when labels are shuffled and the
    ENTIRE procedure (in-fold probe selection + forest) is repeated."""
    rng = np.random.default_rng(seed)
    null = np.zeros(n_perm)
    for i in range(n_perm):
        try:
            y_perm = pd.Series(rng.permutation(y.to_numpy()), index=y.index)
            out = cross_validate(X, y_perm, n_splits, trees, seed + i + 1, nested_k,
                                 compute_shap=False, n_jobs=1)
            null[i] = roc_auc_score(y_perm, out[4])
            del out
        except MemoryError:
            log.warning("Out of memory after %d permutations; using those completed. "
                        "Close other programs and re-run for more.", i)
            null = null[:i]
            break
        if (i + 1) % 10 == 0:
            gc.collect()
        if (i + 1) % 25 == 0:
            log.info("  permutations done: %d/%d", i + 1, n_perm)
    if len(null) == 0:
        return null, float("nan")
    p_value = (1 + int((null >= observed_auc).sum())) / (1 + len(null))
    return null, p_value


def plot_roc(mean_fpr, tprs, aucs, out_path, title):
    mean_tpr = tprs.mean(axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = tprs.std(axis=0)

    sns.set_theme(style="ticks", context="paper", font_scale=1.4)
    fig, ax = plt.subplots(figsize=(7, 6.5))
    for i, t in enumerate(tprs, start=1):
        ax.plot(mean_fpr, t, lw=1, alpha=0.35, label=f"Fold {i} (AUC = {aucs[i - 1]:.2f})")
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color="grey", label="Chance")
    ax.plot(mean_fpr, mean_tpr, color="#08306b", lw=2.5,
            label=f"Mean ROC (AUC = {np.mean(aucs):.2f} $\\pm$ {np.std(aucs):.2f})")
    ax.fill_between(mean_fpr, np.clip(mean_tpr - std_tpr, 0, 1), np.clip(mean_tpr + std_tpr, 0, 1),
                    color="#6baed6", alpha=0.3, label="$\\pm$ 1 SD")
    ax.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02), xlabel="False positive rate",
           ylabel="True positive rate", title=title)
    ax.legend(loc="lower right", frameon=False, fontsize=10)
    sns.despine()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out_path)


def plot_shap_summary(shap_values, X, out_path, max_display):
    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X, max_display=max_display, show=False, plot_type="dot")
    fig = plt.gcf()
    fig.axes[0].set_xlabel("SHAP value (impact on Disease probability)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out_path)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--fig-dir", default="./figures")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--trees", type=int, default=500)
    p.add_argument("--top-n", type=int, default=8, help="Biomarkers to keep (clamped to 5-8).")
    p.add_argument("--nested-top-k", type=int, default=0,
                   help="Leakage-free mode: select the top-K probes by Welch t-test inside each training "
                        "fold (uses all probes in cleaned_expression.csv; ignores script 02's DE list). 0 = off.")
    p.add_argument("--permutations", type=int, default=0,
                   help="With --nested-top-k: number of label permutations for an empirical p-value "
                        "of the pooled out-of-fold AUC (e.g. 100). 0 = off.")
    p.add_argument("--shap-plot-genes", type=int, default=15, help="Genes shown in the SHAP dot plot.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    data_dir, fig_dir = Path(args.data_dir), Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    top_n = int(min(8, max(5, args.top_n)))
    nested = args.nested_top_k > 0

    log.info("[1/5] Loading data")
    X, y = load_inputs(data_dir, nested)
    log.info("Feature matrix: %d samples x %d probes; Disease=%d, Control=%d",
             X.shape[0], X.shape[1], int(y.sum()), int((1 - y).sum()))
    if nested:
        log.info("NESTED mode: top %d probes selected inside each training fold (no label leakage).",
                 args.nested_top_k)
    else:
        log.warning("Probes were selected on ALL samples before CV: AUC is optimistic. "
                    "Use --nested-top-k for an honest estimate.")

    n_splits = min(args.folds, int(y.value_counts().min()))
    if n_splits < 2:
        raise ValueError("Each class needs at least 2 samples for cross-validation.")
    if n_splits < args.folds:
        log.warning("Smallest class has few samples; using %d folds instead of %d.", n_splits, args.folds)

    log.info("[2/5] %d-fold cross-validated Random Forest + per-fold SHAP", n_splits)
    metrics, mean_fpr, tprs, shap_vals, oof_prob, sel_count = cross_validate(
        X, y, n_splits, args.trees, args.seed, args.nested_top_k)
    metrics.to_csv(data_dir / "classification_metrics_per_fold.csv", index=False)

    aucs = metrics["roc_auc"].to_numpy()
    pooled_auc = roc_auc_score(y, oof_prob)
    log.info("Mean ROC-AUC = %.3f +/- %.3f (pooled out-of-fold AUC = %.3f)", aucs.mean(), aucs.std(), pooled_auc)

    perm_p = float("nan")
    if args.permutations > 0:
        if not nested:
            log.warning("--permutations only makes sense with --nested-top-k; skipped.")
        else:
            log.info("Permutation test (%d shuffles of the full nested procedure)...", args.permutations)
            null, perm_p = permutation_test(X, y, n_splits, args.nested_top_k, min(args.trees, 200),
                                            args.seed, args.permutations, pooled_auc)
            pd.DataFrame({"null_pooled_auc": null}).to_csv(data_dir / "permutation_auc_null.csv", index=False)
            log.info("Observed pooled AUC = %.3f; null mean = %.3f (95th pct %.3f); empirical p = %.4f",
                     pooled_auc, null.mean(), np.quantile(null, 0.95), perm_p)

    log.info("[3/5] ROC figure")
    title = (f"{n_splits}-fold nested CV ROC: Random Forest (top-{args.nested_top_k} probes per fold)"
             if nested else f"{n_splits}-fold CV ROC: Random Forest")
    plot_roc(mean_fpr, tprs, aucs, fig_dir / "Figure2_ROC_Curve.png", title)

    log.info("[4/5] SHAP ranking and summary plot")
    mean_abs = np.abs(shap_vals).mean(axis=0)
    ranking = (pd.DataFrame({"probe_id": X.columns, "mean_abs_shap": mean_abs,
                             "selection_frequency": sel_count / n_splits})
               .sort_values(["mean_abs_shap", "selection_frequency"], ascending=False).reset_index(drop=True))
    ranking["shap_rank"] = ranking.index + 1
    top = ranking.head(top_n).copy()
    log.info("Top %d biomarkers:\n%s", top_n, top.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    keep = sel_count > 0
    plot_shap_summary(shap_vals[:, keep], X.loc[:, keep], fig_dir / "Figure3_SHAP_Summary.png",
                      max_display=min(args.shap_plot_genes, int(keep.sum())))

    log.info("[5/5] Exporting signature + metrics")
    for col in ["roc_auc", "accuracy", "balanced_accuracy", "precision", "recall", "f1"]:
        top[f"cv_mean_{col}"] = metrics[col].mean()
        top[f"cv_sd_{col}"] = metrics[col].std(ddof=0)
    top["cv_pooled_oof_auc"] = pooled_auc
    top["perm_p_value"] = perm_p
    top["cv_folds"] = n_splits
    top["cv_mode"] = f"nested_top{args.nested_top_k}" if nested else "preselected_DE_probes"
    top["n_input_probes"] = X.shape[1]
    lead = ["shap_rank", "probe_id", "mean_abs_shap", "selection_frequency"]
    top = top[lead + [c for c in top.columns if c not in lead]]
    out = data_dir / "final_biomarker_signature.csv"
    top.to_csv(out, index=False)
    log.info("Saved %s", out)
    log.info("Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log.error("Pipeline step failed: %s", exc)
        sys.exit(1)
