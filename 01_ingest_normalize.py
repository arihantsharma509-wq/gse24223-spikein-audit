#!/usr/bin/env python3
"""
01_ingest_normalize.py
======================
Download GEO series GSE24223 (PBMC transcriptomics), assign Control / Disease
cohorts from the sample phenotype metadata, then clean and normalise the
expression matrix.

Steps
-----
1. Fetch the series with GEOparse into ./data (cached on re-runs).
2. Parse per-sample metadata and assign Control vs Disease labels.
3. Build the probe x sample expression matrix from the per-sample VALUE column.
4. Detect whether values are already log2-scale (GEO2R heuristic); if not,
   apply log2(x + 1).
5. Quantile-normalise across samples.
6. Drop the 20% lowest-variance probes; impute remaining NaNs with the probe
   (row) median.
7. Write ./data/cleaned_expression.csv and ./data/sample_metadata.csv, plus a
   QC figure (./figures/FigureS1_Normalization_QC.png, 300 DPI).

Cohort labelling is keyword based because GEO phenotype fields are free text.
If the defaults mis-assign samples, override with --control-regex,
--disease-regex and/or --group-field (see --help). The script lists the
available metadata values when it cannot find two usable groups.
"""

import argparse
import logging
import re
import sys
import time
from pathlib import Path

import GEOparse
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("01_ingest")

ACCESSION = "GSE24223"
DEFAULT_CONTROL_REGEX = (
    r"\b(?:control|controls|ctrl|healthy|normal|non[- ]?disease[d]?|"
    r"non[- ]?patient|unaffected|volunteer)\b"
)
DEFAULT_DISEASE_REGEX = (
    r"\b(?:disease[d]?|patient|patients|case|cases|affected|infected|"
    r"cancer|tumou?r|sepsis|lupus|arthritis|asthma|diabet\w*|"
    r"HIV|treated|sick)\b"
)
GROUP_CONTROL, GROUP_DISEASE = "Control", "Disease"


# --------------------------------------------------------------------------- #
# Ingestion
# --------------------------------------------------------------------------- #
def load_series(accession, data_dir, soft_file=None, retries=3):
    """Fetch (or load from cache / local SOFT file) a GEO series."""
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            if soft_file:
                log.info("Parsing local SOFT file: %s", soft_file)
                return GEOparse.get_GEO(filepath=str(soft_file), silent=True)
            log.info("Fetching %s into %s (attempt %d/%d)", accession, data_dir, attempt, retries)
            return GEOparse.get_GEO(geo=accession, destdir=str(data_dir), silent=True)
        except Exception as exc:  # network errors, malformed SOFT, etc.
            last_exc = exc
            log.warning("Attempt %d failed: %s", attempt, exc)
            time.sleep(2 * attempt)
    raise RuntimeError(
        f"Could not load {accession}: {last_exc}. If GEO is unreachable from this "
        "machine, download the *_family.soft.gz file manually into ./data or pass "
        "--soft-file."
    ) from last_exc


def _first(value, default=""):
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        return str(value[0]) if value else default
    return str(value)


def _sanitize(key):
    return re.sub(r"[^0-9a-zA-Z]+", "_", key.strip().lower()).strip("_")


def build_metadata(gse):
    """One row per GSM with title, source, platform and parsed characteristics."""
    rows = {}
    for gsm_id, gsm in gse.gsms.items():
        md = gsm.metadata
        row = {
            "title": _first(md.get("title")),
            "source_name": _first(md.get("source_name_ch1")),
            "platform_id": _first(md.get("platform_id")),
        }
        values = []
        for entry in md.get("characteristics_ch1", []) or []:
            if ":" in entry:
                key, val = entry.split(":", 1)
                row[f"char_{_sanitize(key)}"] = val.strip()
                values.append(val.strip())
            else:
                values.append(entry.strip())
        row["_search_text"] = " | ".join([row["title"], row["source_name"]] + values)
        rows[gsm_id] = row
    meta = pd.DataFrame.from_dict(rows, orient="index")
    meta.index.name = "sample_id"
    return meta


def assign_groups(meta, control_regex, disease_regex, group_field=None):
    """Label each sample Control / Disease (NaN when ambiguous or unmatched)."""
    if group_field:
        col = f"char_{_sanitize(group_field)}"
        if col not in meta.columns:
            avail = [c for c in meta.columns if c.startswith("char_")]
            raise ValueError(f"--group-field '{group_field}' not found. Available: {avail}")
        text = meta[col].astype(str)
    else:
        text = meta["_search_text"].astype(str)

    is_ctrl = text.str.contains(control_regex, flags=re.IGNORECASE, regex=True)
    is_dis = text.str.contains(disease_regex, flags=re.IGNORECASE, regex=True)

    labels = pd.Series(np.nan, index=meta.index, dtype=object)
    labels[is_ctrl & ~is_dis] = GROUP_CONTROL
    labels[is_dis & ~is_ctrl] = GROUP_DISEASE
    n_amb = int((is_ctrl & is_dis).sum())
    n_none = int((~is_ctrl & ~is_dis).sum())
    if n_amb:
        log.warning("%d samples matched BOTH patterns and were dropped.", n_amb)
    if n_none:
        log.warning("%d samples matched NEITHER pattern and were dropped.", n_none)
    return labels


def describe_metadata(meta):
    """Log the distinct metadata values to help the user write regexes."""
    for col in [c for c in meta.columns if c.startswith("char_")] + ["source_name"]:
        counts = meta[col].value_counts().head(30)
        log.info("Metadata '%s' values:\n%s", col, counts.to_string())


def apply_filters(meta, filters, case_sensitive=False):
    """Keep samples whose characteristics match every 'key=regex' filter."""
    for f in filters or []:
        if "=" not in f:
            raise ValueError(f"--filter must look like 'cell subtype=CD4', got '{f}'")
        key, pattern = f.split("=", 1)
        col = f"char_{_sanitize(key)}"
        if col not in meta.columns:
            raise ValueError(f"--filter key '{key}' not found. Available: "
                             f"{[c for c in meta.columns if c.startswith('char_')]}")
        flags = 0 if case_sensitive else re.IGNORECASE
        keep = meta[col].astype(str).str.contains(pattern, flags=flags, regex=True)
        log.info("Filter %s: kept %d / %d samples", f, int(keep.sum()), len(meta))
        meta = meta[keep]
    if meta.empty:
        raise ValueError("No samples left after --filter.")
    return meta


def log_group_crosstabs(meta):
    """Show how cohorts are distributed across other characteristics (confounder check)."""
    for col in [c for c in meta.columns if c.startswith("char_")]:
        if 1 < meta[col].nunique() <= 15:
            log.info("Cohort x %s:\n%s", col, pd.crosstab(meta[col], meta["group"].fillna("(unassigned)")).to_string())


def check_spikein_controls(expr, groups, data_dir, thresh=1.0):
    """Affymetrix AFFX-* control probes (e.g. B. subtilis dap/lys/phe/thr poly-A spike-ins)
    should not differ between biological groups. Large differences point to a
    technical (labelling / batch) confound."""
    ctrl = expr[expr.index.astype(str).str.startswith("AFFX")]
    if ctrl.empty:
        return
    g = groups.loc[ctrl.columns]
    dis, con = ctrl.loc[:, g == GROUP_DISEASE], ctrl.loc[:, g == GROUP_CONTROL]
    table = pd.DataFrame({"mean_control": con.mean(axis=1), "mean_disease": dis.mean(axis=1)})
    table["log2_diff_disease_minus_control"] = table["mean_disease"] - table["mean_control"]
    table.to_csv(Path(data_dir) / "affx_control_check.csv", index_label="probe_id")
    diff = table["log2_diff_disease_minus_control"]
    polya = diff[diff.index.str.contains(r"(?i)(?:dap|lys|phe|thr)")]
    n_big = int((diff.abs() > thresh).sum())
    log.info("AFFX control probes: %d total, %d differ > %.1f log2 between cohorts "
             "(poly-A spike-in median diff = %.2f).", len(diff), n_big, thresh,
             polya.median() if len(polya) else float("nan"))
    if len(polya) and abs(polya.median()) > thresh:
        log.warning("Spike-in controls differ strongly between Control and Disease. Cohorts were probably "
                    "labelled/processed differently, so group effects are confounded with batch. "
                    "See %s.", Path(data_dir) / "affx_control_check.csv")


def build_expression_matrix(gse, sample_ids):
    """Probe x sample matrix from each GSM's VALUE column."""
    series = {}
    for sid in sample_ids:
        tbl = gse.gsms[sid].table
        if tbl is None or tbl.empty or "VALUE" not in tbl.columns:
            log.warning("Sample %s has no VALUE column; skipped.", sid)
            continue
        s = pd.to_numeric(tbl.set_index("ID_REF")["VALUE"], errors="coerce")
        series[sid] = s.groupby(level=0).mean()  # collapse duplicated probe IDs
    if not series:
        raise ValueError("No sample contained an expression table with a VALUE column.")
    expr = pd.concat(series, axis=1)
    expr.index.name = "ID_REF"
    return expr


# --------------------------------------------------------------------------- #
# Transformation / normalisation
# --------------------------------------------------------------------------- #
def looks_log2_transformed(expr):
    """GEO2R heuristic: True when the value range is consistent with log2 data."""
    vals = expr.to_numpy(dtype=float).ravel()
    vals = vals[np.isfinite(vals)]
    q = np.quantile(vals, [0.0, 0.25, 0.5, 0.75, 0.99, 1.0])
    needs_log = (q[4] > 100) or ((q[5] - q[0] > 50) and (q[1] > 0))
    log.info("Value quantiles (0/25/50/75/99/100%%): %s", np.round(q, 2).tolist())
    return not needs_log


def quantile_normalize(expr):
    """Rank-mean quantile normalisation across samples (columns).

    Missing values are temporarily filled with the probe median so every sample
    has the same number of ranks, then masked again afterwards.
    """
    arr = expr.to_numpy(dtype=float)
    nan_mask = np.isnan(arr)
    fill = np.nanmedian(arr, axis=1, keepdims=True)
    filled = np.where(nan_mask, fill, arr)

    n_rows = filled.shape[0]
    rank_means = np.sort(filled, axis=0).mean(axis=1)
    out = np.empty_like(filled)
    grid = np.arange(n_rows)
    for j in range(filled.shape[1]):
        ranks = rankdata(filled[:, j], method="average") - 1  # 0-based, ties averaged
        out[:, j] = np.interp(ranks, grid, rank_means)
    out[nan_mask] = np.nan
    return pd.DataFrame(out, index=expr.index, columns=expr.columns)


def filter_low_variance(expr, drop_fraction=0.20):
    var = expr.var(axis=1, skipna=True)
    n_drop = int(np.floor(drop_fraction * len(var)))
    drop_idx = var.fillna(-1).nsmallest(n_drop).index
    kept = expr.drop(index=drop_idx)
    log.info("Variance filter: removed %d / %d probes (lowest %.0f%%).",
             n_drop, len(expr), drop_fraction * 100)
    return kept


def impute_row_median(expr):
    n_missing = int(expr.isna().sum().sum())
    if n_missing:
        medians = expr.median(axis=1)
        expr = expr.apply(lambda col: col.fillna(medians))
    log.info("Imputed %d missing values with probe medians.", n_missing)
    return expr


# --------------------------------------------------------------------------- #
# QC figure
# --------------------------------------------------------------------------- #
def plot_qc(before, after, groups, out_path):
    order = groups.sort_values().index
    palette = {GROUP_CONTROL: "#1f77b4", GROUP_DISEASE: "#d62728"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, df, title in zip(axes, (before, after),
                             ("Before quantile normalisation (log2)", "After normalisation + filtering")):
        data = [df[s].dropna().values for s in order]
        bp = ax.boxplot(data, showfliers=False, patch_artist=True, widths=0.7)
        for patch, s in zip(bp["boxes"], order):
            patch.set_facecolor(palette[groups[s]])
            patch.set_alpha(0.75)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Samples")
        if len(order) > 30:
            ax.set_xticks([])
        else:
            ax.set_xticklabels(order, rotation=90, fontsize=7)
    axes[0].set_ylabel("log2 expression")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.75) for c in palette.values()]
    axes[1].legend(handles, palette.keys(), title="Cohort", frameon=False)
    sns.despine()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out_path)


# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--accession", default=ACCESSION)
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--fig-dir", default="./figures")
    p.add_argument("--soft-file", default=None, help="Use a local *_family.soft(.gz) file instead of downloading.")
    p.add_argument("--control-regex", default=DEFAULT_CONTROL_REGEX)
    p.add_argument("--disease-regex", default=DEFAULT_DISEASE_REGEX)
    p.add_argument("--group-field", default=None,
                   help="Restrict matching to one characteristics key, e.g. 'disease state'.")
    p.add_argument("--filter", action="append", default=[], metavar="KEY=REGEX",
                   help="Keep only samples whose characteristic KEY matches REGEX; repeatable. "
                        "E.g. --filter \"cell subtype=^CD4$\"")
    p.add_argument("--case-sensitive-filters", action="store_true",
                   help="Make --filter regexes case-sensitive (e.g. to separate 'Whole Blood' from 'Whole blood').")
    p.add_argument("--keep-affx", action="store_true",
                   help="Keep Affymetrix AFFX-* control probes (dropped by default).")
    p.add_argument("--min-per-group", type=int, default=3)
    p.add_argument("--variance-drop", type=float, default=0.20)
    return p.parse_args()


def main():
    args = parse_args()
    data_dir, fig_dir = Path(args.data_dir), Path(args.fig_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    log.info("[1/6] Loading GEO series")
    gse = load_series(args.accession, data_dir, args.soft_file)
    log.info("Series %s: %d samples, platform(s): %s", args.accession, len(gse.gsms), list(gse.gpls))

    log.info("[2/6] Assigning cohorts")
    meta = build_metadata(gse)
    meta = apply_filters(meta, args.filter, args.case_sensitive_filters)
    meta["group"] = assign_groups(meta, args.control_regex, args.disease_regex, args.group_field)
    counts = meta["group"].value_counts()
    log_group_crosstabs(meta)
    log.info("Cohort sizes: %s", counts.to_dict())
    if any(counts.get(g, 0) < args.min_per_group for g in (GROUP_CONTROL, GROUP_DISEASE)):
        describe_metadata(meta)
        raise ValueError(
            f"Need >= {args.min_per_group} samples in both cohorts. Adjust --control-regex / "
            "--disease-regex / --group-field using the metadata values listed above."
        )
    meta = meta.dropna(subset=["group"])

    # Keep a single platform so probe IDs are comparable.
    if meta["platform_id"].nunique() > 1:
        top = meta["platform_id"].value_counts().idxmax()
        log.warning("Multiple platforms found; keeping %s only.", top)
        meta = meta[meta["platform_id"] == top]

    log.info("[3/6] Building expression matrix")
    expr = build_expression_matrix(gse, meta.index)
    meta = meta.loc[expr.columns]
    expr = expr.dropna(how="all")
    log.info("Matrix: %d probes x %d samples", *expr.shape)

    log.info("[4/6] Log2 check and quantile normalisation")
    if looks_log2_transformed(expr):
        log.info("Values already appear log2-transformed; no transform applied.")
    else:
        min_val = float(np.nanmin(expr.to_numpy()))
        if min_val < 0:
            log.warning("Negative values (min %.3f) found; shifting data to be non-negative before log2(x+1).", min_val)
            expr = expr - min_val
        expr = np.log2(expr + 1.0)
        log.info("Applied log2(x + 1).")
    check_spikein_controls(expr, meta["group"], data_dir)
    if not args.keep_affx:
        n_before = len(expr)
        expr = expr[~expr.index.astype(str).str.startswith("AFFX")]
        log.info("Removed %d Affymetrix control probes (AFFX-*).", n_before - len(expr))
    pre_norm = expr.copy()
    expr = quantile_normalize(expr)

    log.info("[5/6] Variance filtering and imputation")
    expr = filter_low_variance(expr, args.variance_drop)
    expr = impute_row_median(expr)
    if expr.shape[0] == 0:
        raise ValueError("No probes left after filtering.")

    log.info("[6/6] Saving outputs")
    expr_path = data_dir / "cleaned_expression.csv"
    meta_path = data_dir / "sample_metadata.csv"
    expr.to_csv(expr_path)
    meta.drop(columns=["_search_text"]).to_csv(meta_path)
    log.info("Saved %s (%d probes x %d samples)", expr_path, *expr.shape)
    log.info("Saved %s", meta_path)
    plot_qc(pre_norm, expr, meta["group"], fig_dir / "FigureS1_Normalization_QC.png")
    log.info("Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log.error("Pipeline step failed: %s", exc)
        sys.exit(1)
