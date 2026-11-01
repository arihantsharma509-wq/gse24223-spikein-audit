#!/usr/bin/env python3
"""
05_spikein_audit.py
===================
Dataset-wide technical QC for Affymetrix arrays: do the poly-A spike-in
control probes (B. subtilis lys/phe/dap/thr/trpn, AFFX-*) differ between
sample groups?  They should be flat across samples; large shifts between
groups mean the groups were labelled/processed differently, so any biological
comparison between them is confounded with batch.

For every sample it computes the mean spike-in signal, then summarises by
cell subtype x time point (all samples in the series, not only the two
cohorts used in scripts 01-04).

Outputs
-------
* ./data/spikein_audit_samples.csv   per-sample spike-in level + labels
* ./data/spikein_audit_summary.csv   mean/SD/n per cell subtype x time point
* ./figures/FigureS3_SpikeIn_Audit.png (300 DPI)

Run from the project folder, e.g.
    python 05_spikein_audit.py --soft-file data\\GSE24223_family.soft.gz
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

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("05_audit")

POLYA_REGEX = r"(?i)(?:lys|phe|dap|thr|trpn)"


def sanitize(key):
    return re.sub(r"[^0-9a-zA-Z]+", "_", key.strip().lower()).strip("_")


def time_order(label):
    low = str(label).lower()
    num = re.findall(r"\d+", low)
    return (0 if "control" in low else 1 if low.startswith("pre") else 2, int(num[0]) if num else 0)


def load_series(accession, data_dir, soft_file):
    if soft_file:
        return GEOparse.get_GEO(filepath=str(soft_file), silent=True)
    return GEOparse.get_GEO(geo=accession, destdir=str(data_dir), silent=True)


def collect(gse, time_field, cell_field):
    spike, rows = {}, []
    tkey, ckey = sanitize(time_field), sanitize(cell_field)
    for gsm_id, gsm in gse.gsms.items():
        tbl = gsm.table
        if tbl is None or tbl.empty or "VALUE" not in tbl.columns:
            continue
        sub = tbl[tbl["ID_REF"].astype(str).str.startswith("AFFX")]
        if sub.empty:
            continue
        spike[gsm_id] = pd.to_numeric(sub.set_index("ID_REF")["VALUE"], errors="coerce").groupby(level=0).mean()
        chars = {}
        for entry in gsm.metadata.get("characteristics_ch1", []) or []:
            if ":" in entry:
                k, v = entry.split(":", 1)
                chars[sanitize(k)] = v.strip()
        rows.append({"sample_id": gsm_id, "time_point": chars.get(tkey, "NA"),
                     "cell_subtype_raw": chars.get(ckey, "NA"),
                     "donor": chars.get("donor_number", "NA")})
    if not spike:
        raise ValueError("No AFFX control probes found (is this an Affymetrix platform?).")
    meta = pd.DataFrame(rows).set_index("sample_id")
    # Keep the label exactly as submitted: spelling variants ("Whole Blood" vs "Whole blood")
    # can mark separate submission batches, so they must stay visible.
    meta["cell_subtype"] = meta["cell_subtype_raw"].str.strip()
    return pd.DataFrame(spike), meta


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--accession", default="GSE24223")
    p.add_argument("--soft-file", default=None)
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--fig-dir", default="./figures")
    p.add_argument("--time-field", default="time point")
    p.add_argument("--cell-field", default="cell subtype")
    args = p.parse_args()
    data_dir, fig_dir = Path(args.data_dir), Path(args.fig_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    log.info("[1/4] Loading series")
    gse = load_series(args.accession, data_dir, args.soft_file)
    log.info("[2/4] Extracting AFFX control probes from %d samples", len(gse.gsms))
    spike, meta = collect(gse, args.time_field, args.cell_field)
    polya = [i for i in spike.index if re.search(POLYA_REGEX, i)]
    log.info("%d AFFX probes, %d of them poly-A spike-ins", len(spike), len(polya))
    meta["polyA_spikein_mean"] = spike.loc[polya].mean(axis=0)
    meta["all_affx_mean"] = spike.mean(axis=0)
    meta.to_csv(data_dir / "spikein_audit_samples.csv")

    log.info("[3/4] Summarising by cell subtype x time point")
    summ = (meta.groupby(["cell_subtype", "time_point"])["polyA_spikein_mean"]
            .agg(n="size", mean="mean", sd="std").reset_index())
    summ.to_csv(data_dir / "spikein_audit_summary.csv", index=False)
    overall = meta["polyA_spikein_mean"].median()
    meta["dev"] = meta["polyA_spikein_mean"] - overall
    piv = meta.pivot_table(index="time_point", columns="cell_subtype", values="dev", aggfunc="mean")
    cnt = meta.pivot_table(index="time_point", columns="cell_subtype", values="dev", aggfunc="size")
    order = sorted(piv.index, key=time_order)
    piv, cnt = piv.loc[order], cnt.loc[order]
    for cell in piv.columns:
        col = piv[cell].dropna()
        ctrl = [t for t in col.index if "control" in t.lower()]
        pre = [t for t in col.index if t.lower().startswith("pre")]
        if ctrl and pre:
            log.info("%-12s Control - Pre-Tx spike-in difference: %+.2f log2", cell, col[ctrl[0]] - col[pre[0]])
    log.info("Overall median poly-A spike-in level: %.2f log2", overall)

    log.info("[4/4] Figure")
    sns.set_theme(style="ticks", context="paper", font_scale=1.2)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), gridspec_kw={"width_ratios": [1, 1.3]})
    annot = piv.round(1).astype(str) + "\n(n=" + cnt.fillna(0).astype(int).astype(str) + ")"
    annot = annot.where(piv.notna(), "")
    lim = max(0.5, np.nanmax(np.abs(piv.to_numpy())))
    sns.heatmap(piv, annot=annot, fmt="", cmap="coolwarm", center=0, vmin=-lim, vmax=lim,
                cbar_kws={"label": "Mean poly-A spike-in\n(log2, minus overall median)"}, ax=axes[0],
                linewidths=0.5, linecolor="white")
    axes[0].set_title("Spike-in level by cell subtype and time point")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("")
    sns.stripplot(data=meta, x="time_point", y="polyA_spikein_mean", hue="cell_subtype", order=order,
                  dodge=True, size=5, ax=axes[1], jitter=0.15)
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Mean poly-A spike-in signal (log2)")
    axes[1].set_title("Per-sample spike-in level")
    axes[1].legend(title="Cell subtype", frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.setp(axes[1].get_xticklabels(), rotation=30, ha="right")
    sns.despine()
    fig.tight_layout()
    out = fig_dir / "FigureS3_SpikeIn_Audit.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out)
    log.info("Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log.error("Audit failed: %s", exc)
        sys.exit(1)
