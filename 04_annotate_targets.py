#!/usr/bin/env python3
"""
04_annotate_targets.py
======================
Map biomarker probe IDs to gene symbols and gene names using the GPL platform
annotation table that ships with the GEO series, then build the final table.

Output: ./data/Table1_Biomarker_Signature.csv with columns
  Probe_ID, Gene_Symbol, Gene_Name, log2FC, adj_p_value, SHAP_Importance_Rank
and a summary figure ./figures/Figure4_Biomarker_log2FC.png (300 DPI).

Symbols/names are those provided by the platform's own annotation (manufacturer
or GEO curators). Platforms differ in column naming, so several common layouts
are recognised (Affymetrix 'Gene Symbol'/'Gene Title', Illumina 'Symbol'/
'Definition', Agilent 'GENE_SYMBOL'/'GENE_NAME', Affymetrix exon-style
'gene_assignment'); override with --symbol-col / --name-col if needed.
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
log = logging.getLogger("04_annot")

ACCESSION = "GSE24223"
SYMBOL_CANDIDATES = ["Gene Symbol", "GENE_SYMBOL", "Gene_Symbol", "Symbol", "SYMBOL", "ILMN_Gene",
                     "gene_symbol", "GeneSymbol", "Associated Gene Name"]
NAME_CANDIDATES = ["Gene Title", "GENE_NAME", "Gene_Title", "Definition", "DEFINITION", "Description",
                   "DESCRIPTION", "gene_name", "Gene Name", "GENE_TITLE", "Name", "product"]
MISSING_TOKENS = {"", "---", "--", "na", "n/a", "nan", "none", "null"}


def find_col(columns, candidates):
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def clean_text(value, sep_pattern=r"\s*///\s*"):
    """Take the first entry of multi-valued fields ('A /// B') and normalise blanks."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    first = re.split(sep_pattern, str(value).strip())[0].strip()
    return None if first.lower() in MISSING_TOKENS else first


def parse_gene_assignment(value):
    """'NM_1 // SYMBOL // title // chr // id /// ...' -> (SYMBOL, title)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None, None
    first = str(value).split(" /// ")[0]
    parts = [p.strip() for p in first.split(" // ")]
    symbol = parts[1] if len(parts) > 1 else None
    title = parts[2] if len(parts) > 2 else None
    return (None if (symbol or "").lower() in MISSING_TOKENS else symbol,
            None if (title or "").lower() in MISSING_TOKENS else title)


def load_gse(accession, data_dir, soft_file):
    if soft_file:
        return GEOparse.get_GEO(filepath=str(soft_file), silent=True)
    return GEOparse.get_GEO(geo=accession, destdir=str(data_dir), silent=True)


def build_annotation(gpl, probe_ids, symbol_col=None, name_col=None):
    table = gpl.table
    if table is None or table.empty:
        raise ValueError(f"Platform {gpl.name} has no annotation table.")
    id_col = find_col(table.columns, ["ID", "ID_REF", "Probe ID", "ProbeID"]) or table.columns[0]
    tbl = table.copy()
    tbl[id_col] = tbl[id_col].astype(str)
    tbl = tbl.drop_duplicates(subset=id_col).set_index(id_col)

    sym_col = symbol_col or find_col(tbl.columns, SYMBOL_CANDIDATES)
    nm_col = name_col or find_col(tbl.columns, NAME_CANDIDATES)
    ga_col = find_col(tbl.columns, ["gene_assignment"])

    annot = pd.DataFrame(index=list(probe_ids), columns=["Gene_Symbol", "Gene_Name"], dtype=object)
    sub = tbl.reindex(annot.index)
    if sym_col:
        annot["Gene_Symbol"] = sub[sym_col].map(clean_text)
    if nm_col:
        annot["Gene_Name"] = sub[nm_col].map(clean_text)
    if ga_col and (annot["Gene_Symbol"].isna().any() or annot["Gene_Name"].isna().any()):
        parsed = sub[ga_col].map(parse_gene_assignment)
        annot["Gene_Symbol"] = annot["Gene_Symbol"].fillna(parsed.map(lambda t: t[0]))
        annot["Gene_Name"] = annot["Gene_Name"].fillna(parsed.map(lambda t: t[1]))
    if not (sym_col or nm_col or ga_col):
        raise ValueError(f"No recognised annotation columns in {gpl.name}. Available columns: "
                         f"{list(tbl.columns)}. Use --symbol-col / --name-col.")
    log.info("Platform %s: symbol column=%s, name column=%s%s", gpl.name, sym_col, nm_col,
             f", gene_assignment={ga_col}" if ga_col else "")
    return annot


def plot_table(table, out_path):
    df = table.sort_values("log2FC")
    labels = [f"{s} ({p})" if s != "Unannotated" else p for s, p in zip(df["Gene_Symbol"], df["Probe_ID"])]
    colors = np.where(df["log2FC"] >= 0, "#d62728", "#1f77b4")
    sns.set_theme(style="ticks", context="paper", font_scale=1.4)
    fig, ax = plt.subplots(figsize=(7.5, max(3, 0.55 * len(df) + 1.5)))
    ax.barh(labels, df["log2FC"], color=colors)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("log$_2$ fold change (Disease vs Control)")
    ax.set_title("Biomarker signature")
    sns.despine()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out_path)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--accession", default=ACCESSION)
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--fig-dir", default="./figures")
    p.add_argument("--soft-file", default=None)
    p.add_argument("--symbol-col", default=None)
    p.add_argument("--name-col", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    data_dir, fig_dir = Path(args.data_dir), Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    log.info("[1/4] Loading signature and DE statistics")
    sig_path = data_dir / "final_biomarker_signature.csv"
    deg_path = data_dir / "differentially_expressed_genes.csv"
    meta_path = data_dir / "sample_metadata.csv"
    all_stats_path = data_dir / "all_probe_statistics.csv"
    stats_path = all_stats_path if all_stats_path.exists() else deg_path
    for p in (sig_path, stats_path):
        if not p.exists():
            raise FileNotFoundError(f"{p} not found. Run scripts 01-03 first.")
    sig = pd.read_csv(sig_path, dtype={"probe_id": str})
    deg = pd.read_csv(stats_path, dtype={"probe_id": str})[["probe_id", "log2FC", "adj_p_value"]]
    log.info("Effect sizes / p-values taken from %s", stats_path.name)
    probes = sig["probe_id"].tolist()

    log.info("[2/4] Loading platform annotation")
    gse = load_gse(args.accession, data_dir, args.soft_file)
    if not gse.gpls:
        raise ValueError("No platform (GPL) records found in the series.")
    preferred = None
    if meta_path.exists():
        meta = pd.read_csv(meta_path, index_col=0)
        if "platform_id" in meta.columns and meta["platform_id"].notna().any():
            preferred = meta["platform_id"].mode().iloc[0]
    gpl_names = ([preferred] if preferred in gse.gpls else []) + [g for g in gse.gpls if g != preferred]

    annot, last_err = None, None
    for name in gpl_names:
        try:
            cand = build_annotation(gse.gpls[name], probes, args.symbol_col, args.name_col)
        except Exception as exc:
            last_err = exc
            log.warning("Platform %s unusable: %s", name, exc)
            continue
        if cand["Gene_Symbol"].notna().any() or cand["Gene_Name"].notna().any():
            annot = cand
            break
        log.warning("Platform %s matched no signature probes; trying next.", name)
    if annot is None:
        raise RuntimeError(f"Could not annotate probes from any platform. Last error: {last_err}")

    log.info("[3/4] Building Table 1")
    table = (sig[["shap_rank", "probe_id"]]
             .merge(deg, on="probe_id", how="left")
             .merge(annot.rename_axis("probe_id").reset_index(), on="probe_id", how="left"))
    n_unannot = int(table["Gene_Symbol"].isna().sum())
    if n_unannot:
        log.warning("%d probe(s) have no gene symbol on the platform; labelled 'Unannotated'.", n_unannot)
    table["Gene_Symbol"] = table["Gene_Symbol"].fillna("Unannotated")
    table["Gene_Name"] = table["Gene_Name"].fillna("Not available")
    table = table.rename(columns={"probe_id": "Probe_ID", "shap_rank": "SHAP_Importance_Rank"})
    table = table[["Probe_ID", "Gene_Symbol", "Gene_Name", "log2FC", "adj_p_value", "SHAP_Importance_Rank"]]
    table = table.sort_values("SHAP_Importance_Rank").reset_index(drop=True)
    table["log2FC"] = table["log2FC"].round(3)
    table["adj_p_value"] = table["adj_p_value"].map(lambda v: float(f"{v:.3e}") if pd.notna(v) else v)

    out = data_dir / "Table1_Biomarker_Signature.csv"
    table.to_csv(out, index=False)
    log.info("Saved %s\n%s", out, table.to_string(index=False))

    log.info("[4/4] Summary figure")
    plot_table(table, fig_dir / "Figure4_Biomarker_log2FC.png")
    log.info("Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log.error("Pipeline step failed: %s", exc)
        sys.exit(1)
