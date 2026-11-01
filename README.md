# GSE24223 spike-in audit and blood transcriptome analysis

Python scripts for a reanalysis of the public dataset
[GSE24223](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE24223) (Affymetrix HG-U133 Plus 2.0 arrays of whole
blood and sorted blood cells from 10 kidney transplant patients before and after transplant, plus 5 healthy controls;
original study: Grigoryev et al., *PLoS ONE* 2010, https://doi.org/10.1371/journal.pone.0013358).

The project checks the Affymetrix control (spike-in) probes to find out which comparisons in the dataset are reliable,
then analyses the reliable samples.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\activate
pip install -r requirements.txt
```

The data are not stored in this repository. Script 01 downloads GSE24223 from GEO into `./data` the first time it runs. If GEO is not reachable, download `GSE24223_family.soft.gz` by hand into `./data` and pass
`--soft-file data/GSE24223_family.soft.gz`.

## Scripts

| Script | What it does |
|---|---|
| `01_ingest_normalize.py` | Downloads the series, assigns Control and Disease groups from the sample metadata, removes Affymetrix control probes (after reporting them), checks the log2 scale, quantile-normalizes, filters the 20% lowest-variance probes, and writes the cleaned matrix |
| `02_differential_expression.py` | Welch t-tests, Benjamini-Hochberg correction, volcano plot, and a sample PCA |
| `03_ml_feature_selection.py` | Random forest with 5-fold cross-validation and SHAP; `--nested-top-k` selects probes inside each fold, and `--permutations` gives an empirical p-value |
| `04_annotate_targets.py` | Maps probes to gene symbols and names using the GPL570 annotation |
| `05_spikein_audit.py` | Spike-in control signal for every cell subtype and time point (dataset-wide audit) |
| `06_paired_timecourse.py` | Paired within-patient comparisons of whole blood at each week against before transplant |

Run `python <script>.py --help` for all options.

## Reproducing the analyses in the manuscript

Spike-in audit of the whole dataset (Figure 1):

```bash
python 05_spikein_audit.py
```

Healthy controls versus patients before transplant in whole blood (Table 3; Figure 2 was made with an earlier version of
step 01 that kept the control probes):

```bash
python 01_ingest_normalize.py --group-field "time point" --control-regex "Control Day 0" --disease-regex "Pre-Tx" --filter "cell subtype=^Whole Blood$" --case-sensitive-filters
python 02_differential_expression.py --pca-shape-col char_donor_number --fallback-top-n 30
python 03_ml_feature_selection.py --nested-top-k 50 --permutations 200
python 04_annotate_targets.py
```

CD4 cells (the comparison confounded by sample preparation), in separate folders:

```bash
python 01_ingest_normalize.py --soft-file data/GSE24223_family.soft.gz --data-dir data_cd4 --fig-dir figures_cd4 --group-field "time point" --control-regex "Control Day 0" --disease-regex "Pre-Tx" --filter "cell subtype=^CD4$"
python 02_differential_expression.py --data-dir data_cd4 --fig-dir figures_cd4 --pca-shape-col char_donor_number
python 03_ml_feature_selection.py --data-dir data_cd4 --fig-dir figures_cd4 --nested-top-k 50
python 04_annotate_targets.py --soft-file data/GSE24223_family.soft.gz --data-dir data_cd4 --fig-dir figures_cd4
```

Paired whole blood time course (Table 1, Figure 3; add `--lfc-threshold 0.58` for the 1.5-fold version in Figure 3):

```bash
python 06_paired_timecourse.py --soft-file data/GSE24223_family.soft.gz
python 06_paired_timecourse.py --soft-file data/GSE24223_family.soft.gz --lfc-threshold 0.58
```

## Notes and limitations

- All results are exploratory. With 5 healthy controls and 8 patients before transplant, no probe is significant after
  multiple-testing correction, and the classifier result is a small-sample estimate.
- The classifier selects probes inside each cross-validation fold; the default mode of script 03 (probes chosen from the
  differential expression list on all samples) gives over-optimistic AUC values and is kept only for comparison.
- Large changes in whole blood after transplant mostly reflect changes in the mix of blood cells, not only changes inside
  cells.
- Output tables and figures are written to `./data` and `./figures`; these folders are ignored by git.

## Citation and license

[Add the manuscript citation here after publication.]  
[Add a license, for example MIT, by choosing "Add file > Create new file > LICENSE" on GitHub.]
