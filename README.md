# fm_transfer

Code for analyzing genetic perturbations in single-cell CRISPR screens. The repository has two parts:

* `pke/`: the pair-kernel ensemble (PKE), a kernel ridge regression model that predicts the genetic-interaction (GI)
  residual of an unmeasured double perturbation from the single-perturbation profiles of its two genes, together with
  baselines, ablations, interpretability analyses and external validation.
* `scripts/`, `analysis/`, `figures/`: extraction of frozen single-cell foundation-model embeddings (scGPT, Geneformer,
  scVI) and CPU analyses of their geometry.

No machine-specific paths are hard-coded. All scripts read and write under one directory given by an environment variable:

```bash
export FMTRANSFER_ROOT=/path/to/your/workdir   # defaults to the current directory if unset
```

## Pair-kernel ensemble (`pke/`)

### Method in brief

For genes `a`, `b` with single-perturbation shifts `delta_a = x_a - x_0`, the GI residual of the double is
`r_ab = x_ab - x_a - x_b + x_0`. PKE builds gene kernels `k(a, b)` from the single shifts (9 RBF kernels on whitened
principal components and 4 closed-form erf-network kernels) and turns each into two symmetric kernels on unordered pairs:

* interaction kernel `K_x({a,b},{c,d}) = 0.5 * [k(a,c) k(b,d) + k(a,d) k(b,c)]`
* main-effect kernel `K_+({a,b},{c,d}) = 0.5 * [k(a,c) + k(b,d) + k(a,d) + k(b,c)]`

The 22 pair kernels are averaged with equal weights, and centered kernel ridge regression is fitted with the penalty chosen
by closed-form leave-one-out error. Every prediction splits exactly into an intercept, gene-level terms `g(a) + g(b)` and a
pair-specific term, and into contributions of the measured doubles (representer shares that sum to 1).
The model class is `PKE` in `pke/fm_bio.py`.

### Data

All screens are public in the scPerturb collection (Zenodo record 13350497). Place the files under `$FMTRANSFER_ROOT/data/`:

| Dataset | File |
|---|---|
| Norman et al. 2019 (CRISPRa, K562) | `NormanWeissman2019_filtered.h5ad` |
| Wessels et al. 2023 (Cas13 knockdown) | `WesselsSatija2023.h5ad` |
| Tian et al. 2019 (CRISPRi, iPSC) | `TianKampmann2019_iPSC.h5ad` |
| Tian et al. 2019 (CRISPRi, day-7 neurons) | `TianKampmann2019_day7neuron.h5ad` |

The external validation uses the growth GI table of Norman et al. 2019 (Table S5), placed under `$FMTRANSFER_ROOT/data/external/`.

### Environment

Python 3.10 or later with `numpy`, `scipy`, `scikit-learn`, `anndata`, `scanpy`, `openpyxl` and `matplotlib`.
The GEARS baseline additionally needs `cell-gears==0.1.2`, the Peper reimplementation `torch`, the SLIM baseline `h5py`
and the STRING embeddings distributed with SLIM, and the cell-level baselines GPerturb and `cpa-tools`.

### Running

```bash
cd pke
bash run_all.sh
```

or step by step:

| Script | Purpose | Output (per run directory unless noted) |
|---|---|---|
| `prepare.py --dataset D --seed S` | subsample (at most 500 cells per condition), normalize, select 5,000 highly variable genes, pseudobulk | `pb_hvg.npz`, `subsample.npz`, `summary.json` |
| `fm_pke.py --run R` | PKE, baselines (mean, ridge, kNN, single kernels) and ablations; 5-fold CV under random and held-out-gene splits, 3 fold seeds | `pke.json` |
| `fm_bio.py --run R` | GI-mode prediction, in silico screen of untested pairs, gene propensity, exact explanations, residual programs | `bio/*.json`, `bio/*.tsv` |
| `fm_bio2.py --wd W` | iPSC/neuron transfer of GIs and the gene-similarity versus error analysis | `results/bio2.json` |
| `fm_ext.py --wd W --gi F --format norman` | association of predictions for untested pairs with an independent growth GI map | `results/<out>.json` |
| `fm_ceiling.py --run R` | split-half reliability of the GI residuals | `ceiling.json` |
| `fm_gears.py`, `fm_gears_merge.py` | GEARS baseline on the same folds | `gears.json` |
| `fm_pkefm.py --run R` | foundation-model gene kernels added to, or swapped into, PKE (expects embeddings in `results/emb/`) | `pkefm.json` |
| `fm_rev.py --run R` | label-free evaluation gene sets, 1- and 2-unseen-gene categories, intercept-only and kNN baselines, the zero-mean variant PKE-ANOVA, posterior variance versus error, stability of representer shares, GI modes with Holm correction | `rev.json`, `rev_pd.npz` |
| `fm_rev_al.py --run R` | choice of new doubles by posterior variance, predicted strength or both, against random selection | `rev_al.json` |
| `fm_rev_base.py --run R --emb E [--noisy N]` | Salt (additive), a Peper reimplementation, SLIM and PKE with a STRING kernel; optionally single profiles from N cells | `rev_base.json`, `rev_noisy<N>.json` |
| `fm_dl.py --run R --model gperturb\|cpa --scheme S --fold K` | cell-level GPerturb and CPA baselines on the same folds | `dl_<model>_<scheme>_f<K>.npz` |
| `fm_rev_ext.py --wd W --gi F` | external validation with a gene-label permutation test and gene-level controls | `results/rev_ext.json` |
| `fm_rev_tian.py --wd W` | split-half noise ceiling for the iPSC/neuron comparison | `results/rev_tian.json` |
| `fm_rev_sciplex.py --h5ad F --wd W` | preprocessing of the combinatorial sci-Plex drug screen | `results/Sciplex_s*_mc500/` |
| `fm_rev_summary.py W`, `fm_dl_summary.py W` | dataset-level statistics pooled over subsamples (bootstrap over doubles) | `results/rev_summary.json`, `results/dl_summary.json` |
| `fm_final_table.py W` | summary table across datasets and splits | `results/final_table.txt` |
| `fm_figdata.py`, `make_figures.py` | collect results and draw the figures | `results/figdata.json`, `fig*.pdf` |

`fmcore.py` and `fmpredict.py` hold the shared helpers (condition parsing, the exact GI residual, evaluation genes,
cross-validation splits and ridge pair features). `fm_gira*.py` contain the kernel utilities and the earlier kernel
experiments (single pair kernels, random-feature ensembles, kernel selection by leave-one-out error and pretrained-embedding
kernels) that PKE builds on.

### Evaluation protocol

* Held-out doubles by 5-fold cross-validation, random or held-out-gene splits, fold seeds 0, 1 and 2.
* Evaluation genes per double: the 50 genes with the largest absolute change from control, excluding the two targets.
* Additional evaluation gene sets that use no information from the test double: the 50 genes most changed by either
  single perturbation, and all 5,000 genes.
* `R^2 = 1 - sum ||r - r_hat||^2 / sum ||r||^2` (relative to the additive model) and a centered `R^2` relative to the
  intercept-only model. Dataset-level confidence intervals pool errors over subsamples and fold seeds and resample doubles.

## Foundation-model embedding analyses (`scripts/`, `analysis/`, `figures/`)

```
config/      central path resolution (FMTRANSFER_ROOT)
scripts/     data download and GPU embedding extraction (scGPT, Geneformer, scVI)
analysis/    CPU analyses of the saved embeddings
figures/     figure generation
slurm/       generic SLURM template with placeholders
requirements-*.txt   one environment per model family and one for the analyses
```

Each model family needs its own environment because their dependencies conflict:

```bash
python -m venv env_scgpt && source env_scgpt/bin/activate
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121 && pip install -r requirements-scgpt.txt

python -m venv env_geneformer && source env_geneformer/bin/activate
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121 && pip install -r requirements-geneformer.txt

python -m venv env_scvi && source env_scvi/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121 && pip install -r requirements-scvi.txt

python -m venv env_analysis && source env_analysis/bin/activate && pip install -r requirements-analysis.txt
```

Download the data and extract embeddings, then run the analyses:

```bash
python scripts/download_all.py && python scripts/download_wessels.py
python scripts/extract_scgpt.py norman      # likewise for wessels, extract_geneformer.py, extract_scvi.py
python analysis/exp1_additivity.py norman
python analysis/gi_tracking.py norman
python analysis/program_classification.py
python analysis/criterion_vs_gap.py
python analysis/correction.py
FIGDIR=figures_out python figures/make_figures.py
```

## License

MIT. The datasets and model checkpoints are governed by their own licenses.
