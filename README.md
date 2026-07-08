# What Single-Cell Foundation Models Encode About Genetic Perturbation

Code for the study *"What Single-Cell Foundation Models Encode About Genetic
Perturbation: Composition, Interaction, and the Limits of a Geometric Account."*

This repository probes the **frozen embedding geometry** of single-cell
foundation models (scGPT, Geneformer) — plus a non-pretrained VAE baseline
(scVI) and a raw-expression baseline — to ask what these representations encode
about how genetic perturbations **compose** and **interact**. It reproduces every
figure and number in the paper from public data.

## Summary of findings

- **Compositional additivity.** Foundation-model geometry is substantially more
  additive than raw expression: a double perturbation's embedding is well
  approximated by the sum of its single-perturbation effects.
- **Genetic-interaction (GI) tracking.** The non-additive interaction residual
  tracks measured genetic-interaction strength far above a raw-expression
  baseline (Spearman gap ≈ +0.23, bootstrap P(gap > 0) = 1.0).
- **Not universal, not from pretraining.** GI-tracking replicates across cell
  line and assay for scGPT but **fails for Geneformer under CRISPR interference**,
  and is **reproduced by a non-pretrained VAE** — so it is neither universal
  across architectures nor a product of observational pretraining.
- **No single geometric explanation.** An effective-rank criterion explains the
  failure *within* the transformer family but does **not** generalize across
  architectures, and a post-hoc geometric correction cannot restore lost
  structure (a negative result).
- **Phenotypic program.** Program identity is decodable from the embeddings and
  transfers to perturbations of unseen genes, above a gene-identity ceiling
  (ruling out leakage).

## Repository layout

```
config/            # Central path/config resolution (no hard-coded paths)
  paths.py
scripts/           # Data download + GPU embedding extraction
  download_all.py
  download_wessels.py
  extract_scgpt.py
  extract_geneformer.py
  extract_scvi.py
analysis/          # CPU analyses that consume saved embeddings
  common.py
  exp1_additivity.py
  gi_tracking.py
  program_classification.py
  criterion_vs_gap.py
  correction.py
figures/           # Figure generation
  make_figures.py
slurm/             # Generic SLURM template (no cluster-specific values)
  example_extract.sbatch
requirements-*.txt # One environment per model family + one for analysis
```

> Note: no cluster account codes, usernames, hostnames, or absolute paths appear
> anywhere in this repository. The SLURM template uses placeholders you fill in
> for your own scheduler.

## Configuration: where the data lives

No machine-specific or personal paths are hard-coded anywhere. All scripts
resolve their data directory from a single environment variable:

```bash
export FMTRANSFER_ROOT=/path/to/your/data
```

If unset, it defaults to `./data` next to the repository. See
[`config/paths.py`](config/paths.py). The expected layout under
`FMTRANSFER_ROOT` is:

```
norman/perturb_processed.h5ad          # GEARS-processed Norman 2019
wessels/WesselsSatija2023.h5ad         # scPerturb Wessels / CaRPool-seq
scGPT_human/{best_model.pt,args.json,vocab.json}
hf_cache/                              # HuggingFace cache (Geneformer)
norman_labeled/norman_2019_adata.h5ad  # program labels (for program analysis)
*.npz                                  # extracted embeddings (written here)
```

## Environments

The three model families have **mutually incompatible dependencies**, so each
gets its own environment. Analyses and figures need none of the heavy model
dependencies. We use Python 3.10 and CUDA 12.1 throughout.

```bash
# scGPT extraction
python -m venv env_scgpt && source env_scgpt/bin/activate
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-scgpt.txt

# Geneformer extraction (separate env)
python -m venv env_geneformer && source env_geneformer/bin/activate
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-geneformer.txt

# scVI extraction (separate env)
python -m venv env_scvi && source env_scvi/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-scvi.txt

# Analysis + figures (CPU only; no GPU or model deps)
python -m venv env_analysis && source env_analysis/bin/activate
pip install -r requirements-analysis.txt
```

> Tip: set `PYTHONNOUSERSITE=1` inside each environment to avoid leaking packages
> from a shared user site-packages directory.

## Data sources (all public)

| Dataset / model | Source |
|---|---|
| Norman 2019 (CRISPRa, combinatorial) | via [GEARS](https://github.com/snap-stanford/GEARS) `PertData` |
| Wessels / CaRPool-seq (CRISPRi, Cas13) | [Zenodo record 13350497](https://zenodo.org/records/13350497) (scPerturb) |
| scGPT whole-human checkpoint | [scGPT repo](https://github.com/bowang-lab/scGPT) (public Drive folder) |
| Geneformer V2-104M | [HuggingFace `ctheodoris/Geneformer`](https://huggingface.co/ctheodoris/Geneformer) |
| Norman program labels | public figshare release (`norman_2019_adata.h5ad`) |

Download everything (run on a machine with internet, e.g. a login node):

```bash
python scripts/download_all.py
python scripts/download_wessels.py
```

## Reproducing the study

### 1. Extract embeddings (GPU)

Each extraction takes a dataset argument (`norman` or `wessels`) and writes
`*_pseudobulk.npz` / `*_cell_emb.npz` into `FMTRANSFER_ROOT`.

```bash
# scGPT env
python scripts/extract_scgpt.py norman
python scripts/extract_scgpt.py wessels

# Geneformer env
python scripts/extract_geneformer.py norman
python scripts/extract_geneformer.py wessels

# scVI env (runs both datasets)
python scripts/extract_scvi.py
```

### 2. Run analyses (CPU)

```bash
# analysis env
python analysis/exp1_additivity.py norman
python analysis/exp1_additivity.py wessels
python analysis/gi_tracking.py norman
python analysis/gi_tracking.py wessels
python analysis/program_classification.py
python analysis/criterion_vs_gap.py
python analysis/correction.py
```

Each writes a small JSON of results into `FMTRANSFER_ROOT`.

### 3. Generate figures

```bash
FIGDIR=figures_out python figures/make_figures.py
```

The figure script reproduces all six paper figures. It ships with the reported
values so it runs standalone; point it at your JSON outputs to regenerate from
your own run.

## Method notes

- **Pseudobulking & subsampling.** Cells are subsampled to at most 500 per
  condition (fixed seed) before pseudobulking, for comparability across
  conditions of very different size.
- **Gene-name handling.** Norman variables are Ensembl IDs (scGPT maps via
  symbols; Geneformer is Ensembl-native). Wessels variables are symbols (scGPT
  maps directly; Geneformer maps symbols → Ensembl via its own bundled
  dictionary). The legacy symbol `VPRBP` is aliased to its current symbol
  `DCAF1`, and Wessels guide-barcode pseudo-features are filtered out.
- **GI ground truth.** For each double, the additivity residual in
  log-normalized expression restricted to that condition's top-50 differentially
  expressed genes; applied identically across datasets for like-for-like
  comparison.
- **GI-tracking gap.** Model-minus-raw-PCA difference in Spearman correlation
  between embedding interaction-residual magnitude and expression GI magnitude,
  with a 5,000-sample **paired** bootstrap (same resampled doubles for model and
  baseline).
- **Program leakage control.** Program classification uses a leave-genes-out
  split (test conditions share no gene with training) and a gene-identity
  one-hot baseline, so an FM advantage cannot be gene-identity leakage.

## Reproducibility

All analyses operate on **frozen** embeddings; no model weights are modified.
Datasets and checkpoints are public. Random seeds are fixed (`SEED = 0`).
Bootstraps use 5,000 resamples.

## License

Released under the MIT License. Note that the datasets
and model checkpoints are governed by their own respective licenses; consult the
sources above before redistribution.

## Citation

If you use this code, please cite the paper. A BibTeX entry will be added here
upon publication.
