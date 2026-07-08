"""
Central configuration. All paths are resolved from the FMTRANSFER_ROOT
environment variable (or a local default), so no machine-specific or personal
paths are hard-coded anywhere in the codebase.

Usage:
    export FMTRANSFER_ROOT=/path/to/your/data
    # or leave unset to use ./data relative to the repository

Directory layout expected under FMTRANSFER_ROOT:
    norman/perturb_processed.h5ad          (GEARS-processed Norman 2019)
    wessels/WesselsSatija2023.h5ad         (scPerturb Wessels/CaRPool-seq)
    scGPT_human/{best_model.pt,args.json,vocab.json}
    hf_cache/                              (HuggingFace cache for Geneformer)
    *.npz                                  (extracted embeddings, written here)
"""
import os

# Root data directory: env var first, else ./data next to the repo.
ROOT = os.environ.get(
    "FMTRANSFER_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data")),
)

# Standard subpaths (derived, do not edit).
NORMAN_H5AD = os.path.join(ROOT, "norman", "perturb_processed.h5ad")
WESSELS_H5AD = os.path.join(ROOT, "wessels", "WesselsSatija2023.h5ad")
SCGPT_DIR = os.path.join(ROOT, "scGPT_human")
HF_CACHE = os.path.join(ROOT, "hf_cache")

# Analysis constants used across experiments.
CAP = 500          # max cells per condition when subsampling
SEED = 0
TOPN_DE = 50       # top differentially expressed genes for GI ground truth

def ensure_root():
    if not os.path.isdir(ROOT):
        raise SystemExit(
            f"Data root '{ROOT}' does not exist. Set FMTRANSFER_ROOT to your "
            f"data directory (see config/paths.py and README)."
        )
    return ROOT
