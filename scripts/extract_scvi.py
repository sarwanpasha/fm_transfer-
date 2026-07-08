"""
scVI embedding extraction (non-pretrained VAE baseline).

Trains scVI de novo on each dataset and extracts the latent representation.
scVI has NO observational pretraining -- it learns only from the perturbation
dataset itself, serving as the third architecture type in the study.

Usage:
    python scripts/extract_scvi.py            # runs both norman and wessels
    python scripts/extract_scvi.py norman

Run on a GPU node. Requires FMTRANSFER_ROOT and an environment with scvi-tools,
scanpy, torch (see requirements-scvi.txt).
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import scanpy as sc
import scipy.sparse as sp
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.paths import ROOT, CAP, SEED

N_LATENT = 30
ALIAS = {"VPRBP": "DCAF1"}
rng = np.random.default_rng(SEED)

import scvi
scvi.settings.seed = SEED


def norm_cond_wessels(p):
    if p == "control":
        return "ctrl"
    parts = [ALIAS.get(g, g) for g in p.split("_")]
    return f"{parts[0]}+ctrl" if len(parts) == 1 else "+".join(parts)


def prep(ds):
    if ds == "norman":
        os.environ.setdefault("GEARS_DATA", ROOT)
        ad = sc.read_h5ad(os.path.join(ROOT, "norman", "perturb_processed.h5ad"))
        raw = ad.layers["counts"] if "counts" in ad.layers else ad.X
        ad.layers["counts"] = sp.csr_matrix(raw)
    else:
        ad = sc.read_h5ad(os.path.join(ROOT, "wessels", "WesselsSatija2023.h5ad"))
        good = np.array([(":" not in g) and ("_g1" not in g) and ("_g2" not in g)
                         for g in ad.var.index.astype(str)])
        ad = ad[:, good].copy()
        ad.obs["condition"] = [norm_cond_wessels(p)
                               for p in ad.obs["perturbation"].astype(str)]
        ad.layers["counts"] = sp.csr_matrix(ad.X)
    return ad


def run(ds):
    print(f"===== scVI on {ds} =====", flush=True)
    ad = prep(ds)
    cond = ad.obs["condition"].astype(str).values
    keep = []
    for c in np.unique(cond):
        idx = np.where(cond == c)[0]
        if len(idx) > CAP:
            idx = rng.choice(idx, size=CAP, replace=False)
        keep.append(idx)
    keep = np.sort(np.concatenate(keep))
    ad = ad[keep].copy()
    cond = cond[keep]
    print(f"[scVI/{ds}] {ad.shape} after subsample", flush=True)

    scvi.model.SCVI.setup_anndata(ad, layer="counts")
    model = scvi.model.SCVI(ad, n_latent=N_LATENT)
    model.train(max_epochs=200, early_stopping=True,
                accelerator="gpu" if torch.cuda.is_available() else "cpu")
    Z = model.get_latent_representation()
    print(f"[scVI/{ds}] latent {Z.shape}", flush=True)

    isd = np.array([("+" in c and "ctrl" not in c) for c in cond])
    np.savez(os.path.join(ROOT, f"scvi_{ds}_cell_emb.npz"),
             emb=Z.astype(np.float32), condition=cond.astype(str), is_double=isd)
    uniq = np.unique(cond)
    pb = np.zeros((len(uniq), Z.shape[1]), np.float32)
    pbd = np.zeros(len(uniq), bool)
    for i, c in enumerate(uniq):
        m = cond == c
        pb[i] = Z[m].mean(0)
        pbd[i] = ("+" in c and "ctrl" not in c)
    np.savez(os.path.join(ROOT, f"scvi_{ds}_pseudobulk.npz"),
             emb=pb, condition=uniq.astype(str), is_double=pbd)
    print(f"[scVI/{ds}] pseudobulk {pb.shape}; DONE", flush=True)


if __name__ == "__main__":
    print("torch", torch.__version__, "cuda", torch.cuda.is_available(), flush=True)
    targets = sys.argv[1:] if len(sys.argv) > 1 else ["norman", "wessels"]
    for ds in targets:
        run(ds)
    print("ALL DONE", flush=True)
