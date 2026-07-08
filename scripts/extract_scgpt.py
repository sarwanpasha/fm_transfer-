"""
scGPT embedding extraction.

Produces per-cell (subsampled) and per-condition pseudobulk embeddings, plus
static gene-token embeddings, from the frozen scGPT whole-human checkpoint.

Usage:
    python scripts/extract_scgpt.py norman
    python scripts/extract_scgpt.py wessels

Run on a GPU node. Requires FMTRANSFER_ROOT and an environment with scgpt,
gears (cell-gears), scanpy, torch (see requirements-scgpt.txt).
"""
import os
import sys
import json
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import scanpy as sc
import scipy.sparse as sp
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.paths import ROOT, SCGPT_DIR, CAP, SEED

ALIAS = {"VPRBP": "DCAF1"}
rng = np.random.default_rng(SEED)


def norm_cond_wessels(p):
    if p == "control":
        return "ctrl"
    parts = [ALIAS.get(g, g) for g in p.split("_")]
    return f"{parts[0]}+ctrl" if len(parts) == 1 else "+".join(parts)


def load_dataset(name):
    if name == "norman":
        os.environ.setdefault("GEARS_DATA", ROOT)
        from gears import PertData
        pd_ = PertData(ROOT)
        pd_.load(data_name="norman")
        adata = pd_.adata.copy()
        if "gene_name" in adata.var.columns:
            adata.var["gene_symbol"] = adata.var["gene_name"].astype(str)
        else:
            raise SystemExit("No gene_name column in Norman var; cannot map to scGPT vocab.")
        return adata
    elif name == "wessels":
        adata = sc.read_h5ad(os.path.join(ROOT, "wessels", "WesselsSatija2023.h5ad"))
        good = np.array([(":" not in g) and ("_g1" not in g) and ("_g2" not in g)
                         for g in adata.var.index.astype(str)])
        adata = adata[:, good].copy()
        adata.var["gene_symbol"] = adata.var.index.astype(str)
        adata.obs["condition"] = [norm_cond_wessels(p)
                                  for p in adata.obs["perturbation"].astype(str)]
        return adata
    raise ValueError(name)


def main(name):
    prefix = "" if name == "norman" else f"{name}_"
    adata = load_dataset(name)
    print(f"[scGPT/{name}] loaded {adata.shape}", flush=True)
    cond = adata.obs["condition"].astype(str).values

    vocab = json.load(open(os.path.join(SCGPT_DIR, "vocab.json")))
    in_vocab = adata.var["gene_symbol"].isin(set(vocab.keys()))
    print(f"[scGPT/{name}] genes in vocab: {int(in_vocab.sum())}/{adata.n_vars}", flush=True)

    # subsample <= CAP cells/condition
    keep = []
    for c in np.unique(cond):
        idx = np.where(cond == c)[0]
        if len(idx) > CAP:
            idx = rng.choice(idx, size=CAP, replace=False)
        keep.append(idx)
    keep = np.sort(np.concatenate(keep))
    sub = adata[keep].copy()
    cond_sub = cond[keep]
    print(f"[scGPT/{name}] subsampled to {sub.n_obs} cells", flush=True)

    import scgpt as scg
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[scGPT/{name}] device = {device}", flush=True)

    emb_adata = scg.tasks.embed_data(
        sub, SCGPT_DIR, gene_col="gene_symbol", batch_size=64, device=device,
        use_fast_transformer=False, return_new_adata=True,
    )
    cell = np.asarray(emb_adata.X, dtype=np.float32)
    print(f"[scGPT/{name}] cell embeddings {cell.shape}", flush=True)

    isd = np.array([("+" in c and "ctrl" not in c) for c in cond_sub])
    np.savez(os.path.join(ROOT, f"{prefix}scgpt_cell_emb.npz"),
             emb=cell, condition=cond_sub.astype(str), is_double=isd)

    uniq = np.unique(cond_sub)
    pb = np.zeros((len(uniq), cell.shape[1]), np.float32)
    pbd = np.zeros(len(uniq), bool)
    for i, c in enumerate(uniq):
        m = cond_sub == c
        pb[i] = cell[m].mean(0)
        pbd[i] = ("+" in c and "ctrl" not in c)
    np.savez(os.path.join(ROOT, f"{prefix}scgpt_pseudobulk.npz"),
             emb=pb, condition=uniq.astype(str), is_double=pbd)
    print(f"[scGPT/{name}] pseudobulk {pb.shape}; DONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "norman")
