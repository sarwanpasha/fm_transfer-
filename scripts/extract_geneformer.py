"""
Geneformer V2-104M embedding extraction.

Produces per-cell (subsampled) and per-condition pseudobulk embeddings from the
frozen Geneformer V2-104M model.

Usage:
    python scripts/extract_geneformer.py norman
    python scripts/extract_geneformer.py wessels

Run on a GPU node. Requires FMTRANSFER_ROOT and an environment with geneformer,
scanpy, torch (see requirements-geneformer.txt). Geneformer needs RAW COUNTS.
"""
import os
import sys
import glob
import pickle
import shutil
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import scipy.sparse as sp
import scanpy as sc
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.paths import ROOT, HF_CACHE, CAP, SEED

ALIAS = {"VPRBP": "DCAF1"}
rng = np.random.default_rng(SEED)


def gf_paths():
    snap = glob.glob(os.path.join(HF_CACHE, "models--ctheodoris--Geneformer",
                                  "snapshots", "*"))[0]
    model_dir = os.path.join(snap, "Geneformer-V2-104M")
    name_id = pickle.load(open(os.path.join(snap, "geneformer",
                                            "gene_name_id_dict_gc104M.pkl"), "rb"))
    return model_dir, name_id


def norm_cond_wessels(p):
    if p == "control":
        return "ctrl"
    parts = [ALIAS.get(g, g) for g in p.split("_")]
    return f"{parts[0]}+ctrl" if len(parts) == 1 else "+".join(parts)


def looks_like_counts(M):
    s = M[:50].toarray() if sp.issparse(M) else np.asarray(M[:50])
    return np.allclose(s, np.round(s)) and s.min() >= 0


def load_dataset(name, sym2ens):
    if name == "norman":
        os.environ.setdefault("GEARS_DATA", ROOT)
        from gears import PertData
        pd_ = PertData(ROOT)
        pd_.load(data_name="norman")
        adata = pd_.adata.copy()
        raw = adata.layers["counts"] if "counts" in adata.layers else adata.X
        adata.var["ensembl_id"] = adata.var.index.astype(str)  # Norman var is Ensembl
        adata.X = sp.csr_matrix(raw)
        return adata
    elif name == "wessels":
        adata = sc.read_h5ad(os.path.join(ROOT, "wessels", "WesselsSatija2023.h5ad"))
        good = np.array([(":" not in g) and ("_g1" not in g) and ("_g2" not in g)
                         for g in adata.var.index.astype(str)])
        adata = adata[:, good].copy()
        adata.obs["condition"] = [norm_cond_wessels(p)
                                  for p in adata.obs["perturbation"].astype(str)]
        # map symbols -> Ensembl via Geneformer's own dictionary
        syms = adata.var.index.astype(str)
        ens = np.array([sym2ens.get(s, None) for s in syms], dtype=object)
        mappable = np.array([e is not None for e in ens])
        adata = adata[:, mappable].copy()
        adata.var["ensembl_id"] = [sym2ens[s] for s in adata.var.index.astype(str)]
        adata.X = sp.csr_matrix(adata.X)
        return adata
    raise ValueError(name)


def main(name):
    prefix = "" if name == "norman" else f"{name}_"
    model_dir, sym2ens = gf_paths()
    print(f"[GF/{name}] model {model_dir}", flush=True)

    h5 = os.path.join(ROOT, f"{prefix}gf_h5ad")
    tok = os.path.join(ROOT, f"{prefix}gf_tok")
    emb = os.path.join(ROOT, f"{prefix}gf_emb")
    for d in (h5, tok, emb):
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)

    adata = load_dataset(name, sym2ens)
    print(f"[GF/{name}] genes mapped: {adata.n_vars}", flush=True)
    raw = sp.csr_matrix(adata.X)
    if not looks_like_counts(raw):
        print("[GF] WARNING: X does not look like raw counts; Geneformer rank "
              "encoding requires raw counts. Results may be invalid.", flush=True)

    cond = adata.obs["condition"].astype(str).values
    keep = []
    for c in np.unique(cond):
        idx = np.where(cond == c)[0]
        if len(idx) > CAP:
            idx = rng.choice(idx, size=CAP, replace=False)
        keep.append(idx)
    keep = np.sort(np.concatenate(keep))
    sub = sc.AnnData(X=raw[keep].copy(), obs=adata.obs.iloc[keep].copy(),
                     var=adata.var.copy())
    sub.obs["n_counts"] = np.asarray(sub.X.sum(1)).ravel()
    sub.obs["cell_index"] = np.arange(sub.n_obs)
    print(f"[GF/{name}] subsampled {sub.n_obs}", flush=True)
    sub.write(os.path.join(h5, "data.h5ad"))

    from geneformer import TranscriptomeTokenizer, EmbExtractor
    tk = TranscriptomeTokenizer(
        custom_attr_name_dict={"cell_index": "cell_index", "condition": "condition"},
        nproc=8, model_input_size=4096, special_token=True)
    tk.tokenize_data(h5, tok, "data", file_format="h5ad")
    ds = os.path.join(tok, "data.dataset")

    embex = EmbExtractor(model_type="Pretrained", num_classes=0, emb_mode="cell",
                         max_ncells=None, emb_layer=-1,
                         emb_label=["cell_index", "condition"],
                         forward_batch_size=32, nproc=8, model_version="V2")
    df = embex.extract_embs(model_dir, ds, emb, "cell", output_torch_embs=False)
    label_cols = {"cell_index", "condition"}
    emb_cols = [c for c in df.columns if c not in label_cols]
    df = df.sort_values("cell_index")
    cell = df[emb_cols].to_numpy(dtype=np.float32)
    cond_out = df["condition"].astype(str).to_numpy()
    isd = np.array([("+" in c and "ctrl" not in c) for c in cond_out])
    print(f"[GF/{name}] cell emb {cell.shape}", flush=True)
    np.savez(os.path.join(ROOT, f"{prefix}geneformer_cell_emb.npz"),
             emb=cell, condition=cond_out, is_double=isd)

    uniq = np.unique(cond_out)
    pb = np.zeros((len(uniq), cell.shape[1]), np.float32)
    pbd = np.zeros(len(uniq), bool)
    for i, c in enumerate(uniq):
        m = cond_out == c
        pb[i] = cell[m].mean(0)
        pbd[i] = ("+" in c and "ctrl" not in c)
    np.savez(os.path.join(ROOT, f"{prefix}geneformer_pseudobulk.npz"),
             emb=pb, condition=uniq.astype(str), is_double=pbd)
    print(f"[GF/{name}] pseudobulk {pb.shape}; DONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "norman")
