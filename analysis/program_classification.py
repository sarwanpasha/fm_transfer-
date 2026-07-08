"""
Experiment: phenotypic-program classification with leakage control.

Classifies a Norman condition's author-assigned gene program from its pseudobulk
embedding, using a leave-genes-out protocol (test conditions share no constituent
gene with training) and a gene-identity-only baseline to rule out leakage.

Requires the Norman labeled AnnData with a 'gene_program' obs column and a
'guide_merged' obs column for mapping (from the public figshare release), placed
at $FMTRANSFER_ROOT/norman_labeled/norman_2019_adata.h5ad.

Usage:
    python analysis/program_classification.py
"""
import os
import sys
import json
import warnings
from collections import Counter

warnings.filterwarnings("ignore")
import numpy as np
import scanpy as sc
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.paths import ROOT
from analysis.common import load_pb, single_gene, double_genes

rng = np.random.default_rng(0)
PRIMARY = {"Pioneer factors", "Granulocyte/apoptosis", "Erythroid"}


def genes_of(c):
    return [g for g in c.split("+") if g != "ctrl"]


def load_labels():
    lab = sc.read_h5ad(os.path.join(ROOT, "norman_labeled", "norman_2019_adata.h5ad"))
    gm = lab.obs["guide_merged"].astype(str).values
    gp = lab.obs["gene_program"].astype(str).values
    tmp = {}
    for g, p in zip(gm, gp):
        tmp.setdefault(g, []).append(p)
    return {g: Counter(ps).most_common(1)[0][0] for g, ps in tmp.items()}


def build(space, cond_prog, classes):
    rows = []
    for c, prog in cond_prog.items():
        if prog == "Ctrl" or prog not in classes or c not in space:
            continue
        rows.append((c, space[c], prog, genes_of(c)))
    return rows


def lgo_cv(rows, k=5):
    genes = sorted({g for (_, _, _, gs) in rows for g in gs})
    rng.shuffle(genes)
    groups = [set(genes[i::k]) for i in range(k)]
    X = np.stack([r[1] for r in rows])
    y = np.array([r[2] for r in rows])
    bas, f1s, ntest = [], [], []
    for gg in groups:
        te = [i for i, (_, _, _, gs) in enumerate(rows) if any(g in gg for g in gs)]
        tr = [i for i, (_, _, _, gs) in enumerate(rows) if all(g not in gg for g in gs)]
        if len(te) < 2 or len(tr) < 6 or len(set(y[tr])) < 2:
            continue
        sca = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced")
        clf.fit(sca.transform(X[tr]), y[tr])
        pred = clf.predict(sca.transform(X[te]))
        bas.append(balanced_accuracy_score(y[te], pred))
        f1s.append(f1_score(y[te], pred, average="macro"))
        ntest.append(len(te))
    if not bas:
        return None
    return dict(bal_acc=float(np.mean(bas)), bal_acc_std=float(np.std(bas)),
                macro_f1=float(np.mean(f1s)), n_folds=len(bas), n_test=int(sum(ntest)))


def gene_onehot_lgo(cond_prog, classes, k=5):
    rows = [(c, prog, genes_of(c)) for c, prog in cond_prog.items()
            if prog != "Ctrl" and prog in classes]
    allg = sorted({g for (_, _, gs) in rows for g in gs})
    gidx = {g: i for i, g in enumerate(allg)}
    X = np.zeros((len(rows), len(allg)), np.float32)
    y = np.array([r[1] for r in rows])
    for i, (_, _, gs) in enumerate(rows):
        for g in gs:
            X[i, gidx[g]] = 1.0
    rng.shuffle(allg)
    groups = [set(allg[i::k]) for i in range(k)]
    bas = []
    for gg in groups:
        te = [i for i, (_, _, gs) in enumerate(rows) if any(g in gg for g in gs)]
        tr = [i for i, (_, _, gs) in enumerate(rows) if all(g not in gg for g in gs)]
        if len(te) < 2 or len(tr) < 6 or len(set(y[tr])) < 2:
            continue
        clf = LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced")
        clf.fit(X[tr], y[tr])
        bas.append(balanced_accuracy_score(y[te], clf.predict(X[te])))
    return float(np.mean(bas)) if bas else float("nan")


def main():
    cond_prog = load_labels()
    scgpt = load_pb(os.path.join(ROOT, "scgpt_pseudobulk.npz"))
    gf = load_pb(os.path.join(ROOT, "geneformer_pseudobulk.npz"))
    # rawPCA on the same conditions
    from analysis.common import expr_pseudobulk, rawpca_pb
    from config.paths import NORMAN_H5AD
    expr, uniq = expr_pseudobulk(NORMAN_H5AD, wessels=False)
    rawpca = rawpca_pb(expr, uniq)
    spaces = {"scGPT": scgpt, "Geneformer": gf, "rawPCA": rawpca}

    print("=== program classification (leave-genes-out, 3-class) ===")
    print(f"{'space':<12}{'balAcc':>10}{'macroF1':>10}{'nfold':>7}{'ntest':>7}")
    results = {}
    for sn, space in spaces.items():
        rows = build(space, cond_prog, PRIMARY)
        r = lgo_cv(rows)
        if r:
            results[sn] = r
            print(f"{sn:<12}{r['bal_acc']:>10.3f}{r['macro_f1']:>10.3f}"
                  f"{r['n_folds']:>7}{r['n_test']:>7}")
    gb = gene_onehot_lgo(cond_prog, PRIMARY)
    results["gene_onehot"] = {"bal_acc": gb}
    print(f"{'gene-1hot':<12}{gb:>10.3f}")
    print(f"(chance ~ {1/len(PRIMARY):.3f})")
    json.dump(results, open(os.path.join(ROOT, "program_classification.json"), "w"),
              indent=2, default=float)
    print("[program] wrote program_classification.json")


if __name__ == "__main__":
    main()
