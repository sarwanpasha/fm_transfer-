"""GenePert-style baseline on the folds and evaluation genes of fm_rev.py. GenePert predicts perturbation effects by ridge
regression from GenePT gene embeddings (text-embedding-ada-002 embeddings of NCBI gene summaries). For GI residuals, each
double (a, b) is represented by the symmetric features [e_a + e_b, e_a * e_b, |e_a - e_b|] of the 2 gene embeddings, reduced
to 64 whitened principal components computed over the perturbed genes (no labels used), and the residual is regressed on
these features with the penalty chosen by efficient LOO. Genes without an embedding get the mean embedding (counted in the
output). PKE, ridge on single-perturbation profiles and the training mean are recomputed on the same folds.
Output: <run>/genepert.json and <run>/genepert_pd.npz (same layout as strong_pd.npz).
Usage: fm_genepert.py --run R --emb GenePT_gene_embedding_ada_text.pickle [--seeds 0 1 2]"""
import argparse, json, os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fm_bio import PKE
from fm_gira import gene_feats, ridge_pairs


def emb_feats(emb, genes, k=64):
    E = []; miss = []
    for g in genes:
        v = emb.get(str(g).upper())
        if v is None: miss.append(g)
        E.append(None if v is None else np.asarray(v, np.float64))
    mean = np.mean([e for e in E if e is not None], 0); E = np.stack([mean if e is None else e for e in E])
    E = E - E.mean(0); U, S, _ = np.linalg.svd(E, full_matrices=False); k = min(k, len(genes) - 1)
    Z = U[:, :k] * S[:k]; Z /= np.maximum(Z.std(0, keepdims=True), 1e-12)
    return {g: Z[i] for i, g in enumerate(genes)}, miss


def pair_feat(za, zb):
    return np.concatenate([za + zb, za * zb, np.abs(za - zb)])


def main():
    p = argparse.ArgumentParser(); p.add_argument("--run", required=True); p.add_argument("--emb", required=True)
    p.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2]); a = p.parse_args()
    import fmcore as F, fmpredict as FP
    Z = np.load(os.path.join(a.run, "pb_hvg.npz"), allow_pickle=True)
    conds = list(Z["conds"]); expr = {c: Z["pb"][i].astype(np.float64) for i, c in enumerate(conds)}; hn = np.asarray(Z["genes"])
    singles, doubles, _ = F.parse_conditions(conds); ctrl = expr["control"]
    R = np.asarray(F.residual_exact(expr, "control", doubles), np.float64); n, G = R.shape
    raw = {s: expr[s] - ctrl for s in singles}; pairs = [(x, y) for _, x, y in doubles]
    GS = {"de_obs": FP.de_sets(expr, "control", doubles, hn, 50)}
    emb = pickle.load(open(a.emb, "rb")); ez, miss = emb_feats(emb, singles)
    Fe = np.stack([pair_feat(ez[x], ez[y]) for x, y in pairs])
    f20 = gene_feats(raw, 20); Fraw = np.stack([FP.pair_features(f20[x], f20[y]) for x, y in pairs]); base = PKE(raw, singles)
    meth = ["genepert", "pke", "ridge_raw", "mean_resid"]
    out = dict(n_doubles=n, doubles=[d for d, _, _ in doubles], missing_embedding=miss, n_singles=len(singles)); pd = {}
    for scheme in ("random", "genegrp"):
        acc = {(m, g, c): np.zeros((3, n)) for m in meth for g in GS for c in ("all", "u1", "u2")}
        for sd in a.seeds:
            for tr, te in FP.splits(doubles, scheme, 5, sd):
                trg = {g for i in tr for g in pairs[i]}; ptr, pte = [pairs[i] for i in tr], [pairs[i] for i in te]
                mu = R[tr].mean(0); P = {"mean_resid": np.repeat(mu[None], len(te), 0), "ridge_raw": ridge_pairs(Fraw[tr], Fraw[te], R[tr]),
                                         "genepert": ridge_pairs(Fe[tr], Fe[te], R[tr])}
                m = PKE.__new__(PKE); m.__dict__.update(base.__dict__); P["pke"] = m.fit(ptr, R[tr]).predict(pte)
                for j, i in enumerate(te):
                    nu = sum(g not in trg for g in pairs[i]); cats = ["all"] + ([f"u{nu}"] if scheme == "genegrp" and nu in (1, 2) else [])
                    for gname, sets in GS.items():
                        ix = sets[i]; t = R[i, ix]; tc = t - mu[ix]
                        for mm in meth:
                            e = t - P[mm][j, ix]
                            for c in cats: acc[(mm, gname, c)][:, i] += (e @ e, t @ t, tc @ tc)
        res = {}
        for (mm, gname, c), A in acc.items():
            ok = A[1] > 0
            if ok.sum() == 0: continue
            res.setdefault(gname, {}).setdefault(c, {})[mm] = dict(r2=float(1 - A[0, ok].sum() / A[1, ok].sum()), n=int(ok.sum()))
            pd[f"{scheme}|{gname}|{c}|{mm}"] = A
        out[scheme] = res
        for gname in res:
            for c in res[gname]:
                print(f"GENEPERT {scheme:8s} {gname:7s} {c:3s} " + " ".join(f"{k}={v['r2']:+.3f}" for k, v in res[gname][c].items()), flush=True)
    print(f"missing embeddings: {len(miss)} of {len(singles)}", flush=True)
    json.dump(out, open(os.path.join(a.run, "genepert.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(a.run, "genepert_pd.npz"), **{k.replace("|", "__"): v for k, v in pd.items()}, doubles=np.array(out["doubles"]))
    print("GENEPERT_DONE", flush=True)


if __name__ == "__main__":
    main()
