"""GrnPFN on the disjoint-cell folds. GrnPFN (a prior-data fitted network) is trained from its public code (the trained
weights are not distributed), with screen templates from fm_prior_stats.py, and evaluated here next to PKE, ridge and the
training mean on identical folds. As in the GrnPFN benchmark, cells of every condition are split into thirds (inputs,
training residuals, test residuals; 3 rotations) and the 256 genes with the largest single-perturbation change in the input
third are kept; all methods use these genes, and R^2 is computed on the 50 genes most changed by each double in its test
third (and on all 256 genes).
Output: results/grnpfn/<dataset>.json.  Usage: fm_grnpfn_eval.py --wd W --model M.pt --grnpfn-src DIR --dataset D"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fmcore as F, fmpredict as FP
from fm_bio import PKE
from fm_gira import gene_feats, ridge_pairs
from fm_disjoint import load, part_means

G_KEEP = 256
METHODS = ["grnpfn", "pke", "ridge", "intercept"]


def main():
    p = argparse.ArgumentParser(); p.add_argument("--wd", default="."); p.add_argument("--model", required=True); p.add_argument("--grnpfn-src", required=True)
    p.add_argument("--dataset", required=True); p.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2]); p.add_argument("--fseeds", type=int, nargs="*", default=[0, 1, 2])
    p.add_argument("--min-cells", type=int, default=30); p.add_argument("--pergene", action="store_true"); a = p.parse_args()
    sys.path.insert(0, a.grnpfn_src)
    import torch
    from grnpfn.model import load_grnpfn, predict
    _load = torch.load; torch.load = lambda *x, **k: _load(*x, **{**k, "weights_only": False})   # checkpoint stores its config
    torch.set_num_threads(8); model = load_grnpfn(a.model)
    genes, singles, doubles, C, cells = load(a.wd, a.dataset, a.min_cells)
    pairs = [(x, y) for _, x, y in doubles]; n = len(doubles); sidx = {s: i for i, s in enumerate(singles)}
    nS = np.array([len(cells[s]) for s in singles], float); nP = np.array([len(cells[d]) for d, _, _ in doubles], float); nC = float(len(cells["control"]))
    schemes = ["random", "genegrp"]; gsets = ["de", "all"]
    err = {s: np.zeros((len(METHODS), len(gsets), n)) for s in schemes}; den = {s: np.zeros((len(gsets), n)) for s in schemes}
    pg = {}   # per gene (all-gene set, random split): [err per method..., den]
    for sd in a.seeds:
        parts, _, _ = part_means(C, cells, 1000 + sd)
        for r in range(3):
            A_, B_, C_ = parts[r], parts[(r + 1) % 3], parts[(r + 2) % 3]
            Dall = np.stack([A_[s] - A_["control"] for s in singles]); keep = np.sort(np.argsort(-np.abs(Dall).max(0))[:G_KEEP])
            raw = {s: Dall[i, keep] for i, s in enumerate(singles)}; mu = A_["control"][keep]
            RB = np.asarray(F.residual_exact(B_, "control", doubles), np.float64)[:, keep]; RC = np.asarray(F.residual_exact(C_, "control", doubles), np.float64)[:, keep]
            de = []
            for d, x, y in doubles:
                ch = np.abs(C_[d][keep] - C_["control"][keep]); de.append(np.argsort(-ch)[:50])
            GS = dict(de=de, all=[np.arange(len(keep))] * n)
            base = PKE(raw, singles); f20 = gene_feats(raw, 20); Fr = np.stack([FP.pair_features(f20[x], f20[y]) for x, y in pairs])
            D = Dall[:, keep].astype(np.float32); pidx = np.array([(sidx[x], sidx[y]) for x, y in pairs])
            for sc in schemes:
                for fs in a.fseeds:
                    for tr, te in FP.splits(doubles, sc, 5, fs):
                        ptr, pte = [pairs[i] for i in tr], [pairs[i] for i in te]; P = {}
                        P["intercept"] = np.repeat(RB[tr].mean(0)[None], len(te), 0)
                        m = PKE.__new__(PKE); m.__dict__.update(base.__dict__); P["pke"] = m.fit(ptr, RB[tr]).predict(pte)
                        P["ridge"] = ridge_pairs(Fr[tr], Fr[te], RB[tr])
                        P["grnpfn"] = predict(model, D, mu.astype(np.float32), pidx[tr], RB[tr].astype(np.float32), pidx[te], nS, nP[tr], nP[te], nC)
                        for gi, gs in enumerate(gsets):
                            for j, i in enumerate(te):
                                ix = GS[gs][i]; t = RC[i, ix]; den[sc][gi, i] += t @ t
                                for mi, mm in enumerate(METHODS):
                                    e = t - P[mm][j, ix]; err[sc][mi, gi, i] += e @ e
                                if a.pergene and sc == "random" and gs == "all":
                                    for q, gq in enumerate(keep):
                                        v = pg.setdefault(str(genes[gq]), np.zeros(len(METHODS) + 1)); v[-1] += t[q] ** 2
                                        for mi, mm in enumerate(METHODS): v[mi] += (t[q] - P[mm][j, q]) ** 2
            print(f"{a.dataset} seed {sd} rotation {r} done", flush=True)
    out = dict(dataset=a.dataset, model=os.path.basename(a.model), n_doubles=n, seeds=a.seeds, fseeds=a.fseeds, pairs=[list(p_) for p_ in pairs])
    for sc in schemes:
        for gi, gs in enumerate(gsets):
            d0 = den[sc][gi].sum(); out[f"{sc}|{gs}"] = {mm: float(1 - err[sc][mi, gi].sum() / d0) for mi, mm in enumerate(METHODS)}
            print(sc, gs, " ".join(f"{mm}={out[f'{sc}|{gs}'][mm]:+.3f}" for mm in METHODS), flush=True)
    od = os.path.join(a.wd, "results", "grnpfn"); os.makedirs(od, exist_ok=True); tag = os.path.splitext(os.path.basename(a.model))[0]
    json.dump(out, open(os.path.join(od, f"{a.dataset}_{tag}.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(od, f"{a.dataset}_{tag}.npz"), methods=np.array(METHODS), gsets=np.array(gsets),
                        **{f"err_{s}": err[s] for s in schemes}, **{f"den_{s}": den[s] for s in schemes}, pairs=np.array(pairs))
    if a.pergene:
        tgt = {s.upper() for s in singles}; rows = sorted(pg.items(), key=lambda kv: -kv[1][-1]); tot = sum(v for _, v in pg.items())
        istgt = np.array([g.upper() in tgt for g, _ in rows]); M = np.stack([v for _, v in rows])
        summ = {}
        for lab, mask in (("targets", istgt), ("other", ~istgt), ("all", np.ones(len(rows), bool))):
            Mm = M[mask].sum(0); summ[lab] = dict(n_genes=int(mask.sum()), share_of_signal=float(Mm[-1] / tot[-1]), **{mm: float(1 - Mm[mi] / Mm[-1]) for mi, mm in enumerate(METHODS)})
            print("PERGENE", lab, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in summ[lab].items()}), flush=True)
        for g, v in rows[:15]:
            print("TOPGENE", g, "target" if g.upper() in tgt else "-", " ".join(f"{mm}={1 - v[mi] / v[-1]:+.3f}" for mi, mm in enumerate(METHODS)), f"share={v[-1] / tot[-1]:.3f}", flush=True)
        json.dump(dict(summary=summ, genes={g: v.tolist() for g, v in rows}, methods=METHODS), open(os.path.join(od, f"{a.dataset}_{tag}_pergene.json"), "w"), indent=1)
    print("GRNPFN_EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
