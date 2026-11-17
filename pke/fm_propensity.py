"""Interaction propensity as a property of a gene.
Per run (dataset subsample):
 (1) gene-level structure of the observed residuals: 5-fold CV (random doubles, 3 seeds) of a one-hot gene-additive model
     r_ab = mu + h_a + h_b (ridge, penalty by LOO), PKE main-effect only, PKE and the intercept-only model; centered R2 on the
     50 most changed genes. Also the split-half reliability ceiling of the residuals on the same genes.
 (2) reproducibility of propensity across disjoint doubles: doubles are split into random halves (50 times), PKE is fitted on each
     half, and the Spearman correlation of ||g(x)|| between halves is computed over all single genes; the propensity from one half
     is also correlated with the observed hubness (mean residual norm over the gene's doubles) in the other half, alone and as a
     partial correlation given the single-perturbation magnitude ||delta_x||.
Across runs (--wd, subsample s0):
 (3) cross cell type: propensity (PKE fitted on all doubles) and observed hubness of the genes shared by Tian iPSC and Tian neuron;
 (4) cross assay: Norman propensity and observed transcriptional hubness versus growth hubness of the same genes in the growth GI
     map (mean |GI| over each gene's pairs), with a gene-label permutation P value and partial correlation given ||delta_x||.
Output: <run>/propensity.json and results/propensity_cross.json.
Usage: fm_propensity.py --run R  |  fm_propensity.py --wd W --gi growth.xlsx"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fm_bio import PKE
from scipy.stats import spearmanr, rankdata


class MainPKE(PKE):
    """PKE with the interaction kernels removed (main-effect kernels only)."""
    def kparts(self, P, Q):
        Ks, Kp = super().kparts(P, Q); return Ks, np.zeros_like(Ks)


def partial_rho(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    Z = np.column_stack([np.ones_like(rz), rz])
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]; ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    return float(np.corrcoef(ex, ey)[0, 1])


def load(run):
    import fmcore as F, fmpredict as FP
    Z = np.load(os.path.join(run, "pb_hvg.npz"), allow_pickle=True)
    conds = list(Z["conds"]); expr = {c: Z["pb"][i].astype(np.float64) for i, c in enumerate(conds)}; hn = np.asarray(Z["genes"])
    singles, doubles, _ = F.parse_conditions(conds); R = np.asarray(F.residual_exact(expr, "control", doubles), np.float64)
    DE = FP.de_sets(expr, "control", doubles, hn, 50); raw = {s: expr[s] - expr["control"] for s in singles}
    return singles, doubles, R, DE, raw, FP


def onehot_fit_predict(pairs_tr, Y, pairs_te, genes):
    gi = {g: i for i, g in enumerate(genes)}
    def X(ps):
        M = np.zeros((len(ps), len(genes)))
        for k, (a, b) in enumerate(ps): M[k, gi[a]] += 1; M[k, gi[b]] += 1
        return M
    Xt, Xe = X(pairs_tr), X(pairs_te); mu = Y.mean(0); Yc = Y - mu
    best = None
    for lam in [0.1, 0.3, 1, 3, 10, 30]:
        H = Xt @ np.linalg.solve(Xt.T @ Xt + lam * np.eye(Xt.shape[1]), Xt.T); res = (Yc - H @ Yc) / (1 - np.diag(H))[:, None]
        e = float((res ** 2).sum())
        if best is None or e < best[0]: best = (e, lam)
    W = np.linalg.solve(Xt.T @ Xt + best[1] * np.eye(Xt.shape[1]), Xt.T @ Yc); return mu + Xe @ W


def per_run(run):
    singles, doubles, R, DE, raw, FP = load(run); pairs = [(x, y) for _, x, y in doubles]; n = len(pairs)
    out = {}
    acc = {m: np.zeros(2) for m in ("intercept", "gene_onehot", "pke_main", "pke")}
    for sd in (0, 1, 2):
        for tr, te in FP.splits(doubles, "random", 5, sd):
            ptr, pte = [pairs[i] for i in tr], [pairs[i] for i in te]
            m = PKE(raw, singles).fit(ptr, R[tr]); mm = MainPKE(raw, singles).fit(ptr, R[tr])
            P = {"intercept": np.repeat(R[tr].mean(0)[None], len(te), 0), "gene_onehot": onehot_fit_predict(ptr, R[tr], pte, singles),
                 "pke_main": mm.predict(pte), "pke": m.predict(pte)}
            for k, v in P.items():
                for j, i in enumerate(te):
                    e = R[i, DE[i]] - v[j, DE[i]]; acc[k] += (e @ e, ((R[i, DE[i]] - P["intercept"][j, DE[i]]) ** 2).sum())
    out["gene_level_r2c"] = {k: float(1 - v[0] / acc["intercept"][0]) for k, v in acc.items()}
    # (2) split-half propensity
    mag = np.array([np.linalg.norm(raw[s]) for s in singles]); rng = np.random.default_rng(0); rows = []
    rn = np.array([np.linalg.norm(R[i, DE[i]] - R[:, DE[i]].mean(0)) for i in range(n)])
    for b in range(50):
        perm = rng.permutation(n); A, B = perm[: n // 2], perm[n // 2:]
        props = []
        for half in (A, B):
            m = PKE(raw, singles).fit([pairs[i] for i in half], R[half]); props.append(np.linalg.norm(m.propensity(singles), axis=1))
        rho_half = spearmanr(props[0], props[1])[0]
        # hubness observed in half B for genes with >= 2 doubles there, predicted by propensity from half A
        hub = {}
        for i in B:
            for g in pairs[i]: hub.setdefault(g, []).append(rn[i])
        gs = [g for g in singles if len(hub.get(g, [])) >= 2]; ix = [singles.index(g) for g in gs]
        if len(gs) >= 8:
            hb = np.array([np.mean(hub[g]) for g in gs])
            rows.append((rho_half, spearmanr(props[0][ix], hb)[0], partial_rho(props[0][ix], hb, mag[ix]), spearmanr(mag[ix], hb)[0]))
        else:
            rows.append((rho_half, np.nan, np.nan, np.nan))
    Rw = np.array(rows, float)
    out["split_half"] = dict(rho_propensity_halves=float(np.nanmedian(Rw[:, 0])), rho_prop_vs_heldout_hub=float(np.nanmedian(Rw[:, 1])),
                             partial_given_magnitude=float(np.nanmedian(Rw[:, 2])), rho_magnitude_vs_heldout_hub=float(np.nanmedian(Rw[:, 3])),
                             frac_prop_hub_positive=float(np.nanmean(Rw[:, 1] > 0)))
    json.dump(out, open(os.path.join(run, "propensity.json"), "w"), indent=1); print(os.path.basename(run.rstrip("/")), json.dumps(out), flush=True)


def full_props(run):
    singles, doubles, R, DE, raw, FP = load(run); pairs = [(x, y) for _, x, y in doubles]
    m = PKE(raw, singles).fit(pairs, R); prop = dict(zip(singles, np.linalg.norm(m.propensity(singles), axis=1)))
    rn = [np.linalg.norm(R[i, DE[i]] - R[:, DE[i]].mean(0)) for i in range(len(pairs))]; hub = {}
    for i, (a, b) in enumerate(pairs):
        for g in (a, b): hub.setdefault(g, []).append(rn[i])
    return prop, {g: (float(np.mean(v)), len(v)) for g, v in hub.items()}, {s: float(np.linalg.norm(raw[s])) for s in singles}


def cross(wd, gi_path):
    R = os.path.join(wd, "results"); out = {}
    pi, hi, mi = full_props(os.path.join(R, "TianIPSC_s0_mc500")); pn, hn_, mn = full_props(os.path.join(R, "TianNeuron_s0_mc500"))
    sh = sorted(set(pi) & set(pn)); out["tian_shared_genes"] = len(sh)
    out["tian_propensity_rho"] = float(spearmanr([pi[g] for g in sh], [pn[g] for g in sh])[0])
    out["tian_magnitude_rho"] = float(spearmanr([mi[g] for g in sh], [mn[g] for g in sh])[0])
    out["tian_propensity_partial_given_magnitudes"] = partial_rho(np.array([pi[g] for g in sh]), np.array([pn[g] for g in sh]),
                                                                  np.array([mi[g] * mn[g] for g in sh]))
    hs = [g for g in sh if g in hi and g in hn_ and hi[g][1] >= 3 and hn_[g][1] >= 3]; out["tian_hub_genes"] = len(hs)
    out["tian_observed_hub_rho"] = float(spearmanr([hi[g][0] for g in hs], [hn_[g][0] for g in hs])[0]) if len(hs) > 5 else None
    # cross assay, Norman vs growth hubness
    import openpyxl
    ws = openpyxl.load_workbook(gi_path, read_only=True).worksheets[0]; GI = {}
    for i, r in enumerate(ws.iter_rows(values_only=True)):
        if i < 2 or r[0] is None or r[1] is None or r[0] == r[1] or r[4] is None: continue
        GI[frozenset((str(r[0]).upper(), str(r[1]).upper()))] = abs(float(r[4]))
    val = "replicate-averaged GI score"
    genes = sorted({g for k in GI for g in k}); ghub = {g: np.mean([v for k, v in GI.items() if g in k]) for g in genes}
    pN, hN, mN = full_props(os.path.join(R, "Norman_s0_mc500")); up = {g.upper(): g for g in pN}
    ov = [g for g in genes if g in up]; x = np.array([pN[up[g]] for g in ov]); y = np.array([ghub[g] for g in ov]); z = np.array([mN[up[g]] for g in ov])
    rho = spearmanr(x, y)[0]; rng = np.random.default_rng(0); null = [spearmanr(x, rng.permutation(y))[0] for _ in range(5000)]
    out["norman_growth"] = dict(n_genes=len(ov), value_column=str(val), rho_propensity_growth_hub=float(rho),
                                p_perm=float((np.sum(np.abs(null) >= abs(rho)) + 1) / 5001), partial_given_magnitude=partial_rho(x, y, z),
                                rho_magnitude_growth_hub=float(spearmanr(z, y)[0]))
    oh = [g for g in ov if up[g] in hN and hN[up[g]][1] >= 2]
    out["norman_growth"]["rho_observed_hub_growth_hub"] = float(spearmanr([hN[up[g]][0] for g in oh], [ghub[g] for g in oh])[0])
    out["norman_growth"]["n_observed_hub_genes"] = len(oh)
    top = sorted(ov, key=lambda g: -pN[up[g]])[:10]; out["norman_growth"]["top10_propensity_growth_hub_rank"] = {g: int(sorted(genes, key=lambda h: -ghub[h]).index(g)) + 1 for g in top}
    json.dump(out, open(os.path.join(R, "propensity_cross.json"), "w"), indent=1); print("CROSS", json.dumps(out), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--run"); p.add_argument("--wd"); p.add_argument("--gi"); a = p.parse_args()
    if a.run: per_run(a.run)
    else: cross(a.wd, a.gi)
    print("PROP_DONE", flush=True)
