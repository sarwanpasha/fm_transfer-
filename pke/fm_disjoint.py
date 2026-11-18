"""Disjoint-cell reanalysis. The residual r_ab = x_ab - x_a - x_b + x_0 contains the sampling noise of the single profiles,
and the same noise enters every double of a gene. To remove this shared noise, the cells of every condition are split at random
into 3 disjoint parts. In each of the 3 cyclic rotations, part A gives the single-perturbation inputs (gene kernels and ridge
features), part B the residuals used for training and part C the residuals used for testing, so inputs, training targets and
test targets share no cells. The same folds are also scored with the standard pipeline (all cells for every role).

Per (dataset, split seed) the script reports
 (1) held-out accuracy (5-fold CV, random and held-out-gene splits, fold seeds 0-2) of PKE, ridge, PKE with main-effect kernels
     only, the one-hot gene-additive model and the intercept-only model, on 3 evaluation gene sets: the 50 most changed genes of
     the test double (from part C), the 50 genes most changed by either single (from part A) and all genes;
 (2) interaction propensity with disjoint cells: propensity from (A inputs, B targets) against hubness observed in part C, the
     single-perturbation magnitude from part A against the same hubness, and split-half reproducibility with B targets on one half
     of the doubles and C targets on the other;
 (3) a noise-only null for the pair-specific share of PKE-ANOVA: the model is fitted to N = (r_B - r_C)/sqrt(2), which has no GI
     signal but the same gene structure of noise, and its pair-specific variance fraction is compared with that of r_B;
 (4) for Norman with --gi, the association of propensity with growth hubness in the growth GI map, with partial correlations
     given magnitude and given the number of measured doubles per gene, and growth hubness computed without measured pairs.
Per-double squared errors are saved for pooled and gene-block bootstraps.
Output: results/disjoint/<dataset>_s<seed>[_in<k>].{json,npz}
Usage: fm_disjoint.py --wd W --dataset Norman --seeds 0 1 2 3 4 [--gi growth.xlsx] [--input-cells 10]"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fmcore as F, fmpredict as FP
from fm_bio import PKE
from fm_rev import PKEF
from fm_gira import gene_feats, ridge_pairs
from fm_propensity import MainPKE, partial_rho, onehot_fit_predict
from scipy.stats import spearmanr

FILES = {"Norman": "NormanWeissman2019_filtered.h5ad", "Wessels": "WesselsSatija2023.h5ad",
         "TianIPSC": "TianKampmann2019_iPSC.h5ad", "TianNeuron": "TianKampmann2019_day7neuron.h5ad"}
METHODS = ["pke", "ridge", "pke_main", "onehot", "intercept"]
GSETS = ["de", "singles", "all"]


def load(wd, ds, min_cells):
    import anndata as ad, scipy.sparse as sp, collections
    genes = np.asarray(np.load(os.path.join(wd, "results", f"{ds}_s0_mc500", "pb_hvg.npz"), allow_pickle=True)["genes"]).astype(str)
    A = ad.read_h5ad(os.path.join(wd, "data", FILES[ds]), backed="r")
    lab = A.obs["perturbation"].astype(str).values; cnt = collections.Counter(lab)
    singles, doubles, _ = F.parse_conditions(lab)
    singles = [s for s in singles if cnt[s] >= min_cells]; ss = set(singles)
    doubles = [(d, x, y) for d, x, y in doubles if cnt[d] >= min_cells and x in ss and y in ss]
    keep = ["control"] + singles + [d for d, _, _ in doubles]
    rows = np.flatnonzero(np.isin(lab, keep))
    X = A[rows].to_memory(); A.file.close(); X.var_names_make_unique()
    M = sp.csr_matrix(X.X if sp.issparse(X.X) else sp.csr_matrix(X.X), dtype=np.float32); M.data = np.round(M.data)
    lib = np.asarray(M.sum(1)).ravel(); col = {g: i for i, g in enumerate(X.var_names)}
    C = M[:, [col[g] for g in genes]].tocsr(); C = C.multiply(1e4 / np.maximum(lib, 1)[:, None]).tocsr(); C.data = np.log1p(C.data)
    lab = lab[rows]
    cells = {c: np.flatnonzero(lab == c) for c in keep}
    return genes, singles, doubles, C, cells


def part_means(C, cells, seed, input_cells=None):
    rng = np.random.default_rng(seed); parts = [{}, {}, {}]; small = [{}, {}, {}]; full = {}
    for c, ix in cells.items():
        p = np.array_split(rng.permutation(ix), 3)
        for k in range(3):
            parts[k][c] = np.asarray(C[p[k]].mean(0)).ravel().astype(np.float64)
            if input_cells is not None and c != "control":
                small[k][c] = np.asarray(C[p[k][:input_cells]].mean(0)).ravel().astype(np.float64)
        full[c] = np.asarray(C[ix].mean(0)).ravel().astype(np.float64)
    return parts, small, full


def gene_sets(expr_test, expr_in, doubles, genes, k=50):
    pos = {g: i for i, g in enumerate(genes)}; de = FP.de_sets(expr_test, "control", doubles, genes, k); sg = []
    for d, x, y in doubles:
        s = np.maximum(np.abs(expr_in[x] - expr_in["control"]), np.abs(expr_in[y] - expr_in["control"]))
        for g in (x, y):
            if g in pos: s[pos[g]] = -np.inf
        sg.append(np.argpartition(-s, k)[:k])
    return dict(de=de, singles=sg, all=[np.arange(len(genes))] * len(doubles))


def evaluate(raw, singles, doubles, Rtr, Rte, GS, acc, schemes, fseeds, methods):
    pairs = [(x, y) for _, x, y in doubles]
    base = PKE(raw, singles); basem = MainPKE(raw, singles)
    Fr = None
    if "ridge" in methods:
        f20 = gene_feats(raw, 20); Fr = np.stack([FP.pair_features(f20[x], f20[y]) for x, y in pairs])
    for sc in schemes:
        for fs in fseeds:
            for tr, te in FP.splits(doubles, sc, 5, fs):
                ptr, pte = [pairs[i] for i in tr], [pairs[i] for i in te]; P = {}
                mu = Rtr[tr].mean(0); P["intercept"] = np.repeat(mu[None], len(te), 0)
                if "pke" in methods:
                    m = PKE.__new__(PKE); m.__dict__.update(base.__dict__); P["pke"] = m.fit(ptr, Rtr[tr]).predict(pte)
                if "pke_main" in methods:
                    m = MainPKE.__new__(MainPKE); m.__dict__.update(basem.__dict__); P["pke_main"] = m.fit(ptr, Rtr[tr]).predict(pte)
                if "ridge" in methods: P["ridge"] = ridge_pairs(Fr[tr], Fr[te], Rtr[tr])
                if "onehot" in methods: P["onehot"] = onehot_fit_predict(ptr, Rtr[tr], pte, singles)
                for gi, gs in enumerate(GSETS):
                    for j, i in enumerate(te):
                        ix = GS[gs][i]; t = Rte[i, ix]
                        acc["den0"][sc][gi, i] += t @ t; e = t - P["intercept"][j, ix]; acc["denc"][sc][gi, i] += e @ e
                        for mi, mname in enumerate(methods):
                            e = t - P[mname][j, ix]; acc["err"][sc][mi, gi, i] += e @ e


def new_acc(nm, n, schemes):
    return dict(err={s: np.zeros((nm, len(GSETS), n)) for s in schemes}, den0={s: np.zeros((len(GSETS), n)) for s in schemes},
                denc={s: np.zeros((len(GSETS), n)) for s in schemes})


def summarize(acc, methods, schemes):
    out = {}
    for s in schemes:
        for gi, gs in enumerate(GSETS):
            d0, dc = acc["den0"][s][gi].sum(), acc["denc"][s][gi].sum()
            out[f"{s}|{gs}"] = {m: dict(r2=float(1 - acc["err"][s][mi, gi].sum() / d0), r2c=float(1 - acc["err"][s][mi, gi].sum() / dc))
                                for mi, m in enumerate(methods)}
    return out


def hubness(R, DE, pairs, genes, idx=None, min_n=2):
    idx = np.arange(len(pairs)) if idx is None else idx; hub = {}
    for i in idx:
        v = np.linalg.norm(R[i, DE[i]] - R[idx][:, DE[i]].mean(0))
        for g in pairs[i]: hub.setdefault(g, []).append(v)
    return {g: float(np.mean(v)) for g, v in hub.items() if len(v) >= min_n}


def fanova_int_frac(raw, singles, pairs, Y):
    m = PKEF(raw, singles).fit(pairs, Y); allp = [(x, y) for k, x in enumerate(singles) for y in singles[k + 1:]]
    _, fm, fi, _ = m.predict(allp, parts=True); vm, vi = float((fm ** 2).sum()), float((fi ** 2).sum())
    return vi / (vm + vi)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--wd", default=os.environ.get("FMTRANSFER_ROOT", "."))
    p.add_argument("--dataset", required=True, choices=sorted(FILES)); p.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    p.add_argument("--min-cells", type=int, default=30); p.add_argument("--gi"); p.add_argument("--input-cells", type=int)
    p.add_argument("--fseeds", type=int, nargs="*", default=[0, 1, 2]); a = p.parse_args()
    od = os.path.join(a.wd, "results", "disjoint"); os.makedirs(od, exist_ok=True)
    genes, singles, doubles, C, cells = load(a.wd, a.dataset, a.min_cells)
    pairs = [(x, y) for _, x, y in doubles]; n = len(doubles); schemes = ["random", "genegrp"]
    print(f"{a.dataset}: {len(singles)} singles, {n} doubles, {C.shape[0]} cells", flush=True)
    GI = None
    if a.gi:
        import openpyxl
        ws = openpyxl.load_workbook(a.gi, read_only=True).worksheets[0]; GI = {}
        for i, r in enumerate(ws.iter_rows(values_only=True)):
            if i < 2 or r[0] is None or r[1] is None or r[0] == r[1] or r[4] is None: continue
            GI[frozenset((str(r[0]).upper(), str(r[1]).upper()))] = abs(float(r[4]))
    for sd in a.seeds:
        tag = f"{a.dataset}_s{sd}" + (f"_in{a.input_cells}" if a.input_cells else "")
        parts, small, full = part_means(C, cells, 1000 + sd, a.input_cells)
        acc_d = new_acc(len(METHODS), n, schemes); std_m = ["pke", "ridge", "intercept"]; acc_s = new_acc(len(std_m), n, schemes)
        prop_rows, null_rows, gro_rows = [], [], []
        for r in range(3):
            A_, B_, C_ = parts[r], parts[(r + 1) % 3], parts[(r + 2) % 3]
            inp = small[r] if a.input_cells else A_
            raw = {s: inp[s] - A_["control"] for s in singles}
            RB = np.asarray(F.residual_exact(B_, "control", doubles), np.float64); RC = np.asarray(F.residual_exact(C_, "control", doubles), np.float64)
            GS = gene_sets(C_, A_, doubles, genes)
            evaluate(raw, singles, doubles, RB, RC, GS, acc_d, schemes, a.fseeds, METHODS)
            # (2) propensity with disjoint cells
            m = PKE(raw, singles).fit(pairs, RB); prop = dict(zip(singles, np.linalg.norm(m.propensity(singles), axis=1)))
            mag = {s: float(np.linalg.norm(raw[s])) for s in singles}; hubC = hubness(RC, GS["de"], pairs, singles)
            hg = sorted(hubC); x, y, z = [np.array([d[g] for g in hg]) for d in (prop, hubC, mag)]
            row = dict(rho_prop_hubC=float(spearmanr(x, y)[0]), rho_mag_hubC=float(spearmanr(z, y)[0]), partial_prop_hubC_given_mag=partial_rho(x, y, z), n_genes=len(hg))
            rng = np.random.default_rng(sd * 10 + r); rh = []
            for b in range(20):
                perm = rng.permutation(n); H1, H2 = perm[: n // 2], perm[n // 2:]
                p1 = np.linalg.norm(PKE(raw, singles).fit([pairs[i] for i in H1], RB[H1]).propensity(singles), axis=1)
                p2 = np.linalg.norm(PKE(raw, singles).fit([pairs[i] for i in H2], RC[H2]).propensity(singles), axis=1)
                h2 = hubness(RC, GS["de"], pairs, singles, H2); gg = [g for g in singles if g in h2]; ix = [singles.index(g) for g in gg]
                hv = np.array([h2[g] for g in gg]); mv = np.array([mag[g] for g in gg])
                rh.append((spearmanr(p1, p2)[0], spearmanr(p1[ix], hv)[0] if len(gg) >= 8 else np.nan,
                           partial_rho(p1[ix], hv, mv) if len(gg) >= 8 else np.nan, spearmanr(mv, hv)[0] if len(gg) >= 8 else np.nan))
            rh = np.array(rh, float)
            row.update(split_half_rho=float(np.nanmedian(rh[:, 0])), split_prop_vs_hub=float(np.nanmedian(rh[:, 1])),
                       split_partial_given_mag=float(np.nanmedian(rh[:, 2])), split_mag_vs_hub=float(np.nanmedian(rh[:, 3])))
            prop_rows.append(row)
            # (3) noise-only null for the pair-specific share
            Nz = (RB - RC) / np.sqrt(2)
            null_rows.append(dict(int_frac_signal=fanova_int_frac(raw, singles, pairs, RB), int_frac_noise=fanova_int_frac(raw, singles, pairs, Nz)))
            # (4) growth map (Norman)
            if GI is not None:
                gl = sorted({g for k in GI for g in k}); up = {g.upper(): g for g in singles}; meas = {frozenset((x.upper(), y.upper())) for x, y in pairs}
                gh = {g: np.mean([v for k, v in GI.items() if g in k]) for g in gl}
                ghx = {g: np.mean([v for k, v in GI.items() if g in k and k not in meas] or [np.nan]) for g in gl}
                ov = [g for g in gl if g in up]; nd = {g: sum(1 for x, y in pairs if g in (x.upper(), y.upper())) for g in ov}
                xs = np.array([prop[up[g]] for g in ov]); ys = np.array([gh[g] for g in ov]); zs = np.array([mag[up[g]] for g in ov])
                ns = np.array([nd[g] for g in ov], float); yx = np.array([ghx[g] for g in ov]); ok = ~np.isnan(yx)
                rho = spearmanr(xs, ys)[0]; null = [spearmanr(xs, np.random.default_rng(k).permutation(ys))[0] for k in range(2000)]
                gro_rows.append(dict(n_genes=len(ov), rho=float(rho), p_perm=float((np.sum(np.abs(null) >= abs(rho)) + 1) / 2001),
                                     partial_given_mag=partial_rho(xs, ys, zs), partial_given_ndoubles=partial_rho(xs, ys, ns),
                                     rho_ndoubles_growth=float(spearmanr(ns, ys)[0]), rho_excl_measured=float(spearmanr(xs[ok], yx[ok])[0]),
                                     rho_mag_growth=float(spearmanr(zs, ys)[0])))
            print(f"{tag} rot{r}: prop {json.dumps({k: round(v, 3) for k, v in prop_rows[-1].items()})} null {json.dumps({k: round(v, 4) for k, v in null_rows[-1].items()})}", flush=True)
        # standard pipeline on the same folds (all cells in every role)
        rawS = {s: full[s] - full["control"] for s in singles}; RS = np.asarray(F.residual_exact(full, "control", doubles), np.float64)
        evaluate(rawS, singles, doubles, RS, RS, gene_sets(full, full, doubles, genes), acc_s, schemes, a.fseeds, std_m)
        out = dict(dataset=a.dataset, seed=sd, n_doubles=n, n_singles=len(singles), input_cells=a.input_cells,
                   disjoint=summarize(acc_d, METHODS, schemes), standard=summarize(acc_s, std_m, schemes),
                   propensity=prop_rows, null_pair_share=null_rows, growth=gro_rows)
        json.dump(out, open(os.path.join(od, tag + ".json"), "w"), indent=1)
        np.savez(os.path.join(od, tag + ".npz"), pairs=np.array(pairs), methods=np.array(METHODS), std_methods=np.array(std_m), gsets=np.array(GSETS),
                 **{f"d_{k}_{s}": acc_d[k][s] for k in acc_d for s in schemes}, **{f"s_{k}_{s}": acc_s[k][s] for k in acc_s for s in schemes})
        D, S = out["disjoint"], out["standard"]
        print(f"{tag} DISJOINT random|de pke {D['random|de']['pke']['r2']:.3f} ridge {D['random|de']['ridge']['r2']:.3f} | "
              f"STANDARD pke {S['random|de']['pke']['r2']:.3f} ridge {S['random|de']['ridge']['r2']:.3f}", flush=True)
    print("DISJOINT_DONE", flush=True)


if __name__ == "__main__":
    main()
