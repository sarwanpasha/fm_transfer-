"""External validation, revised statistics (Norman s0, untested pairs vs |growth GI| of Norman et al. 2019 Table S5).
 - gene-label permutation: growth-map gene labels are shuffled among the overlapping genes, which keeps the pair structure
   (genes shared across pairs) intact; P = fraction of permutations with |rho| >= observed.
 - gene-level controls: partial Spearman given (i) the product of transcriptional single-effect magnitudes, (ii) each gene's
   growth hubness h(g) = mean |GI(g, .)| over the growth map (h(a)+h(b) and h(a)h(b)), and (iii) single-gene growth phenotypes
   when the workbook provides them.
 - scores: PKE total strength, PKE gene-level part, PKE pair-specific part, fANOVA pair-specific part, ridge, propensity.
Usage: fm_rev_ext.py --wd W --gi data/external/aax4438_tables5.xlsx"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fm_bio import PKE
from fm_rev import PKEF
from fm_gira import gene_feats, ridge_pairs
from scipy.stats import rankdata, spearmanr


def partial(x, y, Zs):
    rx, ry = rankdata(x), rankdata(y); Z = np.column_stack([np.ones(len(x))] + [rankdata(z) for z in Zs])
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]; ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    return float(np.corrcoef(ex, ey)[0, 1])


def main():
    p = argparse.ArgumentParser(); p.add_argument("--wd", required=True); p.add_argument("--gi", required=True); p.add_argument("--perm", type=int, default=2000); a = p.parse_args()
    import fmcore as F, fmpredict as FP, openpyxl
    path = a.gi if os.path.isabs(a.gi) else os.path.join(a.wd, a.gi)
    wb = openpyxl.load_workbook(path, read_only=True)
    for ws in wb.worksheets:
        rows = list(ws.iter_rows(values_only=True, max_row=4))
        print("SHEET", ws.title, "|", " || ".join(str(r)[:200] for r in rows), flush=True)
    ws = wb.worksheets[0]; GI = {}
    for i, r in enumerate(ws.iter_rows(values_only=True)):
        if i < 2 or r[0] is None or r[1] is None or r[0] == r[1] or r[4] is None: continue
        GI[(str(r[0]).upper(), str(r[1]).upper())] = float(r[4])
    gmap = sorted({g for k in GI for g in k}); key = lambda x, y: (x, y) if (x, y) in GI else (y, x)
    hub = {g: np.mean([abs(v) for k, v in GI.items() if g in k]) for g in gmap}
    # single growth phenotypes: look for a sheet with a gene column and a numeric phenotype column
    single = {}
    for ws2 in wb.worksheets[1:]:
        rr = list(ws2.iter_rows(values_only=True))
        if len(rr) > 20 and any(isinstance(c, str) and ("phenotype" in c.lower() or "gamma" in c.lower()) for row in rr[:3] for c in row if c):
            for row in rr[2:]:
                if row and isinstance(row[0], str) and len(row) > 1 and isinstance(row[1], (int, float)): single[row[0].upper()] = float(row[1])
            print("SINGLE_PHENO sheet", ws2.title, "n=", len(single), flush=True); break
    run = os.path.join(a.wd, "results", "Norman_s0_mc500"); Z = np.load(os.path.join(run, "pb_hvg.npz"), allow_pickle=True)
    conds = list(Z["conds"]); expr = {c: Z["pb"][i].astype(np.float64) for i, c in enumerate(conds)}
    singles, doubles, _ = F.parse_conditions(conds); R = np.asarray(F.residual_exact(expr, "control", doubles), np.float64)
    raw = {s: expr[s] - expr["control"] for s in singles}; pairs = [(x, y) for _, x, y in doubles]
    ov = [s for s in singles if s.upper() in hub]; seen = {frozenset((x.upper(), y.upper())) for x, y in pairs}
    S = [(x, y) for i, x in enumerate(ov) for y in ov[i + 1:] if key(x.upper(), y.upper()) in GI and frozenset((x.upper(), y.upper())) not in seen]
    m = PKE(raw, singles).fit(pairs, R); f, fm, fi, _ = m.predict(S, parts=True)
    mf = PKEF(raw, singles).fit(pairs, R); _, fmf, fif, _ = mf.predict(S, parts=True)
    f20 = gene_feats(raw, 20); Fr = np.stack([FP.pair_features(f20[x], f20[y]) for x, y in pairs]); Fq = np.stack([FP.pair_features(f20[x], f20[y]) for x, y in S])
    g = m.propensity(singles); prop = {s: float(np.linalg.norm(g[i])) for i, s in enumerate(singles)}
    sc = {"pke_strength": np.linalg.norm(f - m.mu, axis=1), "pke_gene_part": np.linalg.norm(fm, axis=1), "pke_pair_part": np.linalg.norm(fi, axis=1),
          "fanova_pair_part": np.linalg.norm(fif, axis=1), "ridge": np.linalg.norm(ridge_pairs(Fr, Fq, R) - R.mean(0), axis=1),
          "propensity": np.array([prop[x] + prop[y] for x, y in S])}
    y = np.array([abs(GI[key(x.upper(), y_.upper())]) for x, y_ in S])
    mag = np.array([np.linalg.norm(raw[x]) * np.linalg.norm(raw[y_]) for x, y_ in S])
    hs = np.array([hub[x.upper()] + hub[y_.upper()] for x, y_ in S]); hp = np.array([hub[x.upper()] * hub[y_.upper()] for x, y_ in S])
    ctrls = {"transcriptional_magnitude": [mag], "growth_hubness": [hs, hp], "both": [mag, hs, hp]}
    if single:
        sp = np.array([abs(single.get(x.upper(), np.nan)) + abs(single.get(y_.upper(), np.nan)) for x, y_ in S])
        if np.isfinite(sp).all(): ctrls["single_growth"] = [sp]; ctrls["all"] = [mag, hs, hp, sp]
    rng = np.random.default_rng(0); genes_ov = sorted({g_.upper() for pr in S for g_ in pr}); out = dict(n=len(S), n_genes=len(genes_ov), scores={})
    for k, v in sc.items():
        rho = float(spearmanr(v, y)[0]); null = []
        for b in range(a.perm):
            perm = dict(zip(genes_ov, rng.permutation(genes_ov)))
            yb = np.array([abs(GI.get(key(perm[x.upper()], perm[y_.upper()]), np.nan)) for x, y_ in S]); ok = np.isfinite(yb)
            null.append(spearmanr(v[ok], yb[ok])[0])
        null = np.array(null); pg = float((np.sum(np.abs(null) >= abs(rho)) + 1) / (a.perm + 1))
        out["scores"][k] = dict(rho=rho, p_gene_perm=pg, null_sd=float(np.std(null)), partial={c: partial(v, y, z) for c, z in ctrls.items()})
        print(f"EXT {k:17s} rho={rho:+.3f} p_gene={pg:.4f} " + " ".join(f"partial[{c}]={val:+.3f}" for c, val in out['scores'][k]['partial'].items()), flush=True)
    # partial of the pair part given the gene part (does the pair-specific part add anything beyond the gene-level part?)
    out["pair_given_gene"] = dict(pke=partial(sc["pke_pair_part"], y, [sc["pke_gene_part"]]), fanova=partial(sc["fanova_pair_part"], y, [np.linalg.norm(fmf, axis=1)]),
                                  pke_with_hub=partial(sc["pke_pair_part"], y, [sc["pke_gene_part"], hs, hp]))
    print("PAIR_GIVEN_GENE", out["pair_given_gene"], flush=True)
    json.dump(out, open(os.path.join(a.wd, "results", "rev_ext.json"), "w"), indent=1); print("REVEXT_DONE", flush=True)


if __name__ == "__main__":
    main()
