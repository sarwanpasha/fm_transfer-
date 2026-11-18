"""Gene-block bootstrap for dataset-level confidence intervals. Doubles that share a gene are not independent, so instead of
resampling doubles, the perturbed genes are resampled with replacement and every double (a, b) enters with weight c_a * c_b,
where c_g is the number of times gene g was drawn. Squared errors are pooled over subsamples and fold seeds as in
fm_rev_summary.py. Reported for the standard benchmark (rev_pd.npz) and for the disjoint-cell reanalysis (results/disjoint).
Output: results/geneblock.json.  Usage: fm_geneblock.py W [--B 2000]"""
import glob, json, os, sys
import numpy as np

W = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "."
B = int(sys.argv[sys.argv.index("--B") + 1]) if "--B" in sys.argv else 2000
R = os.path.join(W, "results"); out = {}; lines = []


def boot(pairs, E, ref, den, B, seed=0):
    """E, ref: per-double errors of the method and the reference; den: per-double denominators. Returns point estimate of
    R2(E) - R2(ref) and percentile CIs under the double bootstrap and the gene-block bootstrap."""
    rng = np.random.default_rng(seed); genes = sorted({g for p in pairs for g in p}); gi = {g: i for i, g in enumerate(genes)}
    A = np.array([gi[a] for a, _ in pairs]); Bb = np.array([gi[b] for _, b in pairs])
    r2 = lambda e, w: 1 - (w * e).sum(-1) / (w * den).sum(-1)
    pt = float(r2(E, 1.0) - r2(ref, 1.0))
    ii = rng.choice(len(pairs), size=(B, len(pairs))); wd = np.zeros((B, len(pairs)))
    for b in range(B): wd[b] = np.bincount(ii[b], minlength=len(pairs))
    c = np.stack([np.bincount(rng.choice(len(genes), len(genes)), minlength=len(genes)) for _ in range(B)])
    wg = c[:, A] * c[:, Bb]
    dd, dg = r2(E, wd) - r2(ref, wd), r2(E, wg) - r2(ref, wg); ok = (wg * den).sum(-1) > 0
    return dict(d=pt, lo_double=float(np.percentile(dd, 2.5)), hi_double=float(np.percentile(dd, 97.5)),
                lo_gene=float(np.percentile(dg[ok], 2.5)), hi_gene=float(np.percentile(dg[ok], 97.5)), n_doubles=len(pairs), n_genes=len(genes))


# standard benchmark: rev_pd.npz (per double: err, den0, denc)
for ds in ["Norman", "Wessels", "TianIPSC", "TianNeuron", "Joung", "Sciplex"]:
    files = sorted(glob.glob(f"{R}/{ds}_s*_mc500/rev_pd.npz"))
    if not files: continue
    for scheme in ("random", "genegrp"):
        for gs in ("de_obs", "singles50", "hvg"):
            for cat in (("all", "u1", "u2") if scheme == "genegrp" else ("all",)):
                acc = {}
                for f in files:
                    Z = np.load(f, allow_pickle=True); dbl = list(Z["doubles"])
                    for m in ("pke", "pke_fanova", "ridge_raw"):
                        k = f"{scheme}__{gs}__{cat}__{m}"
                        if k not in Z.files: continue
                        for j, d in enumerate(dbl):
                            acc.setdefault(m, {}).setdefault(d, np.zeros(3)); acc[m][d] += Z[k][:, j]
                if "ridge_raw" not in acc or "pke" not in acc: continue
                dbl = sorted(d for d in acc["ridge_raw"] if acc["ridge_raw"][d][1] > 0)
                pairs = []
                for d in dbl:
                    parts = [g for g in str(d).split("_") if g]
                    pairs.append((parts[0], parts[1]) if len(parts) == 2 else (str(d), str(d)))
                den = np.array([acc["ridge_raw"][d][1] for d in dbl]); ref = np.array([acc["ridge_raw"][d][0] for d in dbl])
                res = {m: boot(pairs, np.array([acc[m][d][0] for d in dbl]), ref, den, B) for m in ("pke", "pke_fanova") if m in acc}
                out[f"standard|{ds}|{scheme}|{gs}|{cat}"] = res; r = res["pke"]
                lines.append(f"STD  {ds:10s} {scheme:8s} {gs:9s} {cat:3s} n={r['n_doubles']:4d} genes={r['n_genes']:3d} PKE-ridge={r['d']:+.3f} "
                             f"double[{r['lo_double']:+.3f},{r['hi_double']:+.3f}] gene[{r['lo_gene']:+.3f},{r['hi_gene']:+.3f}]")

# stronger baselines (strong_pd.npz): PKE minus each baseline
for ds in ["Norman", "Wessels", "TianIPSC", "TianNeuron", "Joung", "Sciplex"]:
    files = sorted(glob.glob(f"{R}/{ds}_s*_mc500/strong_pd.npz"))
    if not files: continue
    for scheme in ("random", "genegrp"):
        for gs in ("de_obs", "singles50", "hvg"):
            acc = {}
            for f in files:
                Z = np.load(f, allow_pickle=True); dbl = list(Z["doubles"])
                for m in ("pke", "ridge_raw", "poly_krr", "gp_ml", "mkl_align"):
                    k = f"{scheme}__{gs}__all__{m}"
                    if k not in Z.files: continue
                    for j, d in enumerate(dbl):
                        acc.setdefault(m, {}).setdefault(d, np.zeros(3)); acc[m][d] += Z[k][:, j]
            if "pke" not in acc: continue
            dbl = sorted(d for d in acc["pke"] if acc["pke"][d][1] > 0); pairs = []
            for d in dbl:
                parts = [g for g in str(d).split("_") if g]; pairs.append((parts[0], parts[1]) if len(parts) == 2 else (str(d), str(d)))
            den = np.array([acc["pke"][d][1] for d in dbl]); E = np.array([acc["pke"][d][0] for d in dbl])
            res = {m: boot(pairs, E, np.array([acc[m][d][0] for d in dbl]), den, B) for m in acc if m != "pke"}
            for m in res: res[m]["r2_baseline"] = float(1 - np.array([acc[m][d][0] for d in dbl]).sum() / den.sum())
            res["pke_r2"] = float(1 - E.sum() / den.sum()); out[f"strong|{ds}|{scheme}|{gs}"] = res
            lines.append(f"STR  {ds:10s} {scheme:8s} {gs:9s} PKE={res['pke_r2']:+.3f} " + " ".join(
                f"{m}={res[m]['r2_baseline']:+.3f}(PKE-{m}={res[m]['d']:+.3f} gene[{res[m]['lo_gene']:+.3f},{res[m]['hi_gene']:+.3f}])" for m in res if m != "pke_r2"))

# disjoint-cell reanalysis
for ds in ["Norman", "Wessels", "TianIPSC", "TianNeuron"]:
    for suffix in ("", "_in10"):
        files = sorted(f for f in glob.glob(f"{R}/disjoint/{ds}_s[0-9]{suffix}.npz"))
        if not files: continue
        Z0 = np.load(files[0], allow_pickle=True); pairs = [tuple(p) for p in Z0["pairs"]]; meth = list(Z0["methods"]); gsets = list(Z0["gsets"])
        for scheme in ("random", "genegrp"):
            err = sum(np.load(f)[f"d_err_{scheme}"] for f in files); den = sum(np.load(f)[f"d_den0_{scheme}"] for f in files)
            for gi, gs in enumerate(gsets):
                ok = den[gi] > 0; pp = [p for p, o in zip(pairs, ok) if o]
                res = {m: boot(pp, err[meth.index(m), gi][ok], err[meth.index("ridge"), gi][ok], den[gi][ok], B) for m in ("pke", "pke_main", "onehot")}
                out[f"disjoint{suffix}|{ds}|{scheme}|{gs}"] = res; r = res["pke"]
                lines.append(f"DSJ{suffix:4s} {ds:10s} {scheme:8s} {gs:7s} n={r['n_doubles']:4d} PKE-ridge={r['d']:+.3f} "
                             f"double[{r['lo_double']:+.3f},{r['hi_double']:+.3f}] gene[{r['lo_gene']:+.3f},{r['hi_gene']:+.3f}]")
json.dump(out, open(f"{R}/geneblock.json", "w"), indent=1)
print("\n".join(lines)); print("GENEBLOCK_DONE")
