"""Summarize fm_design.py: per dataset, curves pooled over subsamples and repetitions (units). For each rule: mean R2 on the fixed
test set by number of measured doubles, the number of doubles needed to reach 80% and 90% of the accuracy of the full candidate
pool (linear interpolation of the mean curve), the saving relative to random selection, and the mean R2 difference to random over
the design path (area between curves), with 95% CIs from 2,000 bootstrap resamples of units."""
import glob, json, os, sys
import numpy as np
W = sys.argv[1] if len(sys.argv) > 1 else "."; out = {}; rng = np.random.default_rng(0)


def needed(ns, mc, target):
    for k in range(len(ns)):
        if mc[k] >= target:
            if k == 0: return float(ns[0])
            return float(ns[k - 1] + (target - mc[k - 1]) / (mc[k] - mc[k - 1]) * (ns[k] - ns[k - 1]))
    return float("nan")


for ds in ["Norman", "Wessels", "TianIPSC", "TianNeuron", "Joung"]:
    fs = sorted(glob.glob(f"{W}/results/{ds}_s*_mc500/design.json"))
    if not fs: continue
    curves, full = {}, []
    for f in fs:
        d = json.load(open(f)); full += d["full"]
        for r, cs in d["curves"].items():
            for c in cs: curves.setdefault(r, []).append(c)
    L = min(len(c) for cs in curves.values() for c in cs); ns = np.array([p[0] for p in curves["random"][0][:L]])
    Y = {r: np.array([[p[1] for p in c[:L]] for c in cs]) for r, cs in curves.items()}; F = np.array(full); U = len(F)
    res = dict(n_units=U, n_measured=ns.tolist(), full_r2=float(F.mean()))
    B = rng.integers(0, U, size=(2000, U))
    for frac in (0.8, 0.9):
        tgt = frac * F.mean(); base = needed(ns, Y["random"].mean(0), tgt)
        for r in Y:
            nr = needed(ns, Y[r].mean(0), tgt)
            bs = [needed(ns, Y["random"][b].mean(0), frac * F[b].mean()) - needed(ns, Y[r][b].mean(0), frac * F[b].mean()) for b in B]
            bs = np.array(bs); bs = bs[np.isfinite(bs)]
            res.setdefault(r, {})[f"n_{int(frac*100)}"] = nr
            res[r][f"saving_{int(frac*100)}"] = base - nr
            res[r][f"saving_{int(frac*100)}_ci"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))] if len(bs) > 100 else None
    for r in Y:
        d = (Y[r] - Y["random"]).mean(1); res[r]["mean_curve"] = Y[r].mean(0).tolist()
        res[r]["area_vs_random"] = float(d.mean()); bd = np.array([d[b].mean() for b in B])
        res[r]["area_ci"] = [float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))]
    out[ds] = res
    print(ds, f"units={U} full={F.mean():.3f}", " | ".join(f"{r}: N90={res[r]['n_90']:.1f} save90={res[r]['saving_90']:+.1f} {res[r]['saving_90_ci']} area={res[r]['area_vs_random']:+.3f} {np.round(res[r]['area_ci'],3).tolist()}" for r in Y), flush=True)
json.dump(out, open(f"{W}/results/design_summary.json", "w"), indent=1); print("DESIGN_SUMMARY_DONE")
