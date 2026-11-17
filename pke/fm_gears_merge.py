"""Merge per-fold GEARS accumulators (gears_acc_<scheme>_s<seeds>_f<k>.npz) into R2 / paired d vs ridge with bootstrap CI."""
import glob, json, os, sys
import numpy as np
for run in sys.argv[1:]:
    out = {}
    for scheme in ("random", "genegrp"):
        fs = sorted(glob.glob(os.path.join(run, f"gears_acc_{scheme}_s0_f*.npz")))
        fs = [f for f in fs if not f.endswith("_f-1.npz")]
        if not fs: continue
        Z = [np.load(f) for f in fs]; S = {k: sum(z[k] for z in Z) for k in Z[0].files}
        ii = np.flatnonzero(S["ridge_raw_cnt"] > 0); B = np.random.default_rng(0).choice(ii, size=(1000, len(ii)))
        r2 = lambda m, k: 1 - S[f"{m}_err_mod"][k].sum(-1) / S[f"{m}_err_add"][k].sum(-1)
        d = r2("gears", B) - r2("ridge_raw", B); vg, vr = float(r2("gears", ii)), float(r2("ridge_raw", ii))
        dgi = r2("gears_gi", B) - r2("ridge_raw", B); vgi = float(r2("gears_gi", ii))
        out[scheme] = dict(gears=vg, ridge_raw=vr, d=vg - vr, lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)), gears_gi=vgi, d_gi=vgi - vr, lo_gi=float(np.percentile(dgi, 2.5)), hi_gi=float(np.percentile(dgi, 97.5)), folds=len(fs), n_test=int(len(ii)), n_fallback=int(S["nfall"]))
        print(f"GM {os.path.basename(run.rstrip('/')):18s} {scheme:8s} folds={len(fs)} gears={vg:+.3f} ridge={vr:+.3f} d={vg - vr:+.3f} [{out[scheme]['lo']:+.3f},{out[scheme]['hi']:+.3f}] fallback={int(S['nfall'])} | gears_gi={vgi:+.3f} d_gi={vgi - vr:+.3f} [{out[scheme]['lo_gi']:+.3f},{out[scheme]['hi_gi']:+.3f}]")
    json.dump(out, open(os.path.join(run, "gears.json"), "w"), indent=1)
