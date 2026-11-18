"""Sanity check of the deep learning baselines on Norman (subsamples s0 and s1, random split), pooled over folds.
For GEARS: R2 of the residual implied by the predicted double (observed and own singles), R2 of the predicted double shift
x_ab - x_0 against the additive shift, and the median ratio of predicted to observed residual norms. For GEARS, GPerturb and CPA:
the metric of the GEARS study, i.e. the Pearson correlation between predicted and observed change from control over the 20 most
changed genes of each double, and the mean squared error on these genes, each next to the additive expectation.
Reads gears_acc_random_s0_f*.npz, gears_pd20_random_s0_f*.npy and dl_<model>_random_f*_pd20.npy.
Output: results/sanity_norman.json.  Usage: fm_sanity_summary.py W"""
import glob, json, sys, numpy as np
W = sys.argv[1]; R = f"{W}/results"; out = {}
def pd(files):
    P = np.concatenate([np.load(f) for f in files], 1); P = P[:, ~np.isnan(P[0])]
    return dict(n=int(P.shape[1]), pearson_delta20=float(np.mean(P[0])), pearson_delta20_additive=float(np.mean(P[1])),
                mse20=float(np.mean(P[2])), mse20_additive=float(np.mean(P[3])), frac_better_than_additive=float(np.mean(P[0] > P[1])))
gf = [f for s in (0, 1) for f in sorted(glob.glob(f"{R}/Norman_s{s}_mc500/gears_acc_random_s0_f[0-4].npz"))]
if gf:
    Z = [np.load(f) for f in gf]; S = {k: sum(z[k] for z in Z) for k in Z[0].files if k != "nfall"}
    r2 = lambda m: float(1 - S[m + "_err_mod"].sum() / S[m + "_err_add"].sum())
    nr = S["normr"]; ok = nr[:, 1] > 0
    out["gears"] = dict(files=len(gf), r2_resid=r2("gears"), r2_resid_own=r2("gears_gi"), r2_shift=r2("gears_shift"), r2_shift_additive=r2("additive_shift"),
                        median_resid_norm_ratio=float(np.median(nr[ok, 0] / nr[ok, 1])), nfall=int(sum(int(z["nfall"]) for z in Z)),
                        **pd([f for s in (0, 1) for f in sorted(glob.glob(f"{R}/Norman_s{s}_mc500/gears_pd20_random_s0_f[0-4].npy"))]))
for m in ("cpa", "gperturb"):
    fs = [f for s in (0, 1) for f in sorted(glob.glob(f"{R}/Norman_s{s}_mc500/dl_{m}_random_f[0-4]_pd20.npy"))]
    if fs: out[m] = dict(files=len(fs), **pd(fs))
json.dump(out, open(f"{R}/sanity_norman.json", "w"), indent=1)
for k, v in out.items(): print(k, json.dumps({a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items()}))
print("SANITY_SUMMARY_DONE")
