"""Final comparison table: per dataset x split.
PKE (pke.json, 5 runs x 3 fold-seeds): ridge R2, main method (pke_anova_nngp) R2, d vs ridge, #runs with CI>0, GI-ranking rho (method vs ridge).
GEARS (gears.json from fm_gears_merge.py, runs s0,s1, fold-seed 0): GEARS d vs ridge and GEARS-own-interaction d_gi (mean over runs, #runs CI<0)."""
import glob, json, os, sys
import numpy as np
WD = sys.argv[1] if len(sys.argv) > 1 else "."
M = "pke_anova_nngp"; rows = []
hdr = f"{'dataset':11s} {'split':8s} | {'ridge':>6s} {'PKE':>6s} {'dPKE':>7s} {'win':>4s} {'rhoPKE':>7s} {'rhoRdg':>7s} | {'dGEARS':>7s} {'dGEARSgi':>8s} {'GEARS<0':>7s}"
lines = [hdr, "-" * len(hdr)]
for ds in ("Norman", "Wessels", "TianIPSC", "TianNeuron"):
    for sc in ("random", "genegrp"):
        P = [json.load(open(f))[sc] for f in sorted(glob.glob(os.path.join(WD, "results", f"{ds}_s*_mc500", "pke.json")))]
        G = [json.load(open(f)).get(sc) for f in sorted(glob.glob(os.path.join(WD, "results", f"{ds}_s*_mc500", "gears.json")))]
        G = [g for g in G if g]
        if not P: continue
        rr = np.mean([p["ridge_raw"]["r2"] for p in P]); pm = np.mean([p[M]["r2"] for p in P]); dm = np.mean([p[M]["d"] for p in P])
        win = sum(p[M]["lo"] > 0 for p in P); rp = np.mean([p[M]["gi_rho"] for p in P]); rr_ = np.mean([p["ridge_raw"]["gi_rho"] for p in P])
        if G:
            dg = np.mean([g["d"] for g in G]); dgi = np.mean([g["d_gi"] for g in G]); neg = sum(g["hi_gi"] < 0 for g in G)
            gs = f"{dg:+7.3f} {dgi:+8.3f} {neg}/{len(G):<5d}"
        else: gs = f"{'n/a':>7s} {'n/a':>8s} {'':7s}"
        lines.append(f"{ds:11s} {sc:8s} | {rr:6.3f} {pm:6.3f} {dm:+7.3f} {win}/{len(P):<2d} {rp:+7.2f} {rr_:+7.2f} | {gs}")
lines += ["", "PKE = pke_anova_nngp (ANOVA pair kernel, closed-form erf-NNGP + RBF ensemble, centred KRR). d = R2 minus ridge R2 on top-50 DE genes;",
          "win = runs whose paired bootstrap 95% CI of d is > 0. rho = Spearman(pred, true) interaction strength across held-out doubles.",
          "GEARS: runs s0,s1 only, fold-seed 0; dGEARS uses observed singles as additive reference; dGEARSgi uses GEARS' own single predictions;",
          "GEARS<0 = runs whose 95% CI of dGEARSgi is entirely below 0."]
out = "\n".join(lines); print(out); open(os.path.join(WD, "results", "final_table.txt"), "w").write(out + "\n")
