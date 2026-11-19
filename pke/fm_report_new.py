"""Compact report of the revision analyses: disjoint cells with all or 10 input cells, GenePert, AttentionPert and the
per-gene GrnPFN breakdown. Usage: fm_report_new.py W [part ...] with parts in10, genepert, attnpert, pergene."""
import glob, json, os, sys
import numpy as np

W = sys.argv[1]; parts = sys.argv[2:] or ["in10", "genepert", "attnpert", "pergene"]; R = os.path.join(W, "results")
DS = ["Norman", "Wessels", "TianIPSC", "TianNeuron", "Joung", "Sciplex"]

if "in10" in parts:
    for ds in DS:
        for suf in ("", "_in10"):
            F = sorted(glob.glob(f"{R}/disjoint/{ds}_s[0-9]{suf}.json"))
            if not F: continue
            J = [json.load(open(f))["disjoint"] for f in F]
            for k in ("random|de", "genegrp|de"):
                v = {m: np.mean([j[k][m]["r2"] for j in J]) for m in ("pke", "ridge", "intercept")}
                print(f"IN10 {ds:10s} {suf or 'all':5s} {k:11s} n={len(F)} " + " ".join(f"{m}={x:+.3f}" for m, x in v.items()))

if "genepert" in parts:
    for ds in DS:
        F = sorted(glob.glob(f"{R}/{ds}_s[0-9]_mc500/genepert.json"))
        if not F: continue
        J = [json.load(open(f)) for f in F]
        for sc in ("random", "genegrp"):
            v = {m: np.mean([j[sc]["de_obs"]["all"][m]["r2"] for j in J]) for m in ("genepert", "pke", "ridge_raw", "mean_resid")}
            print(f"GP {ds:10s} {sc:8s} n={len(F)} miss={J[0]['missing_embedding'][:3]}({len(J[0]['missing_embedding'])}) " + " ".join(f"{m}={x:+.3f}" for m, x in v.items()))

if "attnpert" in parts:
    for f in sorted(glob.glob(f"{R}/*_mc500/attnpert.json")):
        J = json.load(open(f))
        for sc, v in J.items():
            print(f"AP {os.path.basename(os.path.dirname(f)):20s} {sc:8s} attnpert={v['attnpert']:+.3f} ridge={v['ridge_raw']:+.3f} d={v['d']:+.3f} [{v['lo']:+.3f},{v['hi']:+.3f}] fallback={v['n_fallback']}")

if "pergene" in parts:
    for f in sorted(glob.glob(f"{R}/grnpfn/*_pergene.json")):
        J = json.load(open(f)); s = J["summary"]
        for lab in ("targets", "other", "all"):
            v = s[lab]
            print(f"PG {os.path.basename(f)[:-13]:30s} {lab:7s} genes={v['n_genes']:3d} share={v['share_of_signal']:.3f} " + " ".join(f"{m}={v[m]:+.3f}" for m in J["methods"]))
