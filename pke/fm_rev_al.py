"""Retrospective design of the next doubles to measure. Start from n0 random measured doubles, add `step` doubles per round,
choosing them by (i) random, (ii) largest PKE posterior variance (GP view of kernel ridge), (iii) largest predicted GI strength,
(iv) variance x predicted strength (upper-confidence style). After each round, PKE is refit and scored on all doubles not yet
measured (R2 on each double's top-50 DE genes, and Spearman of predicted vs observed GI strength).
Output: <run>/rev_al.json.  Usage: fm_rev_al.py --run <run> [--reps 20]"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fm_bio import PKE
from fm_rev import post_var


def main():
    p = argparse.ArgumentParser(); p.add_argument("--run", required=True); p.add_argument("--reps", type=int, default=20)
    p.add_argument("--n0", type=int, default=20); p.add_argument("--step", type=int, default=10); p.add_argument("--rounds", type=int, default=6); a = p.parse_args()
    import fmcore as F, fmpredict as FP
    from scipy.stats import spearmanr
    Z = np.load(os.path.join(a.run, "pb_hvg.npz"), allow_pickle=True)
    conds = list(Z["conds"]); expr = {c: Z["pb"][i].astype(np.float64) for i, c in enumerate(conds)}; hn = np.asarray(Z["genes"])
    singles, doubles, _ = F.parse_conditions(conds); R = np.asarray(F.residual_exact(expr, "control", doubles), np.float64)
    DE = FP.de_sets(expr, "control", doubles, hn, 50); raw = {s: expr[s] - expr["control"] for s in singles}
    pairs = [(x, y) for _, x, y in doubles]; n = len(pairs)
    tn = np.array([np.linalg.norm(R[i, DE[i]]) for i in range(n)])
    def score(m, rest):
        P = m.predict([pairs[i] for i in rest]); sse = sum(((R[i, DE[i]] - P[j, DE[i]]) ** 2).sum() for j, i in enumerate(rest))
        ss = sum((R[i, DE[i]] ** 2).sum() for i in rest); pn = [np.linalg.norm(P[j, DE[i]]) for j, i in enumerate(rest)]
        return 1 - sse / ss, float(spearmanr(pn, tn[rest])[0])
    strat = ["random", "variance", "strength", "var_x_strength"]; out = {s: [] for s in strat}
    for rep in range(a.reps):
        rng = np.random.default_rng(rep); start = list(rng.choice(n, a.n0, replace=False))
        for s in strat:
            meas = list(start); curve = []
            for r in range(a.rounds + 1):
                rest = [i for i in range(n) if i not in meas]
                m = PKE(raw, singles).fit([pairs[i] for i in meas], R[meas]); curve.append((len(meas),) + score(m, rest))
                if r == a.rounds: break
                if s == "random": pick = list(np.random.default_rng(1000 * rep + r).choice(rest, a.step, replace=False))
                else:
                    v = post_var(m, [pairs[i] for i in rest]); f = np.linalg.norm(m.predict([pairs[i] for i in rest]) - m.mu, axis=1)
                    sc = v if s == "variance" else f if s == "strength" else np.sqrt(v) * f
                    pick = [rest[k] for k in np.argsort(-sc)[:a.step]]
                meas += pick
            out[s].append(curve)
    summ = {s: [dict(n_measured=int(np.mean([c[k][0] for c in out[s]])), r2=float(np.mean([c[k][1] for c in out[s]])),
                     r2_sd=float(np.std([c[k][1] for c in out[s]])), rho=float(np.mean([c[k][2] for c in out[s]]))) for k in range(a.rounds + 1)] for s in strat}
    for s in strat: print("AL", s, " ".join(f"{d['n_measured']}:{d['r2']:+.3f}" for d in summ[s]), flush=True)
    json.dump(dict(summary=summ, raw=out), open(os.path.join(a.run, "rev_al.json"), "w"), indent=1, default=float); print("AL_DONE", flush=True)


if __name__ == "__main__":
    main()
