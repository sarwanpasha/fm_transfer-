"""Retrospective experimental design with a fixed test set. For each repetition, 30% of the doubles form a fixed test set and the
rest a candidate pool. Starting from n0 random candidates, `step` candidates are added per round by one of 6 rules, PKE is refit,
and R2 (top-50 DE genes of each double, relative to additivity) is scored on the fixed test set, so that all rules are compared
on the same pairs.
 random          uniform among the remaining candidates
 variance        batch-greedy PKE posterior variance: after each pick the variance is updated as if the pick were measured
                 (the GP posterior variance does not depend on the measured values)
 strength        largest predicted GI strength ||f(u) - ybar||
 ucb             sqrt(variance) x strength
 coverage        pairs whose genes have the fewest measured doubles (ties broken at random)
 kcenter         greedy farthest-point selection in the pair-kernel feature space
Output: <run>/design.json.  Usage: fm_design.py --run R [--reps 10] [--n0 10] [--step 5]"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fm_bio import PKE

RULES = ["random", "variance", "strength", "ucb", "coverage", "kcenter"]


def kmat(m, P, Q):
    Ks, Kp = m.kparts(P, Q); return Ks + Kp


def main():
    p = argparse.ArgumentParser(); p.add_argument("--run", required=True); p.add_argument("--reps", type=int, default=10)
    p.add_argument("--n0", type=int, default=10); p.add_argument("--step", type=int, default=5); p.add_argument("--test", type=float, default=0.3)
    a = p.parse_args()
    import fmcore as F, fmpredict as FP
    Z = np.load(os.path.join(a.run, "pb_hvg.npz"), allow_pickle=True)
    conds = list(Z["conds"]); expr = {c: Z["pb"][i].astype(np.float64) for i, c in enumerate(conds)}; hn = np.asarray(Z["genes"])
    singles, doubles, _ = F.parse_conditions(conds); R = np.asarray(F.residual_exact(expr, "control", doubles), np.float64)
    DE = FP.de_sets(expr, "control", doubles, hn, 50); raw = {s: expr[s] - expr["control"] for s in singles}
    pairs = [(x, y) for _, x, y in doubles]; n = len(pairs)
    base = PKE(raw, singles); Pall = base.idx(pairs); Kall = kmat(base, Pall, Pall)

    def fit(idx):
        m = PKE.__new__(PKE); m.__dict__.update(base.__dict__); return m.fit([pairs[i] for i in idx], R[idx])

    def r2(m, test):
        P = m.predict([pairs[i] for i in test])
        sse = sum(((R[i, DE[i]] - P[j, DE[i]]) ** 2).sum() for j, i in enumerate(test)); ss = sum((R[i, DE[i]] ** 2).sum() for i in test)
        return 1 - sse / ss

    out = {r: [] for r in RULES}; full = []
    for rep in range(a.reps):
        rng = np.random.default_rng(rep); perm = rng.permutation(n); nt = int(round(a.test * n))
        test, pool = list(perm[:nt]), list(perm[nt:]); start = list(rng.choice(pool, a.n0, replace=False))
        full.append(r2(fit(pool), test))
        for rule in RULES:
            meas = list(start); curve = []; rr = np.random.default_rng(10_000 + rep)
            while True:
                m = fit(meas); curve.append((len(meas), r2(m, test)))
                rest = [i for i in pool if i not in meas]
                if not rest: break
                k = min(a.step, len(rest))
                if rule == "random":
                    pick = list(rr.choice(rest, k, replace=False))
                elif rule in ("variance", "ucb"):
                    lam = m.lam; cur = list(meas); pick = []
                    f = np.linalg.norm(m.predict([pairs[i] for i in rest]) - m.mu, axis=1) if rule == "ucb" else np.ones(len(rest))
                    for _ in range(k):
                        Kc = Kall[np.ix_(cur, cur)]; Kq = Kall[np.ix_(rest, cur)]
                        v = np.maximum(np.diag(Kall)[rest] - np.einsum("ij,ij->i", Kq, np.linalg.solve(Kc + lam * np.eye(len(cur)), Kq.T).T), 0)
                        sc = np.sqrt(v) * f
                        for j in pick: sc[rest.index(j)] = -np.inf
                        j = rest[int(np.argmax(sc))]; pick.append(j); cur.append(j)
                elif rule == "strength":
                    f = np.linalg.norm(m.predict([pairs[i] for i in rest]) - m.mu, axis=1); pick = [rest[j] for j in np.argsort(-f)[:k]]
                elif rule == "coverage":
                    cnt = {}
                    for i in meas:
                        for g in pairs[i]: cnt[g] = cnt.get(g, 0) + 1
                    pick = []
                    for _ in range(k):
                        sc = np.array([cnt.get(pairs[i][0], 0) + cnt.get(pairs[i][1], 0) + rr.random() * 0.01 if i not in pick else np.inf for i in rest])
                        j = rest[int(np.argmin(sc))]; pick.append(j)
                        for g in pairs[j]: cnt[g] = cnt.get(g, 0) + 1
                else:  # kcenter
                    cur = list(meas); pick = []; d = np.diag(Kall)
                    for _ in range(k):
                        D = d[rest][:, None] + d[cur][None, :] - 2 * Kall[np.ix_(rest, cur)]; md = D.min(1)
                        for j in pick: md[rest.index(j)] = -np.inf
                        j = rest[int(np.argmax(md))]; pick.append(j); cur.append(j)
                meas += [int(x) for x in pick]
            out[rule].append(curve)
        print(f"rep {rep}: full={full[-1]:.3f} " + " ".join(f"{r}@30={dict(out[r][-1]).get(30, np.nan):.3f}" for r in RULES), flush=True)
    json.dump(dict(n=n, n_test=int(round(a.test * n)), n0=a.n0, step=a.step, full=full, curves=out), open(os.path.join(a.run, "design.json"), "w"), default=float)
    print("DESIGN_DONE", flush=True)


if __name__ == "__main__":
    main()
