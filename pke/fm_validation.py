"""Shortlist of untested Norman pairs for experimental validation.
For each of the 5 subsamples, PKE and PKE-ANOVA are fitted to all measured doubles, and every untested pair of single genes is scored
on its 50 genes most changed by either single perturbation (no double needed):
 strength      ||f(u) - ybar||                         pair_share  PKE-ANOVA pair-specific share <f_int, D>/||D||^2
 pair_norm     ||f_int(u)|| under PKE-ANOVA              rel_var     posterior variance / prior variance K(u,u)
 magnitude     predicted Norman magnitude (synergy > 1, buffering < 1) from delta_a + delta_b + f(u)
 up / down     genes with the largest positive / negative predicted residual
Scores are averaged over subsamples; top50_frac is the fraction of subsamples in which the pair is among the 50 strongest.
Groups (at most 2 pairs per gene within a group, rel_var below the median of all candidates):
 hub        largest strength with pair_share < 0.3
 specific   largest pair_norm with pair_share >= 0.4
 additive   predicted non-interacting controls: strength in the bottom 20%, both single magnitudes above the median
Growth GI scores of the independent fitness screen are attached where available.
Output: results/validation_shortlist.json.  Usage: fm_validation.py --wd W --gi growth.xlsx"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fm_bio import PKE, mode_metrics
from fm_rev import PKEF


def main():
    p = argparse.ArgumentParser(); p.add_argument("--wd", required=True); p.add_argument("--gi", required=True)
    p.add_argument("--n_hub", type=int, default=8); p.add_argument("--n_spec", type=int, default=8); p.add_argument("--n_add", type=int, default=4)
    a = p.parse_args()
    import fmcore as F
    agg = {}
    for sd in range(5):
        run = os.path.join(a.wd, "results", f"Norman_s{sd}_mc500")
        Z = np.load(os.path.join(run, "pb_hvg.npz"), allow_pickle=True)
        conds = list(Z["conds"]); expr = {c: Z["pb"][i].astype(np.float64) for i, c in enumerate(conds)}; hn = np.asarray(Z["genes"])
        singles, doubles, _ = F.parse_conditions(conds); R = np.asarray(F.residual_exact(expr, "control", doubles), np.float64)
        raw = {s: expr[s] - expr["control"] for s in singles}; pairs = [(x, y) for _, x, y in doubles]
        seen = {frozenset(pr) for pr in pairs}; pos = {g: i for i, g in enumerate(hn)}
        cand = [(x, y) for i, x in enumerate(sorted(singles)) for y in sorted(singles)[i + 1:] if frozenset((x, y)) not in seen]
        m = PKE(raw, singles).fit(pairs, R); mf = PKEF(raw, singles).fit(pairs, R)
        f = m.predict(cand); ff, fm_, fi, _ = mf.predict(cand, parts=True)
        Q = m.idx(cand); A_, B_ = Q[:, 0], Q[:, 1]
        kd = (sum(1.0 + m.G[k][A_, B_] for k in m.sum_m) + sum(0.5 * (1.0 + m.G[k][A_, B_] ** 2) for k in m.prod_m)) / m.n
        Ks_, Kp_ = m.kparts(Q, m.P); Kq = Ks_ + Kp_; Ks2, Kp2 = m.kparts(m.P, m.P)
        Hinv = np.linalg.inv(Ks2 + Kp2 + m.lam * np.eye(len(m.P))); v = np.maximum(kd - np.einsum("ij,jk,ik->i", Kq, Hinv, Kq), 0)
        mag = {s: np.linalg.norm(raw[s]) for s in singles}
        for j, (x, y) in enumerate(cand):
            s = np.maximum(np.abs(raw[x]), np.abs(raw[y])).astype(float)
            for g in (x, y):
                if g in pos: s[pos[g]] = -np.inf
            G = np.argpartition(-s, 50)[:50]
            d = f[j, G] - m.mu[G]; dd = ff[j, G] - mf.mu[G]; den = dd @ dd + 1e-12
            mm = mode_metrics(raw[x], raw[y], raw[x] + raw[y] + f[j])
            order = np.argsort(f[j, G] - m.mu[G])
            r = agg.setdefault((x, y), dict(strength=[], pair_share=[], pair_norm=[], rel_var=[], magnitude=[], top=[], up=[], down=[]))
            r["strength"].append(float(np.linalg.norm(d))); r["pair_share"].append(float(fi[j, G] @ dd / den))
            r["pair_norm"].append(float(np.linalg.norm(fi[j, G]))); r["rel_var"].append(float(v[j] / kd[j]))
            r["magnitude"].append(float(mm["magnitude"])); r["up"].append([str(hn[G[k]]) for k in order[::-1][:3]]); r["down"].append([str(hn[G[k]]) for k in order[:3]])
            r.setdefault("mag_a", mag[x]); r.setdefault("mag_b", mag[y])
        st = np.array([np.mean(agg[c]["strength"][-1:]) for c in cand]); top = set(np.argsort(-st)[:50])
        for j, c in enumerate(cand): agg[c]["top"].append(j in top)
        print(f"s{sd}: candidates={len(cand)}", flush=True)
    import openpyxl
    ws = openpyxl.load_workbook(a.gi, read_only=True).worksheets[0]; GI = {}
    for i, r in enumerate(ws.iter_rows(values_only=True)):
        if i < 2 or r[0] is None or r[1] is None or r[0] == r[1] or r[4] is None: continue
        GI[frozenset((str(r[0]).upper(), str(r[1]).upper()))] = float(r[4])
    rows = []
    for (x, y), r in agg.items():
        if len(r["strength"]) < 5: continue
        from collections import Counter
        up = [g for g, _ in Counter(sum(r["up"], [])).most_common(3)]; dn = [g for g, _ in Counter(sum(r["down"], [])).most_common(3)]
        rows.append(dict(pair=f"{x}+{y}", a=x, b=y, strength=float(np.mean(r["strength"])), strength_sd=float(np.std(r["strength"])),
                         pair_share=float(np.mean(r["pair_share"])), pair_norm=float(np.mean(r["pair_norm"])), rel_var=float(np.mean(r["rel_var"])),
                         magnitude=float(np.mean(r["magnitude"])), top50_frac=float(np.mean(r["top"])), up=up, down=dn,
                         mag_a=float(r["mag_a"]), mag_b=float(r["mag_b"]), growth_gi=GI.get(frozenset((x.upper(), y.upper())))))
    vmed = np.median([r["rel_var"] for r in rows]); smag = np.median([r["mag_a"] for r in rows] + [r["mag_b"] for r in rows])
    s20 = np.quantile([r["strength"] for r in rows], 0.2)
    def pick(pool, key, n):
        out, cnt = [], {}
        for r in sorted(pool, key=key):
            if cnt.get(r["a"], 0) >= 2 or cnt.get(r["b"], 0) >= 2: continue
            out.append(r); cnt[r["a"]] = cnt.get(r["a"], 0) + 1; cnt[r["b"]] = cnt.get(r["b"], 0) + 1
            if len(out) == n: break
        return out
    low = [r for r in rows if r["rel_var"] <= vmed]
    hub = pick([r for r in low if r["pair_share"] < 0.3], lambda r: -r["strength"], a.n_hub)
    spec = pick([r for r in low if r["pair_share"] >= 0.4], lambda r: -r["pair_norm"], a.n_spec)
    add = pick([r for r in low if r["strength"] <= s20 and r["mag_a"] > smag and r["mag_b"] > smag], lambda r: r["strength"], a.n_add)
    out = dict(n_candidates=len(rows), rel_var_median=float(vmed), hub=hub, specific=spec, additive=add)
    json.dump(out, open(os.path.join(a.wd, "results", "validation_shortlist.json"), "w"), indent=1)
    for nm in ("hub", "specific", "additive"):
        for r in out[nm]:
            print(nm, r["pair"], f"S={r['strength']:.2f}±{r['strength_sd']:.2f} share={r['pair_share']:.2f} pn={r['pair_norm']:.2f} v={r['rel_var']:.3f} mag={r['magnitude']:.2f} top={r['top50_frac']:.1f} up={','.join(r['up'])} down={','.join(r['down'])} growth={r['growth_gi']}", flush=True)
    print("VALIDATION_DONE", flush=True)


if __name__ == "__main__":
    main()
