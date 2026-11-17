"""Shared evaluation helpers: pair features for ridge, per-double evaluation genes, and cross-validation splits."""
import numpy as np
from sklearn.model_selection import KFold


def pair_features(fa, fb):
    """Symmetric pair features used by the ridge baseline."""
    return np.concatenate([fa + fb, fa * fb, np.abs(fa - fb)])


def de_sets(expr, ctrl, doubles, hvg_names, k=50):
    """Per double: indices of the k genes with the largest |x_ab - x_0|, excluding the two targeted genes."""
    pos = {g: i for i, g in enumerate(hvg_names)}
    out = []
    for d, a, b in doubles:
        s = np.abs(expr[d] - expr[ctrl]).astype(float)
        for g in (a, b):
            if g in pos:
                s[pos[g]] = -np.inf
        out.append(np.argpartition(-s, k)[:k])
    return out


def splits(doubles, scheme, K=5, seed=0):
    """'random': K-fold over doubles. 'genegrp': genes in K groups; fold f tests every double touching group f
    and trains on doubles with no gene in group f."""
    n = len(doubles)
    rng = np.random.default_rng(seed)
    if scheme == "random":
        for tr, te in KFold(K, shuffle=True, random_state=seed).split(np.arange(n)):
            yield tr, te
    else:
        genes = sorted({g for _, a, b in doubles for g in (a, b)})
        grp = dict(zip(genes, rng.permutation(np.arange(len(genes)) % K)))
        for f in range(K):
            te = np.array([i for i, (_, a, b) in enumerate(doubles) if grp[a] == f or grp[b] == f])
            tr = np.array([i for i, (_, a, b) in enumerate(doubles) if grp[a] != f and grp[b] != f])
            if len(te) and len(tr) > 10:
                yield tr, te
