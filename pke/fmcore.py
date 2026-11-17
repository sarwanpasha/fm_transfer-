"""Core helpers: condition parsing and the exact genetic-interaction residual."""
import numpy as np


def parse_conditions(labels, ctrl="control", sep="_"):
    """Split condition labels into singles and doubles.

    Doubles are returned as (label, geneA, geneB) and kept only when both genes are also assayed as singles.
    """
    labels = sorted(set(map(str, labels)))
    parts = {c: [g for g in c.split(sep) if g] for c in labels if c != ctrl}
    singles = sorted(c for c, p in parts.items() if len(p) == 1)
    sset = set(singles)
    doubles, dropped = [], []
    for c, p in sorted(parts.items()):
        if len(p) != 2:
            continue
        a, b = p
        if a in sset and b in sset:
            doubles.append((c, a, b))
        else:
            dropped.append(c)
    return singles, doubles, dropped


def residual_exact(emb, ctrl, doubles):
    """r_ab = x_ab - x_a - x_b + x_0 for every double."""
    c = emb[ctrl]
    return np.stack([emb[d] - emb[a] - emb[b] + c for d, a, b in doubles])
