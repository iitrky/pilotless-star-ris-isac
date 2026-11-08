"""Statistics used in Section 7: bootstrap CIs and Holm-corrected paired Wilcoxon tests."""
import numpy as np
from scipy.stats import wilcoxon


def bootstrap_ci(x, n_boot=2000, alpha=0.05, seed=0):
    x = np.asarray(x, float)
    if x.size == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    means = rng.choice(x, (n_boot, x.size), replace=True).mean(axis=1)
    return float(x.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def holm(pvals):
    names = list(pvals)
    p = np.array([pvals[n] for n in names])
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return {n: float(a) for n, a in zip(names, adj)}


def paired_tests(rows, ref, key):
    """Pair slots by (seed, ep, t) across methods; Wilcoxon signed-rank vs ref; Holm correction.
    For learned methods with several training seeds, per-slot values are averaged over seeds."""
    from collections import defaultdict
    table = defaultdict(dict)
    acc = defaultdict(list)
    for r in rows:
        acc[(r["method"], r["seed"], r["ep"], r["t"])].append(r[key])
    for (m, s, e, t), v in acc.items():
        table[m][(s, e, t)] = float(np.mean(v))
    out = {}
    for m in table:
        if m == ref:
            continue
        common = sorted(set(table[m]) & set(table[ref]))
        if len(common) < 10:
            continue
        a = np.array([table[ref][k] for k in common])
        b = np.array([table[m][k] for k in common])
        out[m] = float(wilcoxon(a, b).pvalue)
    adj = holm(out) if out else {}
    return {m: {"p_raw": out[m], "p_holm": adj[m]} for m in out}
