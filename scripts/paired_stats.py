"""Paired comparison on identical held-out episodes: HA-TD3 (each seed) and baselines vs model-based."""
import glob, json, os, sys
import numpy as np
from scipy import stats
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import scripts.eval_site as ev
from hatd3.site import SITE_A, SITE_B

out = {}
for site, nm in ((SITE_A, "A"), (SITE_B, "B")):
    base = np.array([r["throughput"] for r in ev.eval_baseline("model", site, 6, 1e-3)])
    p5b = np.array([r["p5_rate"] for r in ev.eval_baseline("model", site, 6, 1e-3)])
    for name in (("hatd3", "flat", "ddpg", "mlp") if nm == "A" else ("hatd3",)):
        d, d5 = [], []
        for p in sorted(glob.glob(f"runs/{name}_s[0-9]")):
            rs = ev.eval_agent(p, site, 6, 1e-3)
            d.append(np.array([r["throughput"] for r in rs]) - base)
            d5.append(np.array([r["p5_rate"] for r in rs]) - p5b)
        d = np.mean(d, 0); d5 = np.mean(d5, 0)                       # seed-averaged, per episode
        w = stats.wilcoxon(d)
        out[f"{name}_{nm}"] = {"mean_diff": float(d.mean()), "ci": float(stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))),
                               "rel_pct": float(100 * d.mean() / base.mean()), "wilcoxon_p": float(w.pvalue),
                               "episodes_better": int((d > 0).sum()), "n": len(d),
                               "p5_diff": float(d5.mean()), "p5_wilcoxon_p": float(stats.wilcoxon(d5).pvalue)}
        print(name, nm, out[f"{name}_{nm}"], flush=True)
json.dump(out, open("results/paired_stats.json", "w"), indent=1)
