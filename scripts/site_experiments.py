"""Evaluate all controllers on the held-out episodes and write results/site_results.json.

Run after training (see run_site_training.sh). Methods and rows follow Tables 9-10 of the paper.
"""
import glob, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import scripts.eval_site as ev
from hatd3.site import SITE_A, SITE_B
from hatd3.tracker import pilot_overhead

GAM = 1e-3
FC, TSYM = 7e9, (1 / 120e3) * (1 + 1 / 14)


def ci(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return float(x.mean()), 0.0
    from scipy import stats
    return float(x.mean()), float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def per_episode(rs, key):
    return [r[key] for r in rs]


def summarise_runs(list_of_rs):
    """Across seeds: mean of the per-seed means, CI over seeds (or over episodes if one seed)."""
    keys = list_of_rs[0][0].keys()
    out = {}
    for k in keys:
        seed_means = [np.mean(per_episode(rs, k)) for rs in list_of_rs]
        vals = seed_means if len(seed_means) > 1 else per_episode(list_of_rs[0], k)
        out[k] = ci(vals)
    return out


def count_flops():
    """Analytic multiply-accumulate count of one online decision (actor + one critic pass), x2 for FLOP."""
    from hatd3 import npnn
    import scripts.train_site as ts
    from hatd3.agent_np import featurize
    ag, a, lam = ev.load_agent(sorted(glob.glob("runs/hatd3_s[0-9]"))[0])
    env = ts.make_env(1, 3)
    tok, glob_, zone = featurize(env)
    n = [0]
    orig = npnn.T.__matmul__

    def mm(self, o):
        out = orig(self, o)
        n[0] += 2 * out.v.size * self.v.shape[-1]
        return out
    npnn.T.__matmul__ = mm
    ag.act(tok, glob_, zone)
    npnn.T.__matmul__ = orig
    params = sum(p.v.size for p in ag.actor.params()) + sum(p.v.size for p in ag.critics[0].params())
    return {"flop_per_decision": n[0], "params_actor_plus_critic": int(params)}


def main():
    res = {"gamma": GAM, "n_test_episodes": ev.N_EP, "speed_kmh": 120, "N": 256}
    t0 = time.time()
    base = {}
    for b in ("genie", "model", "lookahead", "rt", "bestpath"):
        base[b] = summarise_runs([ev.eval_baseline(b, SITE_A, 6, GAM)])
        print(b, base[b]["throughput"], f"[{time.time()-t0:.0f}s]", flush=True)
    ov, tau = pilot_overhead(6, 256, 16, 120 / 3.6, FC, TSYM)
    res["pilot_overhead_120kmh"] = {"fraction": ov, "tau_p": tau}
    res["baselines"] = base
    runs = {}
    for name in sorted({os.path.basename(p).rsplit("_s", 1)[0] for p in glob.glob("runs/*_s[0-9]")}):
        paths = sorted(glob.glob(f"runs/{name}_s[0-9]"))
        runs[name] = {"seeds": len(paths), "A": summarise_runs([ev.eval_agent(p, SITE_A, 6, GAM) for p in paths])}
        if name == "hatd3":
            runs[name]["B"] = summarise_runs([ev.eval_agent(p, SITE_B, 6, GAM) for p in paths])
            runs[name]["K10"] = summarise_runs([ev.eval_agent(p, SITE_A, 10, GAM) for p in paths])
            res["model_B"] = summarise_runs([ev.eval_baseline("model", SITE_B, 6, GAM)])
            res["genie_B"] = summarise_runs([ev.eval_baseline("genie", SITE_B, 6, GAM)])
            res["model_K10"] = summarise_runs([ev.eval_baseline("model", SITE_A, 10, GAM)])
            res["genie_K10"] = summarise_runs([ev.eval_baseline("genie", SITE_A, 10, GAM)])
            sweep = {}
            for g in (3e-4, 1e-3, 3e-3, 1e-2):
                sweep[str(g)] = {"hatd3": summarise_runs([ev.eval_agent(p, SITE_A, 6, g) for p in paths]),
                                 "model": summarise_runs([ev.eval_baseline("model", SITE_A, 6, g)]),
                                 "genie": summarise_runs([ev.eval_baseline("genie", SITE_A, 6, g)])}
            res["gamma_sweep"] = sweep
        print(name, runs[name]["A"]["throughput"], f"[{time.time()-t0:.0f}s]", flush=True)
    res["model_coupled"] = summarise_runs([ev.eval_baseline("model", SITE_A, 6, GAM, coupled=True)])
    res["learned"] = runs
    res["flops"] = count_flops()
    os.makedirs("results", exist_ok=True)
    json.dump(res, open("results/site_results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
