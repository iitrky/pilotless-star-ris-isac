"""Evaluation protocol v2 (results/protocol_v2.md): large held-out test sets, paired tests with
Holm-Bonferroni correction, bootstrap effect sizes, generalisation and sensitivity analyses."""
import glob, json, os, sys, time
import numpy as np
from scipy import stats
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import scripts.train_site as ts
import scripts.eval_site as ev
import hatd3.agent_np as anp
from hatd3.config import SysConfig
from hatd3.channels import db2lin
from hatd3.site import SiteEnv, SITE_A, SITE_B, random_site
from hatd3.site_control import epoch, summarise
from hatd3.controllers_td import _steer
from hatd3.site_control import _gw

GAM = 1e-3


def make(seed, K_r, site, v=120, kd=10.0, K=6):
    cfg = SysConfig(Nx=32, Nz=8, v_kmh=(v, v), K_r=K_r, K_t=K - K_r)
    env = SiteEnv(cfg, site, seed=seed, gamma_track=GAM)
    env.sch.Kd = db2lin(kd)
    env.reset(K_r=K_r)
    return env


def testset(name, n=None, **kw):
    if name == "A":
        return [lambda i=i: make(950_000 + i, [2, 3, 4][i % 3], SITE_A, **kw) for i in range(n or 60)]
    if name == "B":
        return [lambda i=i: make(960_000 + i, [2, 3, 4][i % 3], SITE_B, **kw) for i in range(n or 60)]
    if name == "C":
        return [lambda i=i: make(970_000 + i, [2, 3, 4][i % 3], random_site(7_000_000 + i), **kw) for i in range(n or 30)]


def run_baseline(kind, eps):
    out = []
    for mk in eps:
        env = mk()
        infos = []
        for t in range(ts.EPOCHS):
            if kind == "bestpath":
                import hatd3.site_control as sc
                orig = sc.genie_own
                sc.genie_own = lambda e: [_steer(e, [max(e.sch.paths(e.p_veh[k])[0], key=lambda q: abs(q[1]))[0]], _gw(e)) for k in range(e.cfg.K)]
                infos.append(epoch(env, "genie", GAM)); sc.genie_own = orig
            else:
                infos.append(epoch(env, "model" if kind == "lookahead" else kind, GAM, mode="lookahead" if kind == "lookahead" else "myopic"))
        out.append(summarise(infos, GAM))
    return out


def run_agent(run, eps):
    ag, a, lam = ev.load_agent(run)
    out = []
    for mk in eps:
        env = mk()
        env.cfg.coupled_phase = a.coupled
        K = env.cfg.K
        ag.K, ag.E = K, K + env.cfg.L
        ag.actor.K = ag.actor.enc.K = K
        for c in ag.critics:
            c.K = c.enc.K = K
        ag.isveh = np.r_[np.ones(K, bool), np.zeros(env.cfg.L, bool)]
        infos, _ = ts.run_agent_episode(env, ag, a, lam)
        out.append(summarise(infos, GAM))
    return out


def per_ep(rs, key):
    return np.array([r[key] for r in rs])


def seed_avg(list_rs, key):
    return np.mean([per_ep(rs, key) for rs in list_rs], 0)


def boot_ci(d, n=10_000, seed=0):
    r = np.random.default_rng(seed)
    m = r.choice(d, (n, len(d))).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def compare(x, y):
    d = x - y
    p = float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    lo, hi = boot_ci(d)
    return {"mean_x": float(x.mean()), "mean_y": float(y.mean()), "diff": float(d.mean()), "ci95": [lo, hi],
            "rel_pct": float(100 * d.mean() / y.mean()), "p": p, "wins": int((d > 0).sum()), "n": int(len(d))}


def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * ps[i])); adj[i] = run
    return adj


def summ(list_rs):
    keys = list_rs[0][0].keys()
    out = {}
    for k in keys:
        sm = [np.mean(per_ep(rs, k)) for rs in list_rs]
        allv = np.concatenate([per_ep(rs, k) for rs in list_rs])
        ci = (stats.t.ppf(0.975, len(sm) - 1) * np.std(sm, ddof=1) / np.sqrt(len(sm))) if len(sm) > 1 else \
             (stats.t.ppf(0.975, len(allv) - 1) * np.std(allv, ddof=1) / np.sqrt(len(allv)))
        out[k] = [float(np.mean(sm)), float(ci)]
    return out


def main():
    t0 = time.time()
    R = {"protocol": "results/protocol_v2.md"}
    runs = {k: sorted(glob.glob(p)) for k, p in {
        "hatd3": "runs/hatd3_s[0-9]", "flat": "runs/flat_s[0-9]", "ddpg": "runs/ddpg_s[0-9]", "mlp": "runs/mlp_s[0-9]",
        "hatd3R": "runs2/hatd3R_s[0-9]", "hatd3FT": "runs2/hatd3FT_s[0-9]"}.items()}
    raw = {}
    for ts_name, methods in (("A", ["genie", "model", "lookahead", "bestpath", "rt", "mlp", "flat", "ddpg", "hatd3", "hatd3R"]),
                             ("B", ["genie", "model", "bestpath", "hatd3", "hatd3R", "hatd3FT"]),
                             ("C", ["genie", "model", "bestpath", "hatd3", "hatd3R"])):
        eps = testset(ts_name)
        raw[ts_name] = {}
        for m in methods:
            raw[ts_name][m] = [run_agent(p, eps) for p in runs[m]] if m in runs else [run_baseline(m, eps)]
            print(ts_name, m, round(np.mean(seed_avg(raw[ts_name][m], "throughput")), 3), f"[{time.time()-t0:.0f}s]", flush=True)
    R["summary"] = {s: {m: summ(v) for m, v in d.items()} for s, d in raw.items()}
    th = lambda s, m, k="throughput": seed_avg(raw[s][m], k)
    R["H1"] = compare(th("A", "hatd3"), th("A", "model"))
    fam = {"p5_A_hatd3_vs_model": (th("A", "hatd3", "p5_rate"), th("A", "model", "p5_rate")),
           "A_hatd3_vs_flat": (th("A", "hatd3"), th("A", "flat")), "A_hatd3_vs_ddpg": (th("A", "hatd3"), th("A", "ddpg")),
           "A_hatd3_vs_mlp": (th("A", "hatd3"), th("A", "mlp")), "B_hatd3R_vs_model": (th("B", "hatd3R"), th("B", "model")),
           "C_hatd3R_vs_model": (th("C", "hatd3R"), th("C", "model")), "B_hatd3FT_vs_model": (th("B", "hatd3FT"), th("B", "model")),
           "A_hatd3R_vs_model": (th("A", "hatd3R"), th("A", "model"))}
    sec = {k: compare(*v) for k, v in fam.items()}
    adj = holm(np.array([v["p"] for v in sec.values()]))
    for (k, v), a in zip(sec.items(), adj):
        v["p_holm"] = float(a)
    R["secondary"] = sec
    R["extra"] = {"B_hatd3_vs_model": compare(th("B", "hatd3"), th("B", "model")),
                  "C_hatd3_vs_model": compare(th("C", "hatd3"), th("C", "model"))}
    # sensitivity (descriptive)
    sens = {}
    for name, kw, err in (("v90", {"v": 90}, 1.0), ("v150", {"v": 150}, 1.0), ("echo3dB", {}, 3.0), ("K5dB", {"kd": 5.0}, 1.0)):
        anp.ECHO_ERR_DB = err
        eps = testset("A", 30, **kw)
        h = [run_agent(p, eps) for p in runs["hatd3"]]
        m = [run_baseline("model", eps)]
        g = [run_baseline("genie", eps)]
        sens[name] = {"hatd3": summ(h), "model": summ(m), "genie": summ(g),
                      "paired": compare(seed_avg(h, "throughput"), seed_avg(m, "throughput"))}
        print("sens", name, sens[name]["paired"]["rel_pct"], f"[{time.time()-t0:.0f}s]", flush=True)
    anp.ECHO_ERR_DB = 1.0
    R["sensitivity"] = sens
    R["train_time_s"] = {k: [json.load(open(p + "/train.json"))["train_time_s"] for p in v] for k, v in runs.items()}
    json.dump(R, open("results/eval_v2.json", "w"), indent=1)
    print("done", f"[{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
