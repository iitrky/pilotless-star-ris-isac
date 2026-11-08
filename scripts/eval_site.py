"""Evaluate controllers on held-out episodes of the deterministic site (Table 9, Table 10, Figs. 9-10).

Usage: python scripts/eval_site.py --run runs/hatd3_s0 [--site B] [--K 10] [--gamma 1e-3]
       python scripts/eval_site.py --baseline model|lookahead|genie|rt|bestpath
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import scripts.train_site as ts
from hatd3.site import SITE_A, SITE_B
from hatd3.site_control import epoch, summarise, _gw, with_own
from hatd3.controllers_td import _steer
from hatd3.agent_np import TD3, featurize

N_EP = 12


def load_agent(run):
    meta = json.load(open(os.path.join(run, "train.json")))
    ar = meta["args"]
    td3 = ar["algo"] == "td3"
    ag = TD3(ar["encoder"], meta["F"], meta["G"], meta["E"], meta["K"], seed=0, twin=td3, level2=not ar["no_level2"])
    ag.actor.load(list(np.load(os.path.join(run, "actor.npy"), allow_pickle=True)))
    ag.critics[0].load(list(np.load(os.path.join(run, "critic.npy"), allow_pickle=True)))
    return ag, argparse.Namespace(**ar), np.array(meta["lam"])


def episodes(site, K, gam):
    for i in range(N_EP):
        K_r = [2, 3, 4][i % 3] if K == 6 else K // 2
        yield ts.make_env(900_000 + i, K_r, site=site, v=(120, 120), K=K)


def eval_baseline(kind, site, K, gam, coupled=False):
    out = []
    for env in episodes(site, K, gam):
        env.cfg.coupled_phase = coupled
        infos = []
        for t in range(ts.EPOCHS):
            if kind == "bestpath":                       # oracle: steer at the strongest true path
                import hatd3.site_control as sc
                orig = sc.genie_own
                def bp(e):
                    gw = _gw(e)
                    return [_steer(e, [max(e.sch.paths(e.p_veh[k])[0], key=lambda q: abs(q[1]))[0]], gw) for k in range(e.cfg.K)]
                sc.genie_own = bp
                infos.append(epoch(env, "genie", gam))
                sc.genie_own = orig
            else:
                infos.append(epoch(env, kind if kind != "lookahead" else "model", gam,
                                   mode="lookahead" if kind == "lookahead" else "myopic"))
        out.append(summarise(infos, gam))
    return out


def eval_agent(run, site, K, gam, log=None):
    ag, a, lam = load_agent(run)
    ag.K, ag.E = K, K + 2
    ag.actor.K = ag.actor.enc.K = K
    for c in ag.critics:
        c.K = c.enc.K = K
    ag.isveh = np.r_[np.ones(K, bool), np.zeros(2, bool)]
    ts.GAM = gam
    out = []
    for env in episodes(site, K, gam):
        env.cfg.coupled_phase = a.coupled
        infos, _ = ts.run_agent_episode(env, ag, a, lam, log=log)
        out.append(summarise(infos, gam))
    ts.GAM = 1e-3
    return out


def agg(rs):
    keys = rs[0].keys()
    return {k: float(np.mean([r[k] for r in rs])) for k in keys}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run"); ap.add_argument("--baseline")
    ap.add_argument("--site", default="A"); ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--gamma", type=float, default=1e-3)
    a = ap.parse_args()
    site = SITE_A if a.site == "A" else SITE_B
    rs = eval_baseline(a.baseline, site, a.K, a.gamma) if a.baseline else eval_agent(a.run, site, a.K, a.gamma)
    print(json.dumps(agg(rs)))
