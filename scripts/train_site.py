"""Train HA-TD3 (or a belief-state baseline/ablation) on the deterministic site (Algorithm 2).

Example:  python scripts/train_site.py --encoder hier --seed 0 --out runs/hatd3_s0
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hatd3.config import SysConfig
from hatd3.site import SiteEnv, SITE_A, SITE_B
from hatd3.site_control import epoch, summarise
from hatd3.agent_np import TD3, Replay, featurize

GAM, EPOCHS, R_REF = 1e-3, 40, 5.0


def make_env(seed, K_r, site=SITE_A, v=(90, 150), coupled=False, K=6):
    cfg = SysConfig(Nx=32, Nz=8, v_kmh=v, coupled_phase=coupled, K_r=K_r, K_t=K - K_r)
    env = SiteEnv(cfg, site, seed=seed, gamma_track=GAM)
    env.reset(K_r=K_r)
    return env


def beta_map(b, bmin, env):
    """Anchored slow head: the learned split is a correction of up to +-0.15 around the
    model-based rule K_r / K, projected to [bmin, 1 - bmin] (Proposition 1 bound)."""
    return float(np.clip(env.cfg.K_r / env.cfg.K + 0.15 * b, bmin, 1 - bmin))


def run_agent_episode(env, agent, a, lam, train=False, rng=None, warm=False, log=None, explore=0.5):
    """One episode with the learned controller; returns infos and transitions."""
    T_h = 1 if a.no_two_timescale else 10
    bmin = 0.01 if a.no_beta_bound else 0.15
    infos, trans = [], []
    b_raw = 0.0
    for t in range(EPOCHS):
        dec = t % T_h == 0
        tok, glob, zone = featurize(env, lam, beta_map(b_raw, bmin, env), (t % T_h) / T_h)
        t_net = time.perf_counter()
        du, b = agent.act(tok, glob, zone, eps=(1.0 if warm else explore) if train else 0.0)
        t_net = time.perf_counter() - t_net
        if train:
            b = float(np.clip(b + rng.normal(0, 0.3 if warm else 0.1), -1, 1))
        if dec:
            b_raw = b
        pc0 = env.pcrb.copy()
        info = epoch(env, "agent", GAM, beta=beta_map(b_raw, bmin, env), du=du, anchor=not a.no_anchor)
        K = env.cfg.K
        pv, pt = info["pcrb_u"][:K] / GAM, info["pcrb_u"][K:] / GAM
        rv = info["rates"] * 6 / R_REF                                       # per-vehicle rate rewards (~1)
        rg = - lam[0] * (pv.mean() - 1) - lam[1] * (pt.mean() - 1)         # sensing (Lagrangian) reward
        if not a.no_shaping:
            phi = lambda p: -np.mean(np.maximum(0, np.log(np.maximum(p, 1e-12) / GAM)))
            rg += 0.5 * phi(info["pcrb_u"]) - phi(pc0)                        # potential-based, gamma = 0.5
        info["dt"] += t_net
        info["du"], info["p_veh"], info["zone_v"] = du.copy(), env.p_veh.copy(), env.zone_v.copy()
        infos.append(info)
        if log is not None:
            log.append({"x": env.p_veh[:, 0].tolist(), "y": env.p_veh[:, 1].tolist(), "du": du.tolist(),
                        "beta": beta_map(b_raw, bmin, env), "K_r": env.cfg.K_r})
        if train:
            tok2, glob2, zone2 = featurize(env, lam, beta_map(b_raw, bmin, env), ((t + 1) % T_h) / T_h)
            trans.append((tok, glob, zone, np.asarray(du), np.array([b_raw]), float(dec), rv, rg, tok2, glob2, zone2,
                          float(t == EPOCHS - 1)))
    return infos, trans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="hier", choices=["hier", "flat", "mlp"])
    ap.add_argument("--algo", default="td3", choices=["td3", "ddpg"])
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    for f in ("no_level2", "no_two_timescale", "no_anchor", "no_shaping", "no_lagrangian", "no_beta_bound", "coupled"):
        ap.add_argument("--" + f.replace("_", "-"), dest=f, action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    env0 = make_env(0, 3)
    tok, glob, zone = featurize(env0)
    F, G, E, K = tok.shape[1], glob.size, tok.shape[0], env0.cfg.K
    td3 = a.algo == "td3"
    agent = TD3(a.encoder, F, G, E, K, seed=a.seed, twin=td3, delay=2 if td3 else 1, level2=not a.no_level2)
    buf = Replay(100_000, rng)
    lam = np.zeros(2) if not a.no_lagrangian else np.ones(2)
    hist, t0, steps = [], time.time(), 0
    for ep in range(a.episodes):
        env = make_env(10_000 * (a.seed + 1) + ep, int(rng.choice([2, 3, 4])), coupled=a.coupled)
        warm = len(buf.data) < 1000
        explore = 0.5 - 0.3 * ep / a.episodes                       # epsilon over codebook entries
        infos, trans = run_agent_episode(env, agent, a, lam, train=True, rng=rng, warm=warm, explore=explore)
        losses = []
        for tr in trans:
            buf.add(tr)
            if not warm:
                for _ in range(1):
                    losses.append(agent.update(buf.sample(64)))
        s = summarise(infos, GAM)
        Kk = env.cfg.K
        pcv = np.mean([i["pcrb_u"][:Kk] for i in infos[5:]]) / GAM
        pct = np.mean([i["pcrb_u"][Kk:] for i in infos[5:]]) / GAM
        if not a.no_lagrangian:
            lam = np.clip(lam + 0.5 * (np.array([pcv, pct]) - 1), 0, 10)
        hist.append({"ep": ep, "throughput": s["throughput"], "viol": s["viol"], "lam": lam.tolist(),
                     "closs": float(np.mean(losses)) if losses else None})
        if ep % 20 == 0 or ep == a.episodes - 1:
            print(f"ep {ep} thr {s['throughput']:.3f} viol {s['viol']:.1f} lam {lam.round(2)} "
                  f"closs {hist[-1]['closs']} [{time.time()-t0:.0f}s]", flush=True)
    np.save(os.path.join(a.out, "actor.npy"), np.array(agent.actor.state(), dtype=object), allow_pickle=True)
    np.save(os.path.join(a.out, "critic.npy"), np.array(agent.critics[0].state(), dtype=object), allow_pickle=True)
    json.dump({"args": vars(a), "hist": hist, "lam": lam.tolist(), "F": F, "G": G, "E": E, "K": K,
               "train_time_s": time.time() - t0}, open(os.path.join(a.out, "train.json"), "w"))


if __name__ == "__main__":
    main()
