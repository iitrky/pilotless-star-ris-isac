"""Pilot-free geometric steering vs perfect-CSI co-phasing (paper Sec. 5.2, measured eta).
Writes results/anchor_check.json."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hatd3.config import SysConfig
from hatd3.env_belief import BeliefEnv

cfg = SysConfig(Nx=32, Nz=8, K_r=1, K_t=1, L_r=1, L_t=1)
out = {}
for warm in (0, 5, 20):
    ga, gr = [], []
    for s in range(30):
        env = BeliefEnv(cfg, seed=s)
        env.reset()
        _, _, vh = np.linalg.svd(env.G)
        w = vh[0].conj()
        W = np.tile(w[:, None], (1, cfg.K + cfg.L)) * np.sqrt(env.P / (cfg.K + cfg.L))
        for t in range(warm):
            env.step_belief(W, np.full(cfg.N, 0.5), np.zeros(cfg.N), np.zeros(cfg.N))
        th_r, _ = env.geometric_phases()
        gw = env.G @ w
        opt = np.sum(np.abs(env.g[0] * gw)) ** 2                       # perfect-CSI co-phasing
        ga.append(np.abs(np.sum(env.g[0] * np.exp(1j * th_r) * gw)) ** 2 / opt)
        gr.append(np.abs(np.sum(env.g[0] * np.exp(1j * np.random.default_rng(s).uniform(0, 2 * np.pi, cfg.N)) * gw)) ** 2 / opt)
    out[f"after_{warm}_epochs"] = {"anchor_over_csi_median": float(np.median(ga)), "anchor_p10": float(np.quantile(ga, 0.1)),
                                   "random_over_csi_median": float(np.median(gr))}
os.makedirs("results", exist_ok=True)
json.dump(out, open("results/anchor_check.json", "w"), indent=2)
print(json.dumps(out, indent=2))
