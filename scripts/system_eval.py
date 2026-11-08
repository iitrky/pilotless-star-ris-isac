"""Full-system evaluation (statistical channel model, time-division ISAC epochs, NumPy only).

Exp A (Theorem 1): throughput vs speed - genie CSI (no pilot cost), pilot-based (genie x overhead),
                   pilot-free belief control (myopic). Both genie and pilot-free use the same scan
                   fraction rule so that only the source of the surface phases differs.
Exp B (Theorem 2): lookahead vs myopic scan fraction in the full system.
"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hatd3.config import SysConfig
from hatd3.env_belief import BeliefEnv, td_transmit
from hatd3.controllers_td import choose, genie_plan
from hatd3.tracker import pilot_overhead

FC, TSYM = 7e9, (1 / 120e3) * (1 + 1 / 14)


def run(cfg, mode, seed, gam, epochs=40):
    env = BeliefEnv(cfg, seed=seed, anchor=False, gamma_track=gam)
    env.reset()
    R = []
    for n in range(epochs):
        rho, b, comm, scan = choose(env, gam, "myopic" if mode == "genie" else mode)
        env.advance()
        if mode == "genie":
            comm, scan = genie_plan(env)
        info = td_transmit(env, rho, b, comm, scan)
        if n >= 10:
            R.append(info["sum_rate"])
    return float(np.mean(R))


def main():
    seeds, gam = int(os.environ.get("SEEDS", 4)), 1e-3
    out, t0 = {"seeds": seeds, "gamma_track": gam, "model": "statistical, Rician K=6 dB, TD-ISAC"}, time.time()
    A = {}
    for Nx in (8, 32):
        for v in (30, 60, 90, 120, 150):
            cfg = SysConfig(Nx=Nx, Nz=8, v_kmh=(v, v))
            g = [run(cfg, "genie", s, gam) for s in range(seeds)]
            m = [run(cfg, "myopic", s, gam) for s in range(seeds)]
            ov, tp = pilot_overhead(cfg.K, cfg.N, cfg.M, v / 3.6, FC, TSYM)
            A[f"N={cfg.N},v={v}"] = {"genie": float(np.mean(g)), "pilot_based": float(np.mean(g) * (1 - ov)),
                                     "pilot_free": float(np.mean(m)), "eta": float(np.mean(m) / np.mean(g)),
                                     "overhead": float(ov), "tau_p": tp}
            print(f"N={cfg.N} v={v}: {A[f'N={cfg.N},v={v}']} [{time.time()-t0:.0f}s]", flush=True)
    out["expA"] = A
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/system_eval.json", "w"), indent=2)


if __name__ == "__main__":
    main()
