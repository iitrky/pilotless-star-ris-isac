"""Evaluate trained policies and baselines on held-out scenarios; write per-slot results,
summary tables with 95% bootstrap CIs, and Holm-corrected Wilcoxon tests vs HA-TD3.

Example:
  python scripts/evaluate.py --runs hatd3=runs/hatd3_s0,runs/hatd3_s1 mlp=runs/mlptd3_s0 \
      --baselines random pga_genie pga_delayed --eps 0.1 --episodes 5 --out results/main
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hatd3.config import SysConfig, AgentConfig
from hatd3.env import StarRisIsacEnv
from hatd3.baselines import random_phase_policy, pga_benchmark
from hatd3.stats import bootstrap_ci, paired_tests


def load_agent(run_dir, obs, device):
    from hatd3.agent import HATD3
    cfgj = json.load(open(f"{run_dir}/config.json"))
    scfg = SysConfig(**{k: v for k, v in cfgj["sys"].items() if k in SysConfig.__dataclass_fields__})
    acfg = AgentConfig(**{k: v for k, v in cfgj["agent"].items() if k in AgentConfig.__dataclass_fields__})
    ag = HATD3(scfg, acfg, obs["veh"].shape[1], obs["tgt"].shape[1], obs["glob"].shape[0], device)
    ag.load(f"{run_dir}/model.pt")
    return ag, scfg, acfg


def run_policy(name, policy, scfg, seeds, eps, episodes, slots, sub=1):
    rows = []
    for sd in seeds:
        env = StarRisIsacEnv(scfg, seed=10_000 + sd)          # held-out scenario seeds
        for ep in range(episodes):
            env.reset(eps=eps)
            state = {}
            for t in range(slots):
                t0 = time.perf_counter()
                W, beta, thr, tht = policy(env, t, state)
                dt = time.perf_counter() - t0
                _, info, _ = env.step(W, beta, thr, tht)
                if t % sub == 0:
                    rows.append({"method": name, "seed": sd, "ep": ep, "t": t, "sum_rate": info["sum_rate"],
                                 "crb_max_ratio": float(np.max(info["crb"] / scfg.crb_thresh)),
                                 "viol": float(np.any(info["crb"] > scfg.crb_thresh)),
                                 "sqrt_crb_deg": float(np.degrees(np.sqrt(np.mean(info["crb"])))),
                                 "time_ms": dt * 1e3})
    return rows


def agent_policy(agent, scfg, acfg):
    from hatd3.action import ActionLayout
    lay = ActionLayout(scfg)

    def pol(env, t, st):
        if t == 0:
            st["prev"] = np.zeros(lay.D, np.float32)
            st["held"] = np.zeros(lay.D_slow, np.float32)
        obs = env.obs
        g = np.concatenate([obs["glob"], st["prev"], st["held"]])
        a = agent.act(obs["veh"], obs["tgt"], g, noise=0.0)
        if t % scfg.T_h == 0 or not acfg.two_timescale:
            st["held"] = a[lay.sl_slow].astype(np.float32).copy()
        else:
            a[lay.sl_slow] = st["held"]
        st["prev"] = a.astype(np.float32)
        W, beta, thr, tht, _ = lay.to_physical(a, env.P)
        return W, beta, thr, tht
    return pol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[], help="name=dir1,dir2 (one dir per training seed)")
    ap.add_argument("--baselines", nargs="*", default=["random"])
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--slots", type=int, default=200)
    ap.add_argument("--test-seeds", type=int, default=5)
    ap.add_argument("--pga-sub", type=int, default=10, help="evaluate PGA every n-th slot (slow)")
    ap.add_argument("--out", default="results/main")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    scfg = SysConfig()
    seeds = list(range(args.test_seeds))
    rows = []
    rng = np.random.default_rng(0)

    for b in args.baselines:
        if b == "random":
            pol = lambda env, t, st: random_phase_policy(env, rng)
            rows += run_policy("Random phases", pol, scfg, seeds, args.eps, args.episodes, args.slots)
        elif b in ("pga_genie", "pga_delayed"):
            genie = b == "pga_genie"
            def pol(env, t, st, genie=genie):
                if t % args.pga_sub == 0 or "last" not in st:
                    st["last"] = pga_benchmark(env, genie=genie, seed=t)
                return st["last"]
            name = "PGA (genie CSI)" if genie else "PGA (delayed CSI)"
            rows += run_policy(name, pol, scfg, seeds, args.eps, 1, args.slots, sub=args.pga_sub)

    probe = StarRisIsacEnv(scfg, seed=0).reset(eps=0.0)
    for spec in args.runs:
        name, dirs = spec.split("=")
        for i, d in enumerate(dirs.split(",")):
            agent, s2, a2 = load_agent(d, probe, args.device)
            r = run_policy(name, agent_policy(agent, s2, a2), s2, seeds, args.eps, args.episodes, args.slots)
            for x in r:
                x["train_seed"] = i
            rows += r

    import csv
    keys = sorted({k for r in rows for k in r})
    with open(f"{args.out}/per_slot.csv", "w", newline="") as f:
        w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(rows)

    methods = sorted({r["method"] for r in rows})
    summ = {}
    for m in methods:
        R = [r for r in rows if r["method"] == m]
        summ[m] = {k: bootstrap_ci(np.array([r[k] for r in R])) for k in ("sum_rate", "viol", "sqrt_crb_deg", "time_ms")}
    ref = "HA-TD3" if "HA-TD3" in methods else None
    tests = paired_tests(rows, ref, "sum_rate") if ref else {}
    json.dump({"summary": summ, "wilcoxon_holm_vs_HA-TD3": tests, "eps": args.eps}, open(f"{args.out}/summary.json", "w"), indent=2)
    with open(f"{args.out}/table.tex", "w") as f:
        f.write("Method & Sum rate (bit/s/Hz) & Violation (\\%) & $\\sqrt{\\mathrm{CRB}}$ (deg) & Time (ms)\\\\\n")
        for m in methods:
            s = summ[m]
            f.write(f"{m} & {s['sum_rate'][0]:.2f} [{s['sum_rate'][1]:.2f}, {s['sum_rate'][2]:.2f}] & "
                    f"{100*s['viol'][0]:.1f} & {s['sqrt_crb_deg'][0]:.3f} & {s['time_ms'][0]:.2f}\\\\\n")
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
