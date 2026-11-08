"""Train HA-TD3 (or an ablation) on the statistical or ray-traced environment.

Example:
  python scripts/train.py --seed 0 --out runs/hatd3_s0
  python scripts/train.py --seed 0 --no-level2 --out runs/abl_nol2_s0
  python scripts/train.py --seed 0 --encoder mlp --out runs/mlptd3_s0   (flat-MLP TD3 baseline)
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from hatd3.config import SysConfig, AgentConfig
from hatd3.env import StarRisIsacEnv
from hatd3.agent import HATD3, Replay
from hatd3.action import ActionLayout
from hatd3.reward import SensingReward


def build_extra(prev_a, held_slow):
    return np.concatenate([prev_a, held_slow]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/hatd3")
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--no-level2", action="store_true")
    ap.add_argument("--no-two-timescale", action="store_true")
    ap.add_argument("--no-shaping", action="store_true")
    ap.add_argument("--no-lagrangian", action="store_true")
    ap.add_argument("--no-beta-bound", action="store_true")
    ap.add_argument("--coupled", action="store_true")
    ap.add_argument("--phase-bits", type=int, default=0)
    ap.add_argument("--sensing-mode", default="ris_sensors")
    ap.add_argument("--encoder", default="hier", choices=["hier", "flat_attn", "mlp"])
    ap.add_argument("--algo", default="td3", choices=["td3", "ddpg"])
    ap.add_argument("--belief", action="store_true", help="pilot-free 'sense now, serve later' setting (paper v2)")
    ap.add_argument("--gamma-track", type=float, default=1e-4)
    ap.add_argument("--Nx", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    scfg = SysConfig(coupled_phase=args.coupled, phase_bits=args.phase_bits, sensing_mode=args.sensing_mode)
    if args.Nx:
        scfg.Nx = args.Nx
    elif args.belief:
        scfg.Nx = 32                                   # N = 256 default in paper v2
    acfg = AgentConfig(use_level2=not args.no_level2, two_timescale=not args.no_two_timescale,
                       use_shaping=not args.no_shaping, use_lagrangian=not args.no_lagrangian,
                       use_beta_bound=not args.no_beta_bound, encoder=args.encoder)
    if args.algo == "ddpg":      # DDPG = TD3 without twin critics' min, smoothing, or delay
        acfg.policy_delay, acfg.target_noise, acfg.noise_clip = 1, 0.0, 0.0
    if args.episodes:
        acfg.episodes = args.episodes
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    json.dump({"sys": scfg.__dict__, "agent": acfg.__dict__, "args": vars(args)}, open(f"{args.out}/config.json", "w"),
              indent=2, default=str)

    if args.belief:
        from hatd3.env_belief import BeliefEnv
        env = BeliefEnv(scfg, seed=args.seed, gamma_track=args.gamma_track)
        acfg.gamma = 0.95
        LAM_COL = 13
        env_step = env.step_belief
    else:
        env = StarRisIsacEnv(scfg, seed=args.seed)
        LAM_COL = 6
        env_step = env.step
    obs = env.reset()
    lay = ActionLayout(scfg)
    fv, ft, fg = obs["veh"].shape[1], obs["tgt"].shape[1], obs["glob"].shape[0]
    agent = HATD3(scfg, acfg, fv, ft, fg, args.device)
    buf = Replay(min(acfg.buffer_size, 200_000), obs["veh"].shape, obs["tgt"].shape, agent.extra, lay.D)
    rew = (SensingReward(scfg, acfg, n_ent=scfg.K + scfg.L, thresh=args.gamma_track) if args.belief
           else SensingReward(scfg, acfg))
    log = open(f"{args.out}/train_log.jsonl", "w")
    step, t0 = 0, time.time()

    for ep in range(acfg.episodes):
        obs = env.reset()
        gid = ep
        buf.add_glob(gid, obs["glob"])
        # Proposition 2 action bound (offline, per episode geometry)
        if acfg.use_beta_bound and not args.belief:
            nu = 1 if scfg.sensing_mode == "ris_sensors" else 2
            cstar = env.sensing_only_crb_star(iters=50, seed=ep)
            bmin = np.minimum((cstar / scfg.crb_thresh) ** (1.0 / nu), 0.95)
            bmin_r = bmin[env.zone_t == 0].max(initial=0.0)
            bmin_t = bmin[env.zone_t == 1].max(initial=0.0)
            if bmin_r + bmin_t > 1:          # infeasible geometry: fall back to unrestricted
                bmin_r = bmin_t = 0.0
            lo, hi = bmin_r, 1 - bmin_t
        else:
            lo, hi = 0.0, 1.0
        prev_a = np.zeros(lay.D, np.float32)
        held = np.zeros(lay.D_slow, np.float32)
        ep_rate, ep_viol, ep_r = [], [], 0.0
        for t in range(scfg.slots_per_episode):
            if args.belief:
                obs["veh"][:, LAM_COL] = rew.lam[: scfg.K] / acfg.lambda_max
                obs["tgt"][:, LAM_COL] = rew.lam[scfg.K:] / acfg.lambda_max
            else:
                obs["tgt"][:, LAM_COL] = rew.lam / acfg.lambda_max
            ex = build_extra(prev_a, held)
            g_full = np.concatenate([obs["glob"], ex])
            if step < acfg.warmup_steps:
                a = np.random.uniform(-1, 1, lay.D)
            else:
                a = agent.act(obs["veh"], obs["tgt"], g_full, noise=acfg.expl_noise)
            decision = (t % scfg.T_h == 0) or not acfg.two_timescale
            if decision:
                held = a[lay.sl_slow].astype(np.float32).copy()
            else:
                a[lay.sl_slow] = held
            W, beta_r, th_r, th_t, kappa = lay.to_physical(a, env.P, lo, hi)
            crb_prev = env.pcrb.copy() if args.belief else env.last_crb.copy()
            obs2, info, done = env_step(W, beta_r, th_r, th_t)
            crb_now = info["pcrb_u"] if args.belief else info["crb"]
            r, parts = rew(info["sum_rate"], crb_prev, crb_now)
            if args.belief:
                obs2["veh"][:, LAM_COL] = rew.lam[: scfg.K] / acfg.lambda_max
                obs2["tgt"][:, LAM_COL] = rew.lam[scfg.K:] / acfg.lambda_max
            else:
                obs2["tgt"][:, LAM_COL] = rew.lam / acfg.lambda_max
            ex2 = build_extra(a.astype(np.float32), held)
            buf.add((obs["veh"], obs["tgt"], ex), a, r, (obs2["veh"], obs2["tgt"], ex2), float(done), gid, float(decision))
            obs, prev_a = obs2, a.astype(np.float32)
            step += 1
            if step >= acfg.warmup_steps:
                agent.update(buf)
            ep_rate.append(info["sum_rate"]); ep_viol.append(np.mean(crb_now > rew.thresh)); ep_r += r
        lam = rew.end_episode()
        rec = {"ep": ep, "step": step, "reward": ep_r, "sum_rate": float(np.mean(ep_rate)),
               "viol_rate": float(np.mean(ep_viol)), "lambda": lam.tolist(), "eps_csi": env.eps,
               "time_s": time.time() - t0}
        log.write(json.dumps(rec) + "\n"); log.flush()
        if ep % 10 == 0:
            print(rec)
        if ep % 100 == 0 or ep == acfg.episodes - 1:
            agent.save(f"{args.out}/model.pt")


if __name__ == "__main__":
    main()
