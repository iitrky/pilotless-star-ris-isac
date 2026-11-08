"""Controllers and episode runner on the deterministic site (paper Secs. 8-9).

Every controller uses the same time-division ISAC epoch (td_transmit) and the same per-epoch
scan-fraction rule (Algorithm 1, step 5); they differ only in the surface phases and split.
"""
import time
import numpy as np
from .controllers_td import choose, _steer
from .env_belief import td_transmit

CAND_D = np.array([16.0, 20.0, 24.0, 28.0, 32.0, 36.0])   # image-plane distances of the codebook [m]
N_BEAM = 1 + len(CAND_D)                                   # residual codebook: LoS anchor + image sources


def beta_rule(env, bmin):
    """Model-based split: proportional to the number of served vehicles in each zone."""
    return float(np.clip(env.cfg.K_r / env.cfg.K, bmin, 1 - bmin))


def _gw(env):
    _, _, vh = np.linalg.svd(env.G)
    return env.G @ vh[0].conj()


def pred_dirs(env):
    p = np.concatenate([env.trk.x[:, :2], env.p_ent[:, 2:3]], 1)
    return env.geo.unit(p - env.geo.p_ris)


def with_own(env, comm, own):
    """Replace each vehicle's own-zone phases in its sub-slot."""
    out = []
    for k in range(env.cfg.K):
        out.append((own[k], comm[k][1]) if env.zone_v[k] == 0 else (comm[k][0], own[k]))
    return out


def image_dirs(env):
    """Codebook of virtual-source directions for each vehicle: mirror images of its predicted
    position across planes parallel to the surface at distances CAND_D (no site knowledge)."""
    p = np.concatenate([env.trk.x[: env.cfg.K, :2], env.p_ent[: env.cfg.K, 2:3]], 1)
    sgn = np.sign(p[:, 1:2])
    img = np.repeat(p[:, None, :], len(CAND_D), 1)
    img[:, :, 1] = 2 * sgn * CAND_D[None, :] - p[:, 1:2]
    return env.geo.unit(img - env.geo.p_ris)                          # (K, J, 3)


def steer_discrete(env, jvec, anchor=True):
    """Own-zone phases for codebook entry j_k: 0 = geometric anchor (predicted LoS direction),
    j >= 1 = virtual source behind a plane at distance CAND_D[j - 1]. Without the anchor the
    LoS entry is unavailable (entry 0 maps to the nearest image plane)."""
    gw = _gw(env)
    los = pred_dirs(env)[: env.cfg.K]
    imgs = image_dirs(env)
    out = []
    for k in range(env.cfg.K):
        j = int(jvec[k])
        u = los[k] if (j == 0 and anchor) else imgs[k, max(j - 1, 0)]
        out.append(_steer(env, [u], gw))
    return out


def genie_own(env):
    """Perfect CSI: co-phase every element of the cascaded channel with the direct path."""
    _, _, vh = np.linalg.svd(env.G)
    w = vh[0].conj()
    gw = env.G @ w
    return [np.angle(env.hd[k] @ w) - np.angle(env.g[k] * gw) for k in range(env.cfg.K)]


def epoch(env, kind, gam, bmin=0.15, beta=None, du=None, anchor=True, mode="myopic"):
    """One epoch of a controller; returns td_transmit info plus the decision time (s)."""
    t0 = time.perf_counter()
    b = beta_rule(env, bmin) if beta is None else beta
    rho, _, comm, scan = choose(env, gam, mode if kind != "genie" else "myopic", beta=b)
    bvec = np.full(env.cfg.N, b)
    if kind == "rt":                                  # two conventional surfaces of N/2 elements
        half = np.arange(env.cfg.N) < env.cfg.N // 2
        bvec = half.astype(float)
    if kind == "agent":
        comm = with_own(env, comm, steer_discrete(env, du, anchor))
    dt = time.perf_counter() - t0
    env.advance()
    if kind == "genie":
        comm = with_own(env, comm, genie_own(env))
    prior_u, _ = env.trk.prior_u()
    true_u = env._true_u()
    info = td_transmit(env, rho, bvec, comm, scan)
    info["dt"] = dt
    info["rho"] = rho
    info["beta"] = b
    info["ang_err"] = np.arcsin(np.clip(true_u, -1, 1)) - np.arcsin(np.clip(prior_u, -1, 1))
    return info


def summarise(infos, gam, skip=5):
    I = infos[skip:]
    thr = np.mean([i["sum_rate"] for i in I])
    pc = np.concatenate([i["pcrb_u"] for i in I]) / gam
    ae = np.concatenate([i["ang_err"] for i in I])
    rv = np.concatenate([i["rates"] for i in I])
    return {"throughput": float(thr), "p5_rate": float(np.percentile(rv, 5)), "pcrb_ratio": float(np.mean(pc)), "viol": float(100 * np.mean(pc > 1)),
            "angle_rmse_deg": float(np.degrees(np.sqrt(np.mean(ae ** 2)))),
            "ms": float(1e3 * np.mean([i["dt"] for i in I]))}
