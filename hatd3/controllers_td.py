"""Model-based time-division controllers: genie / myopic / lookahead (paper Sec. 9, system results)."""
import numpy as np
from .env_belief import td_epoch_metrics

RHO_GRID = np.array([0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])


def gbar(var_u, Nx):
    m = np.arange(1, Nx)
    var_u = np.atleast_1d(var_u)[:, None]
    return (1.0 / Nx) * (1 + 2 * np.sum((1 - m / Nx) * np.exp(-(np.pi * m) ** 2 * var_u / 2), axis=1))


def _steer(env, dirs_list, gw):
    if len(dirs_list) == 0:
        return np.zeros(env.cfg.N)
    s = np.zeros(env.cfg.N, complex)
    for u in dirs_list:
        s += np.exp(-1j * env.geo.k0 * (env.geo.ris_xz[:, 0] * u[0] + env.geo.ris_xz[:, 1] * u[2]))
    return np.angle(s) - np.angle(gw)


def plan(env, dirs):
    """Phases for each vehicle's sub-slot and for the scan block, from directions 'dirs' (all entities)."""
    cfg = env.cfg
    _, _, vh = np.linalg.svd(env.G)
    gw = env.G @ vh[0].conj()
    zone_all = {z: [dirs[e] for e in range(cfg.K + cfg.L) if env.zone_e[e] == z] for z in (0, 1)}
    scan = (_steer(env, zone_all[0], gw), _steer(env, zone_all[1], gw))
    comm = []
    for k in range(cfg.K):
        own = _steer(env, [dirs[k]], gw)
        other = scan[1] if env.zone_v[k] == 0 else scan[0]   # other zone keeps scanning its entities
        comm.append((own, other) if env.zone_v[k] == 0 else (other, own))
    return comm, scan


def genie_plan(env):
    """Perfect current CSI: co-phase each vehicle's true cascaded channel in its sub-slot."""
    cfg = env.cfg
    _, _, vh = np.linalg.svd(env.G)
    gw = env.G @ vh[0].conj()
    dirs = env.geo.unit(env.p_ent - env.geo.p_ris)
    comm, scan = plan(env, dirs)
    out = []
    for k in range(cfg.K):
        own = -np.angle(env.g[k] * gw)
        out.append((own, comm[k][1]) if env.zone_v[k] == 0 else (comm[k][0], own))
    return out, scan


def _q_u(env):
    out = np.zeros(env.trk.n)
    for e in range(env.trk.n):
        _, H = env.trk._h(env.trk.x[e], e)
        out[e] = H[0] @ env.trk.Q @ H[0]
    return out


def choose(env, gamma, mode, disc=0.95, beta=0.5, served=None, served_next=None):
    """Belief-only choice of the scan fraction rho (myopic or one-step lookahead)."""
    cfg, geo = env.cfg, env.geo
    p_pred = np.concatenate([env.trk.x[:, :2], env.p_ent[:, 2:3]], 1)
    dirs = geo.unit(p_pred - geo.p_ris)
    d = np.linalg.norm(p_pred - geo.p_ris, axis=1)
    rcs = np.r_[np.full(cfg.K, cfg.rcs), np.full(cfg.L, env.rcs_road)]
    alpha = np.sqrt(geo.lam ** 2 * rcs / ((4 * np.pi) ** 3 * np.maximum(d, 1) ** 4))
    comm, scan = plan(env, dirs)
    bvec = np.full(cfg.N, beta)
    _, prior = env.trk.prior_u()
    qu = _q_u(env)
    # predicted aligned SNR of each vehicle from the LoS geometry
    _, _, vh = np.linalg.svd(env.G)
    gw = env.G @ vh[0].conj()
    pl = geo.pathloss(d[: cfg.K]) * env.ch.Kr / (1 + env.ch.Kr)
    s0 = env.P * pl * np.where(env.zone_v == 0, beta, 1 - beta) * np.sum(np.abs(gw)) ** 2 / env.sigma2
    best, best_val, fb = None, -np.inf, None
    for rho in RHO_GRID:
        _, crb_u, _ = td_epoch_metrics(env, rho, bvec, comm, scan, use_true=False, alpha=alpha, dirs=dirs, served=served)
        post = 1 / (1 / np.maximum(prior, 1e-14) + 1 / np.maximum(crb_u, 1e-14))
        fb = (rho, bvec, comm, scan)
        if not np.all(post <= gamma):
            continue
        if mode == "myopic":
            return rho, bvec, comm, scan
        S = list(range(cfg.K)) if served is None else list(served)
        Sn = S if served_next is None else list(served_next)
        now = (1 - rho) / len(S) * np.sum(np.log2(1 + s0[S] * gbar(prior[S], cfg.Nx)))
        pn = post[: cfg.K] + qu[: cfg.K]
        nxt = (1 - rho) / len(Sn) * np.sum(np.log2(1 + s0[Sn] * gbar(pn[Sn], cfg.Nx)))
        val = now + disc * nxt
        if val > best_val:
            best_val, best = val, (rho, bvec, comm, scan)
    return best if best is not None else fb
