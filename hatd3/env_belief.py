"""Pilot-free 'sense now, serve later' environment (paper Secs. 3-4).

Each step is one epoch: the agent acts on tracker beliefs only; the chosen configuration is
evaluated on the true channel; echoes (with CRB set by the configuration) update the EKF.
Phases can be geometry-anchored: theta = theta_geo(belief) + residual (Sec. 6.3).
"""
import numpy as np
from .env import StarRisIsacEnv
from .channels import crb_angle, sensor_resp, db2lin
from .tracker import EKFTracker


class BeliefEnv(StarRisIsacEnv):
    def __init__(self, cfg, seed=0, dT=5e-3, sigma_a=3.0, gamma_track=1e-4, rcs_road=1.0, anchor=True):
        super().__init__(cfg, seed)
        self.dT, self.sigma_a, self.gamma_track, self.anchor = dT, sigma_a, gamma_track, anchor
        self.rcs_road = rcs_road
        self.cfg_slots = cfg.slots_per_episode

    # ------------------------------------------------------------------
    def reset(self, eps=None):
        super().reset(eps=0.0)                     # geometry, channels, targets (CSI never exposed)
        cfg = self.cfg
        # all K vehicles and L road users are tracked entities
        self.p_ent = np.concatenate([self.p_veh, self.p_tgt])
        self.v_ent = np.concatenate([self.v_veh, self.v_tgt])
        self.zone_e = np.concatenate([self.zone_v, self.zone_t])
        rcs = np.concatenate([np.full(cfg.K, cfg.rcs), np.full(cfg.L, self.rcs_road)])
        d = np.linalg.norm(self.p_ent - self.geo.p_ris, axis=1)
        self.alpha_e = np.sqrt(self.geo.lam ** 2 * rcs / ((4 * np.pi) ** 3 * np.maximum(d, 1) ** 4)) \
            * np.exp(1j * self.rng.uniform(0, 2 * np.pi, len(d)))
        self.trk = EKFTracker(cfg.K + cfg.L, self.dT, self.sigma_a, self.geo.p_ris, self.p_ent[:, 2], self.rng)
        x_true = np.concatenate([self.p_ent[:, :2], self.v_ent[:, :2]], 1)
        self.trk.init(x_true)
        self.trk.predict()
        self.pcrb = np.full(cfg.K + cfg.L, 1.0)
        self.last_rate = np.zeros(cfg.K)
        self.t = 0
        self.obs = self._belief_tokens()
        return self.obs

    def _belief_tokens(self):
        cfg = self.cfg
        x, P = self.trk.x, self.trk.P
        u, varu = self.trk.prior_u()
        d = np.linalg.norm(np.concatenate([x[:, :2], self.p_ent[:, 2:3]], 1) - self.geo.p_ris, axis=1)
        sd = np.sqrt(np.maximum(np.diagonal(P, axis1=1, axis2=2), 1e-12))
        rpx = P[:, 0, 2] / (sd[:, 0] * sd[:, 2])
        rpy = P[:, 1, 3] / (sd[:, 1] * sd[:, 3])
        n = cfg.K + cfg.L
        isveh = np.r_[np.ones(cfg.K), np.zeros(cfg.L)]
        rate = np.r_[self.last_rate, np.zeros(cfg.L)]
        tok = np.column_stack([x[:, :2] / 100, x[:, 2:] / 40, np.log10(sd ** 2) / 6, rpx, rpy, u, d / 100,
                               np.log10(self.pcrb / self.gamma_track + 1e-12), np.zeros(n),  # lambda slot
                               np.eye(2)[self.zone_e], isveh, rate / 10]).astype(np.float32)
        glob = np.concatenate([self.geo.p_bs / 50, self.geo.p_ris / 50, [cfg.M / 64, cfg.N / 1024]]).astype(np.float32)
        # split into the agent's veh/tgt token groups (same features for both)
        return {"veh": tok[: cfg.K], "tgt": tok[cfg.K:], "glob": glob, "zone_v": self.zone_v, "zone_t": self.zone_t,
                "prior_u": u, "prior_var_u": varu}

    def geometric_phases(self):
        """Co-phase the BS-surface LoS beam with the predicted directions of each zone's vehicles."""
        cfg, geo = self.cfg, self.geo
        # predicted 3-D direction from the surface (antenna heights are known), not just u
        p_pred = np.concatenate([self.trk.x[:, :2], self.p_ent[:, 2:3]], 1)
        udir = geo.unit(p_pred - geo.p_ris)
        _, _, vh = np.linalg.svd(self.G_hat if hasattr(self, "G_hat") else self.G)
        w = vh[0].conj()
        gw = self.G @ w                                         # BS-surface is static and calibrated
        th = {}
        for z in (0, 1):
            idx = np.where(self.zone_v == z)[0]
            s = np.zeros(cfg.N, complex)
            for k in idx:
                s += np.exp(-1j * geo.k0 * (geo.ris_xz[:, 0] * udir[k, 0] + geo.ris_xz[:, 1] * udir[k, 2]))
            th[z] = np.angle(s) - np.angle(gw) if len(idx) else np.zeros(cfg.N)
        return th[0], th[1]

    # ------------------------------------------------------------------
    def advance(self):
        """Move every entity by one epoch and update the true channels (no observation)."""
        cfg = self.cfg
        steps = max(1, int(round(self.dT / cfg.Ts)))
        for _ in range(steps):
            self.ch.age(np.abs(self.v_veh[:, 0]))
        self.p_ent = self.p_ent + self.v_ent * self.dT
        self.p_veh, self.p_tgt = self.p_ent[: cfg.K], self.p_ent[cfg.K:]
        self._refresh_true()

    def transmit(self, W, th_r, th_t):
        """Serve on the true channel, sense, update the EKF; returns rates, PCRB and CRBs."""
        cfg = self.cfg
        sinr, rates, _, _ = self.evaluate(W, th_r, th_t)
        R = W @ W.conj().T
        phi, el = self.geo.angles_from_ris(self.p_ent)
        crb_u = np.zeros(len(phi)); crb_d = np.zeros(len(phi))
        brms = cfg.bw / np.sqrt(12)
        for e in range(len(phi)):
            th = th_r if self.zone_e[e] == 0 else th_t
            a, ad = self.geo.a_ris_angle(phi[e], el[e])
            b, bd = sensor_resp(self.geo, phi[e], el[e])
            c_phi = crb_angle(self.G, th, a, ad, R, self.alpha_e[e], self.sigma2, cfg.T_snap, b, bd)
            crb_u[e] = (np.cos(phi[e]) * np.cos(el[e])) ** 2 * c_phi        # delta method phi -> u
            v = (th * a) @ self.G
            snr = np.abs(self.alpha_e[e]) ** 2 * np.real(np.vdot(b, b)) * np.real(v @ R.T @ v.conj()) * cfg.T_snap / self.sigma2
            crb_d[e] = (299_792_458.0 ** 2) / (8 * np.pi ** 2 * brms ** 2 * max(snr, 1e-12))
        x_true = np.concatenate([self.p_ent[:, :2], self.v_ent[:, :2]], 1)
        true_u_before = self._true_u()
        prior_u, _ = self.trk.prior_u()
        self.pcrb = self.trk.update(x_true, crb_u, crb_d)
        self.trk.predict()
        self.last_rate = rates
        self.t += 1
        self.obs = self._belief_tokens()
        done = self.t >= self.cfg_slots
        return self.obs, {"rates": rates, "sum_rate": rates.sum(), "pcrb_u": self.pcrb,
                          "crb_u": crb_u, "track_err_u": true_u_before - prior_u}, done

    def step_belief(self, W, beta_r, theta_r, theta_t, residual_scale=np.pi):
        if self.anchor:
            g_r, g_t = self.geometric_phases()
            theta_r = g_r + residual_scale * np.angle(np.exp(1j * theta_r)) / np.pi
            theta_t = g_t + residual_scale * np.angle(np.exp(1j * theta_t)) / np.pi
        th_r, th_t = self.coefficients(beta_r, theta_r, theta_t)
        self.advance()
        return self.transmit(W, th_r, th_t)

    def _true_u(self):
        d = self.p_ent - self.geo.p_ris
        return d[:, 0] / np.linalg.norm(d, axis=1)


# ---------------------------------------------------------------------------
# Time-division ISAC epoch (used by the model-based controllers and the paper's system results)
# ---------------------------------------------------------------------------
def _crb_traces(G, th, a, ad, b, bd, R):
    v = (th * a) @ G
    vd = (th * ad) @ G
    B = np.outer(b, v)
    Bd = np.outer(bd, v) + np.outer(b, vd)
    return (np.real(np.trace(B.conj().T @ B @ R)), np.real(np.trace(Bd.conj().T @ Bd @ R)),
            np.trace(Bd.conj().T @ B @ R))


def td_epoch_metrics(env, rho, beta, comm_phases, scan_phases, use_true=True, alpha=None, dirs=None, served=None):
    """Rates and per-entity CRB_u for one time-division epoch.

    comm_phases[k] = (theta_r, theta_t) used in vehicle k's sub-slot (T(1-rho)/K symbols);
    scan_phases = (theta_r, theta_t) used in the scan sub-block (T*rho symbols).
    Every sub-slot contributes Fisher information (data symbols are known at the transmitter).
    """
    cfg, geo = env.cfg, env.geo
    _, _, vh = np.linalg.svd(env.G)
    w = vh[0].conj()
    R = env.P * np.outer(w, w.conj())
    blocks = [(scan_phases, cfg.T_snap * rho, None)] if rho > 0 else []
    served = list(range(cfg.K)) if served is None else list(served)
    for k in served:
        blocks.append((comm_phases[k], cfg.T_snap * (1 - rho) / len(served), k))
    rates = np.zeros(cfg.K)
    if use_true:
        p = env.p_ent
        dirs_e = geo.unit(p - geo.p_ris)
        al = env.alpha_e
    else:
        dirs_e, al = dirs, alpha
    n_e = len(dirs_e)
    tbb = np.zeros(n_e); tdd = np.zeros(n_e); tdb = np.zeros(n_e, complex)
    for (ph, Tj, k) in blocks:
        c_r, c_t = env.coefficients(beta, ph[0], ph[1])
        if k is not None and use_true:
            th = c_r if env.zone_v[k] == 0 else c_t
            h = env.hd[k] + (env.g[k] * th) @ env.G
            rates[k] = (1 - rho) / len(served) * np.log2(1 + env.P * np.abs(h @ w) ** 2 / env.sigma2)
        for e in range(n_e):
            th = c_r if env.zone_e[e] == 0 else c_t
            phi = np.arctan2(dirs_e[e][0], abs(dirs_e[e][1])); el = np.arcsin(dirs_e[e][2])
            a, ad = geo.a_ris_angle(phi, el)
            b, bd = sensor_resp(geo, phi, el)
            x, y, z = _crb_traces(env.G, th, a, ad, b, bd, R)
            tbb[e] += Tj * x; tdd[e] += Tj * y; tdb[e] += Tj * z
    crb_u = np.zeros(n_e)
    for e in range(n_e):
        phi = np.arctan2(dirs_e[e][0], abs(dirs_e[e][1])); el = np.arcsin(dirs_e[e][2])
        den = tdd[e] * tbb[e] - np.abs(tdb[e]) ** 2
        c_phi = env.sigma2 / (2 * np.abs(al[e]) ** 2) * tbb[e] / max(den, 1e-300)
        crb_u[e] = (np.cos(phi) * np.cos(el)) ** 2 * c_phi
    return rates, crb_u, w


def td_transmit(env, rho, beta, comm_phases, scan_phases, served=None):
    """Serve and sense one time-division epoch on the true channel; update the EKF."""
    cfg = env.cfg
    rates, crb_u, _ = td_epoch_metrics(env, rho, beta, comm_phases, scan_phases, served=served)
    crb_d = np.full(len(crb_u), 0.05 ** 2)          # range is not the bottleneck at 100 MHz
    x_true = np.concatenate([env.p_ent[:, :2], env.v_ent[:, :2]], 1)
    env.pcrb = env.trk.update(x_true, crb_u, crb_d)
    env.trk.predict()
    env.last_rate = rates
    env.t += 1
    env.obs = env._belief_tokens()
    return {"rates": rates, "sum_rate": rates.sum(), "pcrb_u": env.pcrb, "crb_u": crb_u}
