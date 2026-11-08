"""STAR-RIS ISAC vehicular environment (statistical channel model).

The ray-traced variant replaces Channels with stored Sionna snapshots (see sionna_export.py);
the step logic is identical.
"""
import numpy as np
from .channels import (Geometry, Channels, add_csi_error, effective_channels, sinr_rates,
                       crb_angle, echo_matrices, sensor_resp, db2lin)


class StarRisIsacEnv:
    def __init__(self, cfg, seed=0):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.geo = Geometry(cfg, self.rng)
        self.sigma2 = db2lin(cfg.noise_psd_dBm - 30 + cfg.nf_dB) * cfg.bw
        self.P = db2lin(cfg.P_max_dBm - 30)
        self.zone_v = np.array([0] * cfg.K_r + [1] * cfg.K_t)        # 0 = r, 1 = t
        self.zone_t = np.array([0] * cfg.L_r + [1] * cfg.L_t)
        self.eps = 0.0

    # ------------------------------------------------------------------
    def reset(self, eps=None):
        cfg, rng = self.cfg, self.rng
        self.t = 0
        self.eps = rng.uniform(*cfg.csi_eps_train) if eps is None else eps
        self.ch = Channels(self.geo, rng)
        self.ch.init_users(cfg.K)
        lanes = np.where(self.zone_v == 0, rng.choice([10.0, 14.0], cfg.K), rng.choice([-10.0, -14.0], cfg.K))
        self.p_veh = np.stack([rng.uniform(-60, 60, cfg.K), lanes, np.full(cfg.K, 1.5)], axis=1)
        sp = rng.uniform(*cfg.v_kmh, cfg.K) / 3.6
        self.v_veh = np.stack([sp * rng.choice([-1, 1], cfg.K), np.zeros(cfg.K), np.zeros(cfg.K)], axis=1)
        ty = np.where(self.zone_t == 0, 20.0, -20.0)
        self.p_tgt = np.stack([rng.uniform(-30, 30, cfg.L), ty, np.full(cfg.L, 1.0)], axis=1)
        tsp = rng.uniform(1, 5, cfg.L)
        self.v_tgt = np.stack([tsp * rng.choice([-1, 1], cfg.L), np.zeros(cfg.L), np.zeros(cfg.L)], axis=1)
        self.alpha = self.ch.target_gain(self.p_tgt)
        self.last_sinr = np.zeros(cfg.K)
        self.last_echo_snr = np.zeros(cfg.L)
        self.phi_hat = self.geo.angles_from_ris(self.p_tgt)[0]
        self._refresh_true()
        # initial CRB under a neutral configuration (random phases, beta = 0.5, isotropic beams)
        W0 = (rng.standard_normal((cfg.M, cfg.K + cfg.L)) + 1j * rng.standard_normal((cfg.M, cfg.K + cfg.L)))
        W0 *= np.sqrt(self.P) / np.linalg.norm(W0)
        c_r, c_t = self.coefficients(np.full(cfg.N, 0.5), rng.uniform(0, 2 * np.pi, cfg.N), rng.uniform(0, 2 * np.pi, cfg.N))
        _, _, self.last_crb, self.last_echo_snr = self.evaluate(W0, c_r, c_t)
        self._observe()
        return self.obs

    def _refresh_true(self):
        self.G = self.ch.bs_ris()
        self.g = self.ch.ris_users(self.p_veh)
        self.hd = self.ch.direct(self.p_veh)

    def _observe(self):
        """Agent sees CSI from the previous slot with estimation error (set before the move)."""
        self.g_hat = add_csi_error(self.g, self.eps, self.rng)
        self.hd_hat = add_csi_error(self.hd, self.eps, self.rng)
        self.G_hat = self.G  # BS-RIS link is quasi-static; assume well estimated
        self.obs = self._tokens()

    def _tokens(self):
        cfg = self.cfg
        sc_g = 1.0 / (np.sqrt(np.mean(np.abs(self.g_hat) ** 2)) + 1e-30)
        sc_h = 1.0 / (np.sqrt(np.mean(np.abs(self.hd_hat) ** 2)) + 1e-30)
        onehot_v = np.eye(2)[self.zone_v]
        veh = np.concatenate([
            (self.g_hat * sc_g).real, (self.g_hat * sc_g).imag,
            (self.hd_hat * sc_h).real, (self.hd_hat * sc_h).imag,
            self.p_veh[:, :2] / 100.0, self.v_veh[:, :2] / 40.0,
            (10 * np.log10(self.last_sinr + 1e-9) / 30.0)[:, None], onehot_v], axis=1)
        _, el = self.geo.angles_from_ris(self.p_tgt)
        rng_ = np.linalg.norm(self.p_tgt - self.geo.p_ris, axis=1)
        onehot_t = np.eye(2)[self.zone_t]
        tgt = np.concatenate([
            np.cos(self.phi_hat)[:, None], np.sin(self.phi_hat)[:, None], (rng_ / 100.0)[:, None],
            (self.v_tgt[:, 0] / 10.0)[:, None], (10 * np.log10(self.last_echo_snr + 1e-12) / 30.0)[:, None],
            np.log10(self.last_crb / cfg.crb_thresh + 1e-12)[:, None],
            np.zeros((cfg.L, 1)),            # lambda_l slot, filled by the trainer
            onehot_t], axis=1)
        sc_G = 1.0 / (np.sqrt(np.mean(np.abs(self.G_hat) ** 2)) + 1e-30)
        glob = np.concatenate([(self.G_hat * sc_G).real.ravel(), (self.G_hat * sc_G).imag.ravel()])
        return {"veh": veh.astype(np.float32), "tgt": tgt.astype(np.float32),
                "glob": glob.astype(np.float32), "zone_v": self.zone_v, "zone_t": self.zone_t}

    # ------------------------------------------------------------------
    def coefficients(self, beta_r, theta_r, theta_t):
        cfg = self.cfg
        if cfg.coupled_phase:
            d = np.angle(np.exp(1j * (theta_t - theta_r)))
            theta_t = theta_r + np.sign(d + 1e-12) * np.pi / 2
        if cfg.phase_bits > 0:
            q = 2 * np.pi / 2 ** cfg.phase_bits
            theta_r = np.round(theta_r / q) * q
            theta_t = np.round(theta_t / q) * q
        beta_r = np.clip(beta_r, 0, 1)
        th_r = np.sqrt(beta_r) * np.exp(1j * theta_r)
        th_t = np.sqrt(1 - beta_r) * np.exp(1j * theta_t)
        return th_r, th_t

    def evaluate(self, W, th_r, th_t, G=None, g=None, hd=None):
        """Rates and CRBs of a configuration on the given (default: true current) channels."""
        cfg = self.cfg
        G = self.G if G is None else G
        g = self.g if g is None else g
        hd = self.hd if hd is None else hd
        theta_users = np.where(self.zone_v[:, None] == 0, th_r[None, :], th_t[None, :])
        H = effective_channels(hd, g, G, theta_users)
        sinr, rates = sinr_rates(H, W, self.sigma2)
        R = W @ W.conj().T
        phi, el = self.geo.angles_from_ris(self.p_tgt)
        crb = np.zeros(cfg.L)
        snr = np.zeros(cfg.L)
        for l in range(cfg.L):
            th = th_r if self.zone_t[l] == 0 else th_t
            a, ad = self.geo.a_ris_angle(phi[l], el[l])
            b, bd = self._sens(phi[l], el[l])
            crb[l] = crb_angle(G, th, a, ad, R, self.alpha[l], self.sigma2, cfg.T_snap, b, bd)
            Bm, _ = echo_matrices(G, th, a, ad, b, bd)
            nrx = Bm.shape[0]
            snr[l] = np.abs(self.alpha[l]) ** 2 * np.real(np.trace(Bm @ R @ Bm.conj().T)) * cfg.T_snap / (nrx * self.sigma2)
        return sinr, rates, crb, snr

    def _sens(self, phi, el):
        if self.cfg.sensing_mode == 'bs':
            return None, None
        return sensor_resp(self.geo, phi, el)

    def step(self, W, beta_r, theta_r, theta_t):
        cfg = self.cfg
        th_r, th_t = self.coefficients(beta_r, theta_r, theta_t)
        # The configuration chosen from stale/erroneous CSI is applied to the channel that has
        # already evolved: move users and age the channel first, then evaluate.
        self.p_veh = self.p_veh + self.v_veh * cfg.Ts
        self.p_veh[:, 0] = (self.p_veh[:, 0] + 60) % 120 - 60
        self.p_tgt = self.p_tgt + self.v_tgt * cfg.Ts
        self.ch.age(np.abs(self.v_veh[:, 0]))
        self._refresh_true()
        sinr, rates, crb, snr = self.evaluate(W, th_r, th_t)
        # Echo processing: efficient estimator approximated as phi_hat ~ N(phi, CRB).
        phi, _ = self.geo.angles_from_ris(self.p_tgt)
        self.phi_hat = phi + np.sqrt(np.minimum(crb, 1.0)) * self.rng.standard_normal(cfg.L)
        self.last_sinr, self.last_crb, self.last_echo_snr = sinr, crb, snr
        self.t += 1
        self._observe()
        done = self.t >= cfg.slots_per_episode
        return self.obs, {"rates": rates, "sum_rate": rates.sum(), "crb": crb, "echo_snr": snr}, done

    # ------------------------------------------------------------------
    def sensing_only_crb_star(self, iters=20, seed=0):
        """Offline CRB~*_l at beta = 1 used by Proposition 2.

        Surface phases focus the BS's dominant eigen-beam onto target l; the BS beam is matched to
        the cascade a^T Theta G; then a local random search refines the phases. A lightweight
        surrogate for the sensing-only optimum of [9] -- it upper-bounds CRB~*, so the resulting
        beta_min is conservative only if the surrogate is near-optimal; report which you used.
        """
        cfg, rng = self.cfg, np.random.default_rng(seed)
        phi, el = self.geo.angles_from_ris(self.p_tgt)
        _, _, vh = np.linalg.svd(self.G)
        w0 = vh[0].conj()
        out = np.zeros(cfg.L)
        for l in range(cfg.L):
            a, ad = self.geo.a_ris_angle(phi[l], el[l])
            b, bd = self._sens(phi[l], el[l])
            ph = -np.angle(a * (self.G @ w0))
            best = np.inf
            for it in range(iters + 1):
                cand = ph if it == 0 else ph + rng.normal(0, 0.3, cfg.N)
                th = np.exp(1j * cand)
                v = (th * a) @ self.G
                w = v.conj() / np.linalg.norm(v) * np.sqrt(self.P)
                R = np.outer(w, w.conj())
                c = crb_angle(self.G, th, a, ad, R, self.alpha[l], self.sigma2, cfg.T_snap, b, bd)
                if c < best:
                    best, ph = c, cand
            out[l] = best
        return out
