"""Sensing-aware reward: normalised sum rate - Lagrangian CRB term + potential-based shaping."""
import numpy as np


class SensingReward:
    def __init__(self, scfg, acfg, n_ent=None, thresh=None):
        self.s, self.c = scfg, acfg
        self.thresh = scfg.crb_thresh if thresh is None else thresh
        self.lam = np.zeros(scfg.L if n_ent is None else n_ent)
        self.ep_ratio = []

    def potential(self, crb):
        return -self.c.shaping_eta * np.sum(np.maximum(np.log(crb / self.thresh), 0.0))

    def __call__(self, sum_rate, crb_prev, crb_now):
        c, G = self.c, self.thresh
        ratio = np.clip(crb_now / G - 1.0, -1.0, 10.0)          # bounded (assumption A2)
        self.ep_ratio.append(crb_now / G)
        r_comm = sum_rate / c.rate_ref
        if c.use_lagrangian:
            r_sens = -np.sum(self.lam * ratio)
        else:
            r_sens = -c.fixed_weight * np.sum(ratio)
        r_shape = 0.0
        if c.use_shaping:
            r_shape = c.gamma * self.potential(np.maximum(crb_now, 1e-300)) - self.potential(np.maximum(crb_prev, 1e-300))
            r_shape = float(np.clip(r_shape, -10.0, 10.0))
        return r_comm + r_sens + r_shape, {"r_comm": r_comm, "r_sens": r_sens, "r_shape": r_shape}

    def end_episode(self):
        """Projected dual ascent on lambda (slower timescale than the actor)."""
        if self.ep_ratio and self.c.use_lagrangian:
            mean_ratio = np.mean(np.array(self.ep_ratio), axis=0)
            self.lam = np.clip(self.lam + self.c.lr_lambda * (mean_ratio - 1.0), 0.0, self.c.lambda_max)
        self.ep_ratio = []
        return self.lam.copy()
