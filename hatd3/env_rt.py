"""Ray-traced environment: replays Sionna snapshots (scripts/sionna_export.py) instead of the
statistical channel model. Same interface as StarRisIsacEnv."""
import numpy as np
from .env import StarRisIsacEnv
from .channels import Channels


class _RTChannels(Channels):
    def __init__(self, geo, rng, data):
        self.geo, self.cfg, self.rng = geo, geo.cfg, rng
        self.d, self.k = data, 0

    def init_users(self, K):
        pass

    def age(self, speeds):
        self.k = min(self.k + 1, self.d["g"].shape[0] - 1)

    def bs_ris(self):
        return self.d["G"]

    def ris_users(self, p_veh):
        return self.d["g"][self.k]

    def direct(self, p_veh):
        return self.d["hd"][self.k]

    def target_gain(self, p_tgt):
        return Channels.target_gain(self, p_tgt)


class RTStarRisIsacEnv(StarRisIsacEnv):
    def __init__(self, cfg, npz_files, seed=0):
        super().__init__(cfg, seed)
        self.files = list(npz_files)

    def reset(self, eps=None):
        f = self.files[self.rng.integers(len(self.files))]
        data = dict(np.load(f))
        obs = super().reset(eps)                     # sets zones, rng state, targets
        self.ch = _RTChannels(self.geo, self.rng, data)
        pv = data["p_veh"][0]
        self.p_veh = pv[:, :3].copy()
        self.v_veh = np.concatenate([pv[:, 3:5], np.zeros((len(pv), 1))], 1)
        self.p_tgt = data["p_tgt"][0].copy()
        self.alpha = self.ch.target_gain(self.p_tgt)
        self._refresh_true()
        self._observe()
        return self.obs

    def step(self, W, beta_r, theta_r, theta_t):
        obs, info, done = super().step(W, beta_r, theta_r, theta_t)
        k = self.ch.k
        self.p_veh = self.ch.d["p_veh"][k][:, :3].copy()  # trajectories come from the file
        self.p_tgt = self.ch.d["p_tgt"][k].copy()
        return obs, info, done
