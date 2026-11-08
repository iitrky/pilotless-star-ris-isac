"""Default configuration (matches Table 1 of the paper)."""
from dataclasses import dataclass, field


@dataclass
class SysConfig:
    fc: float = 7e9               # carrier [Hz], 6G upper mid-band (FR3)
    bw: float = 100e6             # bandwidth [Hz]
    M: int = 16                   # BS antennas (ULA)
    Nx: int = 8                   # STAR-RIS columns
    Nz: int = 8                   # STAR-RIS rows
    K_r: int = 3                  # vehicles in reflection zone
    K_t: int = 3                  # vehicles in transmission zone
    L_r: int = 1                  # targets in reflection zone
    L_t: int = 1                  # targets in transmission zone
    P_max_dBm: float = 30.0
    noise_psd_dBm: float = -174.0
    nf_dB: float = 7.0
    v_kmh: tuple = (30.0, 120.0)  # speed range sampled per vehicle
    Ts: float = 0.5e-3            # slot [s]
    T_h: int = 20                 # slow-head period [slots]
    T_snap: int = 256             # sensing snapshots per slot
    rician_K_dB: float = 6.0
    pl_exp: float = 2.0           # path-loss exponent (LoS-dominant segments)
    direct_blocked: bool = True   # BS->vehicle direct link blocked
    csi_eps_train: tuple = (0.0, 0.2)
    rcs: float = 10.0             # target RCS [m^2] (car-sized)
    Ns: int = 16                  # sensing elements co-located at the STAR-RIS
    sensing_mode: str = 'ris_sensors'  # 'ris_sensors' (nu=1) or 'bs' (monostatic via surface, nu=2)
    crb_thresh: float = 1.218e-3  # Gamma_l [rad^2] = (2 deg)^2
    coupled_phase: bool = False
    phase_bits: int = 0           # 0 = continuous, 1 = 1-bit hardware
    slots_per_episode: int = 200

    @property
    def N(self):
        return self.Nx * self.Nz

    @property
    def K(self):
        return self.K_r + self.K_t

    @property
    def L(self):
        return self.L_r + self.L_t


@dataclass
class AgentConfig:
    d: int = 128
    heads: int = 4
    layers_l1: int = 2
    hidden: int = 256
    lr_actor: float = 1e-4
    lr_critic: float = 1e-3
    lr_lambda: float = 1e-2
    lambda_max: float = 50.0
    gamma: float = 0.99
    tau: float = 0.005
    policy_delay: int = 2
    expl_noise: float = 0.1
    target_noise: float = 0.2
    noise_clip: float = 0.5
    buffer_size: int = 200_000
    batch: int = 256
    shaping_eta: float = 1.0
    rate_ref: float = 10.0        # R_ref normaliser [bit/s/Hz]
    warmup_steps: int = 5000
    episodes: int = 3000
    use_shaping: bool = True
    use_lagrangian: bool = True
    fixed_weight: float = 1.0     # used when use_lagrangian=False
    use_level2: bool = True
    two_timescale: bool = True
    use_beta_bound: bool = True
    encoder: str = 'hier'         # 'hier' (HA-TD3) | 'flat_attn' (SA-TTD3-style) | 'mlp' (MLP-TD3)
