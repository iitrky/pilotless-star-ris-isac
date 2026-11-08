"""Deterministic site model: first-order image-method ray tracing with knife-edge blockage.

Replaces the statistical Rician channel of env.py with a site-specific one (paper Sec. 8.1):
  * line-of-sight path from the surface to each vehicle;
  * specular reflections from building façades (image method, finite façade extent);
  * single-bounce scattering from street furniture (lamp posts);
  * blockage of every path segment by parked trucks, a bus shelter and tree canopies,
    with ITU-R P.526 single knife-edge diffraction loss J(nu) (capped for foliage);
  * a weak diffuse component (Rician K = 10 dB) aged with Jakes' model.
The multipath is a deterministic function of vehicle position, so a controller that sees only
tracker beliefs can learn it; the diffuse part cannot be learned.
"""
from dataclasses import dataclass, field
import numpy as np
from scipy.special import j0
from .channels import C0, db2lin
from .env_belief import BeliefEnv


@dataclass
class Screen:            # vertical obstacle in the plane y = y, x in [x1, x2], z in [0, h]
    y: float
    x1: float
    x2: float
    h: float
    cap_dB: float        # maximum loss (foliage ~ 12 dB, metal truck ~ 30 dB)


@dataclass
class Facade:            # reflecting vertical plane y = y, x in [x1, x2]
    y: float
    x1: float
    x2: float
    gamma: float


@dataclass
class SiteLayout:
    name: str
    screens: list = field(default_factory=list)
    facades: list = field(default_factory=list)
    scatterers: list = field(default_factory=list)   # (x, y, z, rcs)
    ris_z: float = 6.0


def _trees(y, xs, cap=12.0):
    return [Screen(y, x - 2.5, x + 2.5, 7.5, cap) for x in xs]


SITE_A = SiteLayout(
    "A",
    screens=[Screen(5.0, -34, -24, 4.2, 30), Screen(5.0, 8, 20, 4.2, 30),          # parked trucks (r side)
             Screen(-5.0, -12, -2, 4.2, 30), Screen(-5.0, 30, 42, 4.2, 30),        # parked trucks (t side)
             Screen(4.0, 40, 46, 3.0, 25)] + _trees(6.5, [-52, -8, 30]) + _trees(-6.5, [-40, 14, 55]),
    facades=[Facade(24.0, -70, 70, 0.6), Facade(-24.0, -70, -5, 0.5), Facade(-24.0, 5, 70, 0.5)],
    scatterers=[(x, 3.5, 6.0, 1.0) for x in (-50, -25, 0, 25, 50)] + [(x, -3.5, 6.0, 1.0) for x in (-37, -12, 13, 38)],
)

SITE_B = SiteLayout(                     # unseen site: different street width, gaps, obstacles
    "B",
    screens=[Screen(5.5, -20, -8, 4.0, 30), Screen(5.5, 25, 37, 4.0, 30), Screen(-5.5, -48, -36, 4.0, 30),
             Screen(-5.5, 0, 10, 4.0, 30), Screen(4.0, -5, 1, 3.0, 25)] + _trees(7.0, [-45, 5, 50]) + _trees(-7.0, [-20, 25]),
    facades=[Facade(28.0, -70, -15, 0.7), Facade(28.0, -5, 70, 0.7), Facade(-21.0, -70, 70, 0.4)],
    scatterers=[(x, 4.0, 6.0, 1.5) for x in (-40, -10, 20, 50)] + [(x, -4.0, 6.0, 1.5) for x in (-30, 0, 30)],
)


def knife_edge_dB(nu):
    """ITU-R P.526 approximation of single knife-edge diffraction loss."""
    nu = np.asarray(nu, float)
    return np.where(nu > -0.78, 6.9 + 20 * np.log10(np.sqrt((nu - 0.1) ** 2 + 1) + nu - 0.1), 0.0)


def segment_loss(p, q, screens, lam):
    """Amplitude factor of the segment p -> q through all screens it crosses."""
    loss_db = 0.0
    for s in screens:
        if (p[1] - s.y) * (q[1] - s.y) >= 0:
            continue
        t = (s.y - p[1]) / (q[1] - p[1])
        xi, zi = p[0] + t * (q[0] - p[0]), p[2] + t * (q[2] - p[2])
        depth = min(s.h - zi, xi - s.x1, s.x2 - xi)          # >0 inside the screen
        d1 = np.linalg.norm(np.array([xi, s.y, zi]) - p)
        d2 = np.linalg.norm(q - np.array([xi, s.y, zi]))
        nu = depth * np.sqrt(2 * (d1 + d2) / (lam * d1 * d2))
        loss_db += min(float(knife_edge_dB(nu)), s.cap_dB)
    return 10 ** (-loss_db / 20)


class SiteChannel:
    def __init__(self, site, geo, rng, k_diffuse_dB=10.0):
        self.site, self.geo, self.rng = site, geo, rng
        self.Kd = db2lin(k_diffuse_dB)

    def paths(self, p):
        """List of (unit direction from the surface, complex amplitude) for one vehicle position p."""
        geo, lam = self.geo, self.geo.lam
        o = geo.p_ris
        out = []
        side = np.sign(p[1])
        d = np.linalg.norm(p - o)
        vis = segment_loss(o, p, self.site.screens, lam)
        out.append((geo.unit(p - o), vis * lam / (4 * np.pi * d) * np.exp(-1j * geo.k0 * d)))
        for f in self.site.facades:
            if np.sign(f.y) != side:
                continue
            img = np.array([p[0], 2 * f.y - p[1], p[2]])
            t = (f.y - o[1]) / (img[1] - o[1])
            r = o + t * (img - o)                                  # reflection point
            if not (f.x1 <= r[0] <= f.x2):
                continue
            dd = np.linalg.norm(img - o)
            a = f.gamma * segment_loss(o, r, self.site.screens, lam) * segment_loss(r, p, self.site.screens, lam)
            out.append((geo.unit(r - o), a * lam / (4 * np.pi * dd) * np.exp(-1j * geo.k0 * dd)))
        for (x, y, z, rcs) in self.site.scatterers:
            if np.sign(y) != side:
                continue
            s = np.array([x, y, z])
            d1, d2 = np.linalg.norm(s - o), np.linalg.norm(p - s)
            a = np.sqrt(rcs) * lam / ((4 * np.pi) ** 1.5 * d1 * d2) * segment_loss(s, p, self.site.screens, lam)
            out.append((geo.unit(s - o), a * np.exp(-1j * geo.k0 * (d1 + d2))))
        return out, vis

    def specular(self, P):
        """Deterministic part of g_k (K, N) and LoS visibility factors."""
        rows, vis = [], []
        for p in P:
            ps, v = self.paths(p)
            rows.append(sum(a * self.geo.a_ris(u) for u, a in ps))
            vis.append(v)
        return np.array(rows), np.array(vis)


class SiteEnv(BeliefEnv):
    """Belief environment on a deterministic site. K_r may vary between episodes."""

    def __init__(self, cfg, site=SITE_A, seed=0, gamma_track=1e-3, **kw):
        super().__init__(cfg, seed=seed, gamma_track=gamma_track, anchor=False, **kw)
        self.site = site
        self.geo.p_ris = np.array([0.0, 0.0, site.ris_z])
        self.sch = SiteChannel(site, self.geo, self.rng)

    def reset(self, eps=None, K_r=None):
        cfg = self.cfg
        if K_r is not None:
            K = cfg.K
            cfg.K_r, cfg.K_t = K_r, K - K_r
        self.zone_v = np.array([0] * cfg.K_r + [1] * cfg.K_t)
        if hasattr(self, "alpha0"):
            del self.alpha0
        obs = super().reset()
        self.alpha0 = self.alpha_e.copy()
        self.g_diff = (self.rng.standard_normal((cfg.K, cfg.N)) + 1j * self.rng.standard_normal((cfg.K, cfg.N))) / np.sqrt(2)
        self._refresh_true()
        return self._belief_tokens()

    def _refresh_true(self):
        super()._refresh_true()                     # G (static), hd (blocked direct link)
        if not hasattr(self, "sch"):
            return
        spec, vis_v = self.sch.specular(self.p_veh)
        pw = np.sqrt(np.mean(np.abs(spec) ** 2, 1, keepdims=True))
        self.g = spec + pw / np.sqrt(self.sch.Kd) * self.g_diff if hasattr(self, "g_diff") else spec
        # echoes travel the LoS path twice: blockage applies on both ways
        if not hasattr(self, "alpha0") or len(self.p_ent) != len(self.alpha0):
            return
        vis_e = np.array([segment_loss(self.geo.p_ris, p, self.site.screens, self.geo.lam) for p in self.p_ent])
        self.vis_e = vis_e
        if True:
            self.alpha_e = self.alpha0 * vis_e ** 2

    def advance(self):
        cfg = self.cfg
        fd = np.abs(self.v_veh[:, 0]) * cfg.fc / C0
        rho = j0(2 * np.pi * fd * self.dT)[:, None]
        self.g_diff = rho * self.g_diff + np.sqrt(1 - rho ** 2) * (
            self.rng.standard_normal(self.g_diff.shape) + 1j * self.rng.standard_normal(self.g_diff.shape)) / np.sqrt(2)
        self.p_ent = self.p_ent + self.v_ent * self.dT
        self.p_veh, self.p_tgt = self.p_ent[: cfg.K], self.p_ent[cfg.K:]
        self._refresh_true()
