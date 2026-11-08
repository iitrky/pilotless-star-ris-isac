"""Geometry, array responses, statistical channels with Doppler aging, and the CRB.

Coordinate frame: the STAR-RIS lies in the x-z plane (y = 0) with its normal along +y.
y > 0 is the reflection zone (BS side), y < 0 is the transmission zone.
"""
import numpy as np
from scipy.special import j0

C0 = 299_792_458.0


def db2lin(x):
    return 10.0 ** (np.asarray(x) / 10.0)


class Geometry:
    def __init__(self, cfg, rng):
        self.cfg = cfg
        self.lam = C0 / cfg.fc
        self.k0 = 2 * np.pi / self.lam
        d = self.lam / 2
        ix = np.arange(cfg.Nx) - (cfg.Nx - 1) / 2
        iz = np.arange(cfg.Nz) - (cfg.Nz - 1) / 2
        X, Z = np.meshgrid(ix * d, iz * d, indexing="ij")
        self.ris_xz = np.stack([X.ravel(), Z.ravel()], axis=1)       # (N, 2) local offsets
        self.p_ris = np.array([0.0, 0.0, 8.0])
        self.p_bs = np.array([-15.0, 20.0, 12.0])   # ~25 m from the surface
        self.bs_x = np.arange(cfg.M) * d                              # ULA along x
        self.sens_x = (np.arange(cfg.Ns) - (cfg.Ns - 1) / 2) * d     # sensor ULA along x at the surface
        self.rng = rng

    # ---- array responses -------------------------------------------------
    def a_ris(self, u):
        """STAR-RIS UPA response toward unit direction(s) u (..., 3) -> (..., N)."""
        ph = self.ris_xz[:, 0] * u[..., None, 0] + self.ris_xz[:, 1] * u[..., None, 2]
        return np.exp(1j * self.k0 * ph)

    def b_bs(self, u):
        return np.exp(1j * self.k0 * self.bs_x * u[..., None, 0])

    @staticmethod
    def unit(v):
        return v / np.linalg.norm(v, axis=-1, keepdims=True)

    def pathloss(self, d):
        d = np.maximum(d, 1.0)
        return (self.lam / (4 * np.pi)) ** 2 * d ** (-self.cfg.pl_exp)

    def angles_from_ris(self, p):
        """Azimuth phi (from the surface normal, in the horizontal plane) and elevation."""
        u = self.unit(p - self.p_ris)
        phi = np.arctan2(u[..., 0], np.abs(u[..., 1]))
        el = np.arcsin(np.clip(u[..., 2], -1, 1))
        return phi, el

    def a_ris_angle(self, phi, el):
        """Response and its derivative w.r.t. phi (elevation held fixed)."""
        ux = np.sin(phi) * np.cos(el)
        uz = np.sin(el)
        ph = self.ris_xz[:, 0] * ux + self.ris_xz[:, 1] * uz
        a = np.exp(1j * self.k0 * ph)
        adot = 1j * self.k0 * self.ris_xz[:, 0] * np.cos(phi) * np.cos(el) * a
        return a, adot


def sensor_resp(geo, phi, el):
    """Response of the sensor ULA co-located with the STAR-RIS, and d/dphi."""
    ph = geo.sens_x * np.sin(phi) * np.cos(el)
    b = np.exp(1j * geo.k0 * ph)
    bdot = 1j * geo.k0 * geo.sens_x * np.cos(phi) * np.cos(el) * b
    return b, bdot


class Channels:
    """LoS from geometry (recomputed every slot) + Rician NLoS with Jakes Gauss-Markov aging."""

    def __init__(self, geo, rng):
        self.geo, self.cfg, self.rng = geo, geo.cfg, rng
        self.Kr = db2lin(self.cfg.rician_K_dB)
        N, M = self.cfg.N, self.cfg.M
        self.G_nlos = self._cn((N, M))
        self.g_nlos = None
        self.hd_nlos = None

    def _cn(self, shape):
        return (self.rng.standard_normal(shape) + 1j * self.rng.standard_normal(shape)) / np.sqrt(2)

    def bs_ris(self):
        geo = self.geo
        d = np.linalg.norm(geo.p_ris - geo.p_bs)
        u_ris_to_bs = geo.unit(geo.p_bs - geo.p_ris)
        u_bs_to_ris = geo.unit(geo.p_ris - geo.p_bs)
        los = np.outer(geo.a_ris(u_ris_to_bs), geo.b_bs(u_bs_to_ris).conj())
        a = np.sqrt(geo.pathloss(d))
        return a * (np.sqrt(self.Kr / (1 + self.Kr)) * los + np.sqrt(1 / (1 + self.Kr)) * self.G_nlos)

    def init_users(self, K):
        self.g_nlos = self._cn((K, self.cfg.N))
        self.hd_nlos = self._cn((K, self.cfg.M))

    def age(self, speeds):
        """Advance NLoS components by one slot with per-vehicle Jakes correlation."""
        fd = speeds * self.cfg.fc / C0
        rho = j0(2 * np.pi * fd * self.cfg.Ts)[:, None]
        self.g_nlos = rho * self.g_nlos + np.sqrt(1 - rho ** 2) * self._cn(self.g_nlos.shape)
        self.hd_nlos = rho * self.hd_nlos + np.sqrt(1 - rho ** 2) * self._cn(self.hd_nlos.shape)

    def ris_users(self, p_veh):
        """g_k^H rows (K, N): received = g_k^H Theta G x."""
        geo = self.geo
        u = geo.unit(p_veh - geo.p_ris)
        d = np.linalg.norm(p_veh - geo.p_ris, axis=1)
        los = geo.a_ris(u)
        a = np.sqrt(geo.pathloss(d))[:, None]
        return a * (np.sqrt(self.Kr / (1 + self.Kr)) * los + np.sqrt(1 / (1 + self.Kr)) * self.g_nlos)

    def direct(self, p_veh):
        """h_{d,k}^H rows (K, M); heavily attenuated when blocked."""
        geo = self.geo
        u = geo.unit(p_veh - geo.p_bs)
        d = np.linalg.norm(p_veh - geo.p_bs, axis=1)
        los = geo.b_bs(u).conj()
        block = 1e-3 if self.cfg.direct_blocked else 1.0   # ~30 dB blockage loss
        a = np.sqrt(geo.pathloss(d) * block)[:, None]
        return a * (np.sqrt(self.Kr / (1 + self.Kr)) * los + np.sqrt(1 / (1 + self.Kr)) * self.hd_nlos)

    def target_gain(self, p_tgt):
        """Round-trip RIS-target-RIS reflection coefficient alpha_l (radar equation)."""
        geo = self.geo
        d = np.linalg.norm(p_tgt - geo.p_ris, axis=1)
        amp = np.sqrt(geo.lam ** 2 * self.cfg.rcs / ((4 * np.pi) ** 3 * np.maximum(d, 1.0) ** 4))
        return amp * np.exp(1j * self.rng.uniform(0, 2 * np.pi, size=d.shape))


def add_csi_error(h, eps, rng):
    """h_hat = h + Delta, Delta ~ CN(0, eps^2 ||h||^2 / n I), row-wise."""
    if eps <= 0:
        return h.copy()
    n = h.shape[-1]
    p = np.sum(np.abs(h) ** 2, axis=-1, keepdims=True) / n
    e = (rng.standard_normal(h.shape) + 1j * rng.standard_normal(h.shape)) / np.sqrt(2)
    return h + eps * np.sqrt(p) * e


# ---------------------------------------------------------------------------
# Communication and sensing metrics
# ---------------------------------------------------------------------------
def effective_channels(hd, g, G, theta_vec_per_user):
    """h_k^H = hd_k^H + g_k^H diag(theta_k) G. theta_vec_per_user: (K, N) complex coefficients."""
    return hd + (g * theta_vec_per_user) @ G


def sinr_rates(H, W, sigma2):
    """H: (K, M) rows h_k^H; W: (M, K+L) with the first K columns serving vehicles."""
    K = H.shape[0]
    P = np.abs(H @ W) ** 2
    sig = P[np.arange(K), np.arange(K)]
    interf = P.sum(axis=1) - sig
    sinr = sig / (interf + sigma2)
    return sinr, np.log2(1 + sinr)


def echo_matrices(G, theta_vec, a, adot, b=None, bdot=None):
    """B(phi) and dB/dphi.
    b is None  -> BS monostatic via the surface (nu = 2): B = G^T Theta a a^T Theta G
    b given    -> sensors at the surface (nu = 1):        B = b a^T Theta G
    """
    v = (theta_vec * a) @ G                  # a^T Theta G   (M,)
    vd = (theta_vec * adot) @ G
    if b is None:
        B = np.outer(v, v)
        Bd = np.outer(vd, v) + np.outer(v, vd)
    else:
        B = np.outer(b, v)
        Bd = np.outer(bdot, v) + np.outer(b, vd)
    return B, Bd


def crb_angle(G, theta_vec, a, adot, R, alpha, sigma2, T, b=None, bdot=None):
    """Closed-form CRB of phi for echo Y = alpha B(phi) X + Z with unknown complex alpha.

    Point-target result of Liu et al. (IEEE TSP 2022) with B in place of the array response,
    X X^H / T ~ R.
    """
    B, Bd = echo_matrices(G, theta_vec, a, adot, b, bdot)
    t_bb = np.real(np.trace(B.conj().T @ B @ R))
    t_dd = np.real(np.trace(Bd.conj().T @ Bd @ R))
    t_db = np.trace(Bd.conj().T @ B @ R)
    den = t_dd * t_bb - np.abs(t_db) ** 2
    return sigma2 / (2 * T * np.abs(alpha) ** 2) * t_bb / max(den, 1e-300)


def fim_crb_angle(G, theta_vec, a_fun, phi, R, alpha, sigma2, T, b_fun=None, h=1e-6):
    """Reference CRB from the 3x3 FIM over (phi, Re alpha, Im alpha), with dB/dphi by finite
    differences. Used only to verify crb_angle."""
    def Bof(p):
        a, _ = a_fun(p)
        v = (theta_vec * a) @ G
        return np.outer(v, v) if b_fun is None else np.outer(b_fun(p)[0], v)
    B = Bof(phi)
    Bd = (Bof(phi + h) - Bof(phi - h)) / (2 * h)
    # derivatives of the mean mu = alpha B X w.r.t. (phi, Re a, Im a): D_i X
    D = [alpha * Bd, B, 1j * B]
    F = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            F[i, j] = 2 * T / sigma2 * np.real(np.trace(D[i].conj().T @ D[j] @ R))
    return np.linalg.inv(F)[0, 0]
