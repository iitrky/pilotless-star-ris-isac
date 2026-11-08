"""Action-vector layout and projection onto the feasible set (NumPy only)."""
import numpy as np


class ActionLayout:
    def __init__(self, scfg):
        M, N, K, L = scfg.M, scfg.N, scfg.K, scfg.L
        self.M, self.N, self.K, self.L = M, N, K, L
        self.nW = 2 * M * (K + L)
        self.sl_W = slice(0, self.nW)
        self.sl_pr = slice(self.nW, self.nW + 2 * N)
        self.sl_pt = slice(self.nW + 2 * N, self.nW + 4 * N)
        self.sl_slow = slice(self.nW + 4 * N, self.nW + 5 * N + 1)
        self.D = self.nW + 5 * N + 1
        self.D_slow = N + 1

    def to_physical(self, a, P, beta_lo=0.0, beta_hi=1.0):
        """numpy a (D,) -> W (M, K+L) complex, beta_r (N), theta_r, theta_t (N), kappa."""
        M, N, K, L = self.M, self.N, self.K, self.L
        w = a[self.sl_W].reshape(K + L, 2, M)
        W = (w[:, 0] + 1j * w[:, 1]).T                       # (M, K+L)
        pr = a[self.sl_pr].reshape(N, 2)
        pt = a[self.sl_pt].reshape(N, 2)
        th_r = np.arctan2(pr[:, 1], pr[:, 0] + 1e-9)
        th_t = np.arctan2(pt[:, 1], pt[:, 0] + 1e-9)
        s = a[self.sl_slow]
        beta_r = beta_lo + (beta_hi - beta_lo) * (s[:N] + 1) / 2
        kappa = 0.05 + 0.9 * (s[N] + 1) / 2                  # keep both functions alive
        Wc, Ws = W[:, :K], W[:, K:]
        Wc = Wc * np.sqrt(kappa * P) / (np.linalg.norm(Wc) + 1e-12)
        Ws = Ws * np.sqrt((1 - kappa) * P) / (np.linalg.norm(Ws) + 1e-12)
        return np.concatenate([Wc, Ws], axis=1), beta_r, th_r, th_t, kappa
