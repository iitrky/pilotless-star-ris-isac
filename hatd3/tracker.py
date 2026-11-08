"""EKF tracking of road users from surface-sensor echoes, and the PCRB recursion (paper Sec. 3.2).

State x = [px, py, vx, vy]; measurement z = [u, d], with u = sin(phi) cos(el) the horizontal
direction cosine seen from the surface and d the range. The measurement noise covariance is set
each epoch from the CRBs produced by the current configuration (efficient-estimator assumption).
"""
import numpy as np


def cv_matrices(dT, sigma_a):
    I2 = np.eye(2)
    F = np.block([[I2, dT * I2], [np.zeros((2, 2)), I2]])
    Q = sigma_a ** 2 * np.block([[dT ** 4 / 4 * I2, dT ** 3 / 2 * I2], [dT ** 3 / 2 * I2, dT ** 2 * I2]])
    return F, Q


def meas(x, p_ris):
    """z = [u, d] and its Jacobian H (2x4) w.r.t. the state (target height held fixed)."""
    dx, dy, dz = x[0] - p_ris[0], x[1] - p_ris[1], x[2] - p_ris[2] if len(x) > 4 else -7.0
    d = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    u = dx / d
    H = np.zeros((2, 4))
    H[0, 0] = (dy ** 2 + dz ** 2) / d ** 3
    H[0, 1] = -dx * dy / d ** 3
    H[1, 0] = dx / d
    H[1, 1] = dy / d
    return np.array([u, d]), H


class EKFTracker:
    def __init__(self, n_ent, dT, sigma_a, p_ris, z_ent, rng):
        self.F, self.Q = cv_matrices(dT, sigma_a)
        self.p_ris = np.asarray(p_ris, float)
        self.dz = np.asarray(z_ent, float) - self.p_ris[2]       # fixed heights
        self.n = n_ent
        self.rng = rng
        self.x = np.zeros((n_ent, 4))
        self.P = np.zeros((n_ent, 4, 4))
        self.J = np.zeros((n_ent, 4, 4))                           # PCRB information

    def init(self, x_true, pos_std=2.0, vel_std=5.0):
        """Coarse initial detection (no pilots)."""
        P0 = np.diag([pos_std ** 2, pos_std ** 2, vel_std ** 2, vel_std ** 2])
        for e in range(self.n):
            self.x[e] = x_true[e] + self.rng.multivariate_normal(np.zeros(4), P0)
            self.P[e] = P0.copy()
            self.J[e] = np.linalg.inv(P0)

    def _h(self, x, e):
        dx, dy, dz = x[0] - self.p_ris[0], x[1] - self.p_ris[1], self.dz[e]
        d = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        H = np.zeros((2, 4))
        H[0, 0] = (dy ** 2 + dz ** 2) / d ** 3
        H[0, 1] = -dx * dy / d ** 3
        H[1, 0] = dx / d
        H[1, 1] = dy / d
        return np.array([dx / d, d]), H

    def predict(self):
        for e in range(self.n):
            self.x[e] = self.F @ self.x[e]
            self.P[e] = self.F @ self.P[e] @ self.F.T + self.Q
        return self.x.copy(), self.P.copy()

    def update(self, x_true, crb_u, crb_d):
        """Draw z ~ N(h(x_true), diag(crb)) and run the EKF update; also advance the PCRB."""
        pcrb_u = np.zeros(self.n)
        for e in range(self.n):
            R = np.diag([max(crb_u[e], 1e-12), max(crb_d[e], 1e-8)])
            z_true, Ht = self._h(x_true[e], e)
            z = z_true + self.rng.multivariate_normal(np.zeros(2), R)
            zh, H = self._h(self.x[e], e)
            S = H @ self.P[e] @ H.T + R
            Kg = self.P[e] @ H.T @ np.linalg.inv(S)
            self.x[e] = self.x[e] + Kg @ (z - zh)
            self.P[e] = (np.eye(4) - Kg @ H) @ self.P[e]
            # PCRB (Tichavsky et al.), information form, Jacobian at the true state
            self.J[e] = np.linalg.inv(self.Q + self.F @ np.linalg.inv(self.J[e]) @ self.F.T) + Ht.T @ np.linalg.inv(R) @ Ht
            Jinv = np.linalg.inv(self.J[e])
            pcrb_u[e] = Ht[0] @ Jinv @ Ht[0]                      # PCRB of u by the delta method
        return pcrb_u

    def prior_u(self):
        """Predicted direction cosine and its variance (delta method) for every entity."""
        u = np.zeros(self.n)
        var = np.zeros(self.n)
        for e in range(self.n):
            zh, H = self._h(self.x[e], e)
            u[e] = zh[0]
            var[e] = H[0] @ self.P[e] @ H[0]
        return u, var


def pilot_overhead(K, N, M, v, fc, Tsym):
    """Fraction of a Clarke coherence interval consumed by cascaded-channel pilots [Wang et al. 2020]."""
    tau_p = K + N + max(K - 1, int(np.ceil((K - 1) * N / M)))
    Tc = 0.423 * 299_792_458.0 / (max(v, 1e-3) * fc)
    return min(1.0, tau_p * Tsym / Tc), tau_p
