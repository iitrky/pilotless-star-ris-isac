"""Fast NumPy-only checks of the analytical results used in the paper."""
import os, sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hatd3.config import SysConfig
from hatd3.env import StarRisIsacEnv
from hatd3.channels import crb_angle, fim_crb_angle, sensor_resp
from hatd3.controllers_td import gbar
from hatd3.tracker import pilot_overhead


@pytest.mark.parametrize("Nx,sigma", [(16, 0.01), (32, 0.02), (64, 0.005)])
def test_lemma1_monte_carlo(Nx, sigma):
    rng = np.random.default_rng(0)
    du = rng.normal(0, sigma, 100_000)
    n = np.arange(Nx)
    g = np.abs(np.exp(1j * np.pi * np.outer(du, n)).sum(1)) ** 2 / Nx ** 2
    assert abs(g.mean() - gbar(sigma ** 2, Nx)) / g.mean() < 0.01


def test_theorem1_pilot_length_and_crossover():
    _, tau = pilot_overhead(K=6, N=64, M=16, v=30 / 3.6, fc=7e9, Tsym=8.93e-6)
    assert tau == 90
    v0 = 0.423 * 299_792_458.0 / (7e9 * tau * 8.93e-6) * 3.6
    assert 80 < v0 < 82           # km/h: the 81 km/h pilot wall for N = 64
    frac, _ = pilot_overhead(K=6, N=256, M=16, v=120 / 3.6, fc=7e9, Tsym=8.93e-6)
    assert frac == 1.0            # 256 elements: no data time left at 120 km/h


@pytest.mark.parametrize("mode", ["ris_sensors", "bs"])
def test_crb_closed_form_matches_fim(mode):
    rng = np.random.default_rng(1)
    cfg = SysConfig(sensing_mode=mode)
    env = StarRisIsacEnv(cfg, seed=3)
    env.reset(eps=0.0)
    th = np.sqrt(rng.uniform(0.1, 1, cfg.N)) * np.exp(1j * rng.uniform(0, 2 * np.pi, cfg.N))
    A = rng.standard_normal((cfg.M, cfg.M)) + 1j * rng.standard_normal((cfg.M, cfg.M))
    R = A @ A.conj().T
    R *= env.P / np.real(np.trace(R))
    phi, el = env.geo.angles_from_ris(env.p_tgt)
    a, ad = env.geo.a_ris_angle(phi[0], el[0])
    b, bd = (None, None) if mode == "bs" else sensor_resp(env.geo, phi[0], el[0])
    c1 = crb_angle(env.G, th, a, ad, R, env.alpha[0], env.sigma2, cfg.T_snap, b, bd)
    bfun = None if mode == "bs" else (lambda p: sensor_resp(env.geo, p, el[0]))
    c2 = fim_crb_angle(env.G, th, lambda p: env.geo.a_ris_angle(p, el[0]), phi[0], R,
                       env.alpha[0], env.sigma2, cfg.T_snap, bfun, h=1e-7)
    assert abs(c1 - c2) / c2 < 1e-6


@pytest.mark.parametrize("mode,nu", [("ris_sensors", 1), ("bs", 2)])
def test_proposition1_crb_scales_beta_minus_nu(mode, nu):
    rng = np.random.default_rng(2)
    cfg = SysConfig(sensing_mode=mode)
    env = StarRisIsacEnv(cfg, seed=5)
    env.reset(eps=0.0)
    ph = np.exp(1j * rng.uniform(0, 2 * np.pi, cfg.N))
    R = np.eye(cfg.M) * env.P / cfg.M
    phi, el = env.geo.angles_from_ris(env.p_tgt)
    a, ad = env.geo.a_ris_angle(phi[0], el[0])
    b, bd = (None, None) if mode == "bs" else sensor_resp(env.geo, phi[0], el[0])
    betas = np.array([0.1, 0.3, 0.9])
    crbs = [crb_angle(env.G, np.sqrt(bt) * ph, a, ad, R, env.alpha[0], env.sigma2, cfg.T_snap, b, bd) for bt in betas]
    slope = np.polyfit(np.log(betas), np.log(crbs), 1)[0]
    assert abs(slope + nu) < 1e-3
