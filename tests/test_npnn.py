"""Finite-difference check of the NumPy autodiff used by the HA-TD3 networks."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hatd3.npnn import T, Param, Linear, concat


def test_gradients_match_finite_differences():
    rng = np.random.default_rng(0)
    x = T(rng.standard_normal((3, 5, 4)))
    mask = rng.random((3, 5, 5)) > 0.3
    mask[:, np.arange(5), np.arange(5)] = True
    l1, l2 = Linear(4, 6, rng), Linear(6, 2, rng)
    W = Param(rng.standard_normal((6, 6)) * 0.3)

    def f():
        h = l1(x).tanh()
        a = (h @ W @ h.T()).softmax(mask)
        z = concat([(a @ h).relu(), h], -1)[:, 1:4, :6]
        return (l2(z) * l2(z)).mean() + z.sum(1).mean()

    params = l1.params() + l2.params() + [W]
    out = f(); out.backward()
    for p in params:
        g = p.g.copy()
        num = np.zeros_like(p.v)
        it = np.nditer(p.v, flags=["multi_index"])
        for _ in it:
            i = it.multi_index
            old = p.v[i]
            p.v[i] = old + 1e-6; fp = f().v
            p.v[i] = old - 1e-6; fm = f().v
            p.v[i] = old
            num[i] = (fp - fm) / 2e-6
        assert np.max(np.abs(num - g)) < 1e-6 * max(1, np.max(np.abs(num))), p.v.shape
