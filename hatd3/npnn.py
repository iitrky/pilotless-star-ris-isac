"""Minimal reverse-mode automatic differentiation in NumPy (no PyTorch needed).

Supports the operations used by the HA-TD3 networks: batched matmul with broadcasting,
elementwise add/mul with broadcasting, relu, tanh, masked softmax, sum/mean, transpose and
slicing. Gradients are checked against finite differences in tests/test_npnn.py.
"""
import numpy as np


def _unbroadcast(g, shape):
    while g.ndim > len(shape):
        g = g.sum(0)
    for i, s in enumerate(shape):
        if s == 1 and g.shape[i] != 1:
            g = g.sum(i, keepdims=True)
    return g


class T:
    __slots__ = ("v", "g", "parents", "back", "req")

    def __init__(self, v, parents=(), back=None, req=False):
        self.v = np.asarray(v, dtype=np.float64)
        self.g = None
        self.parents = parents
        self.back = back
        self.req = req or any(p.req for p in parents)

    # ----------------------------------------------------------------- ops
    def __add__(self, o):
        o = o if isinstance(o, T) else T(o)
        out = T(self.v + o.v, (self, o))
        out.back = lambda g: (_unbroadcast(g, self.v.shape), _unbroadcast(g, o.v.shape))
        return out

    __radd__ = __add__

    def __sub__(self, o):
        o = o if isinstance(o, T) else T(o)
        return self + o * (-1.0)

    def __mul__(self, o):
        o = o if isinstance(o, T) else T(o)
        out = T(self.v * o.v, (self, o))
        out.back = lambda g: (_unbroadcast(g * o.v, self.v.shape), _unbroadcast(g * self.v, o.v.shape))
        return out

    __rmul__ = __mul__

    def __matmul__(self, o):
        out = T(self.v @ o.v, (self, o))

        def back(g):
            ga = g @ np.swapaxes(o.v, -1, -2)
            gb = np.swapaxes(self.v, -1, -2) @ g
            return _unbroadcast(ga, self.v.shape), _unbroadcast(gb, o.v.shape)
        out.back = back
        return out

    def relu(self):
        m = self.v > 0
        out = T(self.v * m, (self,))
        out.back = lambda g: (g * m,)
        return out

    def tanh(self):
        y = np.tanh(self.v)
        out = T(y, (self,))
        out.back = lambda g: (g * (1 - y ** 2),)
        return out

    def softmax(self, mask=None):
        """Softmax over the last axis; mask (bool, broadcastable) marks allowed entries."""
        x = self.v if mask is None else np.where(mask, self.v, -1e9)
        x = x - x.max(-1, keepdims=True)
        e = np.exp(x)
        y = e / e.sum(-1, keepdims=True)
        out = T(y, (self,))
        out.back = lambda g: (y * (g - (g * y).sum(-1, keepdims=True)),)
        return out

    def sum(self, axis=None, keepdims=False):
        out = T(self.v.sum(axis=axis, keepdims=keepdims), (self,))

        def back(g):
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            return (np.broadcast_to(g, self.v.shape).copy(),)
        out.back = back
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.v.size if axis is None else self.v.shape[axis]
        return self.sum(axis, keepdims) * (1.0 / n)

    def T(self):
        out = T(np.swapaxes(self.v, -1, -2), (self,))
        out.back = lambda g: (np.swapaxes(g, -1, -2),)
        return out

    def __getitem__(self, idx):
        out = T(self.v[idx], (self,))

        def back(g):
            z = np.zeros_like(self.v)
            z[idx] += g
            return (z,)
        out.back = back
        return out

    # ----------------------------------------------------------------- backprop
    def backward(self, g=None):
        topo, seen = [], set()

        def visit(t):
            if id(t) in seen or not t.req:
                return
            seen.add(id(t))
            for p in t.parents:
                visit(p)
            topo.append(t)
        visit(self)
        self.g = np.ones_like(self.v) if g is None else g
        for t in reversed(topo):
            if t.back is None or t.g is None:
                continue
            for p, gp in zip(t.parents, t.back(t.g)):
                if p.req:
                    p.g = gp if p.g is None else p.g + gp


def concat(ts, axis=-1):
    out = T(np.concatenate([t.v for t in ts], axis), tuple(ts))
    sizes = np.cumsum([t.v.shape[axis] for t in ts])[:-1]
    out.back = lambda g: tuple(np.split(g, sizes, axis))
    return out


class Param(T):
    def __init__(self, v):
        super().__init__(v, req=True)


class Module:
    def params(self):
        out = []
        for v in self.__dict__.values():
            if isinstance(v, Param):
                out.append(v)
            elif isinstance(v, Module):
                out += v.params()
            elif isinstance(v, (list, tuple)):
                for x in v:
                    if isinstance(x, Module):
                        out += x.params()
        return out

    def state(self):
        return [p.v.copy() for p in self.params()]

    def load(self, st):
        for p, v in zip(self.params(), st):
            p.v = v.copy()


class Linear(Module):
    def __init__(self, i, o, rng, scale=1.0, bias=True):
        self.W = Param(rng.standard_normal((i, o)) * scale * np.sqrt(2.0 / (i + o)))
        self.b = Param(np.zeros(o)) if bias else None

    def __call__(self, x):
        y = x @ self.W
        return y + self.b if self.b is not None else y


class MLP(Module):
    def __init__(self, sizes, rng, out_scale=1.0):
        self.layers = [Linear(a, b, rng, scale=(out_scale if k == len(sizes) - 2 else 1.0))
                       for k, (a, b) in enumerate(zip(sizes[:-1], sizes[1:]))]

    def __call__(self, x):
        for k, l in enumerate(self.layers):
            x = l(x)
            if k < len(self.layers) - 1:
                x = x.relu()
        return x


class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, clip=10.0):
        self.p, self.lr, self.b1, self.b2, self.eps, self.clip = params, lr, b1, b2, eps, clip
        self.m = [np.zeros_like(p.v) for p in params]
        self.s = [np.zeros_like(p.v) for p in params]
        self.t = 0

    def zero(self):
        for p in self.p:
            p.g = None

    def step(self):
        self.t += 1
        gs = [np.zeros_like(p.v) if p.g is None else p.g for p in self.p]
        n = np.sqrt(sum(float((g ** 2).sum()) for g in gs))
        c = min(1.0, self.clip / (n + 1e-12))
        for i, (p, g) in enumerate(zip(self.p, gs)):
            g = g * c
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.s[i] = self.b2 * self.s[i] + (1 - self.b2) * g * g
            mh = self.m[i] / (1 - self.b1 ** self.t)
            sh = self.s[i] / (1 - self.b2 ** self.t)
            p.v = p.v - self.lr * mh / (np.sqrt(sh) + self.eps)
        self.zero()
