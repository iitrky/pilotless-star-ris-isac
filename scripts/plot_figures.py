"""Paper figures from training logs and evaluation outputs.

  Fig. 2  convergence:  python scripts/plot_figures.py conv  --runs HA-TD3=runs/hatd3_s0,... MLP-TD3=...
  Fig. 9  CSI sweep:    python scripts/plot_figures.py sweep --dirs results/eps_0.0 results/eps_0.1 ...
"""
import argparse, json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def smooth(x, w=25):
    x = np.asarray(x, float)
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")


def conv(args):
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for spec in args.runs:
        name, dirs = spec.split("=")
        curves_r, curves_v = [], []
        for d in dirs.split(","):
            L = [json.loads(l) for l in open(f"{d}/train_log.jsonl")]
            curves_r.append(smooth([x["sum_rate"] for x in L]))
            curves_v.append(smooth([x["viol_rate"] for x in L]))
        n = min(map(len, curves_r))
        R = np.stack([c[:n] for c in curves_r]); V = np.stack([c[:n] for c in curves_v])
        x = np.arange(n)
        for ax, Y in ((axs[0], R), (axs[1], 100 * V)):
            m, s = Y.mean(0), Y.std(0)
            ax.plot(x, m, label=name, lw=1.2)
            ax.fill_between(x, m - s, m + s, alpha=0.2)
    axs[0].set_ylabel("Sum rate (bit/s/Hz)"); axs[1].set_ylabel("CRB violation (%)")
    for ax in axs:
        ax.set_xlabel("Episode"); ax.grid(lw=0.3)
    axs[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(args.out, dpi=300)


def sweep(args):
    data = {}
    for d in args.dirs:
        S = json.load(open(f"{d}/summary.json"))
        for m, v in S["summary"].items():
            data.setdefault(m, []).append((S["eps"], v["sum_rate"], v["viol"]))
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for m, pts in data.items():
        pts.sort()
        e = [p[0] for p in pts]
        axs[0].errorbar(e, [p[1][0] for p in pts], yerr=[[p[1][0] - p[1][1] for p in pts], [p[1][2] - p[1][0] for p in pts]],
                        label=m, capsize=2, lw=1)
        axs[1].plot(e, [100 * p[2][0] for p in pts], marker="o", ms=3, label=m)
    for ax in axs:
        ax.axvspan(0, 0.2, color="0.9", zorder=0); ax.set_xlabel(r"CSI error $\epsilon$"); ax.grid(lw=0.3)
    axs[0].set_ylabel("Sum rate (bit/s/Hz)"); axs[1].set_ylabel("CRB violation (%)"); axs[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(args.out, dpi=300)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["conv", "sweep"])
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--dirs", nargs="*", default=[])
    ap.add_argument("--out", default="fig.png")
    a = ap.parse_args()
    {"conv": conv, "sweep": sweep}[a.kind](a)
