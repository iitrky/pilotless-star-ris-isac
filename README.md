# Pilotless STAR-RIS ISAC with belief-state reinforcement learning

Code for the paper

> S. Marisargunam and G. Arul Dalton, *Belief State Reinforcement Learning for Pilotless Intelligent Surface Control in Vehicular Sensing and Communication*, submitted to SN Computer Science.

A STAR-RIS (simultaneously transmitting and reflecting reconfigurable intelligent surface) serves vehicles on both sides of a façade while sensors on the surface track them from echoes. No pilots are sent: an extended Kalman filter (EKF) turns echoes into beliefs, a geometric anchor steers the surface, and **HA-TD3** (hierarchical-attention TD3) learns a site-specific residual from those beliefs under a posterior Cramér–Rao bound (PCRB) constraint.

## Repository layout

| Path | Contents |
| --- | --- |
| `hatd3/config.py` | System and agent parameters (Table 5 of the paper) |
| `hatd3/channels.py` | Geometry, Rician channels, STAR-RIS energy-splitting model, closed-form and Fisher-information CRB |
| `hatd3/tracker.py` | Constant-velocity EKF, PCRB recursion, pilot-overhead formula (Theorem 1) |
| `hatd3/env.py`, `hatd3/env_belief.py` | CSI-state environment and the pilotless belief-state environment (time-division ISAC epoch, geometric anchor) |
| `hatd3/controllers_td.py` | Model-based controllers: genie, pilot-based, pilot-free myopic and look-ahead (Algorithm 1 without learning) |
| `hatd3/agent.py`, `action.py`, `reward.py` | HA-TD3 (zone-masked attention, two-timescale actor, twin critics), action layout, PCRB-Lagrangian reward with shaping (Algorithm 2) |
| `hatd3/baselines.py`, `stats.py` | PGA benchmark, statistics (mean, 95% CI) |
| `hatd3/env_rt.py`, `scripts/sionna_export.py` | Ray-traced environment from Sionna RT exports |
| `scripts/` | Training, evaluation, theory checks, system evaluation, figures |
| `matlab/make_trajectories.m` | Vehicle trajectories for the ray-traced scenes |
| `results/` | JSON outputs behind every reported number |
| `figures/` | Fig1–Fig8 of the paper (600 dpi TIFF and PNG) |
| `tests/` | Fast NumPy checks of Lemma 1, Theorem 1, the CRB and Proposition 1 |

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # numpy, scipy, matplotlib, torch; sionna is optional (ray tracing)
pytest -q                           # 8 tests, under 2 s
```

## Reproduce the paper

| Paper item | Command | Output |
| --- | --- | --- |
| Lemma 1, Theorem 1, Theorem 2 (Table 3, Fig. 3, 4, 6) | `python scripts/theory_v2.py` | `results/theory_v2.json` |
| CRB vs Fisher information, Proposition 1 (Fig. 5), link budget | `python scripts/verify_theory.py` | `results/theory_checks.json` |
| Geometric anchor vs perfect CSI | `python scripts/anchor_check.py` | `results/anchor_check.json` |
| Speed sweep (Table 6, Fig. 8), look-ahead test, headroom (Table 7) | `python scripts/system_eval.py` | `results/system_eval.json` |
| All figures | `python scripts/make_figures.py` | `figures/Fig1–Fig8.{tif,png}` |

Each command runs on a CPU in seconds to a few minutes.

### Training HA-TD3 (belief state)

```bash
for s in 0 1 2 3 4; do
  python scripts/train.py --belief --seed $s --out runs/hatd3_s$s                      # proposed
  python scripts/train.py --belief --seed $s --encoder flat_attn --out runs/flat_s$s   # flat attention
  python scripts/train.py --belief --seed $s --encoder mlp       --out runs/mlp_s$s    # MLP-TD3
  python scripts/train.py --belief --seed $s --algo ddpg         --out runs/ddpg_s$s   # DDPG
done
python scripts/train.py --seed 0 --out runs/csi_hatd3_s0                               # CSI-state baseline
```

### Ray tracing

```bash
python scripts/sionna_export.py --scene simple_street_canyon --out rt/train   # training scene
python scripts/sionna_export.py --scene munich --out rt/test                  # unseen test scene
```

## Status

| Part | Status |
| --- | --- |
| Theory checks, model-based controllers, full-system speed sweep, figures, unit tests | Run; outputs in `results/` and `figures/` |
| HA-TD3 and DRL baselines (PyTorch) | Implemented and syntax-checked; training runs for the paper's Tables 8–10 are in progress |
| Sionna RT export and ray-traced environment | Implemented for Sionna RT ≥ 1.0; check API names against your installed version |
| `scripts/evaluate.py` | Evaluates the CSI-state setting; belief-state evaluation is done inside `train.py` (`--belief`) |
| USRP testbed | Hardware specific and not included |

## Citation

See `CITATION.cff`. Please cite the paper if you use this code.

## License

MIT, see `LICENSE`.
