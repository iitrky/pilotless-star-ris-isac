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
| `hatd3/site.py`, `hatd3/site_control.py` | Deterministic site model (image-method reflections, knife-edge blockage, two sites) and the controllers evaluated on it |
| `hatd3/npnn.py`, `hatd3/agent_np.py` | NumPy autodiff and the HA-TD3 / flat-attention / MLP / DDPG agents used for all reported learning results |
| `hatd3/env_rt.py`, `scripts/sionna_export.py` | Optional Sionna RT path (not used for the reported results) |
| `scripts/` | Training, evaluation, theory checks, system evaluation, figures |
| `matlab/make_trajectories.m` | Vehicle trajectories for the ray-traced scenes |
| `results/` | JSON outputs behind every reported number |
| `figures/` | Fig1–Fig8 of the paper (600 dpi TIFF and PNG) |
| `tests/` | Fast NumPy checks of Lemma 1, Theorem 1, the CRB and Proposition 1 |

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # numpy, scipy, matplotlib (torch and sionna optional)
pytest -q                           # 9 tests, a few seconds
```

## Reproduce the paper

| Paper item | Command | Output |
| --- | --- | --- |
| Lemma 1, Theorem 1, Theorem 2 (Table 3, Fig. 3, 4, 6) | `python scripts/theory_v2.py` | `results/theory_v2.json` |
| CRB vs Fisher information, Proposition 1 (Fig. 5), link budget | `python scripts/verify_theory.py` | `results/theory_checks.json` |
| Geometric anchor vs perfect CSI | `python scripts/anchor_check.py` | `results/anchor_check.json` |
| Speed sweep (Table 6, Fig. 8), look-ahead test, headroom (Table 7) | `python scripts/system_eval.py` | `results/system_eval.json` |
| Learned controllers, baselines, ablations (Tables 8–9, Figs. 9–11) | `bash run_site_training.sh 2` | `runs/`, `results/site_results.json`, `figures/Fig9–Fig11` |
| Figures 1–8 | `python scripts/make_figures.py` | `figures/Fig1–Fig8.{tif,png}` |

Each command runs on a CPU in seconds to a few minutes.

### Training HA-TD3 on the site model

```bash
python scripts/train_site.py --encoder hier --seed 0 --out runs/hatd3_s0     # proposed (about 11 min on one CPU core)
python scripts/eval_site.py --run runs/hatd3_s0                              # held-out site-A episodes
python scripts/eval_site.py --run runs/hatd3_s0 --site B                     # unseen site
python scripts/eval_site.py --baseline model                                 # pilot-free model-based reference
```
Everything runs on NumPy; PyTorch is not needed for the reported results. `run_site_training.sh` trains all 19 runs (3 seeds of four learners and 7 ablations) and then writes `results/site_results.json` and Figs. 9–11.

## Status

| Part | Status |
| --- | --- |
| Theory checks, statistical-model system results, site-model learning results, figures, unit tests | Run; outputs in `results/`, `runs/` and `figures/` |
| PyTorch agent (`hatd3/agent.py`, `scripts/train.py`) and Sionna RT path | Earlier implementation, kept for reference; not used for the reported results |
| USRP testbed | Not part of this study |

## Citation

See `CITATION.cff`. Please cite the paper if you use this code.

## License

MIT, see `LICENSE`.
