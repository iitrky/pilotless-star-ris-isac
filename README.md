# Pilotless STAR-RIS ISAC with belief-state reinforcement learning
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24305221.svg)](https://doi.org/10.5281/zenodo.24305221)
Complete code for the paper

> G. Arul Dalton and S. Marisargunam, *Belief State Reinforcement Learning for Pilotless Intelligent Surface Control in Vehicular Sensing and Communication*, submitted to SN Computer Science.

A simultaneously transmitting and reflecting reconfigurable intelligent surface (STAR-RIS) serves vehicles on both sides of a façade while sensors on the surface track them from radar echoes. **No pilots are sent and no channel is estimated.** An extended Kalman filter (EKF) turns echoes into beliefs, a geometric anchor steers the surface from those beliefs, and **HA-TD3** (hierarchical-attention twin-delayed deep deterministic policy gradient) learns a site-specific correction under a posterior Cramér–Rao bound (PCRB) constraint.

Everything in the paper — theory checks, system simulations, the site model, all learned agents, the evaluation protocol, every table and every figure — is produced by the code in this repository. The learning stack is written in NumPy (with its own small automatic-differentiation module), so **no GPU, PyTorch or ray-tracing licence is needed**.


## Contents

1. [Quick start](#1-quick-start)
2. [Repository structure](#2-repository-structure)
3. [Implementation details](#3-implementation-details)
4. [Hyperparameters](#4-hyperparameters)
5. [Reproducing the paper](#5-reproducing-the-paper)
6. [Main results](#6-main-results)
7. [Compute requirements](#7-compute-requirements)
8. [Scope and limitations](#8-scope-and-limitations)
9. [Citation and licence](#9-citation-and-licence)

---

## 1. Quick start

```bash
git clone https://github.com/iitrky/pilotless-star-ris-isac
cd pilotless-star-ris-isac
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt                         # numpy, scipy, matplotlib, pytest
pytest -q                                               # 9 tests, a few seconds

# evaluate the trained HA-TD3 agent (seed 0) on held-out site-A episodes
python scripts/eval_site.py --run runs/hatd3_s0
python scripts/eval_site.py --baseline model           # pilot-free model-based reference
```

Python 3.9 or later. Trained agents are included in `runs/` and `runs2/`, so every evaluation can be repeated without retraining.

---

## 2. Repository structure

| Path | Contents | Paper |
| --- | --- | --- |
| `hatd3/config.py` | System and simulation parameters (carrier, bandwidth, array sizes, geometry, noise, epochs) | Table 6 |
| `hatd3/channels.py` | Geometry, array responses, Rician channels, STAR-RIS energy-splitting and coupled-phase models, closed-form and Fisher-information CRB | Sec. 3, eq. (2), Prop. 1 |
| `hatd3/tracker.py` | Constant-velocity EKF, PCRB recursion, cascaded-channel pilot length and overhead | Sec. 3.2, Thm. 1 |
| `hatd3/env.py` | CSI-state environment (statistical channel) | Sec. 3 |
| `hatd3/env_belief.py` | Pilotless belief-state environment: time-division ISAC epoch, scan + data sub-slots, echo-driven EKF updates, geometric anchor | Sec. 3.1, 5.4 |
| `hatd3/controllers_td.py` | Model-based controllers: genie, pilot-based, pilot-free myopic and look-ahead scan allocation; expected gain ḡ(σ²) | Lemma 1, Sec. 5.4 |
| `hatd3/site.py` | Deterministic site model: image-method façade reflections, knife-edge blockage (ITU-R P.526), lamp-post scatterers, diffuse component; sites A and B; procedural random sites | Sec. 8.1 |
| `hatd3/site_control.py` | Controllers on the site model, image-source codebook, strongest-path oracle, per-epoch metrics | Sec. 6.3, 8.2 |
| `hatd3/npnn.py` | Minimal reverse-mode automatic differentiation in NumPy (checked against finite differences) | — |
| `hatd3/agent_np.py` | Belief tokens, zone-structured hierarchical attention, hybrid actor, decomposed twin critics, TD3/DDPG updates, replay buffer | Sec. 6, Alg. 2 |
| `hatd3/agent.py`, `action.py`, `reward.py`, `baselines.py` | Earlier PyTorch implementation, kept for reference (not used for reported results) | — |
| `hatd3/env_rt.py`, `scripts/sionna_export.py` | Optional Sionna RT path (not used for reported results) | — |
| `scripts/theory_v2.py`, `scripts/verify_theory.py` | Lemma 1, Theorems 1–2, Proposition 1, CRB vs Fisher information, link budget | Sec. 5 |
| `scripts/system_eval.py`, `scripts/anchor_check.py` | Full-system speed sweep, look-ahead test, headroom, anchor accuracy | Sec. 9.1–9.2 |
| `scripts/train_site.py` | Training of HA-TD3 and all learned baselines/ablations/variants | Alg. 2, Sec. 6.6 |
| `scripts/eval_site.py`, `scripts/site_experiments.py`, `scripts/paired_stats.py` | First evaluation (12 episodes) used for Figs. 10–12 | Sec. 9.4–9.5 |
| `scripts/eval_v2.py`, `scripts/eval_abl_v2.py` | Pre-registered evaluation: 150 held-out episodes, paired tests, Holm correction, bootstrap CIs, sensitivity, ablations | Tables 9–13 |
| `scripts/make_figures.py`, `scripts/plot_site.py` | All figures (600 dpi TIFF + PNG) | Figs. 1–12 |
| `results/protocol_v2.md` | Evaluation protocol, committed **before** the final test runs | Sec. 8.3 |
| `results/*.json` | Raw outputs behind every number in the paper | — |
| `runs/`, `runs2/` | Trained agents (actor and critic weights, training logs) | — |
| `figures/` | Fig1–Fig12 as submitted | — |
| `tests/` | Unit tests: autodiff gradients, Lemma 1, Theorem 1, CRB, Proposition 1 | — |
| `run_all.sh` | Reproduces everything end to end | — |

---

## 3. Implementation details

### 3.1 System and frame (Sec. 3, `channels.py`, `env_belief.py`)
- BS with M = 16 antennas; STAR-RIS with N = N_x × N_z = 32 × 8 elements, energy-splitting protocol Θ_χ = diag(√β_n^χ e^{jθ_n^χ}), β^r + β^t = 1; 16 sensing elements on the surface.
- 7 GHz carrier, 100 MHz bandwidth, 120 kHz subcarrier spacing; epoch ΔT = 5 ms; ISAC block of T = 256 symbols split into a scan sub-block (ρT symbols, all entities illuminated) and K data sub-slots (surface steered at one vehicle each). Echoes of **all** sub-blocks update the tracker.
- Direct BS–vehicle link attenuated by 30 dB; BS beam = dominant right singular vector of the static, calibrated BS–surface channel.

### 3.2 Tracking (Sec. 3.2, `tracker.py`)
- State [x, y, v_x, v_y] per entity, constant-velocity model with manoeuvre noise σ_a = 3 m/s².
- Measurements: direction cosine u and range from the surface, with covariance R_e = diag(CRB_u, CRB_d) computed from the actual echo SNR of each sub-block.
- PCRB recursion (Tichavský et al.) gives the tracking-accuracy constraint PCRB_e^u ≤ Γ (Γ = 10⁻³).

### 3.3 Theory (Sec. 5, `controllers_td.py`, `tracker.py`, `scripts/theory_v2.py`)
- **Lemma 1:** expected normalised array gain under Gaussian direction error, ḡ(σ²) = (1/N_x)[1 + 2Σ_m(1 − m/N_x)e^{−π²m²σ²/2}] (function `gbar`).
- **Theorem 1:** cascaded-channel pilot length τ_p = K + N + max(K − 1, ⌈(K − 1)N/M⌉) and crossover speed v₀ = 0.423 c/(f_c τ_p T_sym) (function `pilot_overhead`).
- **Proposition 1:** CRB ∝ β^−ν with ν = 1 for surface-mounted sensors (checked in `tests/test_core.py`).
- **Theorem 2:** value-iteration test of look-ahead vs myopic sensing allocation.

### 3.4 Site model (Sec. 8.1, `site.py`)
- First-order image-method ray tracing: line of sight, specular reflections from façades whose reflection point lies on the façade, single-bounce paths via lamp posts.
- Every path segment is attenuated by vertical obstacle screens with ITU-R P.526 knife-edge loss J(ν) (trucks/shelters capped at 30/25 dB, tree canopies at 12 dB); the edge (top or side) with the smallest loss is used.
- Diffuse Rician component (K = 10 dB) aged with Jakes' model; echo amplitude attenuated by the same blockage on both ways.
- **Site A** (training), **site B** (unseen: different street width, façade gaps, reflection coefficients and obstacles), and `random_site(seed)` (procedural: façade distance 18–34 m, 0–2 gaps, Γ 0.35–0.75, 2–3 trucks and 1–4 trees per side).

### 3.5 HA-TD3 (Sec. 6, `agent_np.py`, `site_control.py`)
- **Belief tokens** per entity: predicted state, log-variances, correlation coefficients, predicted u and range, log(PCRB/Γ), multiplier, zone and type flags, reported rate, sinusoidal position encoding (periods 1.5–48 m), predicted direction, and the **measured echo attenuation** (echo amplitude relative to the free-space radar equation, 1 dB error). No channel coefficients.
- **Encoder:** type-specific MLPs → zone-masked self-attention (level 1) → attention pooling per zone → level-2 attention over {zone r, zone t, global} → context c. Width d = 32.
- **Hybrid action:**
  - per vehicle, a discrete **residual codebook** entry: j = 0 is the geometric anchor (predicted line of sight), j = 1…6 are virtual sources mirrored across planes at 16, 20, …, 36 m (generic street-canyon prior, no site map), chosen greedily from the per-vehicle Q-values;
  - a continuous **slow head** that corrects the occupancy rule β^r = K_r/K by up to ±0.15 every T_h = 10 epochs, projected to [0.15, 0.85].
- **Critics:** twin critics with decomposed value Q = Σ_k q_k(s, j_k, b) + q_g(s, b); q_k trained on vehicle k's own rate, q_g on the sensing reward; clipped double Q-learning with target-policy smoothing on the slow action.
- **Reward:** r_k = (K/R_ref)·R_k, r_g = −Σ_c λ_c(mean PCRB_c/Γ − 1) + potential-based shaping; multipliers by projected dual ascent once per episode.
- **Baselines:** MLP-TD3 (flattened tokens), flat self-attention TD3, DDPG (single critic, no smoothing, no delay) — same tokens, codebook, anchor and budget.
- **Deployment variants (Sec. 6.6):** HA-TD3-R (new random site every training episode) and HA-TD3-FT (site-A agent fine-tuned for 100 episodes at site B).

### 3.6 Evaluation protocol (Sec. 8.3, `results/protocol_v2.md`, `scripts/eval_v2.py`)
- Test sets with seeds disjoint from training: **T_A** (60 episodes, site A), **T_B** (60, site B), **T_C** (30 episodes on 30 unseen random sites); 40 epochs per episode at 120 km/h, first 5 excluded.
- Primary hypothesis H1 (HA-TD3 > model-based on T_A) and eight secondary comparisons with Holm–Bonferroni correction; two-sided Wilcoxon signed-rank tests on paired per-episode differences; 95% bootstrap CIs (10 000 resamples).
- Sensitivity: speed 90/150 km/h, echo error 3 dB, diffuse K = 5 dB.

---

## 4. Hyperparameters

| Item | Value |
| --- | --- |
| Encoder width d; attention heads; hidden width | 32; 1; 64 |
| Codebook | anchor + 6 image planes at 16–36 m |
| Optimiser | Adam; actor 3 × 10⁻⁴, critic 10⁻³; gradient-norm clip 10 |
| Discount γ_d; target rate τ; policy delay | 0.5; 0.01; 2 |
| Target smoothing noise; clip | 0.2; 0.5 |
| Replay buffer; minibatch; updates per step | 10⁵; 64; 1 |
| Episodes × epochs; warm-up | 400 × 40; 1000 transitions (random actions) |
| Exploration | ε over codebook entries from 0.5 to 0.2; Gaussian noise 0.1 on the slow head |
| Slow period T_h; β bound | 10 epochs; [0.15, 0.85] |
| Multiplier step η_λ; cap | 0.5; 10 |
| Seeds | 0, 1, 2 (ablations: 0) |

---

## 5. Reproducing the paper

**Everything at once** (about 4 hours on a 2-core CPU, most of it training):
```bash
bash run_all.sh
```

**Step by step:**

| Paper item | Command | Output |
| --- | --- | --- |
| Lemma 1 (Fig. 3), Theorem 1 (Table 3, Fig. 4), Theorem 2 (Table 4, Fig. 6) | `python scripts/theory_v2.py` | `results/theory_v2.json` |
| CRB vs Fisher information, Proposition 1 (Fig. 5), link budget | `python scripts/verify_theory.py` | `results/theory_checks.json` |
| Geometric anchor vs perfect CSI | `python scripts/anchor_check.py` | `results/anchor_check.json` |
| Speed sweep, look-ahead test, headroom (Tables 7–8, Fig. 9) | `python scripts/system_eval.py` | `results/system_eval.json` |
| Train all learned agents (19 runs) | `bash run_site_training.sh 2` | `runs/` |
| Train HA-TD3-R and HA-TD3-FT | see `run_all.sh` (step 4) | `runs2/` |
| First evaluation, Γ sweep (Fig. 11), training curves (Fig. 12), learned behaviour (Fig. 10) | `python scripts/site_experiments.py && python scripts/plot_site.py` | `results/site_results.json`, `figures/Fig10–12` |
| Pre-registered evaluation (Tables 9, 11, 12, 13) | `python scripts/eval_v2.py` | `results/eval_v2.json` |
| Ablations on T_A, K = 10, coupled phases (Table 10) | `python scripts/eval_abl_v2.py` | `results/eval_abl_v2.json` |
| Figures 1–9 | `python scripts/make_figures.py` | `figures/Fig1–Fig9` |

Single agent:
```bash
python scripts/train_site.py --encoder hier --seed 0 --out runs/hatd3_s0           # HA-TD3, ~11 min
python scripts/train_site.py --site random --seed 0 --out runs2/hatd3R_s0           # site-randomised
python scripts/train_site.py --site B --init runs/hatd3_s0 --episodes 100 \
       --warmup 200 --seed 10 --out runs2/hatd3FT_s0                                # fine-tuning
python scripts/train_site.py --no-anchor --seed 0 --out runs/abl_noanchor_s0        # an ablation
```
Ablation flags: `--no-level2`, `--no-two-timescale`, `--no-anchor`, `--no-shaping`, `--no-lagrangian`, `--no-beta-bound`, `--coupled`; baselines: `--encoder flat`, `--encoder mlp`, `--algo ddpg`.

All runs are seeded; results are reproducible on the same platform up to floating-point differences.

---

## 6. Main results

On the 60 held-out site-A episodes (protocol fixed in advance), at 120 km/h with N = 256:

| Method | Throughput (bit/s/Hz) | 5th-percentile vehicle rate |
| --- | --- | --- |
| Genie (perfect CSI, upper bound) | 5.62 | 0.57 |
| Pilot-based (pilots exceed the coherence time) | 0 | 0 |
| Pilot-free model-based | 4.58 | 0.21 |
| Oracle strongest-path steering (bound) | 4.85 | 0.35 |
| MLP-TD3 / flat-attention TD3 / DDPG | 4.58 / 4.75 / 4.76 | 0.23 / 0.31 / 0.31 |
| **HA-TD3** | **4.79** (+4.5%, p < 0.001) | **0.31** (+48%) |

HA-TD3 obtains 76% of the oracle's steering gain from beliefs alone. The learned correction does **not** transfer to unseen sites (site B −1.4%, 30 random sites −0.8%); site-randomised training does not fix this, and 100 episodes of on-site fine-tuning restore parity with model-based control. Full statistics: `results/eval_v2.json`.

---

## 7. Compute requirements

- CPU only. Training one agent: 7–12 minutes on one core (400 episodes × 40 epochs); fine-tuning: under 3 minutes.
- Pre-registered evaluation: about 40 minutes; ablation evaluation: about 15 minutes.
- Memory: under 2 GB. Disk: about 45 MB including trained agents.

---

## 8. Scope and limitations

- All learning results use the deterministic site model implemented here (first-order image method with knife-edge blockage), not measured channels or full ray tracing. The optional Sionna RT path (`env_rt.py`, `sionna_export.py`) and the PyTorch agent (`agent.py`, `train.py`) are earlier implementations kept for reference and were not used for the reported results.
- No hardware testbed is part of this study.

---

## 9. Citation and licence

If you use this code, please cite the paper (see `CITATION.cff`). Released under the MIT licence (`LICENSE`).

Contact: G. Arul Dalton (corresponding author), aruldalton@saveetha.ac.in
