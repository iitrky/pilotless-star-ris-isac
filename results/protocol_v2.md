# Evaluation protocol v2 (fixed before any v2 result was computed)

Written on 17 Nov 2026, before training the new agents or evaluating any method on the v2 test sets.
Committed to the repository before the runs; the analysis follows it without change.

## Test sets (all disjoint from every training and v1 test seed)
- **T_A**: site A, 60 episodes, episode seeds 950000–950059, K_r = 2, 3, 4 cycled, all vehicles at 120 km/h, 40 epochs, first 5 excluded.
- **T_B**: unseen site B, 60 episodes, seeds 960000–960059, otherwise as T_A.
- **T_C**: 30 held-out procedurally generated sites (site seeds 7000000–7000029, never used in training), one episode each, seeds 970000–970029.

## Methods
Genie, pilot-free model-based, look-ahead model-based, oracle strongest-path steering,
reflect + transmit RIS, MLP-TD3, flat-attention TD3, DDPG, HA-TD3 (site A), and two new variants:
- **HA-TD3-R**: trained on a new random site every episode (site randomisation), 3 seeds, 400 episodes.
- **HA-TD3-FT**: the three site-A HA-TD3 agents fine-tuned on site B for 100 episodes.

## Primary hypothesis
H1: HA-TD3 (site A) achieves higher mean throughput than pilot-free model-based control on T_A.
Test: two-sided Wilcoxon signed-rank test on per-episode differences (learned results averaged over seeds), alpha = 0.05.

## Secondary comparisons (Holm–Bonferroni across the family, alpha = 0.05)
1. HA-TD3 vs model-based, 5th-percentile vehicle rate, T_A.
2. HA-TD3 vs flat-attention TD3, T_A.  3. HA-TD3 vs DDPG, T_A.  4. HA-TD3 vs MLP-TD3, T_A.
5. HA-TD3-R vs model-based, T_B.  6. HA-TD3-R vs model-based, T_C.  7. HA-TD3-FT vs model-based, T_B.
8. HA-TD3-R vs model-based, T_A.

Effect sizes: mean paired difference with 95% bootstrap CI (10 000 resamples) and relative change.

## Sensitivity (descriptive, first 30 episodes of T_A)
Speed 90 and 150 km/h; echo-amplitude error 3 dB instead of 1 dB; diffuse Rician K = 5 dB instead of 10 dB.
