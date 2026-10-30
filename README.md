# Deep RL for Optimal Trading with Partial Information — Replication

PyTorch replication of Macrì, Jaimungal & Lillo (2025), *Deep reinforcement learning for optimal trading with partial information*. An agent trades a mean-reverting (Ornstein–Uhlenbeck) signal whose mean-reversion level switches between hidden regimes. It has to infer the regime from prices while managing inventory.

We implement all three of the paper's GRU-based agents, each with PPO, DDPG and TD3 learners:

| Variant | Idea |
|---|---|
| **hid** | One step: the GRU's hidden state feeds straight into the RL policy and is trained end to end. |
| **prob** | Two steps: a GRU classifier estimates posterior regime probabilities, and the RL policy trades on them. |
| **reg** | Two steps: a GRU regressor forecasts the next signal value, and the RL policy trades on the forecast. |

## Extension: a convex training reward

The paper's per-step reward (Eq. 4) is linear in the trade, `r = I·ΔS − λ|q|`. That puts the optimum at a corner, so agents learn bang-bang policies that slam between ±I_max. We train on a strictly concave version with two extra costs, a temporary-impact cost `−η·q²` and an inventory holding cost `−ψ·I²` (see [`src/env/reward.py`](src/env/reward.py)). This gives a unique interior optimum and smoother positions. Setting `η = ψ = 0` recovers Eq. 4 exactly.

To keep results comparable, **every agent is scored on the paper's Eq. 4 reward**, whatever it was trained on.

## Results

These are out-of-sample cumulative rewards (paper Eq. 4) for PPO agents trained with η = 0.03 and ψ = 5e-4. Each cell is the mean ± std across **5 training seeds**, all evaluated on the same test paths (eval seed 999).

| Variant | Scenario 1 (θ) | Scenario 2 (θ, κ) | Scenario 3 (θ, κ, σ) |
|---|---|---|---|
| prob | 13.50 ± 1.17 | **18.74** ± 1.63 | **32.35** ± 2.14 |
| hid | 12.23 ± 2.88 | 16.77 ± 2.13 | 27.44 ± 1.71 |
| reg | **13.62** ± 1.74 | 15.22 ± 2.97 | 26.48 ± 3.38 |

The two-step **prob** agent leads in Scenarios 2 and 3 and is within noise of the best in Scenario 1. Its edge widens as more parameters switch regime: it is +18% over hid in Scenario 3. The paper reports DDPG trained on the linear reward, so absolute numbers here aren't directly comparable to its Table 4. Per-seed results are in [`results/runs/`](results/runs).

### Holding-cost sweep

Sweeping ψ from 0 to 2e-3 (η = 0.03, 3 seeds per point) costs Eq. 4 reward in every scenario as positions shrink. prob leads in 14 of the 15 scenario × ψ cells; the exception is Scenario 2 at ψ = 5e-4, where all three agents are within noise.

![Eq. 4 reward vs holding-cost ψ](figures/psi_sweep_multiseed.png)

## Repository layout

```
src/
  data/    ou_simulator.py          regime-switching OU price paths
  env/     trading_env.py, reward.py
  models/  gru.py, gru_classifier.py, regressor.py, ppo.py, ddpg.py, td3.py, agents.py
  train/   train_hid.py, train_classifier.py, train_prob.py, train_regressor.py, train_reg.py
  eval/    eval_hid.py, eval_prob.py, eval_reg.py, sweeps and plotting
configs/   one YAML per scenario × variant
scripts/   capstone.sh — full train + eval grid
results/   per-seed CSVs, ψ sweep, sweep logs
figures/   reward curves, P&L distributions, ψ sweep
artifacts/ trained checkpoints (.pt)
```

## Running it

Requires Python 3.8+.

```bash
pip install -r requirements.txt

# Train and evaluate one cell
python3 -m src.train.train_hid --scenario 1 --model ppo --seed 1
python3 -m src.eval.eval_hid  --scenario 1 --model ppo --seed 999

# prob: pre-train the regime classifier, then the RL agent
python3 -m src.train.train_classifier --scenario 1 --theta-only --seed 1
python3 -m src.train.train_prob       --scenario 1 --model ppo --theta-only --seed 1
python3 -m src.eval.eval_prob         --scenario 1 --model ppo --theta-only --seed 999

# reg: pre-train the price regressor, then the RL agent
python3 -m src.train.train_regressor --scenario 1 --seed 1
python3 -m src.train.train_reg       --scenario 1 --model ppo --seed 1
python3 -m src.eval.eval_reg         --scenario 1 --model ppo --seed 999

# Full results grid (3 variants × 3 scenarios × N seeds)
./scripts/capstone.sh 5
```

Use `--model ddpg` or `--model td3` to switch learners.

## Team

This was a four-person project (Summer 2026).

- **Michael Bhang** ([@mbhang2](https://github.com/mbhang2)) built the trading environment and reward, including the convex-reward extension. He also wrote the DDPG actor–critic and training machinery, the PPO agents, the hid pipeline for all three scenarios, the prob classifier pipeline and the reg pipeline.
- **Emre Kırcı** ([@kirciemre](https://github.com/kirciemre)) wrote the initial prototype and restructured the modular `src/` layout and documentation.
- **Shiyu Zhang** contributed the Scenario 1 reg code and made the ψ sweep reproducible.
- **@kikigwq** contributed the Scenario 3 diagnostics.

## Reference

Macrì, A., Jaimungal, S., & Lillo, F. (2025). *Deep reinforcement learning for optimal trading with partial information.*
