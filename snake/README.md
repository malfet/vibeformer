# Snake RL — pixel-based PPO on QBasic Nibbles

Reinforcement-learning experiments on a faithful Python port of
**QBASIC NIBBLES.BAS** (Microsoft, 1990). The plan mirrors the
[../digger-rl](../digger-rl) project's recipe: build pixel-only agents
with PPO + a heuristic teacher for BC bootstrap, learning the lessons
from that project (pixel-from-scratch PPO is hopeless; BC anchor is
the lever).

The eventual goal is to swap the Python sim out for the *real*
nibbles.bas running inside DOSBox-Pure via the libretro binding from
digger-rl. Until then, the Python sim lets us iterate fast.

## TL;DR scoreboard

| Approach | Mean ep score | Notes |
|---|---:|---|
| BFS heuristic teacher | **2100** | 5 eps, level 40+, never game-overs in 20k steps |
| Pixel BC + 1 DAGGER iter (run01) | **-47** | 15k warmup + 15k DAGGER on 84×84 RGB; dies in ~450 steps. Acc on labels 80%+ but stochastic sample at eval can't survive. Suspect snake at 2×2 native cells gets blurred to sub-pixel by the 100→84 area downscale. |

## Layout

| File | Purpose |
| --- | --- |
| `nibbles_sim.py` | Pure game logic. 50x80 arena, 9 level wall layouts, snake mechanics. Faithful to BAS semantics (direction codes, no-reverse rule, length growth on eat, death penalty -10, lives=5). |
| `nibbles_env.py` | `NibblesEnv` (single env, RGB framebuffer, score/lives in info dict) and `NibblesVecEnv` (in-proc for `num_envs=1`, subprocess workers for >1). Same API shape as `digger_env.py`. |
| `tools/heuristic_agent.py` | BFS-to-number teacher with self / wall avoidance. The teacher policy for BC pretrain + DAGGER. |
| `tools/play_human.py` | ncurses front-end — play the sim with arrow keys (Unicode half-blocks render the 50-row arena into 25 terminal rows). |
| `train_bc.py` | First-iteration pixel BC trainer. Warmup with BFS-teacher labels → optional DAGGER iterations → stochastic eval. NatureCNN trunk, shared actor/critic heads (PPO not yet ported). |
| `interaction-log.txt` | Full chronological log of every prompt; the journey. |
| `nibbles/` | (gitignored) QBASIC.EXE + NIBBLES.BAS for the eventual DOSBox-hosted env. |

## Setup

```bash
cd snake
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

PyTorch is included for the eventual PPO trainer; the sim and env only
need numpy + matplotlib.

## How to run (so far)

```bash
# Smoke test the sim
python -c "from nibbles_sim import NibblesGame; g = NibblesGame(rng_seed=0); print(g.head, g.length, g.level)"

# Watch the heuristic play (matplotlib)
python -m tools.heuristic_agent --live

# Play it yourself in the terminal (ncurses, arrow keys, q to quit)
python -m tools.play_human

# First iteration: BC warmup + 1 DAGGER iter, eval 5 episodes
python train_bc.py --warmup-steps 15000 --warmup-epochs 5 \
    --dagger-iters 1 --dagger-collect-steps 15000 --dagger-epochs 5 \
    --eval-eps 5 --env-max-steps 8000 --run-name run01
```

## Open paths

1. **BFS heuristic teacher** — first deliverable; the upper bound for
   per-level score that BC will imitate.
2. **Pixel PPO + BC pretrain** — port `train_ppo.py` from digger-rl with
   the env swapped.
3. **Symbolic obs ablation** — digger-rl saw 3x gains symbolic-vs-pixel;
   worth knowing the ceiling for Nibbles too.
4. **Swap to DOSBox-hosted real nibbles.bas** — once the recipe works on
   the Python sim, reuse `_libretro.cpython-312-darwin.so` from
   digger-rl, point at `nibbles/QBASIC.EXE /RUN NIBBLES.BAS`.

## Key constants worth knowing

- Arena: 50 rows x 80 cols (1-indexed), border walls at row 3 / 50 and col 1 / 80.
- Action space: `{NOOP, UP, DOWN, LEFT, RIGHT}` (BAS direction codes 1=up, 2=down, 3=left, 4=right).
- Snake start length 2, lives 5; eat number n → score += n, length += n*4.
- Eat '9' → next level. Die → -10 score, respawn at level start.
