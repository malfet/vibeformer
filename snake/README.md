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
| Pixel BC + 1 DAGGER iter (run01) | **-47** | 15k warmup + 15k DAGGER on 84×84 RGB (area downscale). Train acc 81% but eval mean -47; snake invisible after area-interp blurs 2×2 cells to sub-pixel. |
| Pixel BC + 1 DAGGER iter (run02) | **-46** | Same as run01 but obs_size=168 + nearest downscale. Train acc 88%, much sharper probabilities. Snake plays first 15-20 steps perfectly with >0.95 confidence, then degrades after body grows long. Late-game routing is the open problem. |
| Pixel BC + 1 DAGGER iter (run03) | **-50** | Same as run02 but with the `NibblesEnv` seed-diversity bug fixed (every reset now draws a fresh game seed). Train acc 87% (down from 90% — fitting more diverse data is harder). Eval went WORSE: never ate, all 5 lives lost from random navigation. The previous runs were memorizing one specific trajectory; diversity exposed lack of generalization. |
| Symbolic BC + 1 DAGGER iter (run04) | **-49** | Same hyperparams as run03 but feeding the (5, 50, 80) one-hot symbolic obs (cell-type grid: empty/wall/body/head/number) through a small NatureCNN (477k params). Train acc 76%, eval no better than pixel. Probing shows the model picks UP when the number is literally one cell to the right of the head — encoder gets the spatial info in its receptive field but BC over 40k samples doesn't teach navigation. **The bottleneck is BC's data appetite, not the obs encoding.** |
| Tiny-snake + plain BC (run05) | **+3** | New 12x12 textbook snake env with 3-action relative space (STRAIGHT/LEFT/RIGHT) — no NOOP, no reverse-direction ambiguity. Symbolic (5, 12, 12) one-hot in, 3 logits out. 10k warmup + 5k DAGGER, 4.78M params. Big leap: student actually eats food (0-7 per episode), vs teacher ~31. The win was env simplification + 3-action space, not network architecture. |
| Tiny-snake + MC-credit BC (run06) | **+4** | Same as run05 but each training sample weighted by `next_positive_reward / steps_to_event` (user's "retroactive uniform credit"). Min score 2 (vs 0 for plain) — fewer zero-eat episodes. Modest gain on top of plain BC. |
| Tiny-snake + smaller net (runs 07-09) | **0-1** | Tried encoder_width=0.5 (1.2M params) and stride-2 conv downsampling (1.24M params). Both collapse to the modal-class policy (always STRAIGHT). MC-weighted CE is sparse enough that ~1.2M params lacks the redundancy to learn navigation under noisy gradients. **Empirically the 4.78M FC is the right size for this task at this data scale.** |
| Tiny-snake + discounted MC (run10) | **0** | Variant where credit = sum γ^k r — produces negative weights for death-leading trajectories; weighted CE then **anti-imitates** the teacher at those samples. Blew up (CE→-7.4, immediate-death policy). uniform-pos's "deaths get zero weight" is the correct shape. |
| **Tiny-snake + PPO + BC anchor (ppo01)** | **8.2** | Loaded the run06 BC checkpoint, ran PPO for 200k steps with `--bc-anchor-coef 0.5` (extra CE on teacher's action mixed into each PPO minibatch). **2× over BC** — first time RL's reward signal beats the BC ceiling. Peak eval mean 10.1 at update 775, final 8.2 (10-ep eval is noisy). KL stayed ~0.001, anchor CE ~0.1: BC anchor + small clip kept the policy close to teacher behavior while reward shaping pulled it higher. |
| **Tiny-snake + PPO, anneal anchor 0.5→0 (ppo02)** | **10.6** | Same as ppo01 but `--bc-anchor-final 0.0`: BC weight decays linearly to zero by training end. Best final result — the policy starts anchored to teacher behavior, then PPO is free to push past it. Peak mean 11.0 at update 700. |
| Tiny-snake + PPO, ent_coef=0.05 (ppo03) | 10.0 | Same as ppo01 but 5× the entropy bonus. **Highest peak** (mean 11.9 at update 675) but noisier — extra exploration finds better basins but doesn't stay there. Would pair well with a `--save-best` checkpoint sweep. |
| Tiny-snake + PPO from scratch, width=0.25 (ppo04) | 1.4 | 78k-param agent, no BC init, ent_coef=0.05, 300k steps. Final eval 1.4 (max 3). Cannot learn navigation from cold PPO at this capacity — but one episode survived 409 steps with 0 food, so it *did* find a "don't die" local optimum. Confirms the small-network finding from runs 07-09 holds even with reward signal: the BC checkpoint is doing the heavy lifting in ppo01-03. |
| Tiny-snake + micro CNN (15k) plain BC (run11) | **+4** | Hand-picked minimal CNN: 5→8→16→32 conv with stride 1/2/2 + 32-d FC = ~15.5k params. Plain CE (no MC weighting) + 100k teacher samples + 30 epochs + 1 DAGGER iter. **Matches the 4.78M-param baseline at 310× fewer parameters.** Lesson: under plain CE the model size doesn't matter much; under MC-weighted CE small nets collapse to modal class. |

## Findings so far

The story arc, condensed:

1. **Full Nibbles (50×80, 5 absolute actions) + BC was a dead end** — runs 01-04 plateau at eval ≈ -50 regardless of obs format (pixel area-down, pixel nearest-down, symbolic) or network size (1.7M to 9.5M). Two distinct bugs masked it for a while: (a) area-interp downscale erased the 2-px snake to a sub-pixel smear; (b) `NibblesEnv.reset()` was passing the same `rng_seed` every time, so every "episode" had an identical number-spawn sequence and the model just memorized one trajectory.

2. **Diversity fix made things look worse, which was actually progress** — once spawns were varied, train-acc dropped from 90→87 and eval dropped from -46→-50. The drop revealed that the previous runs were memorizing the deterministic trajectory; diversity was the honest baseline. Probing the trained policy showed it predicted the wrong action even when the number was one cell to the right of the head.

3. **The big lever was env + action-space simplification, not encoder capacity** — moving to a 12×12 textbook snake with a 3-action relative space (STRAIGHT/LEFT/RIGHT) made plain BC immediately produce something that plays (eval 3-4). Same model architecture as the failing Nibbles runs.

4. **MC-weighted credit and smaller-net interact badly** — `--mc-credit uniform-pos` (zero-credit on death) gave a small bump on the 4.78M model (eval 3→4), but the same weighting collapsed the 1.2M / 311k / 78k variants to modal-class predictions because the credit was sparse enough that the small heads couldn't push through. `discounted` mode was strictly worse: negative weights on death-leading samples anti-imitated the teacher.

5. **PPO above BC was the second big lever** — 200k steps of PPO on top of the run06 BC checkpoint, with a BC-anchor CE in every minibatch, climbed eval 4 → 8.2. Annealing the anchor toward zero (ppo02) gave the best final result (10.6); high entropy (ppo03) hit the highest peak (11.9) but settled noisier. From scratch with no BC init (ppo04), the same small network couldn't crack navigation — the BC checkpoint was doing the heavy lifting.

6. **Network size was almost irrelevant under plain BC** — once weighted CE was off, a 15.5k-parameter micro CNN matched the 4.78M baseline (run11 vs run06, both eval mean 4). The BC ceiling is set by data and training procedure, not parameter count.

Open question: how to climb above the BC ceiling without RL. Current attempt is a distance-map feature (see [#feature-engineering](#feature-engineering)).

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
