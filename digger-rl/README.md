# Digger RL — status & next steps

Reinforcement-learning experiments on the 1983 DOS game **DIGGER** running
inside DOSBox Pure via a libretro pybind11 binding. The goal is to learn
to play it from minimal supervision (score + survival time as reward),
ultimately with a *pixel-only* deployable agent.

## TL;DR — current scoreboard

| Approach | n | Mean (sem) | Max ep | Notes |
|---|---:|---:|---:|---|
| **SmartHeuristic v5 (teacher)** | — | **1475** | — | hand-coded BFS + LoS-fire + dir-aware + falling-bag avoidance |
| Symbolic PPO + BC pretrain + per-step BC anchor + penalties | 20 | 891 | 1475 | **best ML so far**; beats teacher max on some episodes |
| **Pixel BC-only (100k labels, 20 epochs), 50-ep eval** | 50 | **629.5 ±32** | **1175** | true pixel ceiling at 92.1% teacher acc |
| Pixel DAGGER (50k warmup + 3×25k student-collected), 50-ep eval | 50 | 625.0 ±34 | 1275 | within noise of pure BC; DAGGER iters did not help |
| Audit's pixel DAGGER (multi-iter, GreedyEmerald teacher) | — | 783 | — | the older audit number; not 50-ep checked |
| Pixel PPO + warmup + live teacher anchor + penalties | 20 | 364 | 750 | PPO **degrades** the BC policy |
| Pixel PPO + live teacher anchor (no warmup) | 20 | 324 | 825 | live anchor only |
| Symbolic PPO + shaping only (vanilla, seed=1) | 20 | 290 | 1050 | local-optimum basin at ~250 |
| Symbolic PPO + shaping only (vanilla, seed=2) | 20 | 235 | 500 | same basin |
| Symbolic Dreamer V3 (500k frames) | 20 | 250 | — | actor entropy collapse |
| Pixel PPO from scratch (exp4_long_500k) | 20 | 75 | — | classic NatureCNN, no teacher |

**Caveat on evaluation noise.** Most numbers above are 10–20 episode
in-trainer ep_end averages, which have ±~100 standard error around the
true mean (we measured this: single-episode std is ~230). The pixel BC
result was reported as "812" from a 10-episode eval; a proper 50-ep
re-eval brought it to **629.5 ±32**. Treat 10-ep numbers as ±100;
prefer the `eval_checkpoint.py` 50-ep numbers for any comparison
worth acting on.

Conclusion: the symbolic GameState extracted from the framebuffer is
**fully sufficient** to play Digger level 1 well, but training-time
pixels-to-action *can* be learned to ~80% of teacher quality via pure
imitation. The bottleneck for the RL agents was never policy learning —
it was always perception and the chicken-and-egg problem of needing
good perception to get a reward signal. See **Lessons from PPO/DAGGER**
below.

## Layout

| File | Purpose |
| --- | --- |
| `digger_env.py` | `DiggerEnv` (single env, RGBA frames, score/lives via RAM at 0x282E0/0x259F2) and `DiggerVecEnv` (in-proc for `num_envs=1`, subprocess workers for >1). Now also exposes `save_state` / `load_state` via libretro `retro_serialize`. |
| `train_ppo.py` | Pixel PPO (NatureCNN). Supports BC from `.npz` traces *and* live teacher labeling (`--teacher-policy {smart,greedy,dodge}`), warmup phase (`--warmup-steps`), and BC-only mode (`--total-timesteps 0`). |
| `train_ppo_symbolic.py` | Symbolic-obs PPO with the same recipe: BC traces + per-minibatch anchor + death/time penalty + resume-from-state. |
| `train_dagger.py` / `train_dagger_pixel.py` | Earlier DAGGER trainers (multi-iter aggregating teacher rollouts). Superseded by the live-teacher anchor path in `train_ppo.py`. |
| `train_dreamer*.py` | Dreamer V3-ish online and symbolic-obs world models. Plateaued; see lessons. |
| `run_digger.py` | Manual play (matplotlib `--live`) + trace recording. **S / L** keys snapshot and restore emulator state (in-memory or disk via `--save-slot`). `--resume` boots straight into a saved scenario. |
| `tools/game_state.py` | `GameState` dataclass + `extract_state_fast(frame)` — vectorised CV extractor, ~4 ms/frame. Now populates `digger.dir` and `BagPos.moving`. |
| `tools/symbolic_env.py` | `SymbolicDiggerEnv` wraps `DiggerEnv`, emits `(6, 10, 15)` mask tensors. Supports `shaping_coef`, `time_penalty`, and `save_state` / `load_state`. |
| `tools/heuristic_agent.py` | `SmartHeuristic v5`: BFS tunnel-distance for safety scoring, line-of-sight FIRE with dirt-tolerance, turn-to-fire, falling-bag avoidance, direction-aware. Mean 1475. |
| `tools/gen_symbolic_trace.py` | Generates BC traces from a chosen heuristic into `.npz` for offline BC. (Live teacher path in `train_ppo.py` makes this optional.) |
| `probe_*.py` | One-off diagnostics: MPS-vs-CPU correctness, libretro state save/restore. |
| `interaction-log.txt` | Full chronological log of every prompt; the journey. |

## Lessons from PPO / DAGGER experiments

These are the substantive findings from ~10 distinct training runs. Each
took 30–90 minutes; collectively they map out what *actually* matters.

### 1. Pixel-from-scratch RL is hopeless without a teacher

Vanilla pixel PPO with NatureCNN + entropy bonus + shaping plateaued at
~50–75 score across multiple seeds and budgets (exp1, exp4_long_500k,
exp7_wide). The diagnostic signal is consistent: `pg loss ≈ 0`,
`clip_frac ≈ 0`, top action at 60–76% after collapse. PPO can't
extract a learning signal because (a) the encoder hasn't learned what
a digger / monster / emerald *is* yet, and (b) sparse-emerald rewards
average to noise over a random-action rollout.

### 2. Symbolic obs is a 3× improvement out of the box

Same trainer (`train_ppo_symbolic.py`), same recipe, only the
observation changed: pixel-PPO plateaued at 75, symbolic-PPO at 250.
The encoder no longer has to solve "where is everything"; it can spend
all its capacity on "what to do." But 250 is still a **local-optimum
basin** — both seed=1 and seed=2 land there. The diagnostic is the
same `pg ≈ 0, clip ≈ 0` pattern: policy doesn't visit "successful
exploration of level 2" states, so the gradient never points there.

### 3. Reward shaping alone doesn't escape the basin

Adding `--death-penalty 100` and `--time-penalty 0.01` to symbolic
PPO without any imitation did pull the policy out of NOOP-equivalent
attractors, but the run still landed in the 250–400 range. The basin
is *wide*, not just *deep*; small per-step gradients can't bridge it.

### 4. Bootstrap from a teacher is the single biggest lever

Adding **BC pretrain** (10 epochs of CE on 50k SmartHeuristic-driven
samples) **plus per-PPO-minibatch BC anchor** (0.5 → 0.05 anneal) took
symbolic PPO from 250 → 891 mean, with individual episodes hitting
1475 (teacher mean). BC pretrain hits 95.8% teacher-action accuracy
on the symbolic obs; PPO then refines, occasionally exceeding the
teacher's greedy heuristic.

### 5. PPO updates can *degrade* a BC-pretrained pixel policy

The same recipe on pixels collapsed: pixel PPO+warmup+anchor hit only
364, while a **pure BC pixel run with the same data hit 812**. The
diagnostic is clear: the BC pretrain reaches 88–92% teacher accuracy,
but during PPO updates the value-loss and PG noise pull the actor away
from teacher behavior and the score regresses. Death/time penalty,
which *helped* vanilla PPO escape its basin, *hurt* the BC policy.

  - **Pure-BC pixel** = 812 mean
  - **BC + PPO + anchor + penalty pixel** = 364 mean

Why the difference symbolic vs pixel: the symbolic encoder hits 95.8%
BC accuracy and the residual 4.2% errors are *closer* to teacher's own
greedy mistakes, so PPO refinement on top is net-positive. The pixel
encoder caps at 92% accuracy and the residual 8% are perception
errors compounding — PPO refinement can't fix those, and the noise
hurts.

### 6. Auxiliary symbolic-prediction head doesn't help pixel BC

Audit data: `dagger_pixel_aux` (BCE weight 1.0) got 583, `_aux_w03`
got 408, both worse than vanilla pixel DAGGER at 783. The hypothesis
("teach the encoder to recover the tile grid as an auxiliary task")
was tried twice; both weights hurt. The aux head reaches 99.9%
accuracy but the action-prediction features it competes with degrade.

### 7. Wider net hurts at this dataset scale

`dagger_pixel_w2` (NatureCNN width=2, 6.7M params) scored 483 vs
width=1's 783 on the same data. With ~30k aggregated samples that's
~5 samples per parameter; the FC layer memorizes iteration-0 mistakes
and overfits to states the student won't revisit. Wider would help
*if* we had 10× more samples or proper regularization (dropout, weight
decay, augmentation) — none of which the pixel trainer has.

### 8. The pixel BC ceiling is well below the symbolic one

Updated after a proper 50-episode re-eval. Pixel BC at 100k labels +
20 epochs reaches 92.1% teacher-action accuracy → **mean 629.5 score
(±32 sem)** across 50 episodes. The original 812 figure was a
10-episode artifact at the upper end of the ±100 SEM range.

Pure BC and DAGGER iterations on top (50k warmup + 3×25k
student-collected) land at indistinguishable means (629 vs 625, both
±~33 sem). The 8% per-step error rate from a 92% accurate policy
compounds badly over ~1000-step episodes; the teacher needs to be
~98% imitable for the agent to come close to teacher score.

The "pixel perception is the bottleneck" framing holds. The symbolic
BC ceiling (95.8% acc → mean ~890 score from symbolic + PPO) is real
representation headroom, not a fluke.

### 9. DAGGER iterations don't beat pure BC when the teacher is fallible

Standard DAGGER theory predicts iteration helps under covariate
shift. In our setting, with a *greedy* teacher (SmartHeuristic v5),
the teacher's labels on student-visited states are not reliably
better than its labels on its own trajectories — when the student
ends up in a corner the teacher would have avoided, the teacher's
greedy rule still picks reasonable but not necessarily recovery-
optimal actions. Result: 50k warmup + 3×25k DAGGER iters lands within
noise of plain 100k warmup + 20 epochs. Different mixing strategies
(e.g. disagreement-only sampling) could help; we haven't tried them.

### 10. MPS-vs-CPU is a non-issue

A single-seed PPO run on MPS underperformed CPU by 3×, suggesting a
framework bug. A 3-seed sweep showed MPS *winning* on the same
metric. A 50-trial Categorical-sampling probe found no statistically
significant bias on MPS. Conclusion: MPS's `multinomial` uses an
independent PRNG stream, so device-vs-device seed comparisons are not
meaningful, but within-device training is fine on either. The per-op
forward/backward math is bit-identical between MPS and CPU.

For pixel training (where CE on a NatureCNN dominates wall time), MPS
shaves ~40% off the warmup phase. For symbolic training (where DOSBox
is the bottleneck), MPS is a wash. **No reason to use `--force-cpu`
anymore unless reproducibility against a CPU baseline matters.**

### 11. DOSBox save-state is gameplay-valid but not byte-exact

Added libretro `retro_serialize` / `retro_unserialize` to the pybind
binding plus `DiggerEnv.save_state` / `load_state` on top. The visible
game (score, lives, digger position, monsters, dirt) round-trips
faithfully, so we can train from saved scenarios via `--resume-from`.
But two back-to-back restores + the same next action diverge after
~3 frames (1927 pixels differ) — DOSBox's internal timer / audio /
JIT state isn't fully serialized. So Go-Explore-style restart works;
deterministic policy A/B from a saved frame does not.

## How to run

### Watch the heuristic play

```bash
python -m tools.heuristic_agent --live --smart
python -m tools.heuristic_agent --episodes 10 --no-episodic-life --smart   # ~1475 mean
```

### Pure-BC pixel training (the best pixel recipe so far)

```bash
python train_ppo.py \
  --total-timesteps 0 \
  --warmup-steps 100000 --warmup-epochs 20 \
  --teacher-policy smart --bc-batch-size 256 \
  --episodic-life --run-name pixel_bc_only

# Then run a tighter eval (the in-trainer 10-ep number is too noisy):
python eval_checkpoint.py \
  data/checkpoints/pixel_bc_only/ppo_digger_bc_only.pt --episodes 50
```

### DAGGER on top of BC warmup (within-noise of pure BC; see Lesson 9)

```bash
python train_ppo.py \
  --total-timesteps 0 \
  --warmup-steps 50000 --warmup-epochs 15 \
  --dagger-iters 3 --dagger-collect-steps 25000 --dagger-epochs 5 \
  --teacher-policy smart --bc-batch-size 256 \
  --episodic-life --run-name pixel_dagger
```

### Symbolic PPO with BC pretrain + anchor (the best symbolic recipe)

```bash
# 1. Generate offline BC trace from teacher
python -m tools.gen_symbolic_trace --out data/traces/sym_smart_50k.npz \
    --steps 50000 --teacher smart --frame-stack 4

# 2. Train
python train_ppo_symbolic.py \
  --total-timesteps 500000 --num-steps 256 \
  --bc-traces data/traces/sym_smart_50k.npz \
  --bc-epochs 10 --bc-batch-size 256 \
  --bc-anchor-coef 0.5 --bc-anchor-final 0.05 \
  --death-penalty 100 --time-penalty 0.01 \
  --shaping-coef 0.5 --frame-stack 4 \
  --force-cpu --run-name ppo_sym_bc_v1
```

### Save / restore game state

```bash
# Play to an interesting scenario, press S, close window
python run_digger.py --live --save-slot scenarios/right_edge.pkl

# Boot straight back into that scenario
python run_digger.py --live --save-slot scenarios/right_edge.pkl --resume

# Train PPO from that scenario
python train_ppo_symbolic.py --resume-from scenarios/right_edge.pkl --force-cpu \
  --total-timesteps 500000 --run-name ppo_sym_right_edge
```

## Open paths

1. **DAGGER iterations on the BC-only pixel model** — the natural
   covariate-shift fix. Run the BC student in the env, label its
   visited states with the teacher, aggregate, retrain. With our 92%
   starting point we'd plausibly reach 1000+.
2. **Higher-resolution pixel obs** (`--obs-size 168`) — 4× more pixels
   per tile; digger sprite goes from 1–2 px to 4 px. Audit suggests
   this is where the 92% pixel BC ceiling could move.
3. **Curriculum from saved scenarios** — capture "stuck at right edge"
   states via `--live S`, train from those. `--resume-from` already
   wired in `train_ppo_symbolic.py`.
4. **Symbolic PPO past teacher mean** — current symbolic best (891) hits
   teacher max (1475) on individual episodes but averages below. Longer
   schedule with `--bc-anchor-final 0.0` (full PPO decoupling at end)
   *might* let the value head learn multi-step planning the teacher's
   greedy rule misses. Failed twice on pixels — symbolic might be
   different.

## Known open problems (from the heuristic era)

### Smart heuristic phantom monsters at borders

CV occasionally tags col 0/14 / row 0/9 tiles as nobbin. Latent issue
that triggers unnecessary dodges. Fix: exclude border tiles in
`extract_state_fast` or raise `dgrn_c` threshold.

### Bag-crushing avoidance

`_greedy_step` treats intact bags as obstacles. Still missing: refusing
to dig dirt directly below a bag (the dig opens a gap, bag falls,
crushes digger). Falling-bag detection (`BagPos.moving`) lands but the
"don't dig under" rule isn't wired.

## Key constants worth knowing

- Tile grid: 15 cols × 10 rows (`MWIDTH`, `MHEIGHT` in `game_state.py`).
- Score RAM offset: `0x282E0` (int32 LE). Lives: `0x259F2` (uint8).
- Frame dimensions: 640×400 RGBA. Score bar takes top 32 px; play area is `(368, 640)`.
- Palette: exactly 11 RGB triples, no antialiasing. See `PAL_*` constants in `game_state.py`.
- `frame_skip = 4`, `frame_stack = 4`, `obs_size = 84` for pixel; 64 for Dreamer; none for symbolic.
- Action space: `{NOOP, LEFT, RIGHT, UP, DOWN, FIRE}` with FIRE = F1 in the libretro keyboard layout.
- `BASE_OBS_CHANNELS = 6` for symbolic: dirt / emerald / digger / monster / bag / cherry.

## Recent significant commits

- `743ce33` — DAGGER iterations (`--dagger-iters`) for BC-only mode
- `f01ff07` — BC-only mode (`--total-timesteps 0` skips PPO)
- `0d3ee6d` — BC warmup phase before PPO (`--warmup-steps`, `--warmup-epochs`)
- `75f84f5` — live teacher labeling (`--teacher-policy`) for pixel DAGGER
- `9bdcf11` — symbolic PPO: BC pretrain + anchor + reward shaping knobs
- `3c7dd75` — symbolic PPO `--resume-from <pickle>` for scenario training
- `ca15ddf` — `run_digger.py --live` S/L key bindings for state save/restore
- `d0747e1` — `DiggerEnv.save_state` / `load_state` via libretro serialize
- `3e4d51f` — MPS-vs-CPU diagnostic probes (rules out framework bug)
- `0bc186e` — symbolic-obs PPO trainer
- `d5566e1` — SmartHeuristic v5: dirt-aware FIRE LoS + safe turn-to-fire
- `bb41a87` — digger direction + falling-bag detection + turn-to-fire

The `interaction-log.txt` has every prompt in chronological order if
you want to retrace specific decisions.
