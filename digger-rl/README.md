# Digger RL — status & next steps

Reinforcement-learning experiments on the 1983 DOS game **DIGGER** running
inside DOSBox Pure via a libretro pybind11 binding. The goal is to learn
to play it from minimal supervision (score + survival time as reward).

## TL;DR

After a long detour through pixel-based PPO and Dreamer, the headline
results today are:

| Approach | Mean ep score | Max ep | Notes |
|---|---:|---:|---|
| **Greedy heuristic on symbolic state, anti-jitter** | **1645** | **3075** | 50 LOC, no learning. Current SOTA. |
| Greedy heuristic, plain | 1410 | 2425 | Re-evaluates each step; hesitates at tile boundaries |
| BC + PPO from color trace (pixels, 20k frames) | ~70 sustained | 1625 | Wider net + 2 traces + 30 BC epochs |
| BC + PPO from color trace (pixels, 500k frames) | ~50 | 1175 | Long training collapses to one action |
| Dreamer online (world model + AC, 500k frames) | 225 | 875 | Actor entropy collapsed to 0 |
| Symbolic PPO from scratch + shaping (20k frames) | ~2 | 8 | Sparse-reward death spiral |

Conclusion: the symbolic GameState extracted from the framebuffer is
**fully sufficient** to play Digger level 1 well. The bottleneck for the
RL agents was always representation learning on raw pixels, not policy
learning. The path forward is **imitation-learn from the heuristic, then
PPO** on the symbolic state.

## Layout

| File | Purpose |
| --- | --- |
| `digger_env.py` | `DiggerEnv` (single env, RGBA frames, score/lives via RAM at 0x282E0/0x259F2) and `DiggerVecEnv` (in-proc for `num_envs=1`, subprocess workers for >1). |
| `train_ppo.py` | Pixel PPO (NatureCNN, width-configurable). Supports BC from `.npz` traces, BC anchor during PPO, color RGB obs. |
| `train_symbolic.py` | Tiny symbolic-state PPO. `--bc-steps N` collects N transitions from the heuristic and BC-pretrains the actor; `--bc-anchor-coef` keeps PPO near the heuristic. |
| `train_dreamer.py` | Offline world-model training on color traces. Phase 1 / Phase 2 done. |
| `train_dreamer_online.py` | Full Dreamer V3-ish online loop (replay buffer + WM + AC, real env). |
| `run_digger.py` | Manual play (matplotlib `--live`) + trace recording (`--record-playtrace`, `--color`). R resets to level 1. |
| `run_ppo.py` / `run_dreamer.py` | Live playback of saved PPO / Dreamer checkpoints. |
| `tools/game_state.py` | `GameState` dataclasses + `extract_state_fast(frame)` — vectorised CV extractor running at ~2 ms/frame (no BFS, just `np.bincount` per-tile per-color). |
| `tools/symbolic_env.py` | `SymbolicDiggerEnv` wraps `DiggerEnv` and emits `(6, 10, 15)` mask tensors: dirt / emerald / digger / monster / bag / cherry. Optional `shaping_coef` for potential-based reward shaping toward nearest emerald. |
| `tools/heuristic_agent.py` | The current SOTA: greedy walk to nearest emerald (anti-jitter). `--smart` adds dodge + fire (currently regresses, see "Known issues"). `--live` opens a matplotlib viewer. |
| `tools/dreamer.py` | Dreamer-V3-ish model code: Encoder, RSSM, Decoder, Actor, Critic, ReplayBuffer. |
| `tools/analyze_ppo.py` | Headless per-step rollout analyzer for PPO checkpoints (action distribution, entropy, value estimates). |
| `tools/tile_classifier.py` | Nearest-prototype tile classifier (alternative to CV thresholds, partial). |
| `interaction-log.txt` | Full chronological log of every prompt; the journey. |

## How to run

### Watch the heuristic play

```bash
python -m tools.heuristic_agent --live
python -m tools.heuristic_agent --live --overlay              # mark detected monsters/bags
python -m tools.heuristic_agent --live --overlay --show-digger --show-emeralds
python -m tools.heuristic_agent --live --smart                # regresses; see below
```

### Headless benchmark (10 episodes, no episodic-life so full game-over)

```bash
python -m tools.heuristic_agent --episodes 10 --no-episodic-life          # greedy = 1645 mean
python -m tools.heuristic_agent --episodes 10 --no-episodic-life --smart  # smart  =  925 mean (worse)
```

### Train symbolic PPO with BC from heuristic (recommended next experiment)

```bash
python train_symbolic.py \
  --total-timesteps 100000 \
  --bc-steps 5000 --bc-epochs 5 --bc-anchor-coef 0.5 \
  --shaping-coef 1.0 --ent-coef 0.01 --ent-coef-final 0.001 \
  --episodic-life --run-name symbolic_bc
```

### Manual play / record traces

```bash
python run_digger.py --live                                 # play
python run_digger.py --live --record-playtrace data/t.npz   # record (color by default)
```

## Known open problems

### 1. CV bag detection is broken (high priority)

`extract_state_fast` never reports `bags`. Yellow `$` glyphs are small
(~5-10 px) and look like emerald inner-highlights to the size+colour
threshold. Without bag positions, the heuristic can't avoid them and
**death-by-falling-bag is the most common cause** of episode end.

Two paths to fix:
- Tighten the bag signature: `$` glyph + dark-gray sack body, but with
  a more careful spatial check (e.g. yellow blob bounded by dark pixels
  on all four sides). Investigate by sampling the actual `data/cv_samples/`
  PNG at known bag coordinates.
- Use `tile_classifier.py` (nearest-prototype on pixel patches) — already
  scaffolded; just needs prototype patches for `BAG_INTACT` and
  `BAG_SPILLED` captured from frames where a bag is visible (record a
  playthrough that creates bags and inspect frames).

### 2. Smart heuristic regresses vs greedy (medium priority)

`--smart` scores ~925 vs greedy's 1410. Two suspected causes:
- **Phantom monsters at borders**: CV occasionally tags col 0/14 or
  row 0/9 tiles as nobbin because the playfield border patterns match
  the dark-green threshold. Each phantom triggers an unnecessary dodge.
  Fix: explicitly exclude border tiles in `extract_state_fast`, or
  require a minimum sprite extent (`dgrn_c >= 30` rather than 20).
- **Dodge logic over-cautious**: moves perpendicular even when the
  monster isn't actually closing on us. Could check monster `dir` /
  velocity, but CV doesn't extract those yet.

Reproduce with `--live --smart --overlay` and watch where blue crosses
appear without a visible green sprite.

### 3. Bag-crushing avoidance not implemented (depends on #1)

Once bag detection works, the heuristic should refuse to walk *up
into* the column below a bag, and refuse to dig dirt below a bag.
Roughly: `if state.bags[any].col == digger.col and bag.row == digger.row - 1: don't go UP`.

### 4. BC-from-heuristic run hasn't been launched yet (the actual experiment)

Code is in place (`train_symbolic.py --bc-steps`). Hasn't been run.
Hypothesis: BC will give the agent a strong starting point and PPO will
improve on the heuristic (specifically by learning bag-avoidance from
its death signal). This is the natural next experiment.

### 5. Pixel-RL agents collapse to one action (low priority — abandoned path)

PPO and Dreamer both eventually collapse to a single argmax action.
Symbolic state route bypasses this entirely so probably not worth
fixing. See commits before `c51ebe9` (vectorised extractor) for the
historical struggle.

## Key constants worth knowing

- Tile grid: 15 cols × 10 rows (`MWIDTH`, `MHEIGHT` in `game_state.py`).
- Score RAM offset: `0x282E0` (int32 LE). Lives: `0x259F2` (uint8).
- Frame dimensions: 640×400 RGBA. Score bar takes top 32 px; play area is `(368, 640)`.
- Palette: exactly 11 RGB triples, no antialiasing. See `PAL_*` constants in `game_state.py`.
- `frame_skip = 4` everywhere; `frame_stack = 4`; `obs_size = 84` for pixel agents, 64 for Dreamer, none for symbolic.
- Action space: `{NOOP, LEFT, RIGHT, UP, DOWN, FIRE}` with FIRE = F1 in the libretro keyboard layout.

## Branch / commit pointers

Recent significant commits:

- `bf7ebdf` — overlay tweaks + digger/nobbin discriminator (improved heuristic to 1410 mean)
- `21ce642` — smart heuristic + `--live` viewer
- `0a4440b` — first greedy baseline (1170 mean)
- `c51ebe9` — vectorised extractor (39 ms → 2 ms per frame)
- `edbce90` — nearest-prototype tile classifier scaffold
- `1042947` — first CV extractor (partial, BFS-based)
- `565072b` — Dreamer phase 3 (online training loop)

The `interaction-log.txt` has every prompt in chronological order with
dates if you want to retrace specific decisions.
