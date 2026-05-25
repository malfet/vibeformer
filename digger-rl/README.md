# Digger RL — status & next steps

Reinforcement-learning experiments on the 1983 DOS game **DIGGER** running
inside DOSBox Pure via a libretro pybind11 binding. The goal is to learn
to play it from minimal supervision (score + survival time as reward).

## TL;DR

After a long detour through pixel-based PPO and Dreamer, the headline
results today are:

| Approach | Mean ep score | Max ep | Notes |
|---|---:|---:|---|
| **Greedy heuristic, anti-jitter, bag-blind** | **1645** | **3075** | 50 LOC, no learning. Previous SOTA — relied on accidental bag-break bonuses. |
| Greedy heuristic + bag-aware step | 1472 | — | Routes around bags; forfeits the accidental 500-pt break bonus. |
| Smart heuristic + bag-aware + fire-cooldown + closest-threat | 1082 | — | Dodge path still parks on bags (open issue). |
| DAGGER on symbolic state (8 iters, 29k labels) | 960 | 1850 | BC acc 99.8%, but score caps below teacher; no FIRE labels. |
| Greedy heuristic, plain | 1410 | 2425 | Re-evaluates each step; hesitates at tile boundaries. |
| BC + PPO from color trace (pixels, 20k frames) | ~70 sustained | 1625 | Wider net + 2 traces + 30 BC epochs. |
| BC + PPO from color trace (pixels, 500k frames) | ~50 | 1175 | Long training collapses to one action. |
| Dreamer online (world model + AC, 500k frames) | 225 | 875 | Actor entropy collapsed to 0. |
| Symbolic PPO from scratch + shaping (20k frames) | ~2 | 8 | Sparse-reward death spiral. |

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

### 1. CV bag detection — RESOLVED (commit `1a7a743`)

Exact-pixel template match (`BAG_TEMPLATE`, 30×32 bool) against the
black mask of the framebuffer. Finds all 7 bags on frame 0 with zero
false positives in tunnel cavities. End-to-end `extract_state_fast`
runtime: 4.4 ms/frame (was 41 ms with naive sliding window).
`BagPos.moving` and `BagPos.broken` still default to False — those
sub-states (mid-fall, broken sack with gold pile) are **not** yet
decoded, which limits the smart heuristic (see §2, §3).

### 2. Smart heuristic still regresses vs greedy (medium priority)

Latest 20-ep means at frame_skip=4, no-episodic-life (commit `4839372`):

| Variant | Mean |
|---|---:|
| bare greedy + bag-avoid | 1472 |
| smart + bag-avoid + fire-spam (no cooldown) | 892 |
| smart + bag-avoid + 50-step cooldown | 1020 |
| smart + bag-avoid + 50-cd + closest-threat | 1082 |

Remaining gap vs the 1645 historical greedy is dominated by:
- **Bag-break bonus forfeited.** The old greedy occasionally pushed a
  bag off a ledge and crushed a monster underneath (+500). Bag-aware
  step refuses to push, losing that accidental scoring path.
  Recovering it cleanly requires `BagPos.moving` / `BagPos.broken`
  from CV (so we can push only when the drop kills a monster).
- **Dodge parks on bags.** `_escape_axis` ignores `state.bags`, so the
  perpendicular dodge can send the digger straight under a falling
  bag. Small standalone fix.
- **Phantom monsters at borders** (older issue, still latent): CV
  occasionally tags col 0/14 / row 0/9 tiles as nobbin. Each phantom
  triggers an unnecessary dodge. Fix: exclude border tiles in
  `extract_state_fast`, or require `dgrn_c >= 30` rather than 20.

Reproduce with `--live --smart --overlay`.

### 3. Bag-crushing avoidance — PARTIALLY DONE (commit `4839372`)

`_greedy_step` now treats intact bags as obstacles and routes around
them. Still missing: refusing to dig dirt directly below a bag (the
dig opens a gap, the bag falls and crushes the digger on the way back).
Needs the simple column-scan `if any(b.col == dc and b.row < dr for b
in state.bags): don't go DOWN`. Dodge-path bag awareness is the other
half (see §2).

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

- `4839372` — bag-aware greedy step + SmartHeuristic fire-cooldown + closest-threat targeting
- `2d6889d` — DAGGER imitation trainer on symbolic state
- `1a7a743` — bag detection by exact sprite template (resolves §1)
- `35d5d2d` — heuristic-BC trace generator + DQN trainer on symbolic state
- `771fc48` — anti-jitter greedy heuristic: 1410 → 1645 mean
- `bf7ebdf` — overlay tweaks + digger/nobbin discriminator
- `21ce642` — smart heuristic + `--live` viewer
- `0a4440b` — first greedy baseline (1170 mean)
- `c51ebe9` — vectorised extractor (39 ms → 2 ms per frame)
- `565072b` — Dreamer phase 3 (online training loop)

The `interaction-log.txt` has every prompt in chronological order with
dates if you want to retrace specific decisions.
