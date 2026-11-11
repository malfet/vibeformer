# Digger RL

Learning to play the 1983 DOS game **DIGGER**, running in DOSBox Pure
through a libretro pybind11 binding. The approach: bootstrap a student
from a hand-written teacher (behavioural cloning, then DAgger), then
fine-tune it with PPO to get past the teacher. Learning from first
principles (no teacher) and a pixel-only agent are later goals.

## Current scoreboard (2026-11-11, symbolic state from RAM)

Same session where it matters; cross-session numbers drift by up to
~±400. Score is a full 3-life game; life is agent steps per life (4
frames each); "L1 clean" is games that cleared level 1 without losing a
life (the project goal).

Same session, 20 games each (2 seeds):

| Policy | Score | Life | L1 clean |
|---|---:|---:|---:|
| **PPO fine-tune of `dagger3` (`ppo_ft1`, final)** | 3662 ±161 | **331** | **16/20** |
| **PPO fine-tune of `dagger3` (`ppo_ft1`, best)** | **3752** ±196 | 330 | 11/20 |
| SmartHeuristic v6.4 teacher | 3269 ±241 | 290 | 13/20 |
| DAgger round 3 student (`dagger3`, 700k steps) | 2858 ±175 | 264 | 7/20 |
| Pixel student (`pixel_dagger_v1`, BC + 3 DAgger, 10 games) | 1513 ±109 | 188 | 0/10 |

The earlier BC run (`bc_v1`) scored 2765 against a same-day teacher of
3098 (Lesson 17). The PPO student is the first policy that beats its
teacher (Lesson 19).

Earlier scoreboards (CV state, pixel PPO, Dreamer, DQN) are in the
lessons below; every number before Lesson 15 used the misaligned CV
state.

## Entrypoints

| Command | What it does |
|---|---|
| `python play.py --live` | Play yourself. S/L save/load the emulator state. |
| `python watch.py POLICY` | Watch a policy play (inline in Kitty / Ghostty / WezTerm, else matplotlib). P pauses, +/- change speed. |
| `python evaluate.py POLICY --episodes 20` | Score, life length, and level-1 clears over N full games. |
| `python evaluate.py POLICY --levels data/save/levels` | Per level 1-10: clears, emeralds eaten, score in one life from each saved level start. |
| `python -m tools.level_states` | Build save states at the start of (and inside) levels 1-10 by poking the level counter. |
| `python collect.py --out F.npz` | Record teacher-labelled traces; `--student CKPT` for DAgger. |
| `python train_symbolic.py ...` | Symbolic-state BC (`--total-timesteps 0`) and PPO (`--load-agent`). |
| `python train_pixel.py ...` | Pixel BC / live-teacher DAgger / PPO. |

`POLICY` is `teacher`, `search`, `random`, or a checkpoint from either
trainer (`tools/policies.py` tells them apart by their config).

## Layout

| File | Purpose |
|---|---|
| `csrc/libretro_frontend.cpp`, `setup.py`, `Makefile` | pybind11 `_libretro` binding for the DOSBox Pure core (`vendor/dosbox-pure`). `make` builds both. |
| `digger_env.py` | `DiggerEnv`: frames, score/lives/death onset from RAM, save/load state. `DiggerVecEnv` (subprocess workers), `FrameStack`, `preprocess_uint8`. |
| `tools/game_state.py` | `GameState` and `extract_state_ram`, the exact state read from game RAM (Lesson 15). `heading_tile`. The older CV extractor `extract_state_fast` is kept for reference only. |
| `tools/symbolic_env.py` | `SymbolicDiggerEnv`: 13 planes per frame (below), frame stacking, egocentric option, reward shaping, `env_step_skipped`. |
| `tools/heuristic_agent.py` | `SmartHeuristic` v6.4, the teacher (Lessons 11a, 16, 17). Headless benchmark with death forensics. |
| `tools/search_teacher.py` | `SearchTeacher`: the heuristic overruled by emulator rollouts (Lesson 13). Stronger but 0.4 s/step. |
| `tools/policies.py` | Loads any policy spec for `watch.py` / `evaluate.py`. |
| `tools/display.py` | Kitty graphics + keyboard protocol viewer, matplotlib fallback. |
| `tools/collect_death_states.py` | Save states N steps before each death; `--probe` checks escapability. |
| `tools/make_report_figs.py` | Regenerates `docs/*.png`. |
| `interaction-log.txt` | Every prompt, in order. |

## Model architectures

**Symbolic observation** (`state_to_tensor`), 13 planes of 10x15 per
frame, 4 frames stacked into (52, 10, 15): dirt, emerald, digger,
monster, bag, cherry (not decoded yet), heading tile, facing (4
whole-board planes: R, U, L, D), fire ready (whole-board), bag moving.
Everything the teacher decides from is in it (Lesson 17).

**`SymbolicAgent`** (`train_symbolic.py`, ~1.3M params at 52 input
channels): 3x conv3x3 (32, 64, 64, ReLU, padding 1) -> flatten -> FC 128
-> linear actor (6 actions) and linear critic heads, sharing the trunk.
`--arch iter` is a weight-tied iterative variant (232k params, used in
the survival experiments).

**Pixel `Agent`** (`train_pixel.py`): NatureCNN on 4 stacked 84x84
frames (gray or color): conv 8x8/4 (32), 4x4/2 (64), 3x3/1 (64) -> FC
512 -> actor and critic heads; `--separate-critic` gives the critic its
own trunk, `--encoder-width` scales the channels.

Actions: NOOP, LEFT, RIGHT, UP, DOWN, FIRE, each held for 4 emulator
frames (`frame_skip`).

## Training approaches

1. **Behavioural cloning.** `collect.py` records (obs, teacher action);
   `train_symbolic.py --total-timesteps 0 --bc-traces ... --bc-val ...`
   fits cross-entropy and keeps the best held-out epoch. 5% noise
   injection (`--explore`) gives off-teacher states with the teacher's
   recovery labelled.
2. **DAgger.** `collect.py --student CKPT` lets the student drive while
   the teacher labels; retrain on the aggregate. Three rounds took
   held-out accuracy from 92.4% to 93.9% (86.6% on the steps where the
   teacher's action changes) and life length past the teacher's.
3. **PPO fine-tune.** `train_symbolic.py --load-agent CKPT`:
   critic-only warmup (`--value-warmup-updates`), a BC anchor to the
   teacher labels annealed 0.3 -> 0.05, score x 0.02 as reward,
   episodes ending at death onset, low lr / entropy, keep the best
   checkpoint.
4. **Pixels** (next): the same teacher labels, but the student sees
   84x84 frames (`train_pixel.py` live-teacher DAgger).

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

Same trainer (`train_symbolic.py`), same recipe, only the
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

### 11a. Teacher quality is the biggest lever (the snake lesson, confirmed)

SmartHeuristic v6 = v5 + three changes: (a) *phantom filter* — monster
detections on dirt tiles are CV artifacts (nobbins can't be inside
dirt) and are ignored, so no more fake dodges or wasted 50-step FIRE
cooldowns; (b) *under-bag transit rule* — dirt directly below an
intact bag may be dug through horizontally but never entered from
below (UP) and never lingered in (NOOP); (c) *Dijkstra emerald
routing* — moves are scored against a multi-source distance field
(dirt cost 2, bags block, under-bag dirt +6 surcharge) instead of
straight-line Manhattan.

The ablation grid (10 eps each) shows a strong interaction: neither
(b) nor (c) helps alone — each is actively harmful — but together
they double the teacher:

| Config | Mean |
|---|---:|
| phantom + transit + Dijkstra (**v6, shipped default**) | **1932** |
| v5 baseline (fresh re-bench) | 1010 |
| phantom only | 945 |
| phantom + transit + Manhattan | 832 |
| strict under-bag block (emeralds under bags unreachable) | 622 |
| Dijkstra only (routes into under-bag dirt) | 592 |

The router is what *exploits* the transit rule: it plans multi-tile
paths that dig under bags safely and detour around them; Manhattan
can't look far enough ahead to use the rule, and Dijkstra without the
rule plans paths that drop bags on the digger.

Downstream, the same BC recipe (identical net, 100k+3×25k budget,
92% train acc both times) went 629.5 → **1042** — the entire delta is
teacher score at constant imitability.

### 11b. DAGGER works once the teacher is worth imitating

With the v5 teacher, DAGGER iterations were within noise of pure BC
(Lesson 9). With v6, mid-evals climbed 755 → 965 → 1190 across three
iterations, and rollout teacher-agreement rose 59.5% → 64.3%. The
covariate-shift gap was always real; v5's labels on student-visited
states were just not good enough to close it.

### 11c. PPO on pixel BC: still net-negative, but the gap narrowed

PPO from the 1042 BC checkpoint with every anti-degradation lever we
now have — `--separate-critic` (value gradients can't touch the BC
encoder), `--bc-value` MC-return critic warmup + head rescale (no
initial value-loss shockwave), live v6 anchor annealed 0.3 → 0,
`--save-best` — evals at 851 ±44 (best ckpt) / 750 ±42 (final).
Training diagnostics were healthy the whole run (pg ≠ 0, clip
0.05–0.35, entropy 0.5–1.3, tiny value loss, rolling score rising
682 → 832), yet the stochastic-eval score is still ~3σ below BC.
Compare the old recipe's 812 → 364 (−55%): the levers cut the
degradation to −18% but didn't flip the sign. The pixel policy's 8%
per-step perception errors remain something PPO refinement cannot
fix — it can only trade teacher-shaped behaviour away against them.
`--save-best` earned its keep: final < best by 100 points.

### 12. DOSBox save-state is gameplay-valid but not byte-exact

Added libretro `retro_serialize` / `retro_unserialize` to the pybind
binding plus `DiggerEnv.save_state` / `load_state` on top. The visible
game (score, lives, digger position, monsters, dirt) round-trips
faithfully, so we can train from saved scenarios via `--resume-from`.
But two back-to-back restores + the same next action diverge after
~3 frames (1927 pixels differ) — DOSBox's internal timer / audio /
JIT state isn't fully serialized. So Go-Explore-style restart works;
deterministic policy A/B from a saved frame does not.

### 12b. Restore is input-live in gameplay states, dead inside the death sequence

Building the near-death curriculum turned up a sharp qualifier on the
above. Restoring an ordinary mid-play state and then pressing a key
works exactly as it should — controlled against a no-restore baseline
from the same position:

```
no-restore, LEFT (control)   [(1,4) ... (1,3), (1,3), (1,3)]
restore + LEFT               [(1,4) ... (1,3), (1,3), (1,3)]   matches control
restore + RIGHT              [(1,4), (1,5), (1,5) ... (1,6)]   diverges as it should
```

But a state captured a few agent steps before the `lives` counter
decrements is **frozen**: LEFT, RIGHT and NOOP produce identical
trajectories and the life is lost at exactly the same step. Those
states are inside Digger's death sequence — the outcome is already
sealed and input is ignored until the respawn.

The trap is that this is invisible from the `lives` counter. (It is *not* invisible in RAM: the death-stage word at `0x259E4` flips the moment the death starts; see 12c.) `lives` only
decrements when the sequence *finishes*, and the death animation still
renders a digger-coloured sprite, so `GameState.digger.present` stays
true throughout. Trying to measure the animation length by watching for
the sprite to vanish returns a median gap of **0** and tells you
nothing. Only replaying a restored state under different actions
exposes it.

Practical rule: **any curriculum built on save-states must validate
escapability by replaying under at least two different action
sequences.** A scenario set where every policy dies at an identical
fixed step is a set of corpses, not a set of hard problems — and it
looks exactly like a set of hard problems in every metric that doesn't
involve replaying it.

### 12c. ~~Save-states lose the digger's position across processes~~ — retracted: it was a stale framebuffer plus a late death anchor

An earlier version of this section claimed a `retro_serialize` blob
loaded into a freshly booted DOSBox brings the digger back at the spawn
point. **That was wrong.** Reading the game's own RAM settles it:

- **Game RAM round-trips byte-exactly across processes** (0 of 649,120
  bytes differ after `load_state` in a new process), and LEFT/RIGHT from a
  cross-process restore produce exactly the trajectories they produce
  in-process.
- **The framebuffer is not part of the state.** For ~3 emulator frames
  after `unserialize`, `get_frame()` still returns whatever the core last
  drew: the spawn screen in a fresh process (the "(9, 7)" in the old
  table), or the pre-restore screen in-process. `DiggerEnv.load_state`
  now runs 3 frames before returning, so the first observation after a
  restore matches the restored game.
- **The CV extractor sometimes reads a nobbin as the digger** (a frame
  with the digger visibly at (1, 4) parsed as (6, 10), where a nobbin
  was). Checking positions through CV made the confusion worse.
- **What actually sealed every scenario (12b) is the death anchor.**
  The digger's record lives right next to `lives` in RAM (`0x259C8` x,
  `0x259CA` y, `0x259CC` column, `0x259CE` row, `0x259E4` death stage:
  1 = alive). A death *starts* ~100 agent steps (~6 s) before `lives`
  goes down: bag or monster hit, death animation, then a 60-tick
  tombstone countdown, all with input ignored. Anchored on the lives
  decrement, lookbacks of 15/25/40/60 were all inside that window:
  **320 of 320** scenarios in `death_bcclone_far.pkl` load with the
  death stage already off "alive".

`DiggerEnv.step` now returns `digger_rc`, `digger_xy`, `death_stage` and
`dying` in `info`, read straight from RAM. `collect_death_states` anchors
on death onset. Re-collected that way (teacher driving, 12 deaths), the
scenarios respond to input, and escapability rises with lookback as a
real curriculum should:

| lookback before onset | teacher survives 50 steps | random survives |
|---:|---:|---:|
| 3 | 0/12 | 0/12 |
| 5 | 1/12 | 1/12 |
| 10 | 2/12 | 1/12 |
| 20 | **6/12** | 3/12 |

Same-process and cross-process save/restore both work, so a scenario
recorded by hand in `play.py --live` (S with `--save-slot`) is a
valid `--resume-from` start again.

### 12d. Deaths are now detected at onset, not at the lives decrement

`DiggerEnv` reports `death_event` on the frame the RAM death stage
leaves "alive". With `episodic_life` the episode ends there, the death
penalty lands on that step, and `reset()` plays the ~100 input-ignored
steps of the death sequence out before handing back the respawned
digger. A death onset on the last life sets `real_done` at once, since the
game's outcome is already sealed. Both frame-skip helpers OR the one-frame
flag across sub-frames. Without that, 3 deaths in 4 were dropped.
Measured with NOOP: an episodic life went from ~730 to ~350 emulator
frames, and the difference is all dead time.

**Life lengths before this change are inflated.** `evaluate.py`
used to measure from one lives drop to the next, which counts the
previous death sequence. Measured at onset, a random policy's life is
**~96 agent steps** (was 194). The "life length" column in the
survival-reward table below is in the old units: subtract ~100 for
comparison. The ordering is unaffected.

### 12e. PPO with the onset death marker (2026-10-04)

Three 400k-step PPO fine-tunes of the symbolic BC clone, same session,
all with onset-terminated episodes, score x 0.02 (emerald = +0.5),
shaping off, and a v6.4 teacher anchor annealed 0.3 -> 0.05. Near-death
scenarios were collected with the BC clone driving and anchored on death
onset: 320 for training (`onset_bc_train.pkl`) and 120 held out
(`onset_bc_heldout.pkl`, different run). Full games: `evaluate.py`,
30 games (teacher 20). Held-out: survive 60 steps from each scenario.

| Policy | Score | Life (steps) | L1 cleared without dying | L1 cleared | Held-out escapes |
|---|---:|---:|---:|---:|---:|
| Teacher v6.4 | 1304 +-132 | 161 +-11 | 1/20 | 6/20 | 29/120 |
| BC clone | 1229 +-115 | 167 +-10 | 0/30 | 9/30 | 24/120 |
| `onset_ctrl` (no death penalty) | **1341 +-76** | **188 +-11** | 0/30 | 13/30 | 25/120 |
| `onset_death` (-5 at onset) | 1293 +-75 | 187 +-9 | 0/30 | **16/30** | 31/120 |
| `onset_death_scen` (+50% scenario resets) | 943 +-86 | 178 +-10 | 0/30 | 5/30 | **32/120** |
| Random | - | ~96 | - | - | 17/120 |

- **PPO on the BC clone is no longer net-negative.** Both game-start arms
  beat BC by ~1 sem on score and ~1.5-2 sem on life length, and clear
  level 1 more often (13-16/30 vs 9/30). The earlier `bcppo_score` run
  tied BC. Onset-terminated episodes are the main change between the two,
  but one seed each is suggestive, not proof.
- **The -5 onset penalty did little.** Same score and life length as the
  control; held-out escapes 31 vs 25 out of 120 is inside binomial noise
  (+-~5).
- **Scenario resets teach escapes but cost the game.** `onset_death_scen`
  is best on the held-out set at long lookbacks (16/30 at 30 steps vs 9/30
  for BC), but its full-game score fell 30%. With half the episodes
  starting in a death trap with a penalty attached, the policy turns
  timid elsewhere. A lower `--resume-prob` (0.1-0.2) is the obvious next
  knob.
- **The goal metric didn't move: 0/120 clean level-1 clears across the
  ML policies.** Clearing level 1 takes several hundred agent steps on one
  life. The best policy averages 188. A +12% gain in life length doesn't
  close a ~3x gap.

### 13. Search teacher: the first policy that meets the goal (2026-10-04)

`tools/search_teacher.py` keeps SmartHeuristic as its default policy and
overrules it with in-process emulator rollouts. The rollouts are scored on
the RAM death flag (12d) and made possible by in-process save/restore (12c).
Per decision it plays the heuristic's action plus heuristic continuation
for `horizon` steps, `rollouts` times. If no rollout dies it takes the
heuristic's action; otherwise it rolls out all six actions and takes the
best (survival rate, steps survived, score gained). Same session, 7800
cycles:

| Policy | Games | Score | Life (steps) | L1 cleared without dying | L1 cleared | Cost/step |
|---|---:|---:|---:|---:|---:|---:|
| SmartHeuristic v6.4 | 20 | 1504 +-117 | 188 | 1/20 | 8/20 | <1 ms |
| **Search h=24, r=2** | 16 | **3844 +-450** | **670** | **5/16** | **13/16** | 0.42 s |
| Search h=40, r=3 | 12 | 3802 +-380 | 635 | 3/12 | 11/12 | 1.0 s |

- Life length goes up 3.5x and score 2.5x. The full search runs on ~28%
  of steps (40% for h=40) and overrules the heuristic on ~19%.
- Looking deeper didn't help: h=40/r=3 is within noise of h=24/r=2 at
  2.4x the cost. Most fatal mistakes are visible within ~24 steps (96
  frames), matching the near-death escapability curve.
- This was made affordable by two speedups: 7800 emulated cycles (0.55 vs
  3.4 ms/frame, same gameplay timing) and the vectorised CV extractor
  (3.8 vs ~10 ms/frame). Rollouts also run CV only when the heuristic
  decides.
- It is a teacher, not a deployable agent: 0.4 s per step and it reads
  RAM and calls the emulator. Next is distilling it, starting with
  symbolic BC/DAgger on its labels, then pixels.

    python evaluate.py search --episodes 2   # ~10 min/game

### 14. Imitating the search teacher: accuracy rises, survival doesn't (2026-10-04)

~100k search-teacher steps (`tools/gen_search_trace.py`, 14 parallel
collectors). Every decision is stored as 84x84 RGB, native-aspect 100x160
RGB, and the symbolic grid, so one set of labels trains every input type
(`tools/train_bc_traces.py`, 13 files train / 1 held out). 10-game
evals, same session:

| Student | Held-out acc | Score | Life (steps) |
|---|---:|---:|---:|
| symbolic + previous action | **64.7%** | 653 +-85 | 167 |
| symbolic | 55.8% | 598 +-82 | 165 |
| pixel 100x160 | 44.5% | 550 +-106 | 143 |
| pixel 84x84 | 42.0% | 398 +-75 | 159 |
| pixel 100x160 + previous action | 60.8% | 200 +-55 | 157 |
| (heuristic teacher / search teacher) | - | 1504 / 3844 | 188 / 670 |

- **Perception is not the main gap.** A student that reads the exact
  symbolic grid tops out at 56% held-out. Native-aspect pixels add only
  ~2.5 points over 84x84. Every student memorises: train accuracy keeps
  climbing to 75-86% while held-out peaks in epochs 2-5 and then falls.
- **The labels are self-consistent but persistent.** Asked twice on the
  same state the teacher agrees 95% (76% on searched steps), and
  `label[t] == label[t-1]` already scores 58.5%, better than every
  history-free student. Adding the previous action as input lifts
  symbolic to 64.7% and pixel to 60.8%. Stacking 4 steps of symbolic
  history adds nothing beyond that.
- **None of it moves life length.** Every student lives 143-167 steps,
  at or below the plain heuristic (188) and far from the teacher (670).
  The previous-action pixel student is the copycat failure: it learns
  "repeat the last action" and then repeats its own mistakes, scoring
  200.
- **Why:** the teacher's advantage lives in the ~20% of steps where
  lookahead overrules the heuristic. Those are the least predictable from
  the current frame, and BC's average-accuracy objective barely weights
  them. The students learn the persistent, heuristic-like 80% and miss
  the decisions that keep the digger alive.

Next: store the search's per-action survival estimates and searched
flags in the traces, then distil *those*: weight the overruled steps and
add a per-action danger head as a soft target. Then run DAgger so the
student gets labels in the states where it actually dies.

### 15. The CV symbolic state was wrong a third of the time; it now comes from RAM (2026-11-11)

Measured against the digger's RAM record over 3000 heuristic steps,
`extract_state_fast` put the digger on the right tile only 62.6% of the
time. It lost the sprite 7.5% of the time, was off by 2+ tiles 96 times,
and flickered A->B->A on a fixed RAM tile 25 times. The column error
climbs from 2% at col 7 to 50% at col 0 and 92% at col 14. The cause is
the tile grid: the game draws tile (h, v) at x = h*20+12, y = v*18+18 in
320x200, i.e. 40x36 px tiles from (24, 36) on our frame, while the CV
code assumes 42.67x36.8 px tiles from (0, 32).

`extract_state_ram` replaces it everywhere a symbolic state is consumed
(symbolic env, heuristic teacher, search rollouts, pixel-trainer teacher
labels). RAM layout, found by scanning for the level-1 layout and spawn
coordinates and checked against the sobomax struct definitions:

| Offset | What | Layout |
|---|---|---|
| `0x259C8` | digger | int16 x, y, h, v, rx, ry, mdir, dir, _, recharge, fire x, y, dir, _, deathstage ... lives (+0x2A), notfiring (+0x2E) |
| `0x25A3E` | emeralds | int8[150], nonzero = emerald |
| `0x25AD6` | monsters | 6 x 36 B: x, y, h, v, xr, yr, dir, hdir, t, hnt, death, bag, dtime, stime, chase, flag, nob, alive |
| `0x25D1A` | bags (live copy) | 7 x 26 B: x, y, h, v, xr, yr, dir, wt, gt (gold timer), fallh, wobbling, unfallen, exist |
| `0x2600C` | dirt field (live copy) | int16[150], bit 0x2000 set = not dug through |

Objects map to the nearest tile. The cherry is not decoded yet; channel 5
stays zero, as it always was under CV. The ~3-frame framebuffer lag
after a no-refresh restore (12c) no longer matters, because RAM is
current immediately.

The unchanged v6.4 heuristic, fed exact state (20 games, 2 seeds, same
session; the two seeds replay identical first four games):

| Teacher v6.4 on | Score | Life (steps) | L1 cleared | L1 cleared without dying |
|---|---:|---:|---:|---:|
| CV state (Lessons 12e, 13; earlier sessions) | 1304-1504 | 161-188 | 6-8/20 | 1/20 |
| **RAM state** | **2399** (2805 +-325 / 1993 +-197) | **246** (273 / 219) | **20/20** | **0/20** |

The teacher now clears level 1 every game, but still never without
losing a life. All earlier symbolic results (BC clones, PPO, search
traces) were trained on the noisy state and should be re-run.

### 16. The teacher paced back and forth; it now plans from the tile it is heading into (2026-11-11)

Watching `--live` showed the digger reversing over and over from about
step 55. It counted 38-42 reversals in the first 600 agent steps. There
were three causes, all in `SmartHeuristic`:

1. **Planning from the rounded tile.** The game only accepts a turn at a
   tile boundary (`digger.c` updatedigger: vertical needs rx == 0,
   horizontal needs ry == 0). A perpendicular key held mid-tile keeps the
   digger going and turns it at the boundary. The heuristic scored moves
   from the nearest tile, which flips as the sprite crosses a tile
   midpoint, so the plan flipped with it. It now plans from the tile the
   digger is moving into (`_heading_tile`, using the RAM x/y now on
   `DiggerPos`).
2. **An emerald the rules won't let it take.** The no-UP-into-under-bag
   rule blocked the move, but the emerald-distance field still counted the
   emerald one step away. Every legal move then tied and it paced beside
   it. The field now forbids that edge too.
3. **Turning back mid-tile on emerald distance.** Heading into an emerald
   tile, the emerald counts as eaten, so the field can prefer turning
   back. It then turns before eating it, and the emerald is a target
   again. A mid-tile reversal now has to win on safety (hazard / fall
   column / monster distance); it can't win on emerald distance alone.

Reversals dropped to 4-6 in 600 steps, and those are firing turns and
retreats. Same session, 20 games each, RAM state:

| Teacher v6.4 | Score | Life (steps) | L1 cleared | L1 cleared without dying |
|---|---:|---:|---:|---:|
| plan from rounded tile | 2124 (2233 / 2015) | 234 | 20/20 | 1/20 |
| **plan from heading tile** | **2783** (2945 / 2620) | **263** | 20/20 | **6/20** |

### 17. A BC student now plays about as well as the teacher (2026-11-11)

The first pure-imitation run on exact (RAM) state. Changes that made the
teacher fully imitable:

- **The observation carries everything the teacher decides from.**
  `state_to_tensor` grew from 6 to 13 planes: heading tile (what the
  teacher plans from, Lesson 16), facing (4 whole-board planes), fire
  ready (whole-board plane, from the game's recharge state), and bag
  moving. With 4-frame stacking the obs is (52, 10, 15).
- **No hidden teacher state for firing.** On RAM state `SmartHeuristic`
  gates FIRE on the game's own recharge (`DiggerPos.can_fire`) instead of
  its private 50-step countdown. Same session, 20 games: 3098 vs 2931
  score, 278 vs 269 life, 6/20 vs 9/20 clean L1 (within noise).
- **Noise injection when collecting.** `gen_symbolic_trace --explore
  0.05` plays a random action 5% of the time but always records the
  teacher's label. Students see off-teacher states with the recovery
  labelled. Parallel collectors also diverge this way; the emulator is
  deterministic.
- **Held-out BC accuracy.** `train_symbolic --bc-val` reports
  per-epoch held-out accuracy and accuracy on steps where the teacher's
  action changes, and keeps the best epoch.

8 collectors x 25k steps (~2 min). 175k train / 25k held out, default CNN,
20 epochs: train acc 99.2%, held-out 92.4%, held-out at action changes
80.5%. Full games, same session:

| Policy | Score | Life (steps) | L1 cleared | L1 cleared without dying |
|---|---:|---:|---:|---:|
| Teacher v6.4 (RAM, game recharge) | 3098 | 278 | 20/20 | 6/20 |
| **BC student, sampled** | **2765** (89%) | **259** | 20/20 | **8/20** |
| BC student, argmax | 2395 | 248 | 10/10 | 2/10 |

Argmax is deterministic, so both seeds replay the same 10 games. It
is worse than sampling, likely because a wrong argmax repeats in a
deterministic loop, while sampling breaks out. The 99% vs 92% gap says
more data (or DAgger on student states) is the next lever.

### 18. DAgger lifts the student to the teacher's level (2026-11-11)

Three DAgger rounds on top of `bc_v1`. Each round, 8 collectors x 25k
steps let the previous student drive (sampled, 5% random) while the
teacher labels every state (`collect.py --student`). A fresh model is
then trained on all traces so far for 15 epochs. Held out: the teacher's
`bc_v1_7` plus each round's student-driven trace 7.

| Student | Train samples | Held-out acc | Acc at action changes | Score | Life | L1 clean |
|---|---:|---:|---:|---:|---:|---:|
| `bc_v1` | 175k | 92.4% | 80.5% | 2765 | 259 | 8/20 |
| `dagger1` | 350k | 92.0% | 82.7% | 2798 | 257 | 4/10 |
| `dagger2` | 525k | 93.3% | 85.1% | 3280 | 312 | 5/10 |
| `dagger3` | 700k | 93.9% | 86.6% | 3115 | 314 | 6/10 |
| teacher (same day) | | | | 3098 | 278 | 6/20 |

DAgger rounds are 10 games, one seed, so score differences between
rounds are within noise. Life length is the clearest signal: about 13%
above the teacher from round 2 on. The gain sits on the hard steps
(accuracy at action changes +6 points), where the student previously
drifted off the teacher's state distribution.

### 19. PPO takes the DAgger student past the teacher (2026-11-11)

`ppo_ft1`: 2M frames of PPO from `dagger3` (~45 min at 450-880 frames/s).
10 critic-only warmup updates came first. Then a BC anchor on the
`dagger3_*` traces, annealed 0.3 -> 0.05; reward score x 0.02; episodes
ending at death onset; lr 1e-4; entropy 0.001 -> 0.0005; best checkpoint
by rolling score kept. Command in "How to run".

| Policy (same session, 20 games, 2 seeds) | Score | Life | L1 cleared | L1 clean |
|---|---:|---:|---:|---:|
| teacher | 3269 ±241 | 290 | 20/20 | 13/20 |
| `dagger3` (PPO's start) | 2858 ±175 | 264 | 20/20 | 7/20 |
| `ppo_ft1` best | 3752 ±196 | 330 | 20/20 | 11/20 |
| `ppo_ft1` final | 3662 ±161 | 331 | 20/20 | 16/20 |

- **+28-31% over its own start and +12-15% over the teacher on score;
  life +14% over the teacher.** The score gap to the teacher is ~1.5-2
  sem, so it is suggestive at n=20; life length and clean clears agree.
- The value warmup did its job: KL and clip fraction stayed 0 for the
  10 warmup updates. Afterwards entropy rose from ~0.01 to 0.1-0.6, and
  training per-life score climbed from ~1500 to ~2500-3800.
- `dagger3` evaluated at 2858 here vs 3115 in its own run (10 games,
  one seed): the usual session drift; only same-session rows compare.
- The final checkpoint is as good as the "best" one, so the rolling-score
  selection did not pick a better point here.

The pixel student (`train_pixel.py`, 100k teacher-driven + 3 x 25k
DAgger steps, 4 x 84x84 RGB) reached 96.5% training accuracy but plays
at half the teacher's level (1513, life 188, 0/10 clean). Its mid-evals
were still rising at round 3 (1090 -> 1105 -> 1530), and that trainer has
no held-out split yet.

### 20. Restores were broken; later levels now exist on disk; shooting is mostly wasted (2026-11-11)

**Save-state restores stalled the digger.** `DiggerEnv.load_state`
released the held key before `unserialize`, but the release reaches the
guest on the next `run()`, *after* the restore. The game saw the key go
up while the wrapper thought it was still held, so it never pressed it
again: after every restore the digger stood still until the action
changed. Everything restore-based (the search teacher, the old
near-death scenarios) ran on this. Now a restore starts with nothing
held and the next step presses its key fresh.

**Restores are also not timing-exact.** Each one moves the game's tick
phase ~1 frame forward. Restoring every step made the game visibly
faster and dropped the teacher from ~3000 to ~1200 a game. So lookahead
must not restore the *played* game: run probes in a second process
(only one core per process), loading `save_state()` blobs there.

**Later levels.** `gamedat.level` is at `LEVEL_OFFSET = 0x2599E`.
`DiggerEnv.poke_level` patches it inside a serialize blob (RAM is stored
uncompressed there). `tools/level_states.py` pokes N-1 shortly before
level 1 ends, so the game builds level N itself: layout, background,
monster count and speed. Levels 1-8 each have their own plan; 9 and 10
reuse 6 and 7 (`levplan`). `data/save/levels` holds 4 starts per level
1-10 plus mid-level states, 87 in all; `DiggerEnv(start_states=...,
start_prob=p)` and `train_pixel.py --start-states/--start-prob` start
episodes from them.

One life from each level start (`evaluate.py --levels`, 8 runs per level):

| Level | teacher clear / eaten | `ppo_ft1` final | pixel DAgger v1 |
|---|---|---|---|
| 1 | 2/8, 65% | **8/8, 100%** | 1/8, 69% |
| 2 | 0/8, 22% | 0/8, 56% | 0/8, 17% |
| 3 | 0/8, 25% | 0/8, 31% | 0/8, 26% |
| 4 | 0/8, 26% | 0/8, 19% | 0/8, 13% |
| 5-10 | 0/48, 21-36% | 0/48, 11-20% | 0/48, **0-2%** |

- Nobody clears a later level in one life. PPO's edge over the teacher
  is a level-1 specialisation: it is behind the teacher from level 4 on.
- The pixel student stops eating emeralds at level 5: unseen backgrounds
  break it completely, not gradually.

**Shooting**, measured with `tools/fire_oracle.py`: from each step where FIRE
would be accepted and a monster is out, play FIRE then NOOPs vs only
NOOPs for 10 steps in a probe process; "would kill" if FIRE kills more.

| 8 games | fire-ready steps | would kill | fired | fired & killed | missed kills |
|---|---:|---:|---:|---:|---:|
| teacher | 4390 | 21 (0.5%) | 137 | 9 (7%) | 12 |
| `ppo_ft1` final | 3297 | 31 (0.9%) | 91 | 11 (12%) | 20 |

- A killing shot exists ~once per 100-200 steps, and both policies waste
  88-93% of their shots. Each miss costs the recharge (`levof10()*3+60`
  frames after the bullet dies).
- The cause is the teacher's labels, not missing observations. The obs
  already has "fire ready", facing, dirt and monsters. The bullet
  (`digger.c updatefire`) moves 8 px/frame horizontally and 7 px/frame
  vertically, and stops at any non-black pixel: dirt edges of a part-dug
  tile, emeralds, bags. The teacher's tile-level line-of-sight check
  can't see that, BC copies it, and PPO's kill reward is too rare (and
  anchored to the teacher) to unlearn it.

## Lessons ported from ../snake (2026-08-31)

The snake project ran the same recipe on a much cheaper simulator and
got 40× the experiment count out of it. Its README's findings #16-18 are
about *architecture*; here is which of them transfer to Digger and which
do not.

| Snake finding | Transfers? | Why / what it becomes here |
|---|---|---|
| #16 egocentric (head-centred) obs beats allocentric | **yes** | Same failure mode is latent in `SymbolicAgent`: `Flatten -> Linear(64*10*15, 128)` gives every board position private weights, so "monster one tile to my left" has to be relearned at each of the 150 positions. `--egocentric` pastes the 10×15 board into a 19×29 canvas with the digger pinned at the centre, plus an off-board plane. |
| #17 weight-tied conv iterator + head-local readout | **yes** | The 3-conv trunk has a 7-tile receptive field on a 15-wide board — it physically cannot relate the digger to an emerald across the map. `--arch iter` applies one shared residual block `--iter-steps` times (range grows with iterations, not parameters) and reads out a 7×7 window around the centre + a global mean-pool. **232k params vs 4.58M** for the egocentric CNN. |
| #17/#18 canonical (rotate-to-face-up) obs | **no** | Snake is rotation-symmetric; Digger is not. Gravity is real here — bags fall *down*, the under-bag transit rule is a statement about the vertical axis. Rotating the board would destroy the single most safety-relevant structure. Egocentric translation invariance is the part that survives. |
| #18 drop the engineered distance channel, let the net infer routing | **untested here, plausible** | Digger's symbolic obs has no distance channel to drop, but the analogue is `--shaping-coef` (Manhattan-to-emerald reward shaping). If the iterator can route by itself, the shaping term is a crutch at best and a distractor at worst. |
| #9 BC capacity wall — 15k too small, 300k-1.2M memorises, 4.78M generalises | **caution** | Read together with our own Lesson 7 (width-2 NatureCNN scored 483 vs width-1's 783) the resolution is *samples per parameter*, not parameter count: snake's 4.78M win came with 600k samples. Our pixel runs had 30k. Scale the net only alongside the DAgger budget. |
| #11 DAgger overtrains past a saturation point | **yes** | Matches our Lesson 11b, where mid-evals were still climbing at iter 3. The rule is the same: watch rollout agreement, early-stop when it stalls, don't fix the iteration count in advance. |
| #12/#15 `--save-best`, γ=0.997 for long horizons, vectorised envs | **yes** | `--save-best` was already worth 100 points on the pixel PPO run. γ=0.997 matters as soon as the objective is survival (see below). Vectorised symbolic envs are still not wired — one DOSBox process per env makes it the expensive one. |
| #14 eval seeds must be disjoint from collection seeds | **N/A but watch it** | Digger has no seed stream we control; DOSBox timing supplies the variation. Our equivalent hazard is the cross-session drift caveat above. |
| #15 pure reward learns new behaviour *once a navigation core exists* | **the load-bearing one** | Snake's from-scratch PPO never learned to play (ppo04: 1.4), but reward-only *fine-tunes* from a competent base learned genuinely new behaviour (2-apple routing 37 → 53; death rate 82% → 12%). That is exactly the shape of the survival experiment below, and the reason it is run both from scratch and from a BC init. |

## Reward strategies: what is the agent actually being paid for?

Every run before this one was paid in game score (plus emerald-distance
shaping). `SymbolicDiggerEnv` now decomposes the reward so the objective
itself is an experimental variable:

```
reward = score_weight * (game score delta)
       + shaping_coef * (Manhattan distance closed toward an emerald)
       + survival_reward     (per agent step that didn't end a life)
       - time_penalty
       - death_penalty       (on any life loss)
```

`--score-weight 0 --shaping-coef 0 --survival-reward 0.01` is the pure
**"maximise frames survived"** objective: the game's score is invisible
to the agent and the only thing that matters is not dying.

Two mechanics make this work:

- **γ=0.997, not 0.99.** With a constant per-step reward the return *is*
  the (discounted) episode length, so the discount horizon is the
  objective. At γ=0.99 the agent cannot see past ~100 steps — shorter
  than a single teacher life.
- **`--max-episode-steps` + truncation-aware GAE.** A survival-paid
  policy that finds a safe corner would otherwise run one episode for
  the whole job. The cap bounds it, and GAE bootstraps `V(s')` at the
  cut instead of treating it as death — without that the agent is taught
  that surviving to the cap is exactly as bad as dying there.

**The number to beat:** the v6 teacher survives a mean of **280 agent
steps per life** (median 267, max 488, n=60 from
`logs/deathlog_v6_baseline.jsonl`) — and it is *trying* to die, in the
sense that it walks into risk to collect emeralds. A random policy
already survives ~195. Survival-only training that lands between those
has learned nothing; the interesting question is whether it clears the
teacher, and whether it does so by playing well or by hiding.

### Result: survival-only reward is strictly dominated (2026-08-31)

Six 500k-step arms, all launched the same session, evaluated the same
session with `evaluate.py` over 20 full 3-life games each
(stochastic sampling). Life length is agent steps between deaths, n=60
lives per row.

| Policy | Game score | Life length | Notes |
|---|---:|---:|---|
| SmartHeuristic v6.4 (teacher) | **1362 ±133** | **266.6 ±8.8** | same-session reference |
| **Symbolic BC clone, v6.4 teacher** | **1175 ±84** | **259.2 ±8.9** | 95.4% teacher acc, **no PPO at all** |
| `score_ctrl` (old recipe, no BC) | 712 ±21 | 230.6 ±1.1 | see the degenerate-loop note below |
| `surv_ego` (survival only, egocentric) | 312 ±52 | 237.2 ±6.2 | best survival arm |
| `surv_score` (survival + score×0.02) | 282 ±44 | 227.8 ±3.2 | |
| `surv_bcinit` (survival only, **from the BC clone**) | 239 ±65 | 210.8 ±7.0 | started at 1175/259 |
| `surv_iter` (survival only, iterator) | 155 ±28 | 203.3 ±2.9 | collapsed to NOOP @55% |
| `surv_alloc` (survival only, allocentric) | 146 ±29 | 215.8 ±3.8 | never left uniform policy |
| Random policy | 124 ±44 | 194.4 ±4.8 | same-session floor |

Reading the life-length column: **the arms trained to maximise lifetime
are worse at lifetime than the arms that were never trained for it.**
The BC clone, which only ever imitated an emerald-collecting teacher,
survives 259 steps; the best pure-survival arm manages 237, and the
allocentric one 216 against a random floor of 194.

`surv_bcinit` is the decisive run. It *started* from the 1175-score /
259-life BC clone and 500k steps of anchor-free survival PPO took it to
239 / 211 — it made the policy worse at the exact quantity it was being
paid for.

**Why: standing still is the degenerate optimum, and Digger punishes it
only weakly.** A constant `+c` per step is identical across all six
actions, so the only thing that distinguishes them is the truncated tail
at death — a tiny, heavily delayed advantage. Tracking the modal action
across training tells the story:

| Arm | first 20 updates | last 20 updates |
|---|---|---|
| `surv_alloc` | UP @23% | DOWN @24% (still uniform) |
| `surv_ego` | UP @21% | LEFT @27% |
| `surv_iter` | RIGHT @23% | **NOOP @55%** |
| `surv_bcinit` | UP @44% | **NOOP @38%** |
| `score_ctrl` | NOOP @22% | **FIRE @61%** |

The two survival arms that never committed stayed at entropy 1.72-1.74
against a 1.79 maximum — a barely-perturbed uniform policy. The two that
did commit, committed to NOOP. `--max-episode-steps 2000` was never hit
in any of the six runs, so hiding never even survived long enough to be
truncated.

**Snake's finding #15 was mis-ported, and that is the lesson.** Snake's
death-averse fine-tune was `--reward-eat 1 --reward-die -5
--reward-step 0.005` — it *kept the task reward*, and the step bonus was
~8% of the food reward. Snake never ran a zero-task-reward arm. Survival
is a useful *auxiliary* term on top of a task reward; as the sole
objective it selects for inaction.

Two side-findings worth recording:

- **`score_ctrl` is a degenerate loop, not a policy.** Its best-by-score
  checkpoint evals at 712 — respectable next to the README's 250-290 for
  the same recipe — but its life length is 230.6 **±1.1**, median 232,
  max 239. Sixty lives all ending within nine steps of each other is a
  fixed action cycle (FIRE @61%) that dies on a timer, not a policy that
  reacts. In-trainer game score also regressed 175 → 51 over the run;
  `--save-best` is the only reason there is a usable checkpoint at all.
- **The architecture A/B is inconclusive *from these runs*.** Under
  identical survival reward, egocentric CNN (237.2 life) beat
  allocentric (215.8) beat iterator (203.3), which is the predicted
  ordering for the first two — but all three sit in the
  "learned-almost-nothing" band, so the comparison is measuring which
  net degrades most gracefully under a no-signal objective, not which
  encodes Digger better. The architecture question has to be re-asked in
  the BC setting, where there is real signal (in flight).

### The actual headline: symbolic BC on the v6.4 teacher

The control run buried in the table above is the best symbolic result
this project has produced. **A plain BC clone of SmartHeuristic v6.4 —
60k samples, 10 epochs, no PPO, no DAgger — evaluates at 1175 ±84 game
score, 86% of the same-session teacher's 1362.** For comparison the
previous symbolic best was 891 (BC + PPO + anchor, v5-teacher era) and
the best pixel result is 1042.

This confirms Lesson 11a from the other direction: the teacher upgrade
propagates straight through to the student, and on symbolic obs the
imitability is high enough (95.4% action accuracy) that BC alone gets
most of the way. It also re-frames every PPO result here — the bar for
"PPO helped" is now 1175, not 891.

## How to run

Build once: `make` (DOSBox Pure core + pybind11 extension into `.venv`).

### Watch / evaluate

```bash
python watch.py teacher                     # the teacher
python watch.py data/checkpoints/dagger3/ppo_sym_final.pt --overlay
python evaluate.py teacher --episodes 20    # same-session baseline
python evaluate.py data/checkpoints/dagger3/ppo_sym_final.pt --episodes 20
```

### Symbolic BC -> DAgger -> PPO (the current recipe)

```bash
# 1. BC on teacher traces (~2 min to collect 200k steps on 8 cores).
for i in $(seq 0 7); do
  python collect.py --out data/traces/bc_v1_$i.npz --steps 25000 \
      --explore 0.05 --seed $i &
done; wait
python train_symbolic.py --total-timesteps 0 --bc-epochs 20 \
  --bc-traces data/traces/bc_v1_{0..6}.npz --bc-val data/traces/bc_v1_7.npz \
  --run-name bc_v1

# 2. DAgger round k: the previous student drives, the teacher labels,
#    retrain on everything (k = 1, 2, 3; see Lesson 18).
for i in $(seq 0 7); do
  python collect.py --out data/traces/dagger1_$i.npz --steps 25000 \
      --explore 0.05 --seed $((100+i)) \
      --student data/checkpoints/bc_v1/ppo_sym_final.pt &
done; wait
python train_symbolic.py --total-timesteps 0 --bc-epochs 15 \
  --bc-traces data/traces/{bc_v1_{0..6},dagger1_{0..6}}.npz \
  --bc-val data/traces/{bc_v1_7,dagger1_7}.npz --run-name dagger1

# 3. PPO fine-tune.
python train_symbolic.py --total-timesteps 2000000 \
  --load-agent data/checkpoints/dagger3/ppo_sym_final.pt \
  --value-warmup-updates 10 --bc-epochs 0 \
  --bc-traces data/traces/dagger3_{0..7}.npz \
  --bc-anchor-coef 0.3 --bc-anchor-final 0.05 \
  --score-weight 0.02 --shaping-coef 0 --lr 1e-4 \
  --ent-coef 0.001 --ent-coef-final 0.0005 \
  --save-best --best-metric score --run-name ppo_ft1
```

### Near-death scenarios

```bash
# Capture save states N steps before each death, then check that a policy
# can escape them at all before training on them (Lesson 12b).
python -m tools.collect_death_states --out data/scenarios/deaths.pkl \
  --deaths 80 --lookbacks 15,25,40,60 \
  --policy ckpt --checkpoint data/checkpoints/dagger3/ppo_sym_final.pt
python -m tools.collect_death_states --out data/scenarios/deaths.pkl --probe \
  --policy ckpt --checkpoint data/checkpoints/dagger3/ppo_sym_final.pt
# Train with --resume-from data/scenarios/deaths.pkl --resume-prob 0.2
```

### Save / restore game state

```bash
python play.py --live --save-slot data/save/right_edge.pkl   # S saves
python play.py --live --save-slot data/save/right_edge.pkl --resume
python train_symbolic.py --resume-from data/save/right_edge.pkl ...
```

## Key constants

- Tile grid: 15 cols x 10 rows. In game coords (320x200) tile (h, v) is
  at x = h*20 + 12, y = v*18 + 18; on the 640x400 frame that is 40x36 px
  tiles from (24, 36).
- RAM (region 0): score `0x282E0` (int32), digger record `0x259C8`,
  emeralds `0x25A3E`, monsters `0x25AD6`, bags `0x25D1A`, dirt field
  `0x2600C` (layouts in Lesson 15).
- `frame_skip = 4`, `frame_stack = 4`; pixel `obs_size = 84`.
- Actions `{NOOP, LEFT, RIGHT, UP, DOWN, FIRE}`; FIRE is F1.
- The game turns the digger only at tile boundaries (`digger.c`
  updatedigger): plan from `heading_tile`, not the rounded tile.
