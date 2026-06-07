"""Smallest possible state save/restore test.

Save the state, then run a few different lengths of replay from both
the live continuation and a restore. Tell me exactly where divergence
starts (step 0 = pure save vs no-action, step 1 = first action, etc.).
"""

from __future__ import annotations

import numpy as np

from digger_env import DiggerEnv

N_WARMUP = 200


def main() -> None:
    env = DiggerEnv(max_steps=10**9, episodic_life=False)
    env.reset()
    for _ in range(N_WARMUP):
        env.step(DiggerEnv.RIGHT)

    state = env.save_state()
    print(f"saved: held_keys={state['held_keys']}  "
          f"last_action={state['last_action']}  steps={state['steps']}")

    rng = np.random.default_rng(0)
    actions = rng.integers(0, env.NUM_ACTIONS, size=10).tolist()
    print(f"replay actions: {actions}")

    # Pass A: continue from live state (no restore).
    frames_a = []
    for a in actions:
        env.step(int(a))
        frames_a.append(env._core.get_frame().copy())

    # Pass B: restore, then replay.
    env.load_state(state)
    print(f"after load: held_keys={list(env._core.get_held_keys())}  "
          f"last_action={env._last_action}  steps={env._steps}")
    frames_b = []
    for a in actions:
        env.step(int(a))
        frames_b.append(env._core.get_frame().copy())

    for i, (fa, fb) in enumerate(zip(frames_a, frames_b)):
        equal = np.array_equal(fa, fb)
        diff = int(np.sum(np.any(fa != fb, axis=-1)))
        mark = "OK" if equal else "DIFF"
        print(f"  step {i+1:2d}: {mark}  diff_pixels={diff}")

    # Also test: serialize -> unserialize WITHOUT any intervening play.
    # Then run the same 1 step from both. They MUST match.
    env.load_state(state)
    env.step(actions[0])
    fc = env._core.get_frame().copy()
    env.load_state(state)
    env.step(actions[0])
    fd = env._core.get_frame().copy()
    print(f"\ntwo back-to-back restores + same step: equal? "
          f"{np.array_equal(fc, fd)}  diff_pixels="
          f"{int(np.sum(np.any(fc != fd, axis=-1)))}")

    env.close()


if __name__ == "__main__":
    main()
