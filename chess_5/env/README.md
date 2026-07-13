# Gomoku Gymnasium Environment

This package provides a Gymnasium-compatible Gomoku environment for self-play RL.

Runtime dependencies:

- `gymnasium` for the RL interface
- `pygame` only for `render_mode="human"` and `render_mode="rgb_array"`

```python
from env import GomokuEnv, make_vector_env

env = GomokuEnv(board_size=5)
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(0)

vec_env = make_vector_env(num_envs=8, board_size=5)
obs, info = vec_env.reset()
```

The observation is a dict with:

- `board`: `board_size x board_size`, where black is `1`, white is `-1`, empty is `0`
- `current_player`: `1` for black, `-1` for white
- `action_mask`: valid moves as a flat `0/1` vector

Actions are flattened board positions: `action = row * board_size + col`.

`info["legal_actions"]` has a fixed length of `board_size ** 2` so it can be
stacked by Gymnasium vector environments. Unused entries are padded with `-1`;
`obs["action_mask"]` is the preferred way to select valid moves.

## Vector environment autoreset

`make_vector_env` uses Gymnasium's `SameStep` autoreset mode. When an element
of `terminated` or `truncated` is true:

- the returned batched observation at that index is the reset observation of
  the next episode;
- `info["final_obs"][index]` is the terminal observation;
- terminal reward/outcome fields are under `info["final_info"]` (Gymnasium 1.x
  stores this as a dict of batched arrays);
- top-level info fields at that index describe the reset episode and must not
  be used as terminal statistics.

For graphical human play, create only one environment:

```python
env = GomokuEnv(board_size=9, render_mode="human")
obs, info = env.reset()

while True:
    if obs["current_player"].item() == 1:
        action = env.wait_for_human_action()
    else:
        action = agent.act(obs, info["action_mask"])

    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```
