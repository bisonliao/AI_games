# H.E.R.O. Depth Curriculum Environment

The curriculum uses frozen format-v2 ALE checkpoints. The human teacher marks
each desired start explicitly. Stage numbering is:

```text
stage = max_depth_of_current_level - checkpoint_depth + 1
```

Each Level is an independent episode, so each Level has its own maximum depth.
Stage 1 contains the deepest valid checkpoint from every Level. No RAM, room,
or screen transition can change depth or create a checkpoint automatically.

## Record a curriculum

Start a fresh game and play normally:

```bash
python HeroEnv/teacher.py
```

Every curriculum episode starts at depth `-1`. This includes initial startup,
manual restart, entering Level 2 after rescuing the Level 1 miner, and a retry
after losing a life. Uncommitted captures from the previous episode are
discarded. The controls are:

- `D`: increment depth, then capture the exact current ALE state and frame.
- `U`: undo the latest uncommitted `D` capture and decrement depth.
- `R` or `F2`: discard uncommitted candidates and restart.

Only `D` creates candidates. They remain in memory and are committed to
`checkpoints/curriculum.draft.json` only after the miner in that Level is
rescued without losing a life. By default a captured state needs at least two
lives and 60% power. A cloned-state validator also checks 120 NOOP frames,
60-frame action responsiveness, and training sticky-action seeds. Rejected
states are retained under `checkpoints/quarantine`.

Each successful demonstration records its remaining raw frames. The training
budget is twice the demonstrated time after accounting for action repeat,
clamped to 100–5000 DQN decisions.

After all desired `D` captures have been accepted, freeze the dataset:

```bash
python HeroEnv/teacher.py --freeze-curriculum
```

This writes an immutable `curriculum-vNNNN.json` and the active
`curriculum.json`, both with a content hash. Adding data later creates a new
version rather than changing an earlier frozen manifest.

Inspect every exact checkpoint reset frame after freezing:

```bash
python HeroEnv/check.py
```

The script writes every variant to the project-root `tmp/` directory as
`<stage>_<idx>.jpg` and prints its task/checkpoint ID mapping.

## Runtime semantics

```python
from HeroEnv import make_hero_level_1_to_2_env

env = make_hero_level_1_to_2_env(
    training=True,
    curriculum_stage=1,
    checkpoint_reset_probability=1.0,
    include_easier_stages=False,
    frameskip=1,
    repeat_action_probability=0.25,
)
```

Reset first samples a task uniformly within the Stage, then a healthy variant
uniformly within that task. `options={"checkpoint_id": ...}` selects an exact
variant for deterministic evaluation.

- Advancing beyond the reset Level means the miner was rescued and terminates
  successfully before a next-Level frame is exposed.
- `training=False` 从正常游戏起点创建环境时，一个 episode 包含
  Level 1→Level 2，并在进入 Level 3 时成功结束；Level 3 画面不会返回。
- `training=True` 从 curriculum checkpoint reset 时，一个 task episode
  只负责 checkpoint 所在的一个 Level：Level 1 checkpoint 进入 Level 2
  成功，Level 2 checkpoint 进入 Level 3 成功。
- Any loss of life terminates unsuccessfully.
- Exhausting `hero_budget_decisions` terminates unsuccessfully.
- `info` includes Stage, task/checkpoint IDs, depth, budget, miner rescue,
  life-loss, and timeout fields.
