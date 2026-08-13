# Project Agent Rules

## Protect Training Artifacts

- Never delete project-level training data or results wholesale. In particular, do not run broad commands such as `rm -rf runs`, `rm -rf checkpoints`, or equivalent recursive deletion of those directories.
- Treat `runs/`, `checkpoints/`, TensorBoard event files, model checkpoints, evaluation outputs, and user-generated recordings as user data, even when they are untracked by git.
- Do not remove or overwrite existing training artifacts unless the user explicitly identifies the exact files and asks for that operation.
- Before any deletion, resolve and inspect the exact target paths. Prefer recoverable moves or deletion of one precisely identified file over recursive cleanup.
- Test and smoke-training outputs must use an isolated temporary directory such as `/tmp/tetris-test-*`, or a uniquely named test run directory outside the project's existing result directories.
- Never clean a shared project result directory merely because it was created during a test; verify ownership of every target file first.

## Validation

- Keep validation commands non-destructive with respect to existing `runs/` and `checkpoints/`.
- When reporting test cleanup, list the exact files removed. If existing user data may be affected, stop and request confirmation before proceeding.
