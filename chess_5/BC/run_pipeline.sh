#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BOARD_SIZE="${BOARD_SIZE:-5}"
RUN_NAME="${1:-bc-${BOARD_SIZE}x${BOARD_SIZE}-$(date +%Y%m%d-%H%M%S)}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$ROOT/BC}"
EXPERT_GAMES="${EXPERT_GAMES:-10000}"
AGGREGATE_GAMES="${AGGREGATE_GAMES:-5000}"
GEN_WORKERS="${GEN_WORKERS:-16}"
EVAL_WORKERS="${EVAL_WORKERS:-8}"
EVAL_GAMES="${EVAL_GAMES:-200}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-256}"
TRAIN_WORKERS="${TRAIN_WORKERS:-4}"
DEVICE="${DEVICE:-auto}"
SEED="${SEED:-0}"
AGGREGATE_MAX_SAMPLES="${AGGREGATE_MAX_SAMPLES:-100000}"
CACHE_LABELS_PER_STATE="${CACHE_LABELS_PER_STATE:-4}"
MAX_CANDIDATES="${MAX_CANDIDATES:-12}"

DATA_ROOT="$ARTIFACT_ROOT/data/$RUN_NAME"
CHECKPOINT_ROOT="$ARTIFACT_ROOT/checkpoints/$RUN_NAME"
TB_ROOT="$ARTIFACT_ROOT/runs/$RUN_NAME"
EVAL_ROOT="$ARTIFACT_ROOT/evaluations/$RUN_NAME"
STATE_ROOT="$ARTIFACT_ROOT/pipeline_state/$RUN_NAME"

EXPERT_DATA="$DATA_ROOT/01_expert"
BC_V1_DIR="$CHECKPOINT_ROOT/02_bc_v1"
EVAL_V1_JSON="$EVAL_ROOT/03_eval_bc_v1.json"
AGGREGATE_DATA="$DATA_ROOT/04_aggregate"
BC_V2_DIR="$CHECKPOINT_ROOT/05_bc_v2"

mkdir -p "$TB_ROOT" "$EVAL_ROOT" "$STATE_ROOT"

run_logged() {
  local step="$1"
  shift
  mkdir -p "$TB_ROOT/$step"
  echo
  echo "[$(date '+%F %T')] START $step"
  "$@" 2>&1 | tee -a "$TB_ROOT/$step/console.log"
  touch "$STATE_ROOT/$step.done"
  echo "[$(date '+%F %T')] DONE  $step"
}

is_complete_dataset() {
  local metadata="$1/metadata.json"
  [[ -f "$metadata" ]] && rg -q '"status": "complete"' "$metadata"
}

PYTHON=(python)

echo "BC pipeline run: $RUN_NAME"
echo "Board: ${BOARD_SIZE}x${BOARD_SIZE}"
echo "TensorBoard: $TB_ROOT"
echo "Monitor with: tensorboard --logdir $TB_ROOT"

if [[ -f "$STATE_ROOT/01_generate_expert.done" ]] || is_complete_dataset "$EXPERT_DATA"; then
  echo "SKIP 01_generate_expert (already complete)"
else
  run_logged 01_generate_expert "${PYTHON[@]}" BC/generate.py \
    --output "$EXPERT_DATA" --mode expert --board-size "$BOARD_SIZE" \
    --games "$EXPERT_GAMES" --workers "$GEN_WORKERS" --seed "$SEED" \
    --max-candidates "$MAX_CANDIDATES" --cache-labels-per-state "$CACHE_LABELS_PER_STATE" \
    --tb-dir "$TB_ROOT/01_generate_expert"
fi

if [[ -f "$STATE_ROOT/02_train_bc_v1.done" ]]; then
  echo "SKIP 02_train_bc_v1 (already complete)"
else
  RESUME=()
  [[ -f "$BC_V1_DIR/latest.pt" ]] && RESUME=(--resume)
  run_logged 02_train_bc_v1 "${PYTHON[@]}" BC/train.py \
    --data-dir "$EXPERT_DATA" --run-name 02_bc_v1 --output-dir "$CHECKPOINT_ROOT" \
    --board-size "$BOARD_SIZE" --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" \
    --workers "$TRAIN_WORKERS" --device "$DEVICE" --seed "$SEED" \
    --tb-dir "$TB_ROOT/02_train_bc_v1" "${RESUME[@]}"
fi

if [[ -f "$STATE_ROOT/03_eval_bc_v1.done" ]]; then
  echo "SKIP 03_eval_bc_v1 (already complete)"
else
  run_logged 03_eval_bc_v1 "${PYTHON[@]}" BC/eval.py \
    --checkpoint "$BC_V1_DIR/best.pt" --board-size "$BOARD_SIZE" \
    --games-per-color "$EVAL_GAMES" --workers "$EVAL_WORKERS" --seed "$((SEED + 10000))" \
    --max-candidates "$MAX_CANDIDATES" \
    --output "$EVAL_V1_JSON" --tb-dir "$TB_ROOT/03_eval_bc_v1"
fi

if [[ -f "$STATE_ROOT/04_generate_aggregate.done" ]] || is_complete_dataset "$AGGREGATE_DATA"; then
  echo "SKIP 04_generate_aggregate (already complete)"
else
  run_logged 04_generate_aggregate "${PYTHON[@]}" BC/generate.py \
    --output "$AGGREGATE_DATA" --mode aggregate --board-size "$BOARD_SIZE" \
    --checkpoint "$BC_V1_DIR/best.pt" --games "$AGGREGATE_GAMES" \
    --workers "$GEN_WORKERS" --seed "$SEED" --max-candidates "$MAX_CANDIDATES" \
    --cache-labels-per-state "$CACHE_LABELS_PER_STATE" \
    --tb-dir "$TB_ROOT/04_generate_aggregate"
fi

if [[ -f "$STATE_ROOT/05_train_bc_v2.done" ]]; then
  echo "SKIP 05_train_bc_v2 (already complete)"
else
  RESUME=()
  [[ -f "$BC_V2_DIR/latest.pt" ]] && RESUME=(--resume)
  run_logged 05_train_bc_v2 "${PYTHON[@]}" BC/train.py \
    --data-dir "$EXPERT_DATA" "$AGGREGATE_DATA" \
    --aggregate-max-samples "$AGGREGATE_MAX_SAMPLES" \
    --run-name 05_bc_v2 --output-dir "$CHECKPOINT_ROOT" \
    --board-size "$BOARD_SIZE" --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" \
    --workers "$TRAIN_WORKERS" --device "$DEVICE" --seed "$SEED" \
    --tb-dir "$TB_ROOT/05_train_bc_v2" "${RESUME[@]}"
fi

echo
echo "Pipeline complete: $RUN_NAME"
echo "Best BC-v2 checkpoint: $BC_V2_DIR/best.pt"
echo "TensorBoard root: $TB_ROOT"
