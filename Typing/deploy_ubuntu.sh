#!/usr/bin/env bash
set -euo pipefail

APP_HOST="${APP_HOST:-0.0.0.0}"
APP_PORT="${APP_PORT:-8997}"

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$APP_DIR/run"
LOG_DIR="$APP_DIR/logs"
PID_FILE="$RUN_DIR/typing-practice.pid"
LOG_FILE="$LOG_DIR/typing-practice.log"

cd "$APP_DIR"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "ERROR: Python virtual environment is not active."
  echo "Please run: source /path/to/venv/bin/activate"
  exit 1
fi

if ! command -v python >/dev/null 2>&1; then
  echo "ERROR: python command not found in current environment."
  exit 1
fi

mkdir -p "$RUN_DIR" "$LOG_DIR"

if [[ -f "$PID_FILE" ]]; then
  OLD_PID="$(cat "$PID_FILE")"
  if [[ "$OLD_PID" =~ ^[0-9]+$ ]] && kill -0 "$OLD_PID" >/dev/null 2>&1; then
    OLD_CMD=""
    if [[ -r "/proc/$OLD_PID/cmdline" ]]; then
      OLD_CMD="$(tr '\0' ' ' <"/proc/$OLD_PID/cmdline")"
    fi

    if [[ "$OLD_CMD" == *"typing_practice.py"* || "$OLD_CMD" == *"app.py"* ]]; then
      echo "Stopping old typing practice service, pid=$OLD_PID ..."
      kill "$OLD_PID"
      for _ in {1..20}; do
        if ! kill -0 "$OLD_PID" >/dev/null 2>&1; then
          break
        fi
        sleep 0.2
      done
    else
      echo "Ignoring stale pid file: $PID_FILE"
    fi
  fi
fi

echo "Starting typing practice service on ${APP_HOST}:${APP_PORT} ..."
HOST="$APP_HOST" PORT="$APP_PORT" nohup python -u typing_practice.py >>"$LOG_FILE" 2>&1 &
NEW_PID="$!"
echo "$NEW_PID" > "$PID_FILE"

sleep 1

if ! kill -0 "$NEW_PID" >/dev/null 2>&1; then
  echo "ERROR: service failed to start. Last log lines:"
  tail -n 40 "$LOG_FILE" || true
  exit 1
fi

echo "Service started."
echo "PID: $NEW_PID"
echo "URL: http://<server-ip>:${APP_PORT}"
echo "Log: $LOG_FILE"
