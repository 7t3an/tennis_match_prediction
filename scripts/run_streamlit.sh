#!/usr/bin/env bash
set -euo pipefail

# Purpose: Cleanly restart Streamlit using workspace venv on a fixed port.
# Why: Avoid Exit Code 127 (wrong interpreter), reduce file-watcher CPU.
# What could happen: If port 8501 is busy, we free it; if venv is missing, script exits.

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

VENV_PY="${ROOT_DIR}/.venv/bin/python"
PORT="${PORT:-8501}"

if [[ ! -x "$VENV_PY" ]]; then
  echo "[ERROR] venv Python not found at $VENV_PY" >&2
  echo "[HINT] Create venv: python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt" >&2
  exit 1
fi

echo "[INFO] Stopping any existing Streamlit (port $PORT)"
pkill -f "streamlit run app.py" || true
if lsof -ti:"$PORT" >/dev/null 2>&1; then
  kill "$(lsof -ti:"$PORT")" || true
fi

echo "[INFO] Starting Streamlit via venv"
"$VENV_PY" -m streamlit run app.py \
  --server.headless true \
  --server.port "$PORT" \
  --server.fileWatcherType none
