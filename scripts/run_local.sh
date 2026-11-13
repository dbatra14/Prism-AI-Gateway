#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -x .venv/bin/python ]]; then
  echo "Missing .venv. Run: python3.11 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt"
  exit 1
fi

export HF_HOME="$PWD/.cache/huggingface"
export XDG_CACHE_HOME="$PWD/.cache"
mkdir -p "$HF_HOME"

cleanup() {
  kill "$alpha_pid" "$beta_pid" "$gateway_pid" 2>/dev/null || true
  wait "$alpha_pid" "$beta_pid" "$gateway_pid" 2>/dev/null || true
}

.venv/bin/python scripts/mock_provider.py --port 9001 --name alpha &
alpha_pid=$!

.venv/bin/python scripts/mock_provider.py --port 9002 --name beta &
beta_pid=$!

.venv/bin/python -m uvicorn app.main:app --port 8080 &
gateway_pid=$!

trap cleanup EXIT INT TERM

echo "Prism is starting. Open http://localhost:8080/console when it is ready."
echo "Press Control-C in this Terminal window to stop Prism."

wait "$gateway_pid"
