#!/usr/bin/env zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
PORT="${AURALUX_SERVER_PORT:-8765}"

if [ ! -d "$VENV_DIR" ]; then
  "$SCRIPT_DIR/setup_env.sh"
fi

source "$VENV_DIR/bin/activate"
python "$SCRIPT_DIR/server.py" --port "$PORT"
