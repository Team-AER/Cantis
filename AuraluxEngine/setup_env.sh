#!/usr/bin/env zsh
# Auralux — ACE-Step 1.5 environment setup for macOS Apple Silicon
#
# This script:
#   1. Clones ACE-Step 1.5 if not already present
#   2. Installs `uv` if not available
#   3. Runs `uv sync` to install all Python dependencies
#
# Requirements: macOS with Apple Silicon (arm64), Python 3.11+

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ACE_STEP_DIR="$SCRIPT_DIR/ACE-Step-1.5"
ACE_STEP_REPO="https://github.com/ace-step/ACE-Step-1.5.git"

echo "============================================"
echo "  Auralux Engine — Environment Setup"
echo "============================================"
echo

# ── 1. Clone ACE-Step 1.5 if missing ─────────────────────────────────────

if [ ! -d "$ACE_STEP_DIR" ]; then
    echo "[Setup] Cloning ACE-Step 1.5 …"
    git clone --depth 1 "$ACE_STEP_REPO" "$ACE_STEP_DIR"
    echo "[Setup] Clone complete."
    echo
elif [ ! -f "$ACE_STEP_DIR/pyproject.toml" ]; then
    echo "[Setup] ACE-Step-1.5 directory exists but looks incomplete. Re-cloning …"
    rm -rf "$ACE_STEP_DIR"
    git clone --depth 1 "$ACE_STEP_REPO" "$ACE_STEP_DIR"
    echo "[Setup] Clone complete."
    echo
else
    echo "[Setup] ACE-Step 1.5 already present at $ACE_STEP_DIR"
    echo
fi

# ── 2. Ensure `uv` is available ──────────────────────────────────────────

_find_uv() {
    if command -v uv &>/dev/null; then return 0; fi
    if [ -x "$HOME/.local/bin/uv" ]; then export PATH="$HOME/.local/bin:$PATH"; return 0; fi
    if [ -x "$HOME/.cargo/bin/uv" ]; then export PATH="$HOME/.cargo/bin:$PATH"; return 0; fi
    return 1
}

if ! _find_uv; then
    echo "[Setup] Installing uv package manager …"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    if ! command -v uv &>/dev/null; then
        echo "[Error] uv installation failed. Install manually:"
        echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi
    echo "[Setup] uv installed."
    echo
fi

echo "[Setup] Using uv: $(command -v uv)"
echo

# ── 3. Install dependencies via uv sync ──────────────────────────────────

echo "[Setup] Syncing Python environment (this may take several minutes on first run) …"
echo

cd "$ACE_STEP_DIR"

if ! uv sync; then
    echo
    echo "[Retry] Online sync failed, trying offline …"
    if ! uv sync --offline; then
        echo "[Error] Failed to sync environment."
        echo "  Check your internet connection and try again."
        exit 1
    fi
fi

echo
echo "============================================"
echo "  Environment ready!"
echo "============================================"
echo "  ACE-Step 1.5: $ACE_STEP_DIR"
echo "  Virtual env:  $ACE_STEP_DIR/.venv"
echo
