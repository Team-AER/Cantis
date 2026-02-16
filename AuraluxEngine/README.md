# Auralux Engine — ACE-Step v1.5 Inference Backend

This folder contains the local Python API server that bridges the Auralux
macOS app with the ACE-Step v1.5 music generation model.

## Architecture

`server.py` is a thin HTTP adapter that:
- Wraps ACE-Step 1.5's Python inference API (`acestep.inference.generate_music`)
- Keeps the REST contract the Swift app already speaks
- Runs the DiT on PyTorch MPS (Apple Silicon GPU)
- Runs the optional 5Hz LM on MLX for native Apple Silicon acceleration
- Auto-downloads models from HuggingFace on first use

## REST Endpoints

- `GET  /health` — Server and model status
- `POST /generate` — Submit a generation job
- `GET  /jobs/<id>` — Poll job progress
- `POST /jobs/<id>/cancel` — Cancel a running job
- `POST /models/download` — Trigger model download

## Quick Start

```bash
cd AuraluxEngine
./setup_env.sh              # Clone ACE-Step 1.5, install deps via uv
./start_api_server_macos.sh # Start the server on port 8765
```

Models (~4GB total) are downloaded from HuggingFace on the first generation
request and cached in `ACE-Step-1.5/checkpoints/`.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `AURALUX_SERVER_PORT` | `8765` | Server port |
| `ACESTEP_CONFIG_PATH` | `acestep-v15-turbo` | DiT model name |
| `ACESTEP_LM_MODEL_PATH` | `acestep-5Hz-lm-0.6B` | LM model name |
| `ACESTEP_LM_BACKEND` | `mlx` | LM backend (mlx for Apple Silicon) |
| `ACESTEP_DEVICE` | `auto` | Compute device (auto/mps/cpu) |
| `ACESTEP_INIT_LLM` | `auto` | LM init (auto/true/false) |
| `ACESTEP_OFFLOAD_TO_CPU` | `false` | Offload models to CPU when idle |

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.11-3.12 (via `uv` package manager)
- ~4GB disk space for model weights
- ~4GB RAM for inference

## Directory Structure

```
AuraluxEngine/
├── server.py                  # Auralux REST adapter
├── setup_env.sh               # Environment setup script
├── start_api_server_macos.sh  # Server launch script
├── requirements.txt           # Dependency documentation
├── README.md                  # This file
└── ACE-Step-1.5/              # Cloned at setup time (gitignored)
    ├── acestep/               # ACE-Step Python package
    ├── checkpoints/           # Downloaded model weights
    ├── pyproject.toml         # Dependency definitions
    └── .venv/                 # Python virtual environment
```
