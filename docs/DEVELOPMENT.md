# Development Guide

## Prerequisites

- macOS 15+ (Sequoia)
- Xcode 16+ or Swift 6+ toolchain
- Apple Silicon (M1 or later)
- Python 3.11+ (installed automatically by the setup script)
- Internet connection for initial setup

## First-time setup

The app handles setup automatically on first launch via the onboarding flow. For manual development setup:

```bash
git clone <repo-url>
cd auralux

# Set up the Python environment (clones ACE-Step 1.5, installs deps)
cd AuraluxEngine
./setup_env.sh
```

## Running the app

### From Xcode

Open `Package.swift` in Xcode, select the `Auralux` executable target, and run (Cmd+R).

### From terminal

```bash
swift run Auralux
```

The app will automatically detect the AuraluxEngine directory and manage the server lifecycle.

## Running the engine server manually

If you prefer to manage the server separately (useful for debugging):

```bash
cd AuraluxEngine
./start_api_server_macos.sh
```

By default the server binds to `127.0.0.1:8765`. Override with `AURALUX_SERVER_PORT`.

The app detects externally running servers and connects to them automatically.

## Running tests

```bash
swift test
```

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AURALUX_SERVER_PORT` | `8765` | Port for the inference server |
| `ACESTEP_LM_BACKEND` | `mlx` | LM backend (`mlx` for Apple Silicon) |
| `ACESTEP_CONFIG_PATH` | `acestep-v15-turbo` | DiT model configuration |
| `ACESTEP_LM_MODEL_PATH` | `acestep-5Hz-lm-0.6B` | LM model name |
| `ACESTEP_DEVICE` | `auto` | Inference device |
| `ACESTEP_INIT_LLM` | `auto` | Whether to initialize the LLM |

## Project structure

```
Auralux/
├── AuraluxApp.swift           # App entry point, service injection
├── Components/                # Reusable UI components
│   ├── AudioDropZone.swift
│   ├── EngineStatusView.swift # Engine status badge
│   ├── ProgressOverlay.swift
│   ├── SliderControl.swift
│   └── TagChip.swift
├── Models/                    # SwiftData models
├── Services/
│   ├── EngineService.swift    # Engine lifecycle management
│   ├── InferenceService.swift # HTTP client for Python server
│   ├── AudioPlayerService.swift
│   ├── AudioExportService.swift
│   └── ...
├── ViewModels/                # @Observable state management
├── Views/
│   ├── ContentView.swift      # Root view with setup/main switching
│   ├── Onboarding/
│   │   └── SetupView.swift    # First-run setup experience
│   ├── Generation/
│   ├── Player/
│   ├── History/
│   ├── Settings/
│   └── Sidebar/
└── Utilities/
```

## Suggested checks before opening a PR

```bash
swift test
swift build
python3 -m py_compile AuraluxEngine/server.py
```

## Troubleshooting

- **Engine not found**: Run from the repository root, or set the `engine.directoryOverride` UserDefaults key to the AuraluxEngine path.
- **Port conflict on 8765**: Set `AURALUX_SERVER_PORT=<new-port>` before starting server.
- **Python venv issues**: Delete `AuraluxEngine/ACE-Step-1.5/.venv` and rerun `./setup_env.sh`.
- **Model download stalls**: Check your internet connection. Models download from HuggingFace (~4 GB).
- **Server crashes**: Check the setup log in the app or server output in terminal.

## Coding expectations

- Keep PRs scoped and testable.
- Add or update tests for behavior changes.
- Update docs when introducing workflow or API changes.
- Follow Swift 6 concurrency patterns (`@Observable`, `actor`, `Sendable`).
