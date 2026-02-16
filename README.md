# Auralux

Auralux is a native macOS application for AI music generation, powered by ACE-Step v1.5 running entirely on Apple Silicon. Generate music from text prompts and lyrics, with full playback, history, and multi-format export — no cloud required.

## Features

- **Text-to-music generation** — describe the music you want and Auralux generates it locally
- **Lyrics support** — write vocal tracks with verse/chorus/bridge structure
- **Tag system** — genre, instrument, and mood tags for precise control
- **Real-time playback** — waveform visualization and FFT spectrum analyzer
- **Multi-format export** — WAV, FLAC, MP3, AAC, ALAC with configurable settings
- **Preset system** — save and reuse generation configurations
- **Generation history** — browse, search, and favorite past generations
- **Audio-to-audio** — import reference audio for style transfer and remixing
- **LoRA support** — load and manage LoRA models for style customization
- **Generation queue** — queue multiple generation jobs
- **On-device inference** — all processing on Apple Silicon via PyTorch MPS + MLX
- **Guided setup** — first-launch onboarding handles everything automatically
- **Log viewer** — built-in log window for debugging and monitoring

## System Requirements

- macOS 15+ (Sequoia)
- Apple Silicon (M1 or later)
- 8 GB RAM minimum (16 GB recommended)
- ~6 GB disk space (for models and Python environment)
- Internet connection (for initial setup and model download)

## Quick Start

### Option 1: Launch the app (recommended)

Build and run the app. On first launch, the built-in setup flow will:
1. Clone ACE-Step 1.5 and install Python dependencies via `uv`
2. Start the local inference server
3. Download AI models on first generation (~4 GB from HuggingFace)

```bash
swift run Auralux
```

Or open `Package.swift` in Xcode and run the Auralux target.

### Option 2: Manual setup

If you prefer to manage the engine separately:

```bash
# Set up the Python environment
cd AuraluxEngine
./setup_env.sh

# Start the inference server
./start_api_server_macos.sh
```

Then build and run the Swift app:

```bash
swift run Auralux
```

The app detects externally running servers and connects to them automatically.

## Running Tests

```bash
swift test
```

CI also validates the Python server: `python3 -m py_compile AuraluxEngine/server.py`

## Project Layout

```
Auralux/           SwiftUI app
├── Views/           Screens (Onboarding, Generation, Player, History, Settings, Sidebar, AudioToAudio)
├── ViewModels/      @Observable state management
├── Services/        Business logic (Engine, Inference, Audio, History, Presets, Queue)
├── Models/          SwiftData models (GeneratedTrack, Preset, Tag)
├── Components/      Reusable UI (TagChip, SliderControl, EngineStatusView, AudioDropZone)
└── Utilities/       Helpers (AppLogger, Constants, AudioFFT, FileUtilities)

AuraluxEngine/     Python inference server wrapping ACE-Step v1.5
AuraluxTests/      Unit tests (Models, Services, ViewModels)
docs/              Architecture, development, and release documentation
.github/           CI workflow, issue templates, PR template
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md) — system design, engine lifecycle, API contract
- [Development Guide](docs/DEVELOPMENT.md) — setup, workflow, project structure, troubleshooting
- [Pending Plan](docs/PENDING_PLAN.md) — remaining implementation items
- [Release Checklist](docs/RELEASE_CHECKLIST.md) — pre-release verification steps
- [Engine README](AuraluxEngine/README.md) — inference server setup and API reference
- [Contributing](CONTRIBUTING.md) — contribution guidelines and PR checklist
- [Security](SECURITY.md) — vulnerability reporting policy
- [Support](SUPPORT.md) — how to get help
- [Code of Conduct](CODE_OF_CONDUCT.md) — community standards

## Technology

| Layer | Technology |
|-------|-----------|
| UI | SwiftUI (macOS 15+) |
| State | `@Observable` macro, SwiftData |
| Inference | ACE-Step v1.5 (PyTorch MPS + MLX) |
| Audio | AVAudioEngine, Accelerate (vDSP FFT) |
| Export | AVFoundation (WAV, FLAC, MP3, AAC, ALAC) |
| Build | Swift Package Manager (Swift 6) |
| Backend | Python 3.11+ with `uv` package manager |
| CI | GitHub Actions (swift test + Python syntax validation) |

## License

Distributed under the MIT License. See [LICENSE](LICENSE).
