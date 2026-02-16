# Auralux

Auralux is a native macOS application for AI music generation, powered by ACE-Step v1.5 running entirely on Apple Silicon. Generate music from text prompts and lyrics, with full playback, history, and multi-format export.

## Features

- **Text-to-music generation** — describe the music you want and Auralux generates it locally
- **Lyrics support** — write vocal tracks with verse/chorus structure
- **Tag system** — genre, instrument, and mood tags for precise control
- **Real-time playback** — waveform visualization and spectrum analyzer
- **Multi-format export** — WAV, FLAC, MP3, AAC, ALAC with configurable settings
- **Preset system** — save and reuse generation configurations
- **Generation history** — browse, search, and favorite past generations
- **On-device inference** — all processing on Apple Silicon, no cloud required
- **Guided setup** — first-launch onboarding handles everything automatically

## System requirements

- macOS 15+ (Sequoia)
- Apple Silicon (M1 or later)
- 8 GB RAM minimum (16 GB recommended)
- ~6 GB disk space (for models and Python environment)
- Internet connection (for initial setup and model download)

## Quick start

### Option 1: Launch the app (recommended)

Build and run the app. On first launch, the built-in setup flow will:
1. Clone ACE-Step 1.5 and install Python dependencies
2. Start the local inference server
3. Download AI models on first generation (~4 GB)

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

## Running tests

```bash
swift test
```

## Project layout

- `Auralux/` — SwiftUI app (views, view models, services, models, components)
- `AuraluxEngine/` — Python inference server wrapping ACE-Step v1.5
- `AuraluxTests/` — unit tests
- `docs/` — architecture, development, and release documentation

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Development Guide](docs/DEVELOPMENT.md)
- [Pending Plan](docs/PENDING_PLAN.md)
- [Release Checklist](docs/RELEASE_CHECKLIST.md)

## Technology

| Layer | Technology |
|-------|-----------|
| UI | SwiftUI (macOS 15+) |
| State | `@Observable`, SwiftData |
| Inference | ACE-Step v1.5 (PyTorch MPS + MLX) |
| Audio | AVAudioEngine, Accelerate (FFT) |
| Export | AVFoundation |
| Build | Swift Package Manager |

## License

Distributed under the MIT License. See [LICENSE](LICENSE).
