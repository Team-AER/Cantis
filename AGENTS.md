# AGENTS.md — Auralux AI Agent Guide

This document provides comprehensive context for AI coding agents working on the Auralux codebase. It covers architecture, conventions, file layout, and patterns you need to follow when making changes.

## Project Overview

Auralux is a **native macOS application for AI music generation**, powered by ACE-Step v1.5 running entirely on Apple Silicon. Users describe the music they want via text prompts, tags, and lyrics — and the app generates it locally using a bundled Python inference server.

- **Platform:** macOS 15+ (Sequoia), Apple Silicon only
- **Language:** Swift 6 (app), Python 3.11+ (inference backend)
- **Build system:** Swift Package Manager (no Xcode project file)
- **UI framework:** SwiftUI with `@Observable` macro
- **Persistence:** SwiftData (`GeneratedTrack`, `Preset`, `Tag`)
- **Inference:** ACE-Step v1.5 via local HTTP server (PyTorch MPS + MLX)

## Repository Structure

```
auralux/
├── Auralux/                        # SwiftUI macOS application
│   ├── AuraluxApp.swift            # @main entry point, service injection
│   ├── ContentView.swift           # Root view (setup vs. main switching)
│   ├── Info.plist                  # App metadata
│   ├── Entitlements.plist          # macOS entitlements
│   ├── Components/                 # Reusable UI components
│   │   ├── AudioDropZone.swift     # Drag-and-drop audio import zone
│   │   ├── EngineStatusView.swift  # Engine status badge (toolbar)
│   │   ├── ProgressOverlay.swift   # Progress indicator overlay
│   │   ├── SliderControl.swift     # Custom parameter slider
│   │   └── TagChip.swift           # Genre/mood/instrument tag chip
│   ├── Models/                     # SwiftData domain models
│   │   ├── GeneratedTrack.swift    # Generated music metadata + file path
│   │   ├── GenerationParameters.swift # Request parameters (Codable)
│   │   ├── Preset.swift            # Saved generation configurations
│   │   └── Tag.swift               # Reusable tag model
│   ├── Services/                   # Business logic layer
│   │   ├── AudioExportService.swift      # Multi-format export (WAV/FLAC/MP3/AAC/ALAC)
│   │   ├── AudioPlayerService.swift      # AVAudioEngine playback wrapper
│   │   ├── EngineService.swift           # Engine lifecycle (setup → start → health → shutdown)
│   │   ├── GenerationQueueService.swift  # Job queue management
│   │   ├── HistoryService.swift          # SwiftData CRUD for history
│   │   ├── InferenceService.swift        # HTTP client to Python server (actor)
│   │   ├── ModelManagerService.swift     # Model download status tracking
│   │   └── PresetService.swift           # Preset CRUD operations
│   ├── Utilities/                  # Helpers and constants
│   │   ├── AppLogger.swift         # Centralized logging (OSLog wrapper)
│   │   ├── AudioFFT.swift          # vDSP FFT for spectrum visualization
│   │   ├── Constants.swift         # App-wide constants (URLs, tags, dimensions)
│   │   └── FileUtilities.swift     # File path helpers
│   ├── ViewModels/                 # @Observable state management
│   │   ├── GenerationViewModel.swift   # Generation orchestration
│   │   ├── HistoryViewModel.swift      # History browsing state
│   │   ├── PlayerViewModel.swift       # Audio playback state
│   │   ├── SettingsViewModel.swift     # Settings persistence
│   │   └── SidebarViewModel.swift      # Navigation state
│   ├── Views/
│   │   ├── AudioToAudio/
│   │   │   ├── AudioImportView.swift   # Audio file import interface
│   │   │   └── LoRAManagerView.swift   # LoRA model management
│   │   ├── Generation/
│   │   │   ├── GenerationView.swift          # Main generation UI
│   │   │   ├── LyricEditorView.swift         # Lyric editing with structure tags
│   │   │   ├── ParameterControlsView.swift   # Duration, variance, seed sliders
│   │   │   └── TagEditorView.swift           # Tag selection and management
│   │   ├── History/
│   │   │   ├── HistoryBrowserView.swift  # Generation history browser
│   │   │   └── HistoryItemView.swift     # Individual history entry
│   │   ├── Onboarding/
│   │   │   └── SetupView.swift           # First-run setup flow
│   │   ├── Player/
│   │   │   ├── PlayerView.swift          # Audio playback interface
│   │   │   ├── SpectrumAnalyzerView.swift # Real-time FFT spectrum
│   │   │   └── WaveformView.swift        # Waveform visualization
│   │   ├── Settings/
│   │   │   ├── ModelSettingsView.swift   # Model configuration
│   │   │   └── SettingsView.swift        # Settings panel
│   │   ├── Sidebar/
│   │   │   ├── PresetListView.swift      # Preset navigation list
│   │   │   ├── RecentListView.swift      # Recent generations list
│   │   │   └── SidebarView.swift         # Main sidebar navigation
│   │   └── LogViewerView.swift           # Log viewer window
│   └── Resources/                  # Bundled assets
├── AuraluxEngine/                  # Python inference backend
│   ├── server.py                   # REST API wrapping ACE-Step v1.5 (~900 lines)
│   ├── setup_env.sh                # Clones ACE-Step, installs deps via uv
│   ├── start_api_server_macos.sh   # Launches server on port 8765
│   ├── test_generate.py            # Backend test script
│   ├── requirements.txt            # Dependency documentation
│   ├── README.md                   # Engine-specific documentation
│   └── ACE-Step-1.5/              # Cloned at setup time (gitignored)
│       ├── acestep/               # ACE-Step Python package
│       ├── checkpoints/           # Downloaded model weights (~4 GB)
│       ├── pyproject.toml         # Python dependency definitions
│       └── .venv/                 # Virtual environment (via uv)
├── AuraluxTests/                   # Swift unit tests
│   ├── ModelTests.swift            # SwiftData model tests
│   ├── ServiceTests.swift          # Service layer tests
│   └── ViewModelTests.swift        # ViewModel behavior tests
├── docs/                           # Technical documentation
│   ├── ARCHITECTURE.md             # System architecture overview
│   ├── DEVELOPMENT.md              # Development setup and workflow
│   ├── PENDING_PLAN.md             # Remaining implementation items
│   └── RELEASE_CHECKLIST.md        # Pre-release verification
├── .github/
│   ├── workflows/ci.yml            # CI: swift test + Python syntax check
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.yml
│   │   └── feature_request.yml
│   ├── pull_request_template.md
│   └── dependabot.yml
├── Package.swift                   # SPM manifest (Swift 6, macOS 15+)
├── AGENTS.md                       # This file
├── README.md                       # Project overview and quick start
├── CHANGELOG.md                    # Version history
├── CONTRIBUTING.md                 # Contribution guidelines
├── CODE_OF_CONDUCT.md              # Community standards
├── SECURITY.md                     # Security reporting policy
├── SUPPORT.md                      # Support channels
├── PRD.md                          # Original product requirements (reference only)
├── plan.md                         # Original implementation plan (reference only)
└── LICENSE                         # MIT License
```

## Architecture

### Two-Process Model

Auralux runs as two cooperating processes:

1. **Swift app** (`Auralux`) — UI, state management, persistence, audio playback/export, engine lifecycle
2. **Python server** (`AuraluxEngine/server.py`) — wraps ACE-Step v1.5 for music generation via a local HTTP REST API on `127.0.0.1:8765`

The Swift app manages the Python server as a subprocess. `EngineService` handles the full lifecycle: setup detection, environment provisioning, server start, health monitoring, and graceful shutdown.

### Layer Diagram

```
┌──────────────────────────────────────────────────┐
│           SwiftUI Views + ViewModels              │
│    (@Observable, @Environment, @Query)            │
├──────────────────────────────────────────────────┤
│              Service Layer                        │
│  EngineService · InferenceService (actor)         │
│  AudioPlayerService · AudioExportService          │
│  HistoryService · PresetService                   │
│  ModelManagerService · GenerationQueueService      │
├──────────────────────────────────────────────────┤
│           SwiftData Persistence                   │
│  GeneratedTrack · Preset · Tag                    │
├──────────────────────────────────────────────────┤
│        Local HTTP REST API (port 8765)            │
├──────────────────────────────────────────────────┤
│        AuraluxEngine/server.py                    │
│   ACE-Step v1.5 (PyTorch MPS + MLX)              │
└──────────────────────────────────────────────────┘
```

### Engine Lifecycle (EngineService)

The `EngineService` state machine drives the app UX:

```
unknown → notSetup → settingUp(progress) → starting → running → ready
                                                              ↘ error
```

- `unknown` — initial state, checking for existing setup
- `notSetup` — ACE-Step environment not found; triggers onboarding (`SetupView`)
- `settingUp(progress)` — running `setup_env.sh` as subprocess, streaming output
- `starting` — Python server process launched, waiting for `/health` response
- `running` — server is responding to health checks
- `ready` — server and models are loaded, generation is possible
- `error` — any failure (auto-restart attempted for transient failures)

### REST API Contract

The Swift `InferenceService` communicates with the Python server via:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Liveness, model status, device info |
| `/generate` | POST | Enqueue generation job, returns `jobID` |
| `/jobs/<id>` | GET | Poll job status, progress, and audio path |
| `/jobs/<id>/cancel` | POST | Request job cancellation |
| `/models/download` | POST | Trigger model download from HuggingFace |

### Generation Flow

1. User configures prompt, tags, lyrics, and parameters in `GenerationView`
2. `GenerationViewModel` builds a `GenerationRequest` and calls `InferenceService.generate()`
3. `InferenceService` (Swift actor) POSTs to `/generate`, receives a `jobID`
4. Polls `/jobs/<id>` for progress updates
5. Python server runs ACE-Step v1.5 inference (DiT via PyTorch MPS, LM via MLX)
6. On completion, a `GeneratedTrack` is persisted via `HistoryService` (SwiftData)
7. Track appears in history and can be played in `PlayerView`

### Persistence

SwiftData models:

- **`GeneratedTrack`** — audio file metadata, prompt, tags, generation parameters, file path
- **`Preset`** — saved generation configurations (tags, duration, variance, etc.)
- **`Tag`** — reusable tag library

Audio files are stored at `~/Library/Application Support/Auralux/Generated/` using relative paths for sandbox resilience.

### MPS Workarounds (Python Server)

The Python server includes runtime patches for PyTorch MPS bugs on Apple Silicon:

- `masked_fill` — CPU fallback for unsupported MPS operation
- `inference_mode` → `no_grad` — MPS does not support inference mode
- Audio codec decoding — CPU fallback
- Text encoder — CPU fallback
- DiT condition encoder — CPU fallback

These patches are applied at server startup before model loading.

## Coding Conventions

### Swift

- **Swift 6** strict concurrency — use `@Sendable`, `actor`, structured concurrency
- **`@Observable`** macro for all ViewModels and services (not `ObservableObject`)
- **`@Environment`** for dependency injection into views (not singletons)
- **`@MainActor`** on ViewModels, services that touch UI state, and `EngineService`
- **`actor`** for thread-safe services (`InferenceService`)
- **`async/await`** for all asynchronous operations (no Combine publishers)
- **SwiftData** `@Model` macro for persistence (not raw SQLite)
- Use `AppLogger.shared` for all logging with appropriate categories
- Constants live in `AppConstants` enum (no stringly-typed values)
- File naming matches type name: `FooService.swift` contains `FooService`

### Python

- **Python 3.11+** with `uv` package manager
- `server.py` is a single-file HTTP server using `ThreadingHTTPServer`
- All model-specific code stays in `server.py`; the Swift app does not import Python directly
- Environment variables for configuration (see `AuraluxEngine/README.md`)

### Project Patterns

- **MVVM + Services**: Views → ViewModels → Services → external systems
- **Service injection**: Services created in `AuraluxApp.init()`, injected via `.environment()`
- **No singletons** for business logic (except `AppLogger.shared` for logging)
- **SwiftData `ModelContainer`** created once in `AuraluxApp.init()` and passed via `.modelContainer()`
- **Graceful degradation**: Generate button disabled when engine not ready; SetupView shown when engine needs setup

### Git & Branch Conventions

- Branches: `feature/<short-name>`, `fix/<short-name>`, `docs/<short-name>`
- Commits: imperative mood summary (`Add queue retry backoff`, not `Added...`)
- Keep commits logically grouped; prefer small, focused PRs
- CI runs `swift test` and `python -m py_compile AuraluxEngine/server.py`

## Build & Run

```bash
# Build the app
swift build

# Run the app (first launch triggers onboarding)
swift run Auralux

# Run tests
swift test

# Manual engine setup (optional)
cd AuraluxEngine && ./setup_env.sh
cd AuraluxEngine && ./start_api_server_macos.sh
```

Or open `Package.swift` in Xcode and run the `Auralux` target.

## Dependencies

### Swift (Package.swift)

| Package | Import | Purpose |
|---------|--------|---------|
| `apple/swift-collections` (1.1.0+) | `Collections` | Efficient data structures (OrderedDictionary, Deque) |
| `markiv/SwiftUI-Shimmer` (1.5.1+) | `Shimmer` | Loading shimmer effects in UI |

### Python (via ACE-Step 1.5)

Managed by `uv` and ACE-Step's `pyproject.toml`. Key packages:

- `torch` / `torchaudio` — PyTorch with MPS backend
- `mlx` / `mlx-lm` — Apple Silicon native ML
- `transformers` / `diffusers` — HuggingFace model support
- `soundfile` — Audio I/O
- `huggingface-hub` — Model downloading

### System Frameworks

- **SwiftUI** — UI
- **SwiftData** — persistence
- **AVFoundation** / **AVAudioEngine** — audio playback and export
- **Accelerate** — vDSP FFT for spectrum analysis
- **AppKit** — macOS window management, NSApplication promotion

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AURALUX_SERVER_PORT` | `8765` | Inference server port |
| `ACESTEP_LM_BACKEND` | `mlx` | LM backend (`mlx` for Apple Silicon) |
| `ACESTEP_CONFIG_PATH` | `acestep-v15-turbo` | DiT model configuration |
| `ACESTEP_LM_MODEL_PATH` | `acestep-5Hz-lm-0.6B` | LM model name |
| `ACESTEP_DEVICE` | `auto` | Inference device (auto/mps/cpu) |
| `ACESTEP_INIT_LLM` | `auto` | Whether to initialize the LLM |
| `ACESTEP_OFFLOAD_TO_CPU` | `false` | Offload models to CPU when idle |

## Testing

Tests live in `AuraluxTests/` and cover:

- **ModelTests.swift** — SwiftData model creation, mutations, validation
- **ServiceTests.swift** — Service layer behavior, error handling
- **ViewModelTests.swift** — State transitions, preset application, settings persistence

Run with `swift test`. CI also runs these on `macos-latest` via GitHub Actions.

## Key Files to Know

| File | Why it matters |
|------|---------------|
| `Auralux/AuraluxApp.swift` | Entry point; creates all services and injects them |
| `Auralux/ContentView.swift` | Root view; switches between SetupView and main UI |
| `Auralux/Services/EngineService.swift` | Engine lifecycle state machine (~456 lines) |
| `Auralux/Services/InferenceService.swift` | HTTP client to Python server (actor, ~304 lines) |
| `AuraluxEngine/server.py` | Python REST server wrapping ACE-Step (~900 lines) |
| `Auralux/Utilities/Constants.swift` | All app-wide constants |
| `Package.swift` | SPM manifest, dependency versions, target config |

## Common Tasks

### Adding a new View

1. Create `Auralux/Views/<Section>/<Name>View.swift`
2. Use `@Environment` to access needed ViewModels/Services
3. Follow existing SwiftUI patterns in the same section
4. If the view needs new state, add it to the appropriate ViewModel

### Adding a new Service

1. Create `Auralux/Services/<Name>Service.swift`
2. Use `@Observable` (if UI-facing) or `actor` (if thread-safe operations needed)
3. Instantiate in `AuraluxApp.init()` and inject via `.environment()`
4. Add tests in `AuraluxTests/ServiceTests.swift`

### Adding a new SwiftData Model

1. Create `Auralux/Models/<Name>.swift` with `@Model` macro
2. Register in the `ModelContainer` initializer in `AuraluxApp.init()`
3. Add tests in `AuraluxTests/ModelTests.swift`

### Modifying the Python server

1. Edit `AuraluxEngine/server.py`
2. Validate: `python3 -m py_compile AuraluxEngine/server.py`
3. Update `AuraluxEngine/README.md` if endpoints or env vars change
4. Update `InferenceService.swift` if the API contract changes

## Important Notes

- **PRD.md** and **plan.md** are historical reference documents. The PRD proposed CoreML conversion which was rejected in favor of MLX/PyTorch MPS. The plan outlined phases of work. Do not treat them as current specifications — refer to `docs/ARCHITECTURE.md` and this file for the actual architecture.
- **No Xcode project file** — the project uses SPM exclusively (`Package.swift`). Open `Package.swift` in Xcode, not an `.xcodeproj`.
- **Models are not in the repo** — ACE-Step 1.5 and its model weights (~4 GB) are cloned/downloaded at setup time and gitignored.
- **The app runs as an SPM executable** — `AuraluxApp.swift` includes an `AppDelegate` that promotes the process to a regular GUI application (menu bar, Dock icon).
- **No Combine** — the codebase uses Swift structured concurrency (`async/await`, `Task`, `TaskGroup`) instead of Combine publishers.
