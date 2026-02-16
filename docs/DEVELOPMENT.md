# Development Guide

## Prerequisites

- macOS 15+ (Sequoia)
- Xcode 16+ or Swift 6+ toolchain
- Apple Silicon (M1 or later)
- Python 3.11+ (installed automatically by the setup script)
- Internet connection for initial setup

## First-Time Setup

The app handles setup automatically on first launch via the onboarding flow. For manual development setup:

```bash
git clone <repo-url>
cd auralux

# Set up the Python environment (clones ACE-Step 1.5, installs deps via uv)
cd AuraluxEngine
./setup_env.sh
```

## Running the App

### From Xcode

Open `Package.swift` in Xcode, select the `Auralux` executable target, and run (Cmd+R).

### From Terminal

```bash
swift run Auralux
```

The app will automatically detect the AuraluxEngine directory and manage the server lifecycle. On first launch, the onboarding flow guides through environment setup and server start.

## Running the Engine Server Manually

If you prefer to manage the server separately (useful for debugging):

```bash
cd AuraluxEngine
./start_api_server_macos.sh
```

By default the server binds to `127.0.0.1:8765`. Override with `AURALUX_SERVER_PORT`.

The app detects externally running servers and connects to them automatically, skipping subprocess management.

## Running Tests

```bash
swift test
```

Tests cover:
- **ModelTests** — SwiftData model creation, mutations, and validation
- **ServiceTests** — service layer behavior and error handling
- **ViewModelTests** — state transitions, preset application, settings persistence

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AURALUX_SERVER_PORT` | `8765` | Port for the inference server |
| `ACESTEP_LM_BACKEND` | `mlx` | LM backend (`mlx` for Apple Silicon) |
| `ACESTEP_CONFIG_PATH` | `acestep-v15-turbo` | DiT model configuration |
| `ACESTEP_LM_MODEL_PATH` | `acestep-5Hz-lm-0.6B` | LM model name |
| `ACESTEP_DEVICE` | `auto` | Inference device (auto/mps/cpu) |
| `ACESTEP_INIT_LLM` | `auto` | Whether to initialize the LLM |
| `ACESTEP_OFFLOAD_TO_CPU` | `false` | Offload models to CPU when idle |

## Project Structure

```
Auralux/
├── AuraluxApp.swift                    # App entry point, service injection, AppDelegate
├── ContentView.swift                   # Root view with setup/main switching
├── Info.plist                          # App metadata
├── Entitlements.plist                  # macOS entitlements
│
├── Components/                         # Reusable UI components
│   ├── AudioDropZone.swift             # Drag-and-drop audio import zone
│   ├── EngineStatusView.swift          # Engine status badge (red/yellow/green)
│   ├── ProgressOverlay.swift           # Progress indicator overlay
│   ├── SliderControl.swift             # Custom parameter slider
│   └── TagChip.swift                   # Tag display component
│
├── Models/                             # SwiftData domain models
│   ├── GeneratedTrack.swift            # Generated music metadata + file path
│   ├── GenerationParameters.swift      # Generation request parameters (Codable)
│   ├── Preset.swift                    # Saved generation configurations
│   └── Tag.swift                       # Reusable tag model
│
├── Services/                           # Business logic layer
│   ├── AudioExportService.swift        # Multi-format export (WAV/FLAC/MP3/AAC/ALAC)
│   ├── AudioPlayerService.swift        # AVAudioEngine playback wrapper
│   ├── EngineService.swift             # Engine lifecycle management (~456 lines)
│   ├── GenerationQueueService.swift    # Job queue with priority ordering
│   ├── HistoryService.swift            # SwiftData CRUD for generation history
│   ├── InferenceService.swift          # HTTP client to Python server (actor, ~304 lines)
│   ├── ModelManagerService.swift       # Model download status tracking
│   └── PresetService.swift             # Preset CRUD operations
│
├── Utilities/                          # Helpers and constants
│   ├── AppLogger.swift                 # Centralized logging (OSLog wrapper)
│   ├── AudioFFT.swift                  # vDSP FFT for spectrum visualization
│   ├── Constants.swift                 # App-wide constants
│   └── FileUtilities.swift             # File path helpers
│
├── ViewModels/                         # @Observable state management
│   ├── GenerationViewModel.swift       # Generation orchestration and state
│   ├── HistoryViewModel.swift          # History browsing state
│   ├── PlayerViewModel.swift           # Audio playback state
│   ├── SettingsViewModel.swift         # Settings persistence
│   └── SidebarViewModel.swift          # Navigation state
│
├── Views/
│   ├── AudioToAudio/
│   │   ├── AudioImportView.swift       # Audio file import interface
│   │   └── LoRAManagerView.swift       # LoRA model management
│   ├── Generation/
│   │   ├── GenerationView.swift        # Main generation interface
│   │   ├── LyricEditorView.swift       # Lyric editing with structure tags
│   │   ├── ParameterControlsView.swift # Duration, variance, seed controls
│   │   └── TagEditorView.swift         # Tag selection and management
│   ├── History/
│   │   ├── HistoryBrowserView.swift    # Generation history browser
│   │   └── HistoryItemView.swift       # Individual history entry display
│   ├── Onboarding/
│   │   └── SetupView.swift             # First-run setup experience
│   ├── Player/
│   │   ├── PlayerView.swift            # Audio playback interface
│   │   ├── SpectrumAnalyzerView.swift  # Real-time FFT spectrum display
│   │   └── WaveformView.swift          # Waveform visualization
│   ├── Settings/
│   │   ├── ModelSettingsView.swift      # Model configuration panel
│   │   └── SettingsView.swift          # Settings panel
│   ├── Sidebar/
│   │   ├── PresetListView.swift        # Preset navigation list
│   │   ├── RecentListView.swift        # Recent generations list
│   │   └── SidebarView.swift           # Main sidebar navigation
│   └── LogViewerView.swift             # Log viewer window
│
└── Resources/                          # Bundled assets
```

### Python Backend

```
AuraluxEngine/
├── server.py                           # REST API wrapping ACE-Step v1.5 (~900 lines)
├── setup_env.sh                        # Clones ACE-Step 1.5, installs deps via uv
├── start_api_server_macos.sh           # Server launch script (port 8765)
├── test_generate.py                    # Backend test script
├── requirements.txt                    # Dependency documentation
├── README.md                           # Engine-specific docs
└── ACE-Step-1.5/                       # Cloned at setup time (gitignored)
    ├── acestep/                        # ACE-Step Python package
    ├── checkpoints/                    # Downloaded model weights (~4 GB)
    ├── pyproject.toml                  # Python dependency definitions
    └── .venv/                          # Virtual environment (via uv)
```

### Tests

```
AuraluxTests/
├── ModelTests.swift                    # SwiftData model tests
├── ServiceTests.swift                  # Service layer tests
└── ViewModelTests.swift                # ViewModel behavior tests
```

## Suggested Checks Before Opening a PR

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
- **Server crashes**: Check the setup log in the app or server output in terminal. The app's log viewer window (menu: Window > Auralux Logs) can help diagnose issues.
- **App not appearing in Dock**: The app uses an `AppDelegate` to promote the SPM executable to a GUI application. Ensure `AuraluxApp.swift` has the `@NSApplicationDelegateAdaptor`.

## Coding Expectations

- Keep PRs scoped and testable.
- Add or update tests for behavior changes.
- Update docs when introducing workflow or API changes.
- Follow Swift 6 concurrency patterns (`@Observable`, `actor`, `Sendable`, `async/await`).
- Use `@MainActor` for ViewModels and UI-facing services.
- Use `actor` for thread-safe services (e.g., `InferenceService`).
- No Combine — use Swift structured concurrency instead.
- Use `AppLogger.shared` for all logging with appropriate categories.
- Constants go in `AppConstants` — avoid magic strings and numbers.
