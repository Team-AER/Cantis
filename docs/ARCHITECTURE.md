# Architecture

## Overview

Auralux is a two-process macOS application: a SwiftUI frontend and a Python inference backend. They communicate over a local HTTP REST API on `127.0.0.1:8765`.

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

## High-Level Components

- **`Auralux/`** (Swift) — UI, state management, audio playback/export, local persistence, and engine lifecycle management.
- **`AuraluxEngine/`** (Python) — local HTTP server wrapping ACE-Step v1.5 for real AI music generation on Apple Silicon.
- **`AuraluxTests/`** (Swift) — unit tests for models, services, and view models.

## Swift App Structure

### Views

Screens organized by feature area:

- `Onboarding/SetupView` — first-run setup flow (system check, environment setup, server start)
- `Generation/` — main generation interface with tag editor, lyric editor, and parameter controls
- `Player/` — audio playback with waveform visualization and FFT spectrum analyzer
- `History/` — browse, search, and favorite past generations
- `Settings/` — app and model configuration
- `Sidebar/` — navigation with presets and recent tracks
- `AudioToAudio/` — audio import and LoRA model management
- `LogViewerView` — dedicated log viewer window (opened via `Window("Auralux Logs", id: "log-viewer")`)

### ViewModels

All use the `@Observable` macro and `@MainActor` for UI-safe state management:

- `GenerationViewModel` — generation orchestration, request building, progress tracking
- `PlayerViewModel` — playback state (play, pause, scrub, loop)
- `HistoryViewModel` — history browsing, search, filtering
- `SidebarViewModel` — navigation state
- `SettingsViewModel` — settings persistence via UserDefaults

### Services

Business logic boundary, injected via `@Environment`:

- **`EngineService`** — owns the full engine lifecycle (setup detection, environment provisioning, server start/stop, health monitoring, graceful shutdown). See [Engine Lifecycle](#engine-lifecycle) below.
- **`InferenceService`** — Swift `actor` that makes HTTP requests to the Python server. Thread-safe by design.
- **`AudioPlayerService`** — `AVAudioEngine` wrapper for playback, waveform data, and real-time FFT.
- **`AudioExportService`** — multi-format export (WAV 16/24/32-bit, FLAC, MP3, AAC, ALAC) with configurable sample rates.
- **`HistoryService`** — SwiftData CRUD for `GeneratedTrack`, search, and favorites.
- **`PresetService`** — preset CRUD, import/export.
- **`ModelManagerService`** — tracks model download status and triggers downloads via `/models/download`.
- **`GenerationQueueService`** — job queue with priority ordering.

### Models

SwiftData `@Model` types:

- **`GeneratedTrack`** — metadata for generated music (prompt, tags, parameters, audio file path, timestamps, favorite flag)
- **`Preset`** — saved generation configurations (name, tags, duration, variance, lyrics template)
- **`Tag`** — reusable tag library with usage tracking
- **`GenerationParameters`** — `Codable` struct for request parameters (not a SwiftData model)

### Components

Reusable UI building blocks:

- `EngineStatusView` — compact toolbar badge showing engine state (red/yellow/green)
- `TagChip` — genre, instrument, and mood tag display
- `SliderControl` — custom parameter slider with label and value display
- `ProgressOverlay` — indeterminate and determinate progress indicator
- `AudioDropZone` — drag-and-drop target for audio file import

### Utilities

- `AppLogger` — centralized logging via `OSLog` with categories (`.app`, `.engine`, `.inference`, etc.)
- `AudioFFT` — Accelerate vDSP FFT computation for spectrum visualization
- `Constants` — app-wide constants (server URL, model names, suggested tags, window dimensions)
- `FileUtilities` — file path helpers for Application Support directories

## Engine Lifecycle

`EngineService` is the central coordinator for the Python inference backend. It uses a state machine:

```
unknown → notSetup → settingUp(progress) → starting → running → ready
                                                              ↘ error(message)
```

### State Descriptions

| State | Meaning | UI Effect |
|-------|---------|-----------|
| `unknown` | App just launched, checking setup status | Loading indicator |
| `notSetup` | ACE-Step environment not found | Shows `SetupView` onboarding |
| `settingUp(progress)` | Running `setup_env.sh` subprocess | Progress display with log output |
| `starting` | Python server launched, waiting for `/health` | Starting indicator |
| `running` | Server responding to health checks | Engine status badge turns yellow |
| `ready` | Server healthy and models loaded | Badge turns green, generation enabled |
| `error(message)` | Setup or server failure | Error display, retry option |

### Lifecycle Flow

1. On launch, `EngineService` checks if the environment is set up (ACE-Step-1.5 directory, venv).
2. If not set up, the app shows `SetupView` which runs `setup_env.sh` as a subprocess, streaming output to `setupLog`.
3. Once the environment is ready, `EngineService` starts the Python server via `start_api_server_macos.sh`.
4. Health monitoring begins — periodic `GET /health` requests.
5. When the server reports healthy with models loaded, state transitions to `ready`.
6. If the server crashes, `EngineService` attempts auto-restart.
7. On app quit, `shutdown()` terminates the server process gracefully.

### External Server Support

If a user runs the server manually (e.g., for debugging), the app detects the externally running server via `/health` and connects to it directly, skipping subprocess management.

## Concurrency Model

- **`@MainActor`** — all ViewModels and `EngineService` (they update UI state)
- **`actor`** — `InferenceService` (thread-safe HTTP client)
- **`async/await`** — all asynchronous operations use Swift structured concurrency
- **`Task`** — used for fire-and-forget operations (health monitoring loop)
- **No Combine** — the codebase does not use Combine publishers

## Generation Flow

1. User configures prompt, tags, lyrics, and parameters in `GenerationView`.
2. `GenerationViewModel` validates inputs and builds a `GenerationRequest`.
3. `InferenceService.generate()` POSTs the request to `/generate` on the Python server.
4. The server enqueues the job and returns a `jobID`.
5. `InferenceService` polls `GET /jobs/<id>` for progress updates.
6. The Python server runs ACE-Step v1.5 inference:
   - **DiT model** (`acestep-v15-turbo`) — 8-step turbo inference via PyTorch MPS
   - **LM model** (`acestep-5Hz-lm-0.6B`) — metadata/CoT reasoning via MLX
7. On completion, the server writes the audio file and returns the path.
8. A `GeneratedTrack` is created and persisted via `HistoryService`.
9. The track appears in history and can be played in `PlayerView`.

## Inference Engine

The Python server (`AuraluxEngine/server.py`) wraps ACE-Step v1.5:

- **DiT model**: `acestep-v15-turbo` — 8-step turbo inference via PyTorch MPS
- **LM model**: `acestep-5Hz-lm-0.6B` — metadata/CoT reasoning via MLX
- Models auto-download from HuggingFace on first generation (~4 GB total)
- Stub fallback produces silent WAV files when models are not yet available

### MPS Workarounds

The server includes runtime patches for known PyTorch MPS bugs on Apple Silicon:

| Workaround | Description |
|------------|-------------|
| `masked_fill` CPU fallback | MPS does not support `masked_fill` for certain tensor types |
| `inference_mode` → `no_grad` | MPS backend does not fully support `inference_mode` |
| Audio codec CPU fallback | Audio decoding routed to CPU for stability |
| Text encoder CPU fallback | Text encoding runs on CPU to avoid MPS errors |
| DiT condition encoder CPU fallback | Condition encoder routed to CPU |

These patches are applied at server startup before any model loading.

## Persistence

SwiftData stores all structured data:

- **`GeneratedTrack`** — generated music metadata and audio file paths
- **`Preset`** — saved generation parameter configurations
- **`Tag`** — reusable tag library with usage counts

Audio files are stored in `~/Library/Application Support/Auralux/Generated/` using relative paths for sandbox resilience.

The `ModelContainer` is created once in `AuraluxApp.init()` and injected via `.modelContainer()`.

## API Contract

| Endpoint | Method | Request | Response |
|----------|--------|---------|----------|
| `/health` | GET | — | `{ status, models_loaded, device, ... }` |
| `/generate` | POST | `{ prompt, tags, lyrics, duration, variance, seed }` | `{ job_id }` |
| `/jobs/<id>` | GET | — | `{ status, progress, audio_path, error }` |
| `/jobs/<id>/cancel` | POST | — | `{ cancelled }` |
| `/models/download` | POST | — | `{ status }` |

## App Entry Point

`AuraluxApp.swift` serves as the composition root:

1. Creates `InferenceService`, `EngineService`, and all ViewModels
2. Initializes the SwiftData `ModelContainer` for `GeneratedTrack`, `Preset`, `Tag`
3. Injects everything into the SwiftUI environment
4. Defines two windows: the main `WindowGroup` and a `Window` for the log viewer
5. Includes an `AppDelegate` that promotes the SPM executable to a regular GUI application (menu bar, Dock icon)

## Distribution

- **Phase 1 (current):** Direct distribution with notarization. The app launches the Python server as a subprocess. Not compatible with the Mac App Store sandbox.
- **Phase 2 (future):** Mac App Store via XPC service or native mlx-swift inference port, eliminating the Python dependency.
