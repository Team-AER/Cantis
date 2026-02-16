# Architecture

## High-level components

- `Auralux/` (Swift): UI, state, local persistence, and engine lifecycle management.
- `AuraluxEngine/` (Python): local HTTP server wrapping ACE-Step v1.5 for real AI music generation.
- `AuraluxTests/` (Swift tests): behavior checks for key app components.

## Swift app structure

- `Views/`: screens and composition, including onboarding/setup flow.
- `ViewModels/`: user-intent orchestration and state transitions.
- `Services/`: boundary logic for inference, engine lifecycle, persistence, queueing, and playback/export.
- `Models/`: app domain and SwiftData models.
- `Components/`: reusable UI components (tags, sliders, engine status, drop zones).
- `Utilities/`: app constants and reusable helpers.

## Engine lifecycle

1. On first launch, the app detects that the Python environment is not set up.
2. `EngineService` presents the onboarding flow (`SetupView`).
3. The setup flow clones ACE-Step 1.5, installs Python dependencies via `uv`, and starts the server.
4. Once the server is healthy, the app transitions to the main UI.
5. `EngineService` monitors server health and auto-restarts on crash.
6. On app quit, the server process is terminated gracefully.

## Generation flow

1. User configures prompt/tags/parameters.
2. `GenerationViewModel` builds a `GenerationRequest` and calls `InferenceService.generate`.
3. `InferenceService` submits the job to the running Python server and polls status.
4. The Python server runs ACE-Step v1.5 inference (DiT via PyTorch MPS, LM via MLX on Apple Silicon).
5. On completion, a `GeneratedTrack` is stored through `HistoryService`.
6. Track appears in history and can be previewed in player views.

## Inference engine

The Python server (`AuraluxEngine/server.py`) wraps ACE-Step v1.5:

- **DiT model**: `acestep-v15-turbo` — 8-step turbo inference via PyTorch MPS
- **LM model**: `acestep-5Hz-lm-0.6B` — metadata/CoT reasoning via MLX
- Models auto-download from HuggingFace on first generation (~4 GB total)
- Stub fallback produces silent WAV files when models are not yet available

## Persistence

SwiftData stores:

- `GeneratedTrack` — generated music metadata and audio file paths
- `Preset` — saved generation parameter configurations
- `Tag` — reusable tag library

Audio files are stored in `~/Library/Application Support/Auralux/Generated/` using relative paths for sandbox resilience.

## API contract

- `GET /health` => service liveness, model status, device info
- `POST /generate` => enqueue generation and return `jobID`
- `GET /jobs/<id>` => status/progress/audio path
- `POST /jobs/<id>/cancel` => request cancellation
- `POST /models/download` => trigger model download

## Distribution

- **Phase 1 (current)**: Direct distribution with notarization. The app launches the Python server as a subprocess. Not compatible with the Mac App Store sandbox.
- **Phase 2 (future)**: Mac App Store via XPC service or native mlx-swift inference port, eliminating the Python dependency.
