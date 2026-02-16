# Auralux — Pending Items Plan

## Goal

Make Auralux a **single-app experience** where users launch the app and everything
works automatically — no terminal commands, no manual setup. The app should handle
Python environment setup, server lifecycle, and model downloads entirely on its own.

### Distribution Strategy

| Channel | Status | Requirements |
|---------|--------|-------------|
| **Direct distribution** (notarized DMG/zip) | Target for Phase 1 | Hardened Runtime + notarization |
| **Mac App Store** | Future (Phase 2) | XPC service or mlx-swift native port to eliminate Python subprocess |

The App Sandbox disallows `Process()` for arbitrary subprocesses. For direct
(notarized) distribution this is fine — we disable the sandbox. For the Mac App
Store, inference must move to an XPC helper or a native mlx-swift port. This plan
focuses on making Phase 1 rock-solid.

---

## Completed Items

### 1. Engine Lifecycle Service ✅

**File:** `Auralux/Services/EngineService.swift`

A single `@Observable` service that owns the entire engine lifecycle:

- **Setup detection** — checks for ACE-Step-1.5 directory and venv
- **Automatic setup** — runs `setup_env.sh` as a subprocess, streams output to `setupLog`
- **Server start/stop** — manages the Python server process
- **Health monitoring** — periodic health checks via `/health`, auto-restart on crash
- **Model status** — tracks whether models are downloaded and loaded
- **Graceful shutdown** — stops the server when the app quits

States: `unknown → notSetup → settingUp → starting → running → ready → error`

### 2. Onboarding / First-Run View ✅

**File:** `Auralux/Views/Onboarding/SetupView.swift`

Shown automatically when the engine is not yet configured:

1. **System check** — verify Apple Silicon, macOS 15+, disk space
2. **Environment setup** — clone ACE-Step 1.5, install Python deps via `uv`
3. **Server start** — launch the local inference server
4. **Model download** — models auto-download on first generation; show status
5. **Ready** — transition to the main app

Features: animated progress, step-by-step flow, error recovery, skip option
for users who already have the server running externally.

### 3. Engine Status Indicator ✅

**File:** `Auralux/Components/EngineStatusView.swift`

A compact status badge shown in the main app toolbar:

- Red — Not configured / Error
- Yellow — Setting up / Starting / Downloading models
- Green — Ready

Clicking opens the Settings > Models view for details.

### 4. App Entry Point ✅

**Files:** `AuraluxApp.swift`, `ContentView.swift`

- `EngineService` injected into the environment
- On launch: checks engine state, shows onboarding if needed
- On quit: gracefully stops the server via `shutdown()`
- `ContentView` shows `SetupView` overlay when engine needs attention
- Engine status indicator in the toolbar
- Log viewer window available via `Window("Auralux Logs", id: "log-viewer")`

### 5. Generation Guard ✅

**File:** `Auralux/Views/Generation/GenerationView.swift`

- Generate button disabled when the engine is not ready
- Clear message directing users to complete setup
- When engine is starting/downloading, shows appropriate status

### 6. Documentation ✅

- **ARCHITECTURE.md** — documents real ACE-Step 1.5 integration and engine lifecycle
- **README.md** — updated setup instructions and feature list
- **DEVELOPMENT.md** — development workflow details, complete project structure

### 7. Build & Test Verification ✅

- Clean build with `swift build`
- Tests pass with `swift test`
- Python server compiles: `python -m py_compile AuraluxEngine/server.py`
- CI validates both on every push/PR

---

## Remaining Items

### Distribution Preparation

| Item | Status | Notes |
|------|--------|-------|
| Apple Developer certificate | Not started | Requires paid Apple Developer account |
| Code signing & notarization | Not started | Needs Xcode + Developer ID |
| App icon and brand assets | Not started | Asset design task |
| DMG/installer packaging | Not started | Needs create-dmg or similar tool |
| Python runtime bundling in .app | Not started | Optional; py2app or custom framework |

### Phase 2 — Native Inference

| Item | Status | Notes |
|------|--------|-------|
| XPC Service for App Store | Not started | Major architecture change to comply with sandbox |
| mlx-swift native inference port | Not started | Months of porting; eliminates Python dependency |
| INT8/FP16 quantization options | Not started | Reduces model size and memory usage |

### Polish & Advanced Features

| Item | Status | Notes |
|------|--------|-------|
| Multi-track generation (vocal + instrumental) | Not started | Depends on ACE-Step capabilities |
| Stem export | Not started | Individual track export |
| Batch generation with seed arrays | Not started | Queue service exists but needs batch UI |
| Keyboard shortcuts for all actions | Not started | Accessibility improvement |
| URL scheme handler (`auralux://generate?...`) | Not started | Inter-app communication |
| Memory pressure monitoring | Not started | Graceful degradation on low-memory systems |
| Performance profiling with Instruments | Not started | GPU, Memory, Energy profiling |

---

## Architecture (Current)

```
┌────────────────────────────────────────────────────┐
│                  AuraluxApp.swift                   │
│   Creates EngineService, injects into environment  │
├─────────────┬──────────────────────────────────────┤
│  SetupView  │           ContentView                │
│ (onboarding)│  ┌─────────┬──────────┬──────────┐   │
│  shown when │  │Sidebar  │Generation│ Player   │   │
│  engine not │  │         │  View    │  View    │   │
│  ready      │  │EngineStatus badge in toolbar  │   │
├─────────────┴──┴─────────┴──────────┴──────────┘   │
│                   EngineService                     │
│  ┌──────────┐  ┌────────────┐  ┌──────────────┐    │
│  │  Setup   │  │  Server    │  │   Health     │    │
│  │Detection │  │ Lifecycle  │  │  Monitoring  │    │
│  └──────────┘  └────────────┘  └──────────────┘    │
├────────────────────────────────────────────────────┤
│            InferenceService (HTTP client)           │
├────────────────────────────────────────────────────┤
│            AuraluxEngine/server.py                  │
│              ACE-Step v1.5 inference                │
└────────────────────────────────────────────────────┘
```
