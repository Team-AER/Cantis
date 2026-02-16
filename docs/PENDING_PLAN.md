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

## Pending Items

### 1. Engine Lifecycle Service (NEW)

**File:** `Auralux/Services/EngineService.swift`

A single `@Observable` service that owns the entire engine lifecycle:

- **Setup detection** — is AuraluxEngine/ACE-Step-1.5 cloned? Is the venv ready?
- **Automatic setup** — runs `setup_env.sh` as a subprocess, streams output
- **Server start/stop** — manages the Python server process
- **Health monitoring** — periodic health checks, auto-restart on crash
- **Model status** — tracks whether models are downloaded and loaded
- **Graceful shutdown** — stops the server when the app quits

States: `notSetup → settingUp → starting → running → ready → error`

### 2. Onboarding / First-Run View (NEW)

**File:** `Auralux/Views/Onboarding/SetupView.swift`

Shown automatically when the engine is not yet configured:

1. **System check** — verify Apple Silicon, macOS 15+, disk space
2. **Environment setup** — clone ACE-Step 1.5, install Python deps via `uv`
3. **Server start** — launch the local inference server
4. **Model download** — models auto-download on first generation; show status
5. **Ready** — transition to the main app

Features: animated progress, step-by-step flow, error recovery, skip option
for users who already have the server running externally.

### 3. Engine Status Indicator (NEW)

**File:** `Auralux/Components/EngineStatusView.swift`

A compact status badge shown in the main app toolbar:

- 🔴 Not configured / Error
- 🟡 Setting up / Starting / Downloading models
- 🟢 Ready

Clicking opens the Settings > Models view for details.

### 4. App Entry Point Updates

**Files:** `AuraluxApp.swift`, `ContentView.swift`

- Inject `EngineService` into the environment
- On launch: check engine state, show onboarding if needed
- On quit: gracefully stop the server
- `ContentView` shows `SetupView` overlay when engine needs attention
- Engine status indicator in the toolbar

### 5. Generation Guard

**File:** `Auralux/Views/Generation/GenerationView.swift`

- Disable the Generate button when the engine is not ready
- Show a clear message directing users to complete setup
- When engine is starting/downloading, show appropriate status

### 6. Documentation Updates

- **ARCHITECTURE.md** — remove "scaffold-only" and "placeholder" language;
  document the real ACE-Step 1.5 integration and engine lifecycle
- **README.md** — update setup instructions and feature list
- **DEVELOPMENT.md** — add development workflow details

### 7. Build & Test Verification

- Clean build with `swift build`
- Run all tests with `swift test`
- Verify no linter errors
- Confirm the app launches and shows onboarding correctly

---

## Architecture After Implementation

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
│         ProcessServerLauncher (subprocess)          │
├────────────────────────────────────────────────────┤
│            AuraluxEngine/server.py                  │
│              ACE-Step v1.5 inference                │
└────────────────────────────────────────────────────┘
```

---

## Items NOT in Scope (Manual / Future)

| Item | Why | When |
|------|-----|------|
| Apple Developer certificate | Requires paid Apple Developer account | Before distribution |
| Code signing & notarization | Needs Xcode + Developer ID | Before distribution |
| App icon | Asset design task | Before distribution |
| DMG/installer packaging | Needs create-dmg or similar tool | Before distribution |
| XPC Service for App Store | Major architecture work | Phase 2 |
| mlx-swift native inference | Months of porting effort | Phase 2 |
| Python runtime bundling in .app | Requires py2app or custom framework | Optional for Phase 1 |
