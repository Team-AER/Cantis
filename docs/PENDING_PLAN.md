# Cantis — Pending Items Plan

## Goal

Cantis is a single-process, fully native macOS app: launch it, download weights once, generate music — entirely on-device. The architectural plumbing (Python subprocess, HTTP server, IPC) is gone. The remaining work is polish, distribution, and feature breadth.

### Distribution Strategy

| Channel | Status | Requirements |
|---------|--------|-------------|
| Direct distribution (notarized DMG/zip) | In progress | Hardened Runtime + notarization |
| Mac App Store | Feasible (no longer architecturally blocked) | Code signing, provisioning, asset polish |

The App Sandbox is enabled. Because all inference is now in-process via mlx-swift, neither distribution channel needs an XPC helper.

---

## Completed

### Native Swift inference engine

`Cantis/Inference/NativeInferenceEngine.swift` plus `DiT/`, `LM/`, `Text/` subtrees implement the full ACE-Step v1.5 pipeline on mlx-swift:

- ACEStepDiT (2B params) with `TurboSampler` (8-step CFG-distilled) and `CFGSampler` (60-step twin-pass)
- DC-HiFi-GAN VAE for latent → 48 kHz audio
- Qwen3 text encoder for conditioning
- Optional 5 Hz audio-token LM (0.6B), gated behind a Settings toggle
- AceStep audio tokenizer + repaint / cover / extract source loaders

State machine: `notDownloaded → downloading → downloaded → loading → ready` (with `error(message)` recoverable from any state).

### In-app onboarding and download

`SetupView` overlay drives `NativeInferenceEngine.downloadAndLoad()`. `ModelDownloader` (actor) fetches the active variant's manifest from HuggingFace sequentially with weighted progress, skipping files that already exist on disk, and creates symlinks from non-turbo variants into the turbo bundle for shared components.

### Variants and modes

- DiT variants: `turbo`, `sft`, `base` (app-downloadable); `xl-turbo`, `xl-sft`, `xl-base` (require `tools/convert_weights.py`)
- Generation modes: `text2music`, `cover`, `repaint`, `extract` wired end-to-end; `text2musicLM` reserved for the LM-driven hint path

### App shell and UX

- `CantisApp` configures `MLX.Memory.cacheLimit`, builds the SwiftData `ModelContainer`, injects services via `@Environment`
- `AppDelegate` promotes the SPM executable to a regular GUI app and forwards termination
- `EngineStatusView` toolbar badge reflects `modelState`
- Generation, playback, audio export (WAV / AAC / ALAC), history, presets, log viewer

### Build and tests

- `swift build` clean on Swift 6.2 / macOS 26 SDK
- Deterministic Model / Service / ViewModel suites pass on CI (`macos-26`)
- MLX integration suites (`ACEStepDiTTests`, `ACEStepLMTests`, `FeasibilityProbeTests`, `Qwen3ConditioningTests`, `Qwen3RealWeightsTests`) run from Xcode against real weights
- `python -m py_compile modeling_acestep_v15_turbo.py tools/convert_weights.py` runs in CI

### App Sandbox

Sandbox enabled with `network.client`, `files.user-selected.read-write`, and `device.audio-input` entitlements. No subprocess spawning required.

---

## Remaining Items

### Distribution

| Item | Status | Notes |
|------|--------|-------|
| Apple Developer ID certificate | Not started | Paid Apple Developer account |
| Code signing (Hardened Runtime) | Not started | For direct distribution |
| Notarization workflow | Not started | `xcrun notarytool` against signed builds |
| App icon and brand assets | Not started | Asset design |
| DMG packaging | Not started | `create-dmg` or similar |
| Mac App Store provisioning | Not started | Now feasible (no XPC needed) |

### Inference / model breadth

| Item | Status | Notes |
|------|--------|-------|
| Wire `text2musicLM` end-to-end | Not started | Needs upstream-canonical FSQ codebook → detokenize → src_latents path; `ACEStepLMSampler` is the integration point |
| INT8 / FP8 quantization | Not started | `SettingsViewModel.QuantizationMode` currently only exposes `fp16`; expand once mlx-swift quant kernels stabilize |
| LoRA loading | Not started | `LoRAManagerView` exists as a UI shell; engine-side loader / merge not implemented |
| XL variants in-app download | Not started | Currently script-only via `tools/convert_weights.py`; publish as MLX safetensors and add manifest entries to enable in-app download |
| Memory pressure monitoring | Not started | Auto-toggle low-memory mode based on `os_proc_available_memory` / pressure events |

### Polish / advanced features

| Item | Status | Notes |
|------|--------|-------|
| Multi-track generation (vocal + instrumental) | Not started | Depends on upstream support |
| Stem export | Not started | Per-track export |
| Batch generation with seed arrays | Not started | Needs a job queue (removed during cleanup) plus batch UI |
| Keyboard shortcuts coverage | Partial | Audit and document |
| URL scheme handler (`cantis://generate?...`) | Not started | Inter-app integration |
| Performance profiling pass | Not started | Instruments: GPU, Memory, Energy |

---

## Architecture (Current)

```
┌────────────────────────────────────────────────────┐
│                  CantisApp.swift                   │
│  Builds NativeInferenceEngine + ViewModels +       │
│  ModelContainer; injects via @Environment          │
├─────────────┬──────────────────────────────────────┤
│  SetupView  │           ContentView                │
│ (overlay    │  ┌─────────┬──────────┬──────────┐   │
│  while      │  │Sidebar  │Generation│ Player   │   │
│  engine.    │  │         │  View    │  View    │   │
│  isOnboard) │  │EngineStatus badge in toolbar  │   │
├─────────────┴──┴─────────┴──────────┴──────────┘   │
│              NativeInferenceEngine                  │
│  ┌──────────────┐  ┌──────────┐  ┌─────────────┐   │
│  │ ModelDownload│  │ Weight   │  │   Sampler   │   │
│  │   (actor)    │  │ Loaders  │  │ + VAE + Text│   │
│  └──────────────┘  └──────────┘  └─────────────┘   │
├────────────────────────────────────────────────────┤
│                    mlx-swift                        │
│              (Metal / Apple Silicon)                │
└────────────────────────────────────────────────────┘
```
