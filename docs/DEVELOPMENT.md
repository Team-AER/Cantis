# Development Guide

## Prerequisites

- macOS 26+ (Tahoe)
- Xcode 26+ (or Swift 6.2 toolchain) — SDK requires the macOS 26 SDK
- Apple Silicon (M1 or later)
- Internet connection (for the first-launch model download)
- Python 3.11+ — only required if you intend to run `tools/convert_weights.py` to convert XL or custom weights

## Running the App

### From Xcode

Open `Package.swift` in Xcode, select the `Cantis` executable target, and run (Cmd+R).

### From the terminal

```bash
swift run Cantis
```

On first launch the in-app onboarding overlay (`SetupView`) will:

1. Check whether the active `DiTVariant` is already present in `~/Library/Application Support/Cantis/Models/`.
2. If not, download the variant's manifest from HuggingFace via `ModelDownloader` (sequential, resumable, weighted progress).
3. Load weights into MLX on a detached task and transition `modelState` → `ready`.

After that, generation runs locally in the same process — there is no server, no port, no subprocess.

## Converting XL or Custom Weights

The Turbo, SFT, and Base variants are pre-converted and download from HuggingFace directly. The XL variants and any custom checkpoints require a one-shot conversion from the original PyTorch weights:

```bash
python tools/convert_weights.py --variant xl-turbo
```

Output is written into `~/Library/Application Support/Cantis/Models/<variant-directory>/`. See `tools/convert_weights.py` for flags and `modeling_acestep_v15_turbo.py` for the reference PyTorch model used by the converter.

## Running Tests

```bash
swift test
```

CI runs the deterministic suites and skips MLX integration tests, which require a local Metal / GPU runtime:

```
swift test --skip 'ACEStepDiTTests|ACEStepLMTests|FeasibilityProbeTests|Qwen3ConditioningTests|Qwen3RealWeightsTests'
```

Suites:

- **ModelTests** — SwiftData model creation, mutations, validation
- **ServiceTests** — service-layer behavior and error handling
- **ViewModelTests** — state transitions, preset application, settings persistence
- **ACEStepDiTTests / ACEStepLMTests / FeasibilityProbeTests / Qwen3ConditioningTests / Qwen3RealWeightsTests** — MLX integration (Xcode-only)

## Settings and User Defaults

Most knobs surface via `SettingsViewModel` and persist in `UserDefaults` under `SettingsViewModel.Keys.*`:

| Setting | Key | Notes |
|---------|-----|-------|
| Active DiT variant | `settings.ditVariant` | `turbo` / `sft` / `base` / `xl-*` |
| Default generation mode | `settings.defaultMode` | `text2music` etc. |
| Default num steps | `settings.defaultNumSteps` | Clamped to `DiTVariant.maxNumSteps` |
| Default schedule shift | `settings.defaultScheduleShift` | One of {1.0, 2.0, 3.0} |
| Default CFG scale | `settings.defaultCfgScale` | Ignored by CFG-distilled variants |
| Load 5 Hz LM | `settings.useLM` | Off by default; ~1.2 GB resident |
| Quantization mode | `settings.quantizationMode` | Currently `fp16` only |
| Low-memory mode | `settings.lowMemoryMode` | Halves `MLX.Memory.cacheLimit` |

There are no environment variables for runtime configuration — everything is in-app settings or UserDefaults.

## Project Structure

```
Cantis/
├── CantisApp.swift                     # App entry point, MLX cache, ModelContainer, env injection
├── Info.plist                           # App metadata
├── Entitlements.plist                   # App Sandbox + network/files/audio entitlements
│
├── Components/                          # Reusable UI components
│   ├── AudioDropZone.swift
│   ├── EngineStatusView.swift
│   ├── ProgressOverlay.swift
│   ├── SliderControl.swift
│   └── TagChip.swift
│
├── Inference/                           # Native Swift inference (mlx-swift)
│   ├── NativeInferenceEngine.swift      # @MainActor coordinator
│   ├── DiT/
│   │   ├── ACEStepDiT.swift
│   │   ├── DiTWeightLoader.swift
│   │   ├── TurboSampler.swift           # 8-step CFG-distilled sampler
│   │   ├── CFGSampler.swift             # Twin-pass CFG sampler (base/SFT)
│   │   ├── AudioVAE.swift               # DC-HiFi-GAN VAE
│   │   ├── VAEWeightLoader.swift
│   │   ├── AceStepAudioTokenizer.swift  # FSQ audio token codec
│   │   ├── AudioFileLoader.swift        # cover/repaint/extract source loading
│   │   └── SilenceLatentLoader.swift
│   ├── LM/                              # Optional 5 Hz audio-token LM
│   │   ├── ACEStepLM.swift
│   │   ├── ACEStepLMSampler.swift
│   │   ├── BPETokenizer.swift
│   │   └── LMWeightLoader.swift
│   └── Text/                            # Qwen3 text conditioning encoder
│       ├── Qwen3Encoder.swift
│       ├── Qwen3EncoderWeightLoader.swift
│       ├── Qwen3Tokenizer.swift
│       └── PackSequences.swift
│
├── Models/                              # SwiftData models + DTOs + enums
│   ├── DiTVariant.swift
│   ├── GeneratedTrack.swift
│   ├── GenerationMode.swift
│   ├── GenerationParameters.swift
│   ├── Preset.swift
│   └── Tag.swift
│
├── Services/                            # Business logic
│   ├── AudioExportService.swift
│   ├── AudioPlayerService.swift
│   ├── HistoryService.swift
│   ├── ModelDownloader.swift            # actor; HuggingFace sequential downloads
│   ├── ModelManagerService.swift        # MLX artifact registry
│   ├── PlaybackDiagnosticsService.swift
│   └── PresetService.swift
│
├── Utilities/                           # Helpers
│   ├── AppLogger.swift
│   ├── AudioFFT.swift
│   ├── Constants.swift
│   └── FileUtilities.swift
│
├── ViewModels/                          # @Observable state
│   ├── GenerationViewModel.swift
│   ├── HistoryViewModel.swift
│   ├── PlayerViewModel.swift
│   ├── SettingsViewModel.swift
│   └── SidebarViewModel.swift
│
├── Views/
│   ├── ContentView.swift                # Root: bootstraps services, overlays SetupView
│   ├── LogViewerView.swift
│   ├── AudioToAudio/
│   │   ├── AudioImportView.swift
│   │   └── LoRAManagerView.swift
│   ├── Generation/
│   │   ├── GenerationView.swift
│   │   ├── LyricEditorView.swift
│   │   ├── ParameterControlsView.swift
│   │   └── TagEditorView.swift
│   ├── History/
│   │   ├── HistoryBrowserView.swift
│   │   └── HistoryItemView.swift
│   ├── Onboarding/
│   │   └── SetupView.swift
│   ├── Player/
│   │   ├── PlayerView.swift
│   │   ├── SpectrumAnalyzerView.swift
│   │   └── WaveformView.swift
│   ├── Settings/
│   │   ├── ModelSettingsView.swift
│   │   └── SettingsView.swift
│   └── Sidebar/
│       ├── PresetListView.swift
│       ├── RecentListView.swift
│       └── SidebarView.swift
│
└── Resources/

CantisTests/
├── ModelTests.swift
├── ServiceTests.swift
├── ViewModelTests.swift
├── ACEStepDiTTests.swift               # MLX integration (Xcode-only)
├── ACEStepLMTests.swift                # MLX integration (Xcode-only)
├── FeasibilityProbeTests.swift         # MLX integration (Xcode-only)
├── Qwen3ConditioningTests.swift        # MLX integration (Xcode-only)
└── Qwen3RealWeightsTests.swift         # MLX integration (Xcode-only)

tools/
└── convert_weights.py                  # PyTorch → MLX weight converter

modeling_acestep_v15_turbo.py           # Reference PyTorch model (used by converter)
```

## Suggested Checks Before Opening a PR

```bash
swift build
swift test --skip 'ACEStepDiTTests|ACEStepLMTests|FeasibilityProbeTests|Qwen3ConditioningTests|Qwen3RealWeightsTests'
python3 -m py_compile modeling_acestep_v15_turbo.py tools/convert_weights.py
```

Run the MLX integration suites from Xcode if your change touches `Cantis/Inference/`.

## Troubleshooting

- **Setup overlay sticks at "downloading"**: check the Cantis log window (Window > Cantis Logs) for HTTP errors. Downloads are resumable — quit and re-launch and the engine will pick up where it left off.
- **`weightsNotFound` error after download**: confirm every file listed in `NativeInferenceEngine.isDownloaded(_:)` is present under `~/Library/Application Support/Cantis/Models/<variant>/`. Non-turbo variants symlink into the turbo directory; if you deleted the turbo bundle, re-download it.
- **`Run python tools/convert_weights.py …` error in Settings**: you selected an XL variant. Run the converter once and re-launch.
- **Resident memory grows across generations**: enable Settings → Low-memory mode; it halves the MLX cache limit. Also confirm "Load 5 Hz LM" is off unless you actually need it (~1.2 GB extra).
- **App not appearing in Dock**: the SPM executable relies on `AppDelegate` to set `setActivationPolicy(.regular)`. Make sure `CantisApp.swift` still uses `@NSApplicationDelegateAdaptor`.
- **`swift build` complains about SDK version**: you need Xcode 26 or the macOS 26 SDK. CI runs on `macos-26` for the same reason.

## Coding Expectations

- Keep PRs scoped and testable.
- Add or update tests for behavior changes (deterministic suites for logic, MLX integration suites for inference).
- Update docs when introducing user-visible or workflow changes.
- Follow Swift 6 concurrency patterns (`@Observable`, `actor`, `Sendable`, `async/await`).
- Use `@MainActor` for ViewModels and UI-facing services (incl. `NativeInferenceEngine`).
- Use `actor` for thread-safe services (e.g. `ModelDownloader`).
- No Combine — use Swift structured concurrency.
- Use `AppLogger.shared` with appropriate categories.
- Constants go in `AppConstants` — avoid magic strings and numbers.
- For new generation knobs, extend `GenerationParameters` with `Codable` defaulting in the custom `init(from:)` so existing presets keep decoding.
