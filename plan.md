# Auralux — Implementation Plan

## Problem Statement
Build a native macOS application for AI music generation using the ACE-Step v1-3.5B foundation model, running entirely on-device on Apple Silicon. The app must be Xcode-native, beautiful, and use the latest frameworks.

## ⚠️ Critical Finding: CoreML Trap — Avoided

**The PRD proposes: PyTorch → ONNX → CoreML. This is the trap.**

After extensive research, CoreML conversion of ACE-Step is **high-risk and likely to fail** because:
1. **No official CoreML conversion exists** for ACE-Step or any 3.5B-param music diffusion model
2. **Custom operators** (DCAE, linear transformer with rotary PE, MERT encoder, m-hubert) have no CoreML equivalents — would require hand-writing conversion logic for each
3. **CoreML file/layer limits** — 3.5B params exceed practical CoreML model size limits, causing failed imports or silent operator fallback to CPU
4. **Diffusion loop control flow** — CoreML is a static-graph inference engine; ACE-Step's iterative denoising (20-50 steps) with dynamic scheduling requires workarounds that degrade performance
5. **Months of conversion work** with no guarantee of parity — community has failed to convert even smaller diffusion models cleanly

### ✅ Recommended Approach: MLX (Apple's Native ML Framework)

**ACE-Step 1.5 already has built-in MLX support** with Mac-specific launch scripts. MLX is the correct path:

| Factor | CoreML (❌ Risky) | MLX (✅ Recommended) |
|--------|-------------------|----------------------|
| ACE-Step support | None, needs conversion | Built-in, tested |
| 3.5B param models | Impractical | First-class support |
| Custom layers | Must rewrite each | Python-native, works |
| Dynamic control flow | Static graph only | Full dynamic support |
| Apple Silicon optimization | Good (when it works) | Excellent (unified memory, Metal) |
| Swift integration | Native | mlx-swift SPM package |
| Time to working inference | 2-4 months (uncertain) | Days (already works) |

---

## Proposed Architecture

```
┌─────────────────────────────────────────────────────┐
│              SwiftUI Frontend (Liquid Glass)         │
│         @Observable, Swift 6, async/await            │
├─────────────────────────────────────────────────────┤
│            Swift Business Logic Layer                │
│    Presets, History (SwiftData), Export, Queue        │
├──────────────────┬──────────────────────────────────┤
│  Audio Engine    │    MLX Inference Bridge           │
│  AVAudioEngine   │  ┌────────────────────────────┐  │
│  Core Audio      │  │  Option A: mlx-swift        │  │
│  Accelerate FFT  │  │  (native Swift MLX calls)   │  │
│                  │  ├────────────────────────────┤  │
│                  │  │  Option B: Local API Server  │  │
│                  │  │  (bundled ACE-Step + MLX     │  │
│                  │  │   Python via subprocess)     │  │
│                  │  └────────────────────────────┘  │
├──────────────────┴──────────────────────────────────┤
│          MLX Runtime (Metal + Unified Memory)        │
├─────────────────────────────────────────────────────┤
│           Apple Silicon (M1/M2/M3/M4 GPU+ANE)       │
└─────────────────────────────────────────────────────┘
```

### Inference Strategy: Hybrid (Option B recommended for Phase 1)

**Phase 1 — Local API Server (fast path to working app):**
- Bundle ACE-Step 1.5 Python + MLX as a subprocess
- Swift app communicates via local HTTP REST API (localhost)
- ACE-Step already ships `start_api_server_macos.sh` — reuse this
- Pros: Proven inference, all features work, fast to ship
- Cons: Larger app bundle (~2GB), requires embedded Python

**Phase 2 — Native mlx-swift port (performance optimization):**
- Incrementally port ACE-Step inference components to pure Swift using mlx-swift
- Start with vocoder and DCAE decoder (smallest, most self-contained)
- Eliminate Python dependency over time
- Pros: Smaller bundle, faster startup, true native experience

---

## Tech Stack (Updated from PRD)

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| **UI** | SwiftUI + Liquid Glass | Latest macOS design language (WWDC25) |
| **State** | `@Observable` macro, `@Environment` | Modern Swift 6 pattern, replaces ObservableObject |
| **Concurrency** | Swift structured concurrency (`async/await`, `TaskGroup`) | Replaces Combine for most use cases |
| **ML Inference** | MLX (Python) → mlx-swift (future) | Proven ACE-Step support, Apple Silicon optimized |
| **Audio Playback** | AVAudioEngine + Core Audio | Native, low-latency audio pipeline |
| **Audio Viz** | Accelerate.framework (vDSP FFT) | Hardware-accelerated spectral analysis |
| **Storage** | SwiftData | Modern replacement for raw SQLite, Xcode-native |
| **Export** | AVFoundation + AudioToolbox | WAV, FLAC, MP3, AAC, ALAC encoding |
| **Networking** | URLSession (localhost only) | Communication with bundled MLX server |
| **Package Mgr** | Swift Package Manager | Xcode-native dependency management |
| **Min Deploy** | macOS 15.0 (Sequoia) | Required for latest SwiftUI + SwiftData features |
| **Xcode** | 16.0+ | Latest Swift 6, Liquid Glass, @Observable |
| **Swift** | 6.0+ | Strict concurrency, modern macros |

---

## Implementation Workplan

### Phase 0: Project Scaffolding
- [ ] Create Xcode project (macOS App, SwiftUI lifecycle, Swift 6)
- [ ] Configure SPM dependencies (mlx-swift, swift-collections, AudioKit)
- [ ] Set up project structure (MVVM + Services architecture)
- [ ] Configure build settings (arm64 only, macOS 15.0+, sandbox entitlements)
- [ ] Set up Git repository with .gitignore for Xcode/Swift

### Phase 1: MLX Inference Engine Integration
- [ ] Clone ACE-Step 1.5 repo, verify MLX inference on local Mac
- [ ] Create `InferenceService` — Swift wrapper to manage bundled Python/MLX subprocess
- [ ] Implement local API server lifecycle (start/stop/health-check)
- [ ] Build `GenerationRequest` / `GenerationResponse` Codable models
- [ ] Implement text-to-music generation via API bridge
- [ ] Implement progress tracking (poll or SSE from server)
- [ ] Handle errors gracefully (model loading, OOM, timeouts)
- [ ] Test generation on M1 Base (8GB) through M4 Max

### Phase 2: Core UI — SwiftUI Shell
- [ ] Implement `NavigationSplitView` 3-column layout (sidebar/content/detail)
- [ ] Build Sidebar: Presets list, Recent generations, navigation
- [ ] Build main generation view: tag editor, lyrics editor, controls
- [ ] Implement Tag Editor with autocomplete (genre, instrument, mood chips)
- [ ] Implement Lyric Editor with syntax highlighting for `[verse]`, `[chorus]` etc.
- [ ] Build parameter controls: Duration slider, Variance slider, Seed input
- [ ] Add Generate / Cancel / Retake buttons with state management
- [ ] Implement Liquid Glass toolbar and window chrome
- [ ] Dark mode + light mode automatic theming
- [ ] Minimum window size enforcement (1024×768)
- [ ] Accessibility: VoiceOver labels, keyboard navigation, focus management

### Phase 3: Audio Playback & Visualization
- [ ] Build `AudioPlayerService` using AVAudioEngine
- [ ] Implement waveform renderer (custom SwiftUI Canvas view)
- [ ] Implement real-time FFT spectrum analyzer (vDSP)
- [ ] Build transport controls: play, pause, stop, scrub, loop
- [ ] Waveform zoom and scroll with selection
- [ ] Volume normalization pipeline

### Phase 4: Data Layer & History
- [ ] Define SwiftData models: `GeneratedTrack`, `Preset`, `Tag`
- [ ] Implement `HistoryService` with search, filter, favorites
- [ ] Build History Browser view (grid + list toggle, previews)
- [ ] Implement Preset management (create, edit, delete, import/export JSON)
- [ ] Implement generation queue with priority ordering
- [ ] Audio file management (storage, cleanup, cache limits)

### Phase 5: Audio Export
- [ ] Implement multi-format export: WAV (16/24/32-bit), FLAC, MP3, AAC, ALAC
- [ ] Sample rate selection (44.1, 48, 96 kHz)
- [ ] Metadata embedding (title, tags, generation params)
- [ ] Batch export with naming templates
- [ ] Share sheet / export to Music.app integration
- [ ] Drag-and-drop export from waveform view

### Phase 6: Audio-to-Audio & LoRA
- [ ] Implement audio file import (drag-drop, file picker) for reference audio
- [ ] Wire up Audio2Audio API endpoints
- [ ] Build LoRA management view (import, organize, preview)
- [ ] LoRA weight adjustment UI
- [ ] Style transfer, voice cloning, remixing modes
- [ ] Audio trimming and segment selection UI

### Phase 7: Advanced Features
- [ ] Multi-track generation (vocal + instrumental separation)
- [ ] Per-track volume/pan controls
- [ ] Stem export (individual tracks)
- [ ] Batch generation with seed arrays
- [ ] Generation queue with parallel processing
- [ ] Settings panel: model config, performance tuning, export prefs
- [ ] Memory pressure monitoring and graceful degradation

### Phase 8: Polish & Optimization
- [ ] Profile with Instruments (GPU, Memory, Energy)
- [ ] Optimize MLX inference: INT8/FP16 quantization
- [ ] Lazy model loading (load components on-demand)
- [ ] App launch time optimization (<2s cold start)
- [ ] Comprehensive error handling + user-facing error messages
- [ ] Onboarding flow (first launch, model download)
- [ ] Keyboard shortcuts for all major actions
- [ ] Tooltips and contextual help
- [ ] URL scheme handler (`auralux://generate?prompt=...`)

### Phase 9: Testing & QA
- [ ] Unit tests: SwiftData models, Codable types, tag parsing
- [ ] Integration tests: end-to-end generation pipeline
- [ ] Performance benchmarks: generation time per hardware tier
- [ ] Memory profiling: peak usage on 8GB systems
- [ ] Accessibility audit (VoiceOver, keyboard-only navigation)
- [ ] Long-running stability tests

### Phase 10: Release Preparation
- [ ] App icon and brand assets
- [ ] Code signing and notarization
- [ ] Model download mechanism (separate from app bundle for App Store)
- [ ] App Store screenshots and description
- [ ] Privacy policy (no data collection, on-device only)
- [ ] License compliance (ACE-Step license, dependencies)
- [ ] README and user documentation

---

## Key Design Decisions

### 1. Why NOT CoreML
See "Critical Finding" section above. The conversion risk is too high for a 3.5B-param model with custom architecture. MLX is the proven, supported path.

### 2. Why MLX over PyTorch MPS
- MLX has native Swift bindings (mlx-swift) — can eventually become fully native
- MLX is Apple's own framework, optimized specifically for Apple Silicon unified memory
- ACE-Step 1.5 already supports MLX natively
- PyTorch MPS is Python-only, no Swift bridge, heavier runtime

### 3. Why Bundled Python Server (Phase 1) vs Pure Swift
- ACE-Step's inference pipeline is complex (DCAE + transformer + diffusion + vocoder)
- Porting all of this to Swift/mlx-swift would take months and risk bugs
- Shipping with bundled Python+MLX gets a working app fast
- Can iteratively port to pure Swift later (Phase 2 of inference strategy)

### 4. Why SwiftData over raw SQLite
- Native Xcode integration, @Model macro, automatic CloudKit sync potential
- Requires macOS 14+ (we target macOS 15+, so no issue)
- Less boilerplate than manual SQLite, type-safe queries

### 5. Why macOS 15+ (Sequoia) instead of macOS 13+ (PRD)
- Required for: SwiftData, @Observable, latest SwiftUI features, Liquid Glass
- Apple Silicon Macs all support macOS 15
- Targeting older OS means missing modern APIs and beautiful UI

### 6. Model Distribution Strategy
- **Dev/direct download:** Bundle models in app (~2.65GB FP16)
- **App Store:** Download models on first launch from CDN/HuggingFace
- Models stored in `~/Library/Application Support/Auralux/Models/`
- Checksum verification on download

---

## Project Structure

```
Auralux/
├── Auralux.xcodeproj
├── Auralux/
│   ├── AuraluxApp.swift                 # App entry point
│   ├── Info.plist
│   ├── Entitlements.plist
│   │
│   ├── Models/                          # SwiftData models
│   │   ├── GeneratedTrack.swift
│   │   ├── Preset.swift
│   │   ├── Tag.swift
│   │   └── GenerationParameters.swift
│   │
│   ├── Services/                        # Business logic
│   │   ├── InferenceService.swift       # MLX bridge
│   │   ├── AudioPlayerService.swift     # AVAudioEngine
│   │   ├── AudioExportService.swift     # Format conversion
│   │   ├── HistoryService.swift         # SwiftData queries
│   │   ├── PresetService.swift          # Preset management
│   │   ├── ModelManagerService.swift    # Download, verify models
│   │   └── GenerationQueueService.swift # Job scheduling
│   │
│   ├── ViewModels/                      # @Observable state
│   │   ├── GenerationViewModel.swift
│   │   ├── HistoryViewModel.swift
│   │   ├── PlayerViewModel.swift
│   │   ├── SettingsViewModel.swift
│   │   └── SidebarViewModel.swift
│   │
│   ├── Views/                           # SwiftUI views
│   │   ├── ContentView.swift            # Root NavigationSplitView
│   │   ├── Sidebar/
│   │   │   ├── SidebarView.swift
│   │   │   ├── PresetListView.swift
│   │   │   └── RecentListView.swift
│   │   ├── Generation/
│   │   │   ├── GenerationView.swift
│   │   │   ├── TagEditorView.swift
│   │   │   ├── LyricEditorView.swift
│   │   │   └── ParameterControlsView.swift
│   │   ├── Player/
│   │   │   ├── PlayerView.swift
│   │   │   ├── WaveformView.swift
│   │   │   └── SpectrumAnalyzerView.swift
│   │   ├── History/
│   │   │   ├── HistoryBrowserView.swift
│   │   │   └── HistoryItemView.swift
│   │   ├── AudioToAudio/
│   │   │   ├── AudioImportView.swift
│   │   │   └── LoRAManagerView.swift
│   │   └── Settings/
│   │       ├── SettingsView.swift
│   │       └── ModelSettingsView.swift
│   │
│   ├── Components/                      # Reusable UI
│   │   ├── TagChip.swift
│   │   ├── SliderControl.swift
│   │   ├── ProgressOverlay.swift
│   │   └── AudioDropZone.swift
│   │
│   ├── Utilities/
│   │   ├── AudioFFT.swift               # vDSP wrappers
│   │   ├── FileUtilities.swift
│   │   └── Constants.swift
│   │
│   └── Resources/
│       ├── Assets.xcassets
│       └── Presets/                      # Bundled presets JSON
│
├── AuraluxEngine/                       # Bundled MLX inference
│   ├── ace_step/                        # ACE-Step 1.5 source
│   ├── requirements.txt
│   ├── server.py                        # Local API server
│   └── setup_env.sh                     # Python env bootstrap
│
├── AuraluxTests/
│   ├── ModelTests.swift
│   ├── ServiceTests.swift
│   └── ViewModelTests.swift
│
└── Package.swift                        # SPM dependencies
```

---

## Dependencies (SPM)

| Package | Purpose |
|---------|---------|
| `ml-explore/mlx-swift` | MLX Swift bindings (Phase 2 inference) |
| `apple/swift-collections` | OrderedDictionary, Deque for queues |
| `AudioKit/AudioKit` | Audio I/O helpers, waveform rendering utils |
| `scinfu/SwiftUI-Shimmer` | Loading shimmer effects |

---

## Risk Mitigations

| Risk | Mitigation |
|------|-----------|
| Embedded Python bundle size (~500MB+) | Ship Python env separately, download on first launch |
| App Store rejection (embedded runtime) | Distribute via website/Homebrew initially; App Store version later with mlx-swift native port |
| 8GB RAM constraint | INT8 quantization reduces model to ~1.33GB; aggressive memory management |
| MLX operator gaps | ACE-Step 1.5 already tested on MLX; monitor upstream for fixes |
| Audio quality degradation from quantization | Offer FP16 (default) and INT8 (low-memory mode) options |
| Subprocess crash handling | Health-check ping, auto-restart, user notification |

---

## Notes

- The PRD's Xcode 15 / Swift 5.9 / macOS 13 targets are outdated for a 2026 app. Updated to Xcode 16+ / Swift 6 / macOS 15+.
- Liquid Glass design language (WWDC25) gives the app a stunning modern look with minimal custom styling.
- The `@Observable` macro eliminates most Combine boilerplate from the PRD's suggested architecture.
- SwiftData replaces raw SQLite for history — provides @Query, automatic migrations, and type safety.
- The ACE-Step 1.5 codebase (not v1 from PRD) is recommended as it's the latest with MLX support.
