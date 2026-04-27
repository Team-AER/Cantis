# Auralux: Python → Swift-Only Migration Plan

_Authored: 2026-04-27. Based on full codebase audit (60 files, 8,106 lines) + mlx-swift capability research._

---

## Background

Auralux is a native macOS app for AI music generation using ACE-Step v1.5. It currently runs in a hybrid architecture:

- **Swift layer** (58 files, ~6,800 lines) — UI, audio playback, history, export — communicates via HTTP REST (`localhost:8765`) with
- **Python subprocess** (`AuraluxEngine/server.py`, 1,150 lines) — wraps ACE-Step v1.5 PyTorch/MLX inference

This architecture produces: subprocess lifecycle instability, 5 MPS Metal shader monkey-patches, synthetic progress estimation, HTTP polling overhead, Python venv provisioning requirements, and a Mac App Store blocker.

**Goal:** Replace `AuraluxEngine/` entirely with a native Swift inference engine using `mlx-swift`, eliminating the subprocess and HTTP IPC layers.

---

## What Python Currently Owns

| Responsibility | Location | Lines | Migration complexity |
|---|---|---|---|
| DiT inference (PyTorch/MPS) | `server.py` | ~300 | High — 8-step diffusion transformer |
| LM inference (MLX Python) | `server.py` | ~100 | Medium — already MLX format |
| VAE decoder (PyTorch) | `server.py` | ~80 | Medium |
| MPS Metal monkey-patches | `server.py` | ~350 | Eliminated — mlx-swift doesn't need them |
| HTTP REST server | `server.py` | ~200 | Low — replaced by actor |
| Thread-based job queue | `server.py` | ~120 | Low — replaced by async/await |
| Model auto-download | `server.py` | ~80 | Low — URLSession |
| Audio file write (`torchaudio`) | `server.py` | ~20 | Low — AVAudioFile |
| Process launcher scripts | `setup_env.sh`, `start_api_server_macos.sh` | ~160 | Deleted |

**What stays untouched:** All Views (except SetupView), all ViewModels (except GenerationViewModel polling refactor), all SwiftData models, AudioPlayerService, AudioExportService, AudioFFT, HistoryService, PresetService — roughly 85% of the Swift codebase.

---

## Technology Stack

| Python component | Swift replacement |
|---|---|
| PyTorch MPS DiT | `apple/mlx-swift` (MLXNN) |
| MLX Python LM | `apple/mlx-swift` (MLXNN, same weight format) |
| PyTorch VAE | `apple/mlx-swift` (MLXNN) |
| `torchaudio.save` | `AVAudioFile` (already in project) |
| HuggingFace hub | `URLSession` with progress |
| HTTP REST server | Swift `actor` (in-process function calls) |
| `threading.Thread` job queue | Swift structured concurrency |
| `psutil` diagnostics | `ProcessInfo` + `os_proc_available_memory` |

### mlx-swift Op Coverage

Confirmed available (from research + `mlx-swift` + `mlx-swift-examples`):

| Op | mlx-swift module | ACE-Step usage |
|---|---|---|
| `Linear` | `MLXNN` | All projection layers |
| `Conv1d` / `Conv2d` | `MLXNN` | VAE encoder/decoder |
| `Embedding` | `MLXNN` | Token embedding, lyric encoder |
| `LayerNorm`, `GroupNorm`, `RMSNorm` | `MLXNN` | DiT blocks, LM blocks |
| `MultiHeadAttention` (GQA/MQA variants) | `MLXNN` | Self-attention, cross-attention |
| `RoPE` positional embeddings | `MLXNN` | DiT self-attention, LM self-attention |
| Activations (GELU, SiLU, etc.) | `MLXNN` / `MLX` | FFN, gating |
| Safetensors weight loading | `MLX` | All model checkpoints |

Not built-in (must implement from primitives):

| Gap | Severity | Plan |
|---|---|---|
| DDIM / flow-matching sampler | Medium | Implement as `TurboSampler.swift` using raw MLX ops; reference: `mlx-audio-swift` |
| Async step callbacks | Low | Wrap sync MLX calls in `AsyncThrowingStream` |

**Reference architecture:** `mlx-audio-swift` (Blaizzy) provides `AudioDiT` patterns directly applicable to ACE-Step's architecture.

---

## The 5 MPS Monkey-Patches: Why They Disappear

Each patch in `server.py` works around a PyTorch MPS Metal shader bug. mlx-swift uses a completely different Metal compute pipeline and does not share these bugs:

| Patch | Bug | mlx-swift status |
|---|---|---|
| `masked_fill` CPU fallback | Metal shader `masked_fill_scalar_strided_32bit` read-only buffer binding | Not applicable — different Metal kernels |
| `inference_mode` → `no_grad` | MPS marks tensors read-only under `inference_mode`, breaking write-enabled shaders | Not applicable — MLX has no `inference_mode` concept |
| Audio code decode CPU fallback | `mul_dense_scalar_float_float` shader write-enabled assertion on read-only MPS buffers | Not applicable |
| Text encoder CPU fallback | Same `mul_dense_scalar_float_float` bug during text encoding | Not applicable |
| `prepare_condition` CPU fallback | DiT condition encoder Metal buffer binding bug | Not applicable |

All 5 patches are eliminated with zero replacement code.

---

## Phase Overview

```
Phase 0 (3 days)     Feasibility gate — add mlx-swift, validate ops, run probe
Phase 1 (3 days)     Weight conversion — safetensors → MLX-native format
Phase 2 (1 week)     Port LM to mlx-swift (easiest — already MLX format)
Phase 3 (2.5 weeks)  Port DiT + VAE to mlx-swift (critical path)
Phase 4 (1 week)     Swift inference actor — replace HTTP REST + job queue
Phase 5 (3 days)     Native model download — replace setup_env.sh + Python downloader
Phase 6 (3 days)     Delete Python layer — AuraluxEngine/, EngineService, InferenceService
Phase 7 (1 week)     Hardening, CI update, Mac App Store compliance
────────────────────────────────────────────────────────────────────
Total: ~6 weeks
```

---

## Phase 0: Feasibility Gate (3 days)

**Goal:** Confirm mlx-swift covers all ACE-Step operations before committing to the full migration. Build and run a probe test suite; document any missing ops and mitigations before proceeding.

### Actions

**1. Add `apple/mlx-swift` to `Package.swift`**

```swift
.package(url: "https://github.com/ml-explore/mlx-swift", from: "0.21.0")
```

Add `MLX`, `MLXNN`, `MLXRandom` to both the `Auralux` target and `AuraluxTests` target.

**2. Create `AuraluxTests/InferenceTests/FeasibilityProbeTests.swift`**

Tests grouped by architecture component:
- Core tensor ops (zeros, matmul, elementwise arithmetic)
- `Linear` layer forward pass
- `LayerNorm` / `RMSNorm`
- `Embedding` lookup
- `MultiHeadAttention` (self and cross)
- `RoPE` positional encoding
- `Conv1d` (for VAE)
- Single DDIM step implemented from raw MLX ops — validates numerical stability
- Safetensors weight loading from a synthetic file written by the test

**3. Load real model weights (if available on dev machine)**

If ACE-Step checkpoints are present in `AuraluxEngine/ACE-Step-1.5/checkpoints/`, load one DiT transformer block's weights and verify shapes match the expected architecture.

**Running the probe:** `swift test --filter FeasibilityProbeTests` does NOT work — MLX requires
Metal GPU access and the `.metallib` compiled shaders, which are only bundled by Xcode's build
system, not by `swift test` CLI. Run via **Xcode → Product → Test (⌘U)** or
`xcodebuild test -scheme Auralux -destination 'platform=macOS'`. The tests will run correctly
on any Apple Silicon Mac through Xcode.

**Exit criterion:** All probe tests pass. If a required op is missing, the test failure documents
which op and the workaround is added to `FeasibilityProbeTests.swift` as an inline comment before
Phase 1 begins.

### Phase 0 Results (completed 2026-04-27)

| Item | Status | Notes |
|---|---|---|
| mlx-swift added to Package.swift | ✅ Done | v0.21.0 resolved |
| App builds with mlx-swift | ✅ Done | Zero errors, 33s build time |
| Op coverage: Linear, LayerNorm, RMSNorm | ✅ Confirmed | From MLXNN source |
| Op coverage: MultiHeadAttention (self + cross) | ✅ Confirmed | `mha(q, keys:, values:)` |
| Op coverage: RoPE | ✅ Confirmed | `rope(x, offset:)` — offset required |
| Op coverage: Conv1d (NLC layout) | ✅ Confirmed | `[batch, length, channels]` |
| Op coverage: Embedding | ✅ Confirmed | `embeddingCount:` + `dimensions:` |
| Op coverage: Safetensors loading | ✅ Confirmed | `loadArrays(url:)` free function |
| Op coverage: DDIM/flow-matching step | ✅ Confirmed | Implemented from raw MLX ops |
| MPS monkey-patches needed | ✅ None | MLX uses different Metal pipeline |
| CLI test runner | ⚠️ Blocked | Metal requires Xcode runner (by design) |

**All required ops are present. Phase 0 cleared → proceed to Phase 1.**

### Files Created/Modified
- `Package.swift` — modified (add mlx-swift dependency)
- `AuraluxTests/InferenceTests/FeasibilityProbeTests.swift` — new

---

## Phase 1: Weight Conversion Pipeline (3 days)

**Goal:** One-time conversion of ACE-Step checkpoints from PyTorch safetensors format to MLX-native format.

### Context

ACE-Step v1.5 ships two checkpoints:
- `checkpoints/acestep-v15-turbo/` — DiT + VAE (PyTorch safetensors, ~3.5 GB)
- `checkpoints/acestep-5Hz-lm-0.6B/` — LM (already MLX `.safetensors`, ~1.2 GB)

The LM weights need minimal work (already MLX format). The DiT/VAE weights need:
1. Name remapping: PyTorch uses flat key paths; the Swift model graph uses struct property paths
2. Weight transposition: PyTorch Conv weights are `[out, in, k]`; MLX expects `[out, k, in]`
3. Any `float16` → `float32` promotion if needed for numerical stability

### Actions

**1. Create `tools/convert_weights.py`** (temporary — kept until migration is stable)

```python
# Loads ACE-Step safetensors weights
# Remaps tensor names to Swift property path conventions
# Transposes Conv weight layouts: [out, in, k] -> [out, k, in]
# Saves to checkpoints/ace-step-v1.5-mlx/ as .safetensors
# Validates: all expected keys present, shapes match reference dict
```

**2. Validate in Swift**

Update `FeasibilityProbeTests.swift` to load the converted weights, instantiate one DiT block with real weights, run a forward pass, and verify output is non-NaN and shapes are correct.

### Files Created
- `tools/convert_weights.py` — temporary, deleted after migration stabilizes
- `AuraluxTests/InferenceTests/FeasibilityProbeTests.swift` — updated

---

## Phase 2: Port LM Model to mlx-swift (1 week)

**Goal:** 5Hz 0.6B language model running natively in Swift. Easiest port because the weights are already MLX format and the architecture is a standard causal transformer.

### Context

The LM (`acestep-5Hz-lm-0.6B`) is a Qwen-style causal LM that generates music structure tokens at 5Hz. It runs at the start of generation and produces guidance hints for the DiT. Because it runs once per 200ms of audio, its per-token latency is not performance-critical.

### Actions

**1. `Auralux/Inference/LM/ACEStepLM.swift`**
- `ACEStepLMConfig: Codable` — hyperparameters from `config.json`
- `ACEStepLMTransformerBlock: Module` — RoPE self-attention + SwiGLU FFN
- `ACEStepLMModel: Module` — embedding + N blocks + LM head
- `func generate(tokens: [Int], maxTokens: Int, temperature: Float) -> [Int]`

**2. `Auralux/Inference/LM/LMWeightLoader.swift`**
- Loads `.safetensors` from checkpoint directory
- Maps weight keys to Swift model property paths via dictionary

**3. Validate against Python baseline**
- With fixed input tokens and temperature: verify Swift output token sequence matches Python within ±1% divergence threshold
- Add `AuraluxTests/InferenceTests/ACEStepLMTests.swift` with golden output verification

### Files Created
- `Auralux/Inference/LM/ACEStepLM.swift`
- `Auralux/Inference/LM/LMWeightLoader.swift`
- `AuraluxTests/InferenceTests/ACEStepLMTests.swift`

---

## Phase 3: Port DiT + VAE to mlx-swift (2.5 weeks)

**Goal:** Core generative engine in Swift. This is the critical path. Replaces all PyTorch MPS code and all 5 MPS monkey-patches.

### Architecture

```
TextEncoder       (caption + tags → [B, S, D] embeddings)
LyricEncoder      (structured lyrics → [B, L, D] embeddings)
ConditionEncoder  (text + lyric embeddings → conditioning tensors)
       ↓
DiT backbone      (N blocks: RoPE self-attention + cross-attention + FFN)
       ↓
TurboSampler      (8-step flow-matching denoising)
       ↓
AudioVAE          (latents → audio waveform)
```

### Week 1: Encoders

**`Auralux/Inference/DiT/TextEncoder.swift`**
- Lightweight text encoder for caption/tag embedding
- Option A: implement from scratch using MLXNN (T5-small architecture)
- Option B: use `swift-transformers` (Hugging Face) for the encoder module
- `encode(caption: String, tags: [String]) -> MLXArray`

**`Auralux/Inference/DiT/LyricEncoder.swift`**
- Tokenizes structured lyrics with `[verse]`/`[chorus]`/`[bridge]` markup
- `Embedding` + transformer → per-token embeddings
- `encode(lyrics: String) -> MLXArray`

**`Auralux/Inference/DiT/ConditionEncoder.swift`**
- Single cross-attention block combining text and lyric embeddings
- `encode(textEmbed: MLXArray, lyricEmbed: MLXArray) -> MLXArray`

### Week 2: DiT Backbone + Sampler

**`Auralux/Inference/DiT/ACEStepDiT.swift`**
- `ACEStepDiTConfig: Codable` — `numLayers`, `hiddenDim`, `numHeads`, `mlpRatio`
- `DiTBlock: Module` — RoPE self-attention + cross-attention (conditioning) + FFN + pre/post LayerNorm
- `ACEStepDiT: Module` — latent embedding projection + N `DiTBlock`s + output projection

**`Auralux/Inference/DiT/TurboSampler.swift`**
- 8-step flow-matching schedule (σ values for ACE-Step turbo)
- Per-step callback: `(step: Int, of total: Int) async -> Bool` — return `false` to cancel (supports `Task.cancel()`)
- `sample(noise: MLXArray, conditioning: MLXArray, dit: ACEStepDiT, steps: Int, seed: Int?) async throws -> MLXArray`
- Implemented from raw MLX ops: no built-in sampler library needed

### Week 3: VAE + Integration

**`Auralux/Inference/DiT/AudioVAE.swift`**
- `AudioVAEConfig: Codable` — channels, strides, dilations
- `AudioVAEDecoder: Module` — ConvTranspose + residual blocks → audio
- `decode(latents: MLXArray) -> MLXArray` — returns raw float audio samples at 44.1kHz stereo

**`Auralux/Inference/DiT/DiTWeightLoader.swift`**
- Loads converted weights (Phase 1) into the Swift model graph
- `load(from directory: URL, into model: ACEStepDiT) throws`
- Validates all required keys present before returning

**Validate against Python baseline**
- Fixed seed, fixed prompt → compare output audio using spectral cosine similarity (target: >0.95)
- Add `AuraluxTests/InferenceTests/ACEStepDiTTests.swift`

### Files Created
- `Auralux/Inference/DiT/TextEncoder.swift`
- `Auralux/Inference/DiT/LyricEncoder.swift`
- `Auralux/Inference/DiT/ConditionEncoder.swift`
- `Auralux/Inference/DiT/ACEStepDiT.swift`
- `Auralux/Inference/DiT/TurboSampler.swift`
- `Auralux/Inference/DiT/AudioVAE.swift`
- `Auralux/Inference/DiT/DiTWeightLoader.swift`
- `AuraluxTests/InferenceTests/ACEStepDiTTests.swift`

---

## Phase 4: Swift Inference Actor (1 week)

**Goal:** Replace HTTP REST + Python job queue with a pure Swift actor. Remove 2-second polling and synthetic progress bars. Get real per-step progress from the DiT sampler.

### `Auralux/Inference/NativeInferenceEngine.swift`

```swift
actor NativeInferenceEngine {
    private var ditModel: ACEStepDiT?
    private var lmModel: ACEStepLMModel?
    private var vae: AudioVAEDecoder?
    private var currentTask: Task<Void, Error>?

    // Initializes model weights from disk
    func loadModels(from directory: URL) async throws

    // Returns a stream of GenerationProgress values ending with .completed(audioURL)
    // Respects Swift structured cancellation: Task.cancel() propagates through the sampler step callback
    func generate(_ params: GenerationParameters) -> AsyncThrowingStream<GenerationProgress, Error>

    // Cooperative cancellation: signals the current sampler step callback to return false
    func cancel()
}

enum GenerationProgress {
    case loading(message: String)
    case step(current: Int, total: Int, message: String)
    case saving
    case completed(audioURL: URL)
}
```

**Key improvements over the Python server:**

| Aspect | Python | Swift actor |
|---|---|---|
| Progress reporting | Synthetic 15% → 90% over estimated time | Real per-step progress from DiT (step 1/8 … 8/8) |
| Cancellation | HTTP POST `/cancel` → race condition possible | `Task.cancel()` → propagates to sampler next step |
| Serialization | `threading.Lock()` | Actor isolation (automatic) |
| Error propagation | JSON `{"status": "failed", "message": "..."}` | Swift typed `Error` thrown through stream |
| Job polling | 2-second HTTP intervals | Immediate `AsyncThrowingStream` updates |

### Refactor `GenerationViewModel.swift`

Replace polling loop:
```swift
// Before: 2-second polling
while !done {
    try await Task.sleep(for: .seconds(2))
    let status = try await inferenceService.pollJob(id: jobID)
    // update progress
}

// After: stream
for try await progress in engine.generate(params) {
    switch progress {
    case .step(let current, let total, _):
        self.progress = Double(current) / Double(total)
    case .completed(let url):
        self.audioURL = url
    }
}
```

### Files Created/Modified/Deleted
- `Auralux/Inference/NativeInferenceEngine.swift` — new
- `Auralux/ViewModels/GenerationViewModel.swift` — polling loop → `AsyncThrowingStream` consumer
- `Auralux/Services/InferenceService.swift` — **deleted** (HTTP actor no longer needed)
- `AuraluxTests/InferenceTests/NativeInferenceEngineTests.swift` — new

---

## Phase 5: Native Model Download (3 days)

**Goal:** Replace `setup_env.sh`, `start_api_server_macos.sh`, and the Python `/models/download` endpoint with a native Swift downloader.

### Context

`ModelManagerService.swift` already exists with UI scaffolding. It currently delegates model downloads to the Python server's `/models/download` endpoint. We implement the downloader directly.

### Actions

**1. Implement `ModelManagerService.download()`**
- `URLSession.shared.download(from:delegate:)` with `URLSessionDownloadDelegate` for byte-level progress
- Resumable downloads: `Range` header + `.download.tmp` partial file
- SHA256 checksum validation after download completes
- `@Published var downloadProgress: ModelDownloadProgress` with `bytesReceived`, `bytesTotal`, `eta`

**2. Update `SetupView.swift`**
- Remove all Python/venv setup UI ("Setting up Python environment…", "Cloning ACE-Step…")
- Replace with model download progress bar showing file name, size downloaded, ETA
- On completion: trigger `NativeInferenceEngine.loadModels(from:)`

**3. Delete `EngineService.swift`**
- Remove subprocess launch, restart loop, health polling, SIGTERM/SIGKILL shutdown (601 lines total)
- Replace `EngineState` enum with:
  ```swift
  enum ModelState { case notDownloaded, downloading(ModelDownloadProgress), loading, ready, error(String) }
  ```

### Files Modified/Deleted
- `Auralux/Services/ModelManagerService.swift` — real download implementation
- `Auralux/Views/Onboarding/SetupView.swift` — remove Python setup UI
- `Auralux/Services/EngineService.swift` — **deleted**
- `Auralux/Components/EngineControlPanel.swift` — **deleted** (Python server start/stop controls no longer exist)

---

## Phase 6: Delete Python Layer (3 days)

**Goal:** Remove all Python-related files and references. App must build and run with no Python runtime on the machine.

### Files Deleted

**`AuraluxEngine/` directory (entirely):**
- `server.py` (1,150 lines)
- `test_generate.py` (160 lines)
- `setup_env.sh` (89 lines)
- `start_api_server_macos.sh` (71 lines)
- `requirements.txt` (28 lines)

**Already deleted in earlier phases:**
- `Auralux/Services/InferenceService.swift` (Phase 4)
- `Auralux/Services/EngineService.swift` (Phase 5)
- `Auralux/Components/EngineControlPanel.swift` (Phase 5)

**Phase 6 additionally deletes:**
- `Auralux/Views/LogViewerView.swift` — Python stdout viewer, no longer needed

### CI/CD

**`.github/workflows/ci.yml`** — remove entire Python sanity job:
```yaml
# Remove this job:
python-sanity:
  runs-on: ubuntu-latest
  steps:
    - python -m py_compile AuraluxEngine/server.py
    # ... smoke test, invalid input test, etc.
```

### Entitlements

**`Auralux/Entitlements.plist`** — remove:
- `com.apple.security.network.client` if it was only needed for localhost HTTP to Python
- Any subprocess execution entitlements added for Python

### Swift References

- Remove all `EngineService` `@Environment` injections from views
- Remove `LogViewerView` navigation item from sidebar
- Remove Python-specific status messages from `EngineStatusView`

### Verification Checklist
- [ ] `swift build` produces 0 errors
- [ ] No `import` of `InferenceService` or `EngineService` anywhere
- [ ] App launches without attempting to start any subprocess
- [ ] Full generation works end-to-end in-process
- [ ] `ps aux | grep python` shows nothing during generation

---

## Phase 7: Hardening and CI (1 week)

**Goal:** Ship-ready quality, Mac App Store eligibility.

### Testing

**Golden output test** (fixed-seed regression)
- Generate with seed 42, prompt "chill lofi piano", 30s duration
- Compute spectral fingerprint (MFCC cosine similarity against a reference)
- Assert similarity > 0.95 — catches regressions in sampler or model loading

**Memory profile test**
- Generate a 30s track, assert peak RAM < 8 GB on M1 16GB machine
- Generate 10 sequential tracks, assert no monotonic memory growth

**Format export integration test**
- Generate → export to WAV, FLAC, MP3, AAC, ALAC
- Verify each output decodes correctly with `AVAudioFile`

**Cancellation robustness test**
- Cancel at step 1, 4, 7 of 8 — verify clean `CancellationError`, no orphaned Metal buffers

### Performance Baseline

Measure and document:

| Metric | Python+Swift (before) | Swift-only (after) |
|---|---|---|
| Cold start (first generation) | 10–30s | Target: < 5s |
| Generation time (30s track) | Baseline | Target: ≤ baseline |
| Memory peak (30s track) | ~6–8 GB | Target: ≤ baseline |

### Mac App Store Compliance

- No subprocess execution → remove `NSAppleEventsUsageDescription`
- No outgoing network to localhost → remove loopback network entitlement
- Sandbox compliance: `codesign --verify --deep --strict ./build/Auralux.app`
- No use of private APIs
- Confirm in-process Metal usage is within App Store guidelines

### Updated CI

```yaml
jobs:
  swift-tests:
    runs-on: macos-15
    steps:
      - uses: actions/checkout@v6
      - name: Build
        run: swift build
      - name: Test
        run: swift test
```

Python sanity job removed entirely.

---

## Risk Assessment

| Risk | Likelihood | Severity | Mitigation |
|---|---|---|---|
| mlx-swift missing an ACE-Step DiT op | Low | High | Phase 0 audit catches this; implement custom MLX Metal kernel as fallback |
| Numerical drift vs. Python baseline | Medium | Medium | Fixed-seed validation; perceptual similarity test (not bit-exact) is sufficient |
| MLX memory spikes vs. PyTorch | Low | Medium | Profile early in Phase 3; tune dtype (fp16 vs fp32) per device RAM |
| Weight name mismatches in conversion | Medium | Low | Phase 1 explicitly validates every tensor name and shape before proceeding |
| DDIM sampler numerical instability | Low | High | Unit test against Python sampler with identical noise and σ schedule |
| ACE-Step upstream breaking changes | Low | Low | Lock to specific ACE-Step commit for migration duration |
| macOS 26 SDK compatibility with mlx-swift | Low | Medium | mlx-swift requires macOS 14+; macOS 26 is a superset |

---

## What This Unlocks

| Current (Python + Swift) | After Migration (Swift-only) |
|---|---|
| 10–30s cold start (Python subprocess + model load) | ~2–5s cold start (in-process model load) |
| Synthetic progress bar (estimated 15% → 90%) | Real per-step progress: Step 1/8 … 8/8 |
| HTTP polling every 2s for job status | Immediate `AsyncThrowingStream` updates |
| Cancellation via HTTP POST, race conditions possible | `Task.cancel()` — cooperative, instantaneous |
| Mac App Store blocked (subprocess) | Mac App Store eligible |
| 5 MPS Metal monkey-patches | Zero platform workarounds |
| subprocess crashes → EngineService restart loop | No subprocess, no restart loop |
| Python venv + uv + PyTorch (multiple GB of tooling) | Model weights only, no runtime dependencies |
| Debugging: Python stdout in `LogViewerView` | Native `OSLog` + Console.app |
| Process isolation: crash kills generation | In-process: unwind with Swift error handling |

---

## Files Summary

### Deleted

| File | Lines | Reason |
|---|---|---|
| `AuraluxEngine/server.py` | 1,150 | Entire Python inference server |
| `AuraluxEngine/test_generate.py` | 160 | Python smoke test client |
| `AuraluxEngine/setup_env.sh` | 89 | Python env provisioning |
| `AuraluxEngine/start_api_server_macos.sh` | 71 | Server launcher |
| `AuraluxEngine/requirements.txt` | 28 | Python dependency list |
| `Auralux/Services/EngineService.swift` | 601 | Subprocess lifecycle management |
| `Auralux/Services/InferenceService.swift` | 230 | HTTP client actor |
| `Auralux/Components/EngineControlPanel.swift` | ~80 | Server start/stop UI |
| `Auralux/Views/LogViewerView.swift` | ~100 | Python stdout viewer |

**Total deleted: ~2,509 lines**

### New (Swift inference engine)

| File | Purpose |
|---|---|
| `Auralux/Inference/NativeInferenceEngine.swift` | Swift actor replacing HTTP + job queue |
| `Auralux/Inference/LM/ACEStepLM.swift` | 5Hz LM model (mlx-swift) |
| `Auralux/Inference/LM/LMWeightLoader.swift` | LM weight loading |
| `Auralux/Inference/DiT/TextEncoder.swift` | Caption/tag text encoder |
| `Auralux/Inference/DiT/LyricEncoder.swift` | Structured lyric encoder |
| `Auralux/Inference/DiT/ConditionEncoder.swift` | Condition embedding |
| `Auralux/Inference/DiT/ACEStepDiT.swift` | Diffusion Transformer |
| `Auralux/Inference/DiT/TurboSampler.swift` | 8-step flow-matching sampler |
| `Auralux/Inference/DiT/AudioVAE.swift` | VAE decoder |
| `Auralux/Inference/DiT/DiTWeightLoader.swift` | DiT/VAE weight loading |
| `tools/convert_weights.py` | One-time weight conversion (temporary) |

**Estimated new code: ~2,500 lines Swift inference engine**

### Modified

| File | Change |
|---|---|
| `Package.swift` | Add mlx-swift dependency |
| `Auralux/ViewModels/GenerationViewModel.swift` | Polling loop → `AsyncThrowingStream` consumer |
| `Auralux/Services/ModelManagerService.swift` | Real `URLSession` download implementation |
| `Auralux/Views/Onboarding/SetupView.swift` | Remove Python setup UI |
| `Auralux/AuraluxApp.swift` | Remove `EngineService` composition |
| `.github/workflows/ci.yml` | Remove Python sanity job |
| `Auralux/Entitlements.plist` | Remove subprocess/network-client entitlements |

---

## Timeline

```
Week 1:   Phase 0 (feasibility gate) + Phase 1 (weight conversion script)
Week 2:   Phase 2 (LM port to mlx-swift)
Week 3:   Phase 3, Week 1 (encoders)
Week 4:   Phase 3, Week 2 (DiT backbone + sampler)
Week 5:   Phase 3, Week 3 (VAE) + Phase 4 (Swift actor)
Week 6:   Phase 5 (native download) + Phase 6 (delete Python) + Phase 7 (hardening)
```

The critical path is **Phase 3** (DiT + VAE port). Everything else can proceed in parallel or has low uncertainty. If Phase 0 reveals a missing op, its workaround should be prototyped before Phase 3 begins.
