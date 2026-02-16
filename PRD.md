# Product Requirements Document: Auralux Native macOS Application

## Executive Summary

**Product Name:** Auralux for macOS  
**Version:** 1.0  
**Target Platform:** macOS 13.0+ (Apple Silicon required)  
**Product Type:** Native AI Music Generation Application  
**Core Technology:** ACE-Step v1-3.5B Foundation Model with Metal Performance Shaders acceleration

Auralux is a native application that brings the full capabilities of the ACE-Step music generation foundation model to Apple Silicon Macs, leveraging Metal GPU acceleration for real-time music synthesis. The application enables text-to-music generation, audio-to-audio transformation, voice cloning, lyric-driven composition, and advanced audio manipulation—all running locally on Apple Silicon hardware without cloud dependencies.[1][2]

## Product Overview

### Vision
Democratize professional music creation by providing a fast, private, and offline-capable music generation tool that harnesses the full computational power of Apple Silicon GPUs to deliver production-quality music synthesis in seconds.

### Target Users
- Music producers and composers
- Content creators and podcasters
- Game developers requiring dynamic music
- Film and video editors
- Hobbyist musicians and AI enthusiasts
- Technical users requiring local, private music generation

### Key Differentiators
- **Native Apple Silicon optimization** using Metal Performance Shaders and ANE
- **Fully offline operation** with no cloud API dependencies
- **Real-time generation** leveraging M-series GPU cores
- **Privacy-first architecture** with all processing on-device
- **Advanced controllability** via structured lyrics, tags, and audio conditioning

## Core Features

### 1. Text-to-Music Generation (Text2Music)

#### 1.1 Natural Language Prompts
- Free-form text description input supporting multi-sentence prompts
- Structured tag system: genre, instruments, tempo, mood, energy level
- Predefined tag categories:
  - **Genres:** funk, pop, soul, rock, electronic, jazz, classical, hip-hop, ambient, folk, metal, country
  - **Instruments:** guitar, drums, bass, keyboard, piano, synth, violin, saxophone, vocals
  - **Characteristics:** melodic, energetic, upbeat, groovy, vibrant, dynamic, calm, aggressive, dreamy
  - **Technical:** BPM range (40-200), key signature, time signature

#### 1.2 Lyric-Driven Composition
- Structured lyric input with markup support:
  - `[verse]` - Verse sections with narrative progression
  - `[chorus]` - Repeated chorus sections with harmonic emphasis
  - `[bridge]` - Bridge sections for tonal variation
  - `[pre-chorus]` - Pre-chorus build-up sections
  - `[outro]` - Ending sections
  - `[instrumental]` or `[inst]` - Pure instrumental segments
- Real-time lyric-music alignment using MERT/m-hubert semantic representation[3][1]
- Lyric editing and regeneration for specific sections

#### 1.3 Duration Control
- Adjustable audio duration: 30 seconds to 4 minutes
- Slider-based UI with precise second-level control
- Estimated generation time display based on duration and hardware

#### 1.4 Variance and Seed Control
- Variance slider (0.0 - 1.0) for controlling randomness vs. coherence
- Seed value input for reproducible generation
- "Retake" functionality with configurable seed variations
- Batch generation with seed array support

### 2. Audio-to-Audio Generation (Audio2Audio)

#### 2.1 Reference Audio Processing
- Support for input audio formats: WAV, MP3, FLAC, AAC, AIFF, M4A
- Audio file drag-and-drop interface
- Waveform visualization with playback controls
- Audio trimming and segment selection

#### 2.2 LoRA Model Support
- LoRA (Low-Rank Adaptation) model loading from local filesystem
- User-trained LoRA integration for style transfer
- LoRA weight adjustment (0.0 - 2.0 multiplier)
- LoRA model management: import, organize, preview metadata

#### 2.3 Transformation Modes
- **Style transfer:** Apply musical style from reference audio to generated content
- **Voice cloning:** Extract and apply vocal characteristics
- **Remixing:** Generate variations maintaining core musical elements
- **Accompaniment generation:** Extract vocals, generate instrumental backing
- **Vocal extraction:** Generate vocals from instrumental description

### 3. Advanced Control Mechanisms

#### 3.1 Multi-Track Generation
- Separate generation of vocal and instrumental tracks
- Lyric2Vocal: Generate isolated vocal tracks from lyrics
- Singing2Accompaniment: Generate backing tracks for existing vocals
- Per-track volume and pan control
- Export stems separately or mixed

#### 3.2 Preset Management
- Save/load generation configurations as presets
- Community preset sharing (export/import JSON)
- Categorized preset library: genres, moods, use cases
- Preset preview with embedded audio samples

#### 3.3 Generation History
- Persistent history of all generated audio
- Searchable meta prompts, tags, timestamps, parameters
- Favorite/bookmark system
- Batch export functionality
- SQLite database for history management

### 4. Audio Playback and Export

#### 4.1 Playback Controls
- Native AVAudioEngine integration
- Waveform visualization with playback position indicator
- Standard controls: play, pause, stop, scrub, loop
- Real-time FFT spectrum analyzer
- Volume normalization and dynamics processing

#### 4.2 Export Options
- Formats: WAV (16/24/32-bit), FLAC, MP3 (320kbps), AAC, ALAC
- Sample rates: 44.1kHz, 48kHz, 96kHz
- Metadata embedding: title, artist, tags, generation parameters
- Batch export with naming templates
- Direct export to Music.app, GarageBand, Logic Pro

## Technical Architecture

### 5. System Architecture

#### 5.1 Application Stack
```
┌─────────────────────────────────────────┐
│         SwiftUI User Interface          │
├─────────────────────────────────────────┤
│      Audio Playback (AVFoundation)      │
├─────────────────────────────────────────┤
│       Business Logic (Swift/C++)        │
├─────────────────────────────────────────┤
│   Model Inference Engine (CoreML/MPS)   │
├─────────────────────────────────────────┤
│         Metal Compute Kernels           │
├─────────────────────────────────────────┤
│    Apple Silicon GPU/ANE Hardware       │
└─────────────────────────────────────────┘
```

#### 5.2 Model Conversion Pipeline
- **Source:** ACE-Step v1-3.5B PyTorch checkpoint from HuggingFace[4]
- **Conversion Path:** PyTorch → ONNX → CoreML (MLModel format)
- **Components to Convert:**
  1. Deep Compression AutoEncoder (DCAE) - Encoder/Decoder
  2. Linear Transformer backbone
  3. MERT text encoder
  4. m-hubert audio encoder
  5. Diffusion U-Net architecture

#### 5.3 Metal Performance Shaders Integration
- Custom Metal compute shaders for:
  - Matrix multiplication operations in transformer layers[5]
  - Attention mechanism computation
  - Diffusion sampling loops
  - Audio spectrogram processing
- MPS Graph API for operator fusion and optimization
- Automatic ANE fallback for compatible layers

### 6. Performance Optimization

#### 6.1 Target Performance Metrics
| Hardware | 4-Min Audio Generation | Memory Usage | GPU Utilization |
|----------|------------------------|--------------|-----------------|
| M1/M2 Base | < 60 seconds | < 4GB | > 85% |
| M1/M2 Pro | < 40 seconds | < 6GB | > 90% |
| M1/M2 Max | < 25 seconds | < 8GB | > 92% |
| M1/M2 Ultra | < 15 seconds | < 12GB | > 95% |
| M3/M4 Series | 10-40% faster than equivalent M1/M2 | Variable | > 90% |

#### 6.2 Optimization Strategies
- **Model quantization:** INT8/FP16 mixed precision for reduced memory footprint
- **Operator fusion:** Combine sequential Metal operations to minimize memory transfers
- **Compute/memory overlap:** Asynchronous dispatch with Metal command buffers
- **Dynamic batching:** Batch inference when generating multiple variations
- **Lazy loading:** Load DCAE/transformer components on-demand
- **Unified memory utilization:** Leverage Apple Silicon unified memory architecture
- **Background processing:** Utilize Grand Central Dispatch for non-blocking generation

#### 6.3 Memory Management
- Automatic model unloading when app backgrounded
- Configurable memory cache size for generated audio
- Progressive loading for large LoRA models
- Memory pressure monitoring with graceful degradation
- Compression of generated audio in history database

### 7. Model Integration

#### 7.1 CoreML Model Structure
```
ACEStep.mlpackage/
├── encoder_mert.mlmodel          # Text encoding (MERT)
├── encoder_mhubert.mlmodel       # Audio encoding (m-hubert)
├── dcae_encoder.mlmodel          # Deep compression encoder
├── dcae_decoder.mlmodel          # Deep compression decoder
├── transformer.mlmodel           # Linear transformer backbone
├── diffusion_unet.mlmodel        # Diffusion model
├── vocoder.mlmodel               # Audio synthesis
└── metadata.json                 # Model configuration
```

#### 7.2 Inference Pipeline
1. **Text Processing:** MERT encoder → text embeddings (768-dim)
2. **Audio Conditioning (if provided):** m-hubert encoder → audio embeddings
3. **Latent Initialization:** Gaussian noise in compressed latent space
4. **Diffusion Denoising:** 20-50 steps with DDPM/DDIM scheduler
5. **Transformer Processing:** Linear attention across temporal dimension
6. **DCAE Decoding:** Latent → mel-spectrogram (16x compression ratio)
7. **Vocoding:** Mel-spectrogram → waveform (HiFi-GAN/BigVGAN)

#### 7.3 LoRA Integration
- Load LoRA weights as separate `.safetensors` files
- Apply low-rank adapters to transformer attention/FFN layers
- Runtime weight merging with configurable alpha parameter
- Support for multiple LoRA stacking (up to 3 simultaneous)

### 8. Technical Stack

#### 8.1 Development Frameworks
- **UI:** SwiftUI with AppKit integration for advanced controls
- **Audio:** AVFoundation, Core Audio, AudioToolbox
- **ML:** Core ML, Metal Performance Shaders (MPS), Accelerate.framework
- **Compute:** Metal, Metal Performance Shaders Graph
- **Storage:** SQLite (generation history), FileManager (audio files)
- **Utilities:** Combine (reactive programming), async/await (concurrency)

#### 8.2 Third-Party Dependencies
- **CoreMLTools** (Python, for model conversion)
- **coremltools-optimize** (quantization and pruning)
- **libsndfile** or **AudioKit** (audio I/O abstractions)
- **Swift-Collections** (efficient data structures)
- **SwiftFFT** or **Accelerate.framework** (FFT for visualization)

#### 8.3 Build System
- Xcode 15.0+
- Swift 5.9+
- Minimum deployment: macOS 13.0 (Ventura)
- Target architectures: arm64 (Apple Silicon only, no Intel support)
- Sandboxed with entitlements: file access, audio input/output

## User Interface Design

### 9. UI/UX Requirements

#### 9.1 Main Window Layout
```
┌─────────────────────────────────────────────────────────┐
│  [≡] Auralux                     [⚙] Settings  [?] Help │
├──────────────┬──────────────────────────────────────────┤
│              │  ╔══════════════════════════════════╗    │
│   Presets    │  ║  Text-to-Music Generation        ║    │
│   ─────────  │  ╚══════════════════════════════════╝    │
│   • Rock     │                                           │
│   • Pop      │  Tags: [funk, guitar, 105 BPM, upbeat]   │
│   • Jazz     │                                           │
│   • Custom   │  ┌─────────────────────────────────┐     │
│              │  │ Lyrics:                         │     │
│   Recent     │  │ [verse]                         │     │
│   ─────────  │  │ City lights shimmer bright      │     │
│   ▶ Track 1  │  │ [chorus]                        │     │
│   ▶ Track 2  │  │ Dancing through the night       │     │
│              │  └─────────────────────────────────┘     │
│              │                                           │
│              │  Duration: [━━━●─────] 180s               │
│              │  Variance: [━━━━●────] 0.5                │
│              │                                           │
│              │  [Generate] [Retake]                      │
│              │                                           │
│              │  ┌─────────────────────────────────┐     │
│              │  │ ▶ [████████░░░] Generated Audio │     │
│              │  └─────────────────────────────────┘     │
└──────────────┴──────────────────────────────────────────┘
```

#### 9.2 Design Principles
- **Native macOS aesthetics:** Follow Apple Human Interface Guidelines
- **Dark mode support:** Automatic theme switching with system preferences
- **Accessibility:** VoiceOver support, keyboard navigation, high contrast mode
- **Responsive layout:** Adapt to window resizing (min: 1024x768)
- **Visual feedback:** Progress indicators, real-time generation stats
- **Contextual help:** Tooltips, inline documentation, example templates

#### 9.3 Key UI Components
1. **Tag Editor:** Autocomplete with suggestions from predefined categories
2. **Lyric Editor:** Syntax highlighting for structure tags, line numbers
3. **Waveform View:** Zoomable, scrollable, with selection capabilities
4. **Generation Queue:** Multi-threaded batch processing with priority
5. **Settings Panel:** Model selection, performance tuning, export preferences
6. **History Browser:** Grid/list view with audio previews and search

### 10. Advanced Features (Post-MVP)

#### 10.1 Real-Time Generation
- Streaming inference for continuous music generation
- Live parameter adjustment during playback
- Morphing between multiple prompts/styles

#### 10.2 MIDI Integration
- MIDI file import for structure control
- Real-time MIDI input for interactive generation
- MIDI export of generated melodies

#### 10.3 Plugin Architecture
- Audio Unit (AU) plugin version
- VST3 plugin for DAW integration
- Standalone CLI tool for batch processing

#### 10.4 Collaborative Features
- Project file format (.auralux) for sharing complete configurations
- Export to cloud storage providers (iCloud, Dropbox)
- Community model repository integration

## Development Roadmap

### Phase 1: Foundation (Months 1-2)
- [ ] Model conversion from PyTorch to CoreML
- [ ] Metal shader implementation for custom operations
- [ ] Basic text-to-music generation pipeline
- [ ] SwiftUI skeleton with text input and playback
- [ ] Benchmark performance on M1/M2/M3 hardware

### Phase 2: Core Features (Months 3-4)
- [ ] Lyric structure parsing and visualization
- [ ] Tag system with autocomplete
- [ ] Audio2Audio with LoRA support
- [ ] Waveform visualization and editing
- [ ] Generation history with SQLite backend
- [ ] Settings panel with model configuration

### Phase 3: Polish & Optimization (Month 5)
- [ ] Advanced audio export options
- [ ] Preset management system
- [ ] Memory optimization and quantization
- [ ] UI/UX refinement and accessibility
- [ ] Comprehensive error handling
- [ ] Performance profiling with Instruments

### Phase 4: Advanced Features (Month 6)
- [ ] Multi-track generation and stem separation
- [ ] Advanced LoRA management
- [ ] Batch generation and queuing
- [ ] Real-time generation experimentation
- [ ] Documentation and tutorials
- [ ] Beta testing and feedback integration

### Phase 5: Release (Month 7)
- [ ] App Store submission preparation
- [ ] Code signing and notarization
- [ ] Marketing materials and website
- [ ] Open-source repository setup (if applicable)
- [ ] Community engagement and support channels

## Technical Specifications

### 11. System Requirements

#### 11.1 Minimum Requirements
- **Hardware:** Mac with Apple Silicon (M1 or later)
- **RAM:** 8GB (16GB recommended for M1 Base)
- **Storage:** 10GB available (5GB for models, 5GB for cache)
- **OS:** macOS 13.0 (Ventura) or later
- **Display:** 1920x1080 or higher resolution

#### 11.2 Recommended Requirements
- **Hardware:** Mac with M1 Pro/Max, M2 Pro/Max, M3 series, or M4 series
- **RAM:** 16GB or higher
- **Storage:** 20GB+ SSD with fast read/write speeds
- **OS:** macOS 14.0 (Sonoma) or macOS 15.0 (Sequoia)
- **Display:** Retina display for optimal UI experience

#### 11.3 Model Storage
| Component | Size (FP32) | Size (FP16) | Size (INT8) |
|-----------|-------------|-------------|-------------|
| MERT Encoder | 450MB | 225MB | 115MB |
| m-hubert Encoder | 380MB | 190MB | 95MB |
| DCAE Encoder | 620MB | 310MB | 155MB |
| DCAE Decoder | 640MB | 320MB | 160MB |
| Transformer | 1.8GB | 900MB | 450MB |
| Diffusion U-Net | 1.2GB | 600MB | 300MB |
| Vocoder | 210MB | 105MB | 55MB |
| **Total** | **~5.3GB** | **~2.65GB** | **~1.33GB** |

### 12. API and Integration

#### 12.1 Internal Swift API
```swift
// Core generation interface
class AuraluxGenerator {
    func generateMusic(
        prompt: String,
        tags: [String],
        lyrics: String?,
        duration: TimeInterval,
        variance: Float,
        seed: Int?,
        loraPath: URL?,
        progressHandler: @escaping (Float) -> Void
    ) async throws -> AudioFile
    
    func generateFromAudio(
        referenceAudio: URL,
        prompt: String,
        loraPath: URL?,
        strength: Float
    ) async throws -> AudioFile
    
    func cancelGeneration()
}

// Preset management
struct GenerationPreset: Codable {
    var name: String
    var tags: [String]
    var lyricsTemplate: String?
    var defaultDuration: TimeInterval
    var variance: Float
}

// History management
class GenerationHistory {
    func save(audio: AudioFile, meta GenerationMetadata) throws
    func search(query: String) -> [HistoryEntry]
    func delete(id: UUID) throws
}
```

#### 12.2 CLI Interface (Future)
```bash
# Basic text-to-music
auralux generate --prompt "upbeat rock song" --duration 120 --output track.wav

# With lyrics
auralux generate --lyrics lyrics.txt --tags "rock,guitar,energetic" --output song.wav

# Audio-to-audio with LoRA
auralux transform --input vocal.wav --lora ./models/jazz-style.safetensors --output jazz-vocal.wav

# Batch generation
auralux batch --config batch.json --output-dir ./generated/
```

#### 12.3 URL Scheme (Inter-App Communication)
```
auralux://generate?prompt=upbeat%20jazz&duration=180
auralux://open-preset?name=Rock%20Ballad
auralux://import-lora?path=/path/to/model.safetensors
```

### 13. Security and Privacy

#### 13.1 Data Privacy
- **100% on-device processing** - no network requests during generation
- **No telemetry or analytics** without explicit user consent
- **Optional crash reporting** with anonymization
- **Sandboxed architecture** following macOS security guidelines
- **User data ownership** - all generated content belongs to user

#### 13.2 Model Security
- **Signed model packages** to prevent tampering
- **Checksum verification** on model load
- **Secure LoRA loading** with validation
- **No executable code in models** - data-only CoreML format

#### 13.3 Compliance
- **Copyright considerations:** Generated music ownership and licensing
- **EULA and Terms of Service** for acceptable use
- **Open-source licensing** (if applicable) - consider Apache 2.0 or MIT
- **Model license compliance** with ACE-Step's terms

### 14. Testing Strategy

#### 14.1 Unit Testing
- Model loading and initialization
- Audio processing pipeline
- Preset serialization/deserialization
- History database operations
- Tag parsing and validation

#### 14.2 Integration Testing
- End-to-end generation pipeline
- LoRA loading and application
- Multi-threaded generation queue
- Memory management under load
- Export format validation

#### 14.3 Performance Testing
- Generation time benchmarks across hardware
- Memory usage profiling
- GPU utilization metrics
- Battery impact measurement
- Thermal throttling behavior

#### 14.4 User Acceptance Testing
- Usability testing with target users
- Accessibility verification
- Cross-device testing (various M-series Macs)
- Long-running stability tests
- Edge case handling (malformed inputs, very long lyrics)

### 15. Success Metrics

#### 15.1 Performance KPIs
- **Generation speed:** < 30s for 4-min audio on M1 Pro[1]
- **Model load time:** < 5s on app launch
- **Memory footprint:** < 6GB peak during generation
- **GPU utilization:** > 85% during active generation
- **App launch time:** < 2s cold start

#### 15.2 Quality Metrics
- **Musical coherence:** User satisfaction > 4.0/5.0
- **Lyric alignment:** Subjective evaluation > 85% accuracy
- **Audio fidelity:** No artifacts, clipping, or distortion
- **Reproducibility:** Identical outputs for same seed/params

#### 15.3 User Engagement
- **Daily active users:** Target growth metric
- **Generations per session:** Average > 5
- **Preset usage:** > 40% users create custom presets
- **Export rate:** > 70% of generations exported
- **Retention:** 30-day retention > 50%

## Risk Assessment

### 16. Technical Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| CoreML conversion issues for complex diffusion models | High | Incremental conversion with ONNX intermediate, fallback to Metal shaders |
| Performance not meeting targets on M1 base | Medium | Dynamic quality scaling, offer "fast" mode with reduced steps |
| Memory constraints on 8GB systems | Medium | Aggressive quantization, model component unloading, streaming inference |
| Metal shader bugs or incompatibilities | Medium | Extensive testing, fallback to CPU for problematic ops |
| App Store rejection (size, content) | Low | Pre-submission review, content policy compliance, download models separately |

### 17. Business Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| ACE-Step model licensing restrictions | High | Clarify open-source license, engage with ACE Studio/StepFun |
| Competing apps with similar features | Medium | Focus on native performance and UX, open-source advantage |
| Limited market size (Apple Silicon only) | Low | M-series adoption growing rapidly, consider iOS version |
| User-generated content copyright issues | Medium | Clear EULA, disclaimer, educational fair use guidance |

## Appendix

### A. Glossary
- **DCAE:** Deep Compression AutoEncoder - Sana's encoder for latent space compression
- **MERT:** Music Understanding Model for text-audio semantic alignment
- **m-hubert:** Masked HuBERT for audio representation learning
- **LoRA:** Low-Rank Adaptation - efficient fine-tuning technique
- **MPS:** Metal Performance Shaders - Apple's GPU acceleration framework
- **ANE:** Apple Neural Engine - dedicated ML hardware accelerator
- **CoreML:** Apple's machine learning framework for on-device inference

### B. References
- ACE-Step GitHub Repository[1]
- ACE-Step Research Paper[3]
- Apple Metal Documentation[6]
- Metal Performance Shaders Guide[5]
- CoreML Model Conversion Best Practices
- Swift Concurrency and async/await Patterns

### C. Contact and Support
- **Product Owner:** [To be assigned]
- **Technical Lead:** [To be assigned]
- **Design Lead:** [To be assigned]
- **Community:** Discord/GitHub Discussions
- **Documentation:** Wiki and inline help system

***

**Document Version:** 1.0  
**Last Updated:** February 16, 2026  
**Status:** Draft for Review  
**Next Review:** Phase 1 Completion

Sources
[1] ACE-Step: A Step Towards Music Generation Foundation Model https://github.com/ace-step/ACE-Step
[2] ACE-Step: A Step Towards Music Generation Foundation Model https://ace-step.github.io
[3] ACE-Step: A Step Towards Music Generation Foundation Model https://arxiv.org/html/2506.00045v1
[4] ACE-Step/ACE-Step-v1-3.5B - Hugging Face https://huggingface.co/ACE-Step/ACE-Step-v1-3.5B
[5] Fast transformer inference with Metal Performance Shaders https://explosion.ai/blog/metal-performance-shaders
[6] Metal Overview - Apple Developer https://developer.apple.com/metal/
[7] Screenshot-2026-02-16-at-6.55.00-PM.jpeg https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/images/37234058/69d702cf-b1d3-4781-b086-4c4bd3744c39/Screenshot-2026-02-16-at-6.55.00-PM.jpeg
[8] ACE-Step: AI Song Generator for Random Song Creationacestep.io https://acestep.io
[9] Ace-Step AI Music Generation v1-3.5B Model https://acestepai.org
[10] TangoFlux Endless: Real-time Audio Generation on Apple Silicon https://github.com/ochyai/TangoFlux-Endless
[11] FluidInference/diar-streaming-sortformer-coreml https://huggingface.co/FluidInference/diar-streaming-sortformer-coreml

