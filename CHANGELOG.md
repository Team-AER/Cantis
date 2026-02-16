# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added

- **Core application** — SwiftUI macOS app built with Swift 6, `@Observable`, and SwiftData.
- **ACE-Step v1.5 integration** — Python inference server wrapping ACE-Step with PyTorch MPS + MLX for on-device AI music generation on Apple Silicon.
- **Engine lifecycle management** — `EngineService` handles automatic setup detection, environment provisioning, server start/stop, health monitoring, and graceful shutdown.
- **Onboarding flow** — first-run `SetupView` guides users through environment setup, server start, and model download.
- **Engine status indicator** — toolbar badge with red/yellow/green status and click-to-settings.
- **Text-to-music generation** — prompt-based music generation via `GenerationView` with tag editor, lyric editor, and parameter controls (duration, variance, seed).
- **Tag system** — genre, instrument, and mood tags with autocomplete suggestions.
- **Lyrics support** — structured lyric editor with verse/chorus/bridge markup.
- **Audio playback** — `AVAudioEngine`-based player with waveform visualization and real-time FFT spectrum analyzer.
- **Multi-format export** — WAV (16/24/32-bit), FLAC, MP3, AAC, ALAC with configurable sample rates.
- **Generation history** — SwiftData-backed history browser with search and favorites.
- **Preset system** — save, load, and manage generation configurations.
- **Audio-to-audio views** — audio import interface and LoRA model management views.
- **Generation queue** — job queue service with priority ordering.
- **Model management** — model download status tracking and trigger via API.
- **Log viewer** — dedicated log viewer window for debugging and monitoring.
- **Sidebar navigation** — presets list, recent generations, and section navigation.
- **MPS workarounds** — runtime patches for PyTorch MPS bugs (masked_fill, inference_mode, codec/encoder CPU fallbacks).
- **Centralized logging** — `AppLogger` with OSLog categories for structured logging.
- **CI pipeline** — GitHub Actions running `swift test` on macOS and Python syntax validation.
- **Community docs** — contributing guide, code of conduct, security policy, support info.
- **GitHub templates** — issue templates (bug report, feature request), PR template, Dependabot config.
- **Comprehensive documentation** — README, architecture guide, development guide, release checklist, engine README, AGENTS.md.
