# Auralux

Auralux is a native macOS SwiftUI app scaffold for on-device AI music generation using a local ACE-Step API bridge.

## Current implementation

- SwiftUI app shell with `NavigationSplitView`
- `@Observable` view models and SwiftData models
- Service layer: inference bridge, generation queue, history, presets, model manager, audio player/export
- Bundled Phase 1 local Python API server scaffold (`AuraluxEngine/server.py`)
- Starter preset resource bundle
- Unit tests for models, queue ordering, and generation view model behavior

## Run server (dev)

```bash
cd AuraluxEngine
./start_api_server_macos.sh
```

## Build and test

```bash
swift test
```
