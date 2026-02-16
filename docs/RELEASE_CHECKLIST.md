# Release Checklist

## Code and Quality

- [ ] `swift build` succeeds with no warnings.
- [ ] `swift test` passes on default branch.
- [ ] Python server script compiles: `python3 -m py_compile AuraluxEngine/server.py`.
- [ ] No debug code, private paths, or credentials committed.
- [ ] CI pipeline passes (GitHub Actions).

## Engine and Inference

- [ ] `setup_env.sh` runs cleanly on a fresh machine (clones ACE-Step 1.5, installs deps).
- [ ] Server starts and responds to `GET /health` with models loaded.
- [ ] End-to-end generation works: prompt → audio file → playback.
- [ ] Model download from HuggingFace completes successfully (~4 GB).
- [ ] MPS workarounds are active and no PyTorch MPS crashes observed.
- [ ] Server shuts down gracefully on app quit (no orphaned processes).

## Application

- [ ] Onboarding flow completes successfully on first launch.
- [ ] Engine status badge reflects correct state (red/yellow/green).
- [ ] Generation, playback, export, history, and presets all functional.
- [ ] Log viewer window opens and displays logs correctly.
- [ ] App promotes to GUI correctly (menu bar, Dock icon).

## Documentation

- [ ] `README.md` reflects current features and limitations.
- [ ] `CHANGELOG.md` updated with notable changes.
- [ ] `AGENTS.md` is current with codebase structure and conventions.
- [ ] New settings/environment variables documented in `DEVELOPMENT.md` and `AuraluxEngine/README.md`.

## Community and Governance

- [ ] `LICENSE`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md` present and current.
- [ ] Issue and PR templates remain aligned with workflow.

## Release Prep

- [ ] Tag/version prepared.
- [ ] Release notes summarize user-visible changes and known limitations.
- [ ] App version string updated in `AuraluxApp.swift` (`applicationVersion`).
