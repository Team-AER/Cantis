# Contributing to Auralux

Thanks for contributing.

## Ground Rules

- Be respectful and follow the [Code of Conduct](CODE_OF_CONDUCT.md).
- Prefer small pull requests with focused scope.
- Include tests for behavior changes where possible.
- Keep generated files and unrelated refactors out of feature PRs.

## Development Setup

1. Install Xcode 16+ (or Swift 6.0+) and ensure Python 3.11+ is available.
2. Clone the repository.
3. Set up the Python environment:
   ```bash
   cd AuraluxEngine
   ./setup_env.sh
   ```
4. Start the local API server (optional — the app manages this automatically):
   ```bash
   cd AuraluxEngine
   ./start_api_server_macos.sh
   ```
5. Build and run:
   ```bash
   swift run Auralux
   ```
6. Run tests:
   ```bash
   swift test
   ```

See [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for the full development guide, troubleshooting, and project structure.

## Branch and Commit Conventions

- Branches: `feature/<short-name>`, `fix/<short-name>`, `docs/<short-name>`.
- Commits: imperative summary (`Add queue retry backoff`).
- Keep commits logically grouped.

## Pull Request Checklist

- [ ] Tests added/updated for changed behavior.
- [ ] `swift test` passes.
- [ ] `python3 -m py_compile AuraluxEngine/server.py` passes (if server was modified).
- [ ] Docs updated for user-visible changes.
- [ ] No secrets, credentials, or personal paths are introduced.

## Coding Standards

- Follow Swift 6 concurrency patterns (`@Observable`, `actor`, `Sendable`, `async/await`).
- Use `AppLogger.shared` for logging (not `print()`).
- Constants go in `AppConstants` — avoid magic strings.
- No Combine — use Swift structured concurrency.
- See [AGENTS.md](AGENTS.md) for detailed coding conventions and patterns.

## Reporting Bugs and Requesting Features

Use GitHub Issues and include:

- Clear reproduction steps.
- Expected vs actual behavior.
- Environment details (macOS version, Swift version, chip model, RAM).
