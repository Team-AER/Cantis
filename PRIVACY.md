# Privacy Policy

**App:** Cantis
**Developer:** Prakhar Shukla
**Effective date:** 2026-11-18

## Summary

Cantis generates music on your Mac. Your prompts, lyrics, generation history, imported audio and exported tracks stay on your device. The app downloads model weights and, in official monitored builds, sends limited crash and error diagnostics to our GlitchTip service to help fix faults. It has no user accounts, advertising, usage analytics or tracking.

## Information we do not collect

We do not collect, store, or transmit any of the following:

- Personal information (name, email address, phone number, postal address)
- Account credentials — Cantis has no sign-in or account system
- Usage analytics or behavioral profiling
- Advertising identifiers (IDFA, IDFV) or any tracking identifiers
- Contacts, calendars, photos, or other system data
- Location

Cantis does not display advertising, does not use any third-party analytics SDK, and does not share any information with data brokers.

## Information stored locally on your Mac

Cantis stores the following on your computer only. This data never leaves your device.

- **Generation history** (in SwiftData): the prompts, lyrics, tags, generation parameters (duration, variance, seed), and file paths for tracks you have generated.
- **Saved presets**: prompts, lyric templates, tags, and parameters you have saved for reuse.
- **Exported audio**: WAV, AAC, or ALAC files written to locations you choose.
- **App settings** (in `UserDefaults`): preferences such as low-memory mode and the active model.
- **Model weights**: AI model files downloaded into `~/Library/Application Support/Cantis/Models/`.
- **Custom model registry** (if you add your own models): a list of HuggingFace repository identifiers in `~/Library/Application Support/Cantis/CustomModels/registry.json`.
- **Local diagnostic logs**: written via Apple's unified logging system (`OSLog`). These are not attached to remote error reports.

## Network activity

Cantis uses the network for model downloads and the limited diagnostics described below. Music generation runs locally.

The network destinations are:

- `https://huggingface.co` — for model weights published at `Team-AER/ace-step-v1.5-mlx`, and (if you opt into them) `Team-AER/ace-step-v1.5-sft-mlx` and `Team-AER/ace-step-v1.5-base-mlx`.
- Any HuggingFace repository you choose to add via the "custom model" feature. If you add a custom model, Cantis fetches its weights from that HuggingFace repository.

When the app downloads model files, HuggingFace receives the standard request metadata that any web request includes — your IP address, a User-Agent string, and the requested file path — under HuggingFace's own privacy policy: <https://huggingface.co/privacy>. We (the developer of Cantis) do not operate that endpoint, do not control or receive that metadata, and do not send any Cantis-specific information along with the request. Specifically, **no prompts, lyrics, generated audio, imported audio, generation history, settings, or device identifiers are transmitted as part of the model download.**

After model downloads, generation works offline. Configured diagnostics can be sent when a network connection is available. No prompts, lyrics, audio, or generation results are ever sent to any server.

## Crash and error diagnostics

Official monitored builds use the Sentry Cocoa SDK to send diagnostics to our GlitchTip service at `https://glitchtip.brainfc.uk/`. Reports contain error types, diagnostic code stacks, app release information and SDK metadata. Before sending, the app removes request context, users, breadcrumbs, extras, exception messages and NSError user-info and mechanism text. Stored error events have IP-address scrubbing enabled.

Prompts, lyrics, audio, generated tracks and history are not attached to these reports. Reports are not linked to an app account, and we do not configure advertising or persistent user identifiers. We use diagnostics to investigate application faults, not to profile behavior or track people across apps and websites. Network infrastructure necessarily receives connection metadata, including the source IP address, and its access logs may contain that metadata. Diagnostics follow the service's configured retention policy.

Source builds with no monitoring DSN configured do not send reports. The reporting configuration for a distributed build is set before signing.

## Microphone access

Cantis requests microphone access for one feature: audio-to-audio style transfer. The system permission prompt explains this with the following text:

> "Cantis needs microphone access for audio-to-audio style transfer features."

When you use this feature, audio captured from the microphone is processed locally and in memory by the on-device model. Captured audio is not stored persistently unless you explicitly export it. It is never transmitted off your Mac.

You can decline microphone access at install time, revoke it later in **System Settings → Privacy & Security → Microphone**, or simply not use the audio-to-audio feature. Text-to-music generation does not require the microphone.

## Imported audio

Cantis lets you drag and drop audio files (WAV, AAC, ALAC, etc.) as input to the cover, repaint, and extract generation modes. Imported audio is read locally through the macOS App Sandbox's user-selected file access. It is processed on-device and is never uploaded.

## Exported audio

When you export a generated track, Cantis writes a WAV, AAC (`.m4a`), or ALAC (`.m4a`) file to the location you choose. Exports are local file writes; nothing is sent over the network.

## Third-party software

Cantis is built with the following open-source libraries. These are compiled into the app and run in-process:

- [`mlx-swift`](https://github.com/ml-explore/mlx-swift) — Apple's MLX framework for on-device machine learning.
- [`swift-collections`](https://github.com/apple/swift-collections) — Standard Swift data structures.
- [`swift-numerics`](https://github.com/apple/swift-numerics) — Numeric protocols and types.
- [`SwiftUI-Shimmer`](https://github.com/markiv/SwiftUI-Shimmer) — A SwiftUI loading-shimmer effect.

The Sentry Cocoa SDK provides the configured GlitchTip diagnostics described above; no Sentry-hosted service is configured. HuggingFace provides model downloads.

## Children's privacy

Cantis is not directed at children under 13 and does not knowingly collect personal information from anyone, including children. Diagnostic reporting follows the limits described above.

## Your control over your data

Because all your data stays on your Mac, you control it directly:

- **Delete a single track**: remove it from the in-app history.
- **Delete all generation history**: remove the SwiftData store and exported audio under `~/Library/Application Support/Cantis/`.
- **Remove model weights**: delete `~/Library/Application Support/Cantis/Models/`.
- **Uninstall**: drag Cantis to the Trash and remove `~/Library/Application Support/Cantis/`. Local data removal is separate from diagnostics already received by the monitoring service.

## Privacy questions

For questions about local data or diagnostic reports, use the contact channels below. The app has no user account identifier that links a diagnostic report to your local generation history.

## Security

Cantis runs in the macOS App Sandbox and requests only the entitlements it needs:

- `com.apple.security.app-sandbox` — App Sandbox (required for Mac App Store).
- `com.apple.security.files.user-selected.read-write` — read and write files you explicitly choose or drag in.
- `com.apple.security.device.audio-input` — microphone access for audio-to-audio (only when you use it).
- `com.apple.security.network.client` — outbound HTTPS requests for model downloads and configured diagnostics.

If you discover a security or privacy issue, please report it privately using GitHub Security Advisories on the Cantis repository (see [SECURITY.md](SECURITY.md)) rather than a public issue.

## Changes to this policy

If this policy ever changes, the new version is committed to the Cantis repository and the **Effective date** at the top of this document is updated. The full revision history is available in git.

## Contact

Cantis is maintained by an individual developer. For privacy-related questions, please use GitHub:

- **General questions**: open an issue on the Cantis GitHub repository.
- **Security or privacy vulnerabilities**: use GitHub Security Advisories on the Cantis GitHub repository (per [SECURITY.md](SECURITY.md)). Please do not file public issues for vulnerabilities.
