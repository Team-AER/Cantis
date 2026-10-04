# Cantis user guide

Cantis is Team AER's native macOS music studio. The [product page](https://aer.app/cantis/) presents the studio; this guide covers the controls available in the application source.

## First launch

Use an Apple Silicon Mac with macOS 26 or later. Download the app through the [Mac App Store](https://apps.apple.com/in/app/cantis/id6764599986?mt=12), or follow the [source-build instructions](DEVELOPMENT.md). Onboarding downloads the Turbo model from Hugging Face and loads it locally. Leave roughly 6.6 GB for the model files, with extra space for temporary downloads and your tracks.

After setup, existing models can generate offline. Adding or redownloading a model needs internet access. Cantis has no cloud generation account or generation server.

## Create a track

1. Select **Text → Music**.
2. Describe the arrangement and sound in the prompt, for example: “instrumental ambient piano, slow tempo, soft pads, intimate room.” Add genre, instrument, or mood tags.
3. For vocals, enter lyrics and a language hint. Section labels such as verse and chorus help organize the text; they do not guarantee a particular output.
4. Adjust duration and seed. Use Turbo's defaults to start; change sampling steps and schedule shift when exploring variations. CFG scale applies to SFT/Base, not CFG-distilled Turbo.
5. Generate, listen, and inspect the waveform and spectrum. Save useful configurations as presets and favorite successful tracks in history.

Model results vary; compare settings with a fixed seed when exploring a control.

## Work with existing audio

| Mode | Input | Purpose |
| --- | --- | --- |
| Text → Music | Prompt, optional lyrics/tags | Generate from text |
| Cover | Source and reference audio | Condition a new track on existing audio |
| Repaint | Source audio and a selected time range | Regenerate a region using the repaint mask |
| Extract | Reference audio | Generate using the extract conditioning path |

Import or drag an audio file into the audio controls and supply the inputs required by the selected mode. Imported files are read on-device. These modes do not provide a documented multi-file stem-export workflow. **Text → Music (LM hints)** is reserved and currently unimplemented.

## Models and memory

Open Settings → Models to download, load, redownload, or delete models. Turbo is the default; SFT and Base add their own DiT weights and share Turbo's text, VAE, and LM files. Keep the Turbo bundle while using those variants.

The model settings also accept compatible custom Hugging Face repositories. Arbitrary PyTorch checkpoints are not directly loadable: the app expects the native MLX artifact layout described in [architecture](ARCHITECTURE.md). XL conversion exists as developer tooling but XL is not currently selectable in the standard model list.

On Macs with 16 GB or less, low-memory mode is enabled by default. The optional 5 Hz LM adds memory use and does not make the unimplemented LM-hint mode available.

## History, playback, and export

Browse or search history, favorite tracks, and reuse saved presets. The player displays a waveform and live FFT spectrum. Export to WAV, AAC, or ALAC; AAC and ALAC use `.m4a`. WAV export copies the generated WAV and preserves its source sample rate. MP3 and FLAC are not offered as supported export formats.

Prompts, presets, generated audio, and model files stay on your Mac. Terminal builds store model files and audio under `~/Library/Application Support/Cantis/`; sandboxed app builds use Application Support inside their macOS container. Settings → Models shows the actual model directory. Exported files go to the location you choose.

## Troubleshooting

- **Download fails:** check the built-in Cantis log window, internet access, and free space, then retry. Completed files are skipped on subsequent attempts; a partially downloaded file may restart.
- **Model files cannot be found:** check the directory shown in model settings. SFT/Base rely on shared files in the Turbo bundle.
- **Generation runs out of memory:** enable low-memory mode, close other memory-heavy apps, disable the optional LM, and reduce track duration.
- **Need help:** use [support](../SUPPORT.md) and include macOS version, chip, memory, model, settings, and reproduction steps. Review logs and screenshots before sharing personal prompts or file paths.

See [development](DEVELOPMENT.md) for build/toolchain errors and [privacy](../PRIVACY.md) for data-handling details.
