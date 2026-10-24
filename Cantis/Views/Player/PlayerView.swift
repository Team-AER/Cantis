import AVFoundation
import AppKit
import SwiftUI
import UniformTypeIdentifiers

struct PlayerView: View {
    let track: GeneratedTrack

    @Environment(PlayerViewModel.self) private var viewModel
    @Environment(SettingsViewModel.self) private var settings
    @Environment(GenerationViewModel.self) private var generationViewModel

    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    @State private var wideControls = false
    @State private var scoreReveal: CGFloat = 1
    @State private var motionEpoch = 0
    @State private var exportError: String? = nil

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                Label("THE TAKE", systemImage: "record.circle")
                    .font(.caption2.weight(.semibold)).tracking(2).foregroundStyle(.mint)
                VStack(alignment: .leading, spacing: 6) {
                    Text(track.title)
                        .font(.title2.weight(.semibold))
                        .accessibilityIdentifier("track-title")
                    Text(track.createdAt.formatted(date: .abbreviated, time: .shortened))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                if let error = viewModel.errorMessage ?? exportError {
                    Label(error, systemImage: "exclamationmark.triangle")
                        .foregroundStyle(.red)
                        .font(.callout)
                }

                if generationViewModel.state.isBusy {
                    ProgressView(value: generationViewModel.progress)
                        .progressViewStyle(.linear)
                        .accessibilityIdentifier("generation-progress")
                }

                HStack(spacing: 16) {
                    Label("\(Int(track.duration)) sec", systemImage: "clock")
                    Text(track.format.uppercased())
                    if let seed = track.seed { Text("Seed \(seed)") }
                }
                .font(.caption.monospaced()).foregroundStyle(.white.opacity(0.65))

                WaveformView(
                    progress: viewModel.progress,
                    samples: viewModel.waveformSamples,
                    onSeek: { fraction in viewModel.seek(to: fraction) }
                )
                .frame(height: 120)
                .mask(alignment: .leading) {
                    Rectangle().scaleEffect(x: scoreReveal, anchor: .leading).id(motionEpoch)
                }
                .onChange(of: viewModel.waveformSamples.isEmpty) { _, empty in
                    guard !empty else { return }
                    Task { @MainActor in
                        scoreReveal = reduceMotion ? 1 : 0
                        await Task.yield()
                        withAnimation(reduceMotion ? nil : .timingCurve(0.2, 0.8, 0.2, 1, duration: 0.55)) {
                            scoreReveal = 1
                        }
                    }
                }
                .onChange(of: reduceMotion) { _, reduced in
                    if reduced {
                        motionEpoch += 1
                        scoreReveal = 1
                    }
                }

                SpectrumAnalyzerView(
                    magnitudes: viewModel.spectrumBins,
                    isPlaying: viewModel.isPlaying
                )
                .frame(height: 90)

                ViewThatFits(in: .horizontal) {
                    playbackControls(horizontal: true)
                    playbackControls(horizontal: false)
                }

                if !track.prompt.isEmpty {
                    Divider().overlay(.white.opacity(0.15))
                    Text(track.prompt).font(.callout).foregroundStyle(.white.opacity(0.7))
                        .textSelection(.enabled)
                }
                Spacer(minLength: 0)
            }
            .padding(24)
        }
        .onGeometryChange(for: Bool.self) { geometry in
            geometry.size.width >= 560
        } action: { wide in
            wideControls = wide
        }
        .foregroundStyle(.white)
        .background(ArrangementStageView.background)
        .preferredColorScheme(.dark)
        .tint(.mint)
        .task(id: track.id) {
            try? await Task.sleep(for: .milliseconds(200))
            guard !Task.isCancelled else { return }
            viewModel.load(path: track.audioFilePath)
        }
    }

    @ViewBuilder
    private func playbackControls(horizontal: Bool) -> some View {
        let layout =
            horizontal
            ? AnyLayout(HStackLayout(spacing: 12))
            : AnyLayout(VStackLayout(alignment: .leading, spacing: 14))
        layout {
            Button {
                viewModel.playPause()
            } label: {
                Image(systemName: viewModel.isPlaying ? "pause.fill" : "play.fill")
                    .frame(width: 16, height: 16)
            }
            .keyboardShortcut(.space, modifiers: [])
            .accessibilityLabel(viewModel.isPlaying ? "Pause" : "Play")
            .accessibilityIdentifier("play-pause-button")
            .help(viewModel.isPlaying ? "Pause" : "Play")
            .disabled(viewModel.errorMessage != nil)

            Button {
                exportAudio()
            } label: {
                Image(systemName: "square.and.arrow.up")
                    .frame(width: 16, height: 16)
            }
            .disabled(track.audioFilePath == nil)
            .accessibilityLabel("Export")
            .help(
                "Export as \(settings.defaultExportFormat.rawValue.uppercased()) (set the format in Settings)"
            )

            @Bindable var vm = viewModel
            Toggle("Loop", isOn: $vm.isLooping)
                .toggleStyle(.switch)
                .frame(width: 110)

            HStack(spacing: 6) {
                Image(systemName: volumeIconName)
                    .foregroundStyle(.secondary)
                    .frame(width: 16)
                Slider(value: $vm.volume, in: 0...1)
                    .frame(width: 100)
                    .accessibilityLabel("Volume")
                    .accessibilityIdentifier("volume-slider")
            }

            Spacer()

            Text(timecode)
                .monospacedDigit()
                .foregroundStyle(.secondary)
        }
    }

    private func exportAudio() {
        guard let path = track.audioFilePath,
            let sourceURL = FileUtilities.resolveAudioPath(path)
        else { return }

        let format = settings.defaultExportFormat
        let panel = NSSavePanel()
        panel.nameFieldStringValue = "\(track.title).\(format.fileExtension)"
        panel.canCreateDirectories = true
        if let utType = UTType(filenameExtension: format.fileExtension) {
            panel.allowedContentTypes = [utType]
        } else {
            panel.allowedContentTypes = [.audio]
        }

        guard panel.runModal() == .OK, let destinationURL = panel.url else { return }

        let service = AudioExportService()
        Task.detached(priority: .userInitiated) {
            do {
                _ = try await service.export(
                    sourceURL: sourceURL,
                    destinationURL: destinationURL,
                    format: format
                )
                await MainActor.run { self.exportError = nil }
            } catch {
                await MainActor.run { self.exportError = "Export failed: \(error.localizedDescription)" }
            }
        }
    }

    private var volumeIconName: String {
        switch viewModel.volume {
        case ..<0.001: "speaker.slash.fill"
        case ..<0.34: "speaker.wave.1.fill"
        case ..<0.67: "speaker.wave.2.fill"
        default: "speaker.wave.3.fill"
        }
    }

    private var timecode: String {
        let current = Int(viewModel.currentTime)
        let total = Int(viewModel.duration)
        return "\(format(seconds: current)) / \(format(seconds: total))"
    }

    private func format(seconds: Int) -> String {
        String(format: "%02d:%02d", max(0, seconds / 60), max(0, seconds % 60))
    }
}
