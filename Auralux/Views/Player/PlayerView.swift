import SwiftUI

struct PlayerView: View {
    let track: GeneratedTrack

    @Environment(PlayerViewModel.self) private var viewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            VStack(alignment: .leading, spacing: 6) {
                Text(track.title)
                    .font(.title2.weight(.semibold))
                Text(track.createdAt.formatted(date: .abbreviated, time: .shortened))
                    .foregroundStyle(.secondary)
            }

            WaveformView(progress: progress)
                .frame(height: 120)

            SpectrumAnalyzerView()
                .frame(height: 90)

            HStack(spacing: 12) {
                Button(viewModel.playerService.isPlaying ? "Pause" : "Play") {
                    viewModel.playPause()
                }
                .keyboardShortcut(.space, modifiers: [])

                Button("Stop") {
                    viewModel.stop()
                }

                Toggle("Loop", isOn: Bindable(viewModel.playerService).isLooping)
                    .toggleStyle(.switch)
                    .frame(width: 110)

                Spacer()

                Text(timecode)
                    .monospacedDigit()
                    .foregroundStyle(.secondary)
            }

            Spacer(minLength: 0)
        }
        .padding(20)
        .onAppear {
            viewModel.load(path: track.audioFilePath)
        }
        .onChange(of: track.id) { _, _ in
            viewModel.load(path: track.audioFilePath)
        }
    }

    private var progress: Double {
        guard viewModel.playerService.duration > 0 else { return 0 }
        return viewModel.playerService.currentTime / viewModel.playerService.duration
    }

    private var timecode: String {
        let current = Int(viewModel.playerService.currentTime)
        let total = Int(viewModel.playerService.duration)
        return "\(format(seconds: current)) / \(format(seconds: total))"
    }

    private func format(seconds: Int) -> String {
        String(format: "%02d:%02d", max(0, seconds / 60), max(0, seconds % 60))
    }
}
