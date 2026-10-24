import SwiftUI

/// The output destination uses real generation state; the idle score is a
/// static musical motif, never a fabricated waveform or progress indicator.
struct ArrangementStageView: View {
    @Environment(GenerationViewModel.self) private var generation
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    static let background = LinearGradient(
        colors: [
            Color(red: 0.065, green: 0.16, blue: 0.15), Color(red: 0.035, green: 0.07, blue: 0.075),
        ],
        startPoint: .topLeading, endPoint: .bottomTrailing
    )

    var body: some View {
        VStack(alignment: .leading, spacing: 24) {
            Label("THE TAKE", systemImage: "record.circle")
                .font(.caption2.weight(.semibold)).tracking(2).foregroundStyle(.mint)
            Spacer(minLength: 12)
            ZStack {
                Canvas { context, size in
                    for line in 0..<5 {
                        let y = size.height / 2 + CGFloat(line - 2) * 14
                        var path = Path()
                        path.move(to: CGPoint(x: 0, y: y))
                        path.addLine(to: CGPoint(x: size.width, y: y))
                        context.stroke(path, with: .color(.mint.opacity(0.12)), lineWidth: 1)
                    }
                }
                Image(systemName: generation.state.isBusy ? "waveform.path" : "music.quarternote.3")
                    .font(.system(size: 58, weight: .ultraLight)).foregroundStyle(.mint)
                    .symbolEffect(.pulse, isActive: generation.state.isBusy && !reduceMotion)
                    .accessibilityHidden(true)
            }
            .frame(height: 140)
            VStack(alignment: .leading, spacing: 9) {
                Text(generation.state.isBusy ? "Taking shape." : "Your next take.")
                    .font(.system(.largeTitle, design: .rounded, weight: .medium))
                if generation.state.isBusy {
                    Text(generation.progressMessage).font(.callout).foregroundStyle(.white.opacity(0.7))
                    if generation.state == .generating {
                        ProgressView(value: generation.progress).tint(.mint)
                            .accessibilityLabel("Generation progress")
                    } else {
                        ProgressView().controlSize(.small)
                    }
                } else {
                    Text("Generate a track, or choose one from History.")
                        .font(.callout).foregroundStyle(.white.opacity(0.65))
                }
            }
            Spacer(minLength: 12)
            HStack {
                Text(generation.mode.displayName)
                Spacer()
                Text("\(Int(generation.duration)) sec").monospacedDigit()
            }
            .font(.caption).foregroundStyle(.white.opacity(0.55))
        }
        .padding(28)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .leading)
        .foregroundStyle(.white)
        .background(Self.background)
        .preferredColorScheme(.dark)
    }
}
