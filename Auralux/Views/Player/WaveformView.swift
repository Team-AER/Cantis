import SwiftUI

struct WaveformView: View {
    var progress: Double

    var body: some View {
        GeometryReader { geo in
            Canvas { context, size in
                let baseline = size.height / 2
                let width = size.width
                let barWidth: CGFloat = 3
                let spacing: CGFloat = 2
                let count = Int(width / (barWidth + spacing))
                let filled = Int(Double(count) * progress)

                for index in 0..<max(1, count) {
                    let normalized = abs(sin(Double(index) * 0.32))
                    let barHeight = max(8, normalized * (size.height * 0.9))
                    let x = CGFloat(index) * (barWidth + spacing)
                    let rect = CGRect(x: x, y: baseline - barHeight / 2, width: barWidth, height: barHeight)
                    let color: Color = index <= filled ? .accentColor : .secondary.opacity(0.3)
                    context.fill(Path(roundedRect: rect, cornerRadius: 2), with: .color(color))
                }
            }
            .clipShape(RoundedRectangle(cornerRadius: 12))
            .background(.quaternary.opacity(0.35), in: RoundedRectangle(cornerRadius: 12))
        }
    }
}
