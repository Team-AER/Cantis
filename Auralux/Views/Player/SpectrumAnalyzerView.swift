import SwiftUI

struct SpectrumAnalyzerView: View {
    @State private var seed: Double = 0

    var body: some View {
        TimelineView(.animation) { timeline in
            let t = timeline.date.timeIntervalSinceReferenceDate
            Canvas { context, size in
                let bars = 36
                let barWidth = size.width / CGFloat(bars)

                for index in 0..<bars {
                    let x = CGFloat(index) * barWidth
                    let wave = abs(sin((t + seed) * 2.4 + Double(index) * 0.5))
                    let height = max(6, wave * size.height)
                    let rect = CGRect(x: x + 1, y: size.height - height, width: barWidth - 2, height: height)
                    context.fill(Path(roundedRect: rect, cornerRadius: 2), with: .color(.mint.opacity(0.7)))
                }
            }
            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 10))
        }
        .onAppear {
            seed = .random(in: 0...10)
        }
    }
}
