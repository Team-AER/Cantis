import SwiftUI

struct SliderControl: View {
    let label: String
    @Binding var value: Double
    let range: ClosedRange<Double>
    let unit: String
    var warningThreshold: Double? = nil
    var warningMessage: String? = nil

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(label)
                    .frame(width: 100, alignment: .leading)
                Slider(value: $value, in: range)
                    .tint(isWarning ? .red : nil)
                Text(verbatim: valueLabel)
                    .font(.caption.monospacedDigit())
                    .frame(width: 70, alignment: .trailing)
                    .foregroundStyle(isWarning ? .red : .secondary)
            }
            if isWarning, let warningMessage {
                Text(warningMessage)
                    .font(.caption2)
                    .foregroundStyle(.red)
                    .padding(.leading, 100)
                    .transition(.opacity)
            }
        }
        .animation(.easeInOut(duration: 0.15), value: isWarning)
    }

    private var isWarning: Bool {
        guard let warningThreshold else { return false }
        return value > warningThreshold
    }

    private var valueLabel: String {
        if unit == "" {
            return String(format: "%.2f", value)
        }
        return "\(Int(value)) \(unit)"
    }
}
