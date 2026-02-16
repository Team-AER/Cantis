import SwiftUI

struct SliderControl: View {
    let label: String
    @Binding var value: Double
    let range: ClosedRange<Double>
    let unit: String

    var body: some View {
        HStack {
            Text(label)
                .frame(width: 100, alignment: .leading)
            Slider(value: $value, in: range)
            Text(verbatim: valueLabel)
                .font(.caption.monospacedDigit())
                .frame(width: 70, alignment: .trailing)
                .foregroundStyle(.secondary)
        }
    }

    private var valueLabel: String {
        if unit == "" {
            return String(format: "%.2f", value)
        }
        return "\(Int(value)) \(unit)"
    }
}
