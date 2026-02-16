import SwiftUI

struct LoRAManagerView: View {
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("LoRA Manager")
                .font(.title3.weight(.semibold))
            Text("Import and tune LoRA adapters in a future phase.")
                .foregroundStyle(.secondary)
            Spacer()
        }
        .padding(20)
    }
}
