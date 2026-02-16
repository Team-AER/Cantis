import SwiftUI

struct ModelSettingsView: View {
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Models")
                .font(.headline)
            Text("Models are stored in \(FileUtilities.modelDirectory.path).")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }
}
