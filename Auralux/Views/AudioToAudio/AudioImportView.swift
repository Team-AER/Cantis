import SwiftUI

struct AudioImportView: View {
    @State private var selectedFileURL: URL?

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Audio to Audio")
                .font(.title2.weight(.semibold))
            Text("Import a reference clip to drive style transfer and remix generation.")
                .foregroundStyle(.secondary)

            AudioDropZone(selectedFileURL: $selectedFileURL)

            if let selectedFileURL {
                Label(selectedFileURL.lastPathComponent, systemImage: "waveform")
            }

            Spacer()
        }
        .padding(20)
    }
}
