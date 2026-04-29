import SwiftUI

struct LyricEditorView: View {
    @Binding var lyrics: String
    @Environment(SettingsViewModel.self) private var settings

    var body: some View {
        GroupBox("Lyrics") {
            TextEditor(text: $lyrics)
                .font(.system(.body, design: .monospaced))
                .frame(minHeight: 180)
                .overlay(alignment: .topLeading) {
                    if lyrics.isEmpty {
                        Text("[verse]\nWrite your lyrics here...")
                            .foregroundStyle(.secondary)
                            .padding(8)
                            .allowsHitTesting(false)
                    }
                }

            if settings.ditVariant.usesCFGDistillation && !lyrics.isEmpty {
                Label {
                    Text("Turbo models may skip later verses — switch to SFT for closer lyric following.")
                        .font(.caption.weight(.medium))
                        .foregroundStyle(.primary)
                } icon: {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundStyle(.orange)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(8)
                .background(
                    RoundedRectangle(cornerRadius: 6, style: .continuous)
                        .fill(Color.orange.opacity(0.15))
                )
                .overlay(
                    RoundedRectangle(cornerRadius: 6, style: .continuous)
                        .strokeBorder(Color.orange.opacity(0.5), lineWidth: 1)
                )
                .padding(.top, 4)
            }
        }
    }
}
