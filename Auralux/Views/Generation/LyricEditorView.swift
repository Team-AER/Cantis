import SwiftUI

struct LyricEditorView: View {
    @Binding var lyrics: String

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
        }
    }
}
