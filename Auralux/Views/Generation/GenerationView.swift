import SwiftUI

struct GenerationView: View {
    @Environment(GenerationViewModel.self) private var viewModel
    @Environment(\.modelContext) private var modelContext
    @State private var tagText = ""

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                GroupBox("Prompt") {
                    TextEditor(text: Bindable(viewModel).prompt)
                        .font(.body)
                        .frame(minHeight: 100)
                        .accessibilityLabel("Music prompt")
                }

                TagEditorView(tags: viewModel.tags, draftTag: $tagText) { tag in
                    viewModel.addTag(tag)
                } onRemoveTag: { tag in
                    viewModel.removeTag(tag)
                }

                LyricEditorView(lyrics: Bindable(viewModel).lyrics)
                ParameterControlsView(duration: Bindable(viewModel).duration, variance: Bindable(viewModel).variance, seedText: Bindable(viewModel).seedText)

                HStack {
                    Button("Generate") {
                        viewModel.generate(in: modelContext)
                    }
                    .keyboardShortcut("g", modifiers: [.command])
                    .disabled(viewModel.state.isBusy)

                    Button("Cancel") {
                        viewModel.cancel()
                    }
                    .disabled(!viewModel.state.isBusy)

                    Spacer()

                    switch viewModel.state {
                    case .generating, .preparing:
                        ProgressView(value: viewModel.progress)
                            .frame(width: 160)
                    case .completed:
                        Label("Done", systemImage: "checkmark.circle.fill")
                            .foregroundStyle(.green)
                    case .failed(let message):
                        Label(message, systemImage: "exclamationmark.triangle.fill")
                            .foregroundStyle(.red)
                    case .idle:
                        EmptyView()
                    }
                }
            }
            .padding(20)
        }
    }
}
