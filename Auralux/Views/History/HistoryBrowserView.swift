import SwiftData
import SwiftUI

struct HistoryBrowserView: View {
    @Environment(HistoryViewModel.self) private var viewModel
    @Environment(\.modelContext) private var modelContext

    var body: some View {
        VStack(spacing: 12) {
            HStack {
                TextField("Search history", text: Bindable(viewModel).query)
                    .textFieldStyle(.roundedBorder)
                    .onSubmit {
                        viewModel.refresh(context: modelContext)
                    }

                Button("Refresh") {
                    viewModel.refresh(context: modelContext)
                }
            }

            if viewModel.tracks.isEmpty {
                ContentUnavailableView("No Generations Yet", systemImage: "music.note.list", description: Text("Generated tracks will appear here."))
            } else {
                List(viewModel.tracks, selection: Bindable(viewModel).selectedTrack) { track in
                    HistoryItemView(track: track)
                        .tag(track)
                        .contextMenu {
                            Button(track.isFavorite ? "Remove Favorite" : "Favorite") {
                                viewModel.toggleFavorite(track, context: modelContext)
                            }
                        }
                }
                .listStyle(.inset)
            }
        }
        .padding(20)
        .onAppear {
            viewModel.refresh(context: modelContext)
        }
        .onChange(of: viewModel.query) { _, newValue in
            if newValue.isEmpty {
                viewModel.refresh(context: modelContext)
            }
        }
    }
}
