import SwiftData
import SwiftUI

struct ContentView: View {
    @Environment(SidebarViewModel.self) private var sidebarViewModel
    @Environment(HistoryViewModel.self) private var historyViewModel
    @Environment(GenerationViewModel.self) private var generationViewModel
    @Environment(\.modelContext) private var modelContext
    @State private var didBootstrap = false

    var body: some View {
        NavigationSplitView {
            SidebarView()
        } content: {
            Group {
                switch sidebarViewModel.selectedSection ?? .generate {
                case .generate:
                    GenerationView()
                case .history:
                    HistoryBrowserView()
                case .audioToAudio:
                    AudioImportView()
                case .settings:
                    SettingsView()
                }
            }
            .navigationTitle(sidebarViewModel.selectedSection?.title ?? "Auralux")
        } detail: {
            if let selectedTrack = historyViewModel.selectedTrack ?? generationViewModel.lastTrack {
                PlayerView(track: selectedTrack)
            } else {
                ContentUnavailableView("No Track Selected", systemImage: "music.note", description: Text("Generate or select a track to preview it."))
            }
        }
        .toolbar {
            ToolbarItem(placement: .automatic) {
                Text("Auralux")
                    .font(.headline)
            }
        }
        .task {
            guard !didBootstrap else { return }
            didBootstrap = true
            let presetService = PresetService(context: modelContext)
            try? presetService.bootstrapFromBundleIfNeeded()
            historyViewModel.refresh(context: modelContext)
        }
    }
}
