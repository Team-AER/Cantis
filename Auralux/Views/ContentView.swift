import SwiftData
import SwiftUI

struct ContentView: View {
    @Environment(SidebarViewModel.self) private var sidebarViewModel
    @Environment(HistoryViewModel.self) private var historyViewModel
    @Environment(GenerationViewModel.self) private var generationViewModel
    @Environment(EngineService.self) private var engineService
    @Environment(\.modelContext) private var modelContext
    @Environment(\.openWindow) private var openWindow
    @State private var didBootstrap = false

    var body: some View {
        ZStack {
            mainContent

            if engineService.isOnboarding {
                Color.black.opacity(0.3)
                    .ignoresSafeArea()
                    .transition(.opacity)

                SetupView()
                    .transition(.opacity.combined(with: .scale(scale: 0.95)))
            }
        }
        .task {
            guard !didBootstrap else { return }
            didBootstrap = true

            let presetService = PresetService(context: modelContext)
            try? presetService.bootstrapFromBundleIfNeeded()
            try? await HistoryService(context: modelContext).reconcileOrphans()
            historyViewModel.refresh(context: modelContext)

            // Check engine status on launch
            await engineService.checkStatus()
        }
    }

    @ViewBuilder
    private var playerPanel: some View {
        if let selectedTrack = historyViewModel.selectedTrack ?? generationViewModel.lastTrack {
            PlayerView(track: selectedTrack)
        } else {
            ContentUnavailableView("No Track Selected", systemImage: "music.note", description: Text("Generate or select a track to preview it."))
        }
    }

    private var mainContent: some View {
        NavigationSplitView {
            SidebarView()
        } detail: {
            HStack(spacing: 0) {
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
                .frame(maxWidth: .infinity)

                if sidebarViewModel.selectedSection != .settings {
                    Divider()
                    playerPanel
                        .frame(width: 320)
                }
            }
        }
        .toolbar {
            ToolbarItem(placement: .automatic) {
                Button {
                    openWindow(id: "log-viewer")
                } label: {
                    Image(systemName: "terminal")
                }
                .help("Show Logs (Cmd+Opt+L)")
                .keyboardShortcut("l", modifiers: [.command, .option])
            }

            ToolbarSpacer(.fixed)

            ToolbarItem(placement: .automatic) {
                EngineStatusView()
            }
        }
    }
}
