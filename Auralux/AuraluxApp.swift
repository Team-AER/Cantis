import SwiftData
import SwiftUI

@main
struct AuraluxApp: App {
    @State private var sidebarViewModel = SidebarViewModel()
    @State private var generationViewModel: GenerationViewModel
    @State private var historyViewModel = HistoryViewModel()
    @State private var playerViewModel = PlayerViewModel()
    @State private var settingsViewModel = SettingsViewModel()
    @State private var engineService: EngineService

    private let modelContainer: ModelContainer

    init() {
        let inferenceService = InferenceService()
        _generationViewModel = State(initialValue: GenerationViewModel(inferenceService: inferenceService))
        _engineService = State(initialValue: EngineService(inferenceService: inferenceService))

        do {
            modelContainer = try ModelContainer(
                for: GeneratedTrack.self,
                Preset.self,
                Tag.self
            )
        } catch {
            fatalError("Failed to initialize SwiftData container: \(error)")
        }
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
                .frame(minWidth: AppConstants.minimumWindowWidth, minHeight: AppConstants.minimumWindowHeight)
                .environment(sidebarViewModel)
                .environment(generationViewModel)
                .environment(historyViewModel)
                .environment(playerViewModel)
                .environment(settingsViewModel)
                .environment(engineService)
                .onDisappear {
                    engineService.shutdown()
                }
        }
        .modelContainer(modelContainer)
        .defaultSize(width: 1280, height: 840)
    }
}
