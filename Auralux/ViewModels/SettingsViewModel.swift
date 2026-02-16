import Foundation
import Observation

@MainActor
@Observable
final class SettingsViewModel {
    enum QuantizationMode: String, CaseIterable, Identifiable {
        case fp16
        case int8

        var id: String { rawValue }
    }

    var quantizationMode: QuantizationMode = .fp16
    var lowMemoryMode = false
    var autoStartServer = true
    var maxConcurrentJobs = 1
    var defaultExportFormat: AudioExportFormat = .wav
}
