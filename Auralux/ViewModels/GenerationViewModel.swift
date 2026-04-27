import Foundation
import Observation
import SwiftData

enum GenerationState: Equatable {
    case idle
    case preparing
    case generating
    case completed
    case failed(String)

    var isBusy: Bool {
        switch self {
        case .preparing, .generating:
            return true
        case .idle, .completed, .failed:
            return false
        }
    }
}

@MainActor
@Observable
final class GenerationViewModel {
    var prompt = GenerationParameters.default.prompt
    var lyrics = GenerationParameters.default.lyrics
    var tags: [String] = GenerationParameters.default.tags
    var duration: Double = GenerationParameters.default.duration
    var variance: Double = GenerationParameters.default.variance
    var seedText = ""

    var state: GenerationState = .idle
    var progress: Double = 0
    var progressMessage: String = ""
    var currentJobID: String? = nil
    var lastTrack: GeneratedTrack?

    private let engine: NativeInferenceEngine
    private let log = AppLogger.shared
    private var generationTask: Task<Void, Never>?

    init(engine: NativeInferenceEngine) {
        self.engine = engine
    }

    func applyPreset(_ preset: Preset) {
        prompt = preset.prompt
        lyrics = preset.lyricTemplate
        tags = preset.tags
        duration = preset.duration
        variance = preset.variance
    }

    func addTag(_ tag: String) {
        let normalized = tag.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !normalized.isEmpty else { return }
        guard !tags.contains(normalized) else { return }
        tags.append(normalized)
    }

    func removeTag(_ tag: String) {
        tags.removeAll { $0 == tag }
    }

    func generate(in context: ModelContext) {
        guard !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            state = .failed("Prompt is required.")
            return
        }

        generationTask?.cancel()
        state = .preparing
        progress = 0
        progressMessage = "Preparing..."
        currentJobID = nil

        let request = GenerationParameters(
            prompt: prompt,
            lyrics: lyrics,
            tags: tags,
            duration: duration,
            variance: variance,
            seed: Int(seedText)
        )

        log.info("Starting generation: \"\(prompt.prefix(60))\" duration=\(duration)s", category: .generation)

        let stream = engine.generate(request: request)

        generationTask = Task {
            defer { generationTask = nil }
            do {
                let historyService = HistoryService(context: context)
                for try await event in stream {
                    switch event {
                    case .preparing(let message):
                        state = .preparing
                        progressMessage = message
                    case .step(let current, let total):
                        state = .generating
                        progress = Double(current) / Double(total)
                        progressMessage = "Step \(current) / \(total)"
                    case .saving:
                        progressMessage = "Saving audio..."
                    case .completed(let audioURL):
                        let track = GeneratedTrack(
                            title: request.prompt,
                            prompt: request.prompt,
                            lyrics: request.lyrics,
                            tags: request.tags,
                            duration: request.duration,
                            variance: request.variance,
                            seed: request.seed,
                            generationID: UUID().uuidString,
                            audioFilePath: FileUtilities.relativeAudioPath(from: audioURL.path)
                        )
                        try historyService.insert(track)
                        await Task.yield()
                        lastTrack = track
                    }
                }
                state = .completed
                log.info("Generation completed successfully", category: .generation)
            } catch is CancellationError {
                state = .idle
                log.info("Generation cancelled", category: .generation)
            } catch {
                state = .failed(error.localizedDescription)
                log.error("Generation failed: \(error.localizedDescription)", category: .generation)
            }
        }
    }

    func cancel() {
        generationTask?.cancel()
        engine.cancelGeneration()
        log.info("Generation cancelled by user", category: .generation)
        state = .idle
        progress = 0
        progressMessage = ""
        currentJobID = nil
    }
}
