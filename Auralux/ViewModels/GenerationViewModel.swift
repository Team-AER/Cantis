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
    var prompt = ""
    var lyrics = ""
    var tags: [String] = []
    var duration: Double = 30
    var variance: Double = 0.5
    var seedText = ""

    var state: GenerationState = .idle
    var progress: Double = 0
    var currentJobID: String?
    var lastTrack: GeneratedTrack?

    private let inferenceService: InferenceService
    private var generationTask: Task<Void, Never>?

    init(inferenceService: InferenceService) {
        self.inferenceService = inferenceService
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

        let request = GenerationRequest(
            prompt: prompt,
            lyrics: lyrics,
            tags: tags,
            duration: duration,
            variance: variance,
            seed: Int(seedText)
        )

        generationTask = Task {
            do {
                let submission = try await inferenceService.generate(request)
                currentJobID = submission.jobID
                state = .generating
                try await pollUntilComplete(jobID: submission.jobID, request: request, context: context)
                state = .completed
            } catch is CancellationError {
                state = .idle
            } catch {
                state = .failed(error.localizedDescription)
            }
        }
    }

    func cancel() {
        generationTask?.cancel()
        if let currentJobID {
            Task {
                try? await inferenceService.cancel(jobID: currentJobID)
            }
        }
        state = .idle
        progress = 0
        currentJobID = nil
    }

    private func pollUntilComplete(jobID: String, request: GenerationRequest, context: ModelContext) async throws {
        let historyService = HistoryService(context: context)

        while !Task.isCancelled {
            let status = try await inferenceService.poll(jobID: jobID)
            progress = status.progress

            if status.status == "failed" {
                throw NSError(domain: "Auralux", code: -1, userInfo: [NSLocalizedDescriptionKey: status.message ?? "Generation failed"])
            }

            if status.status == "completed" {
                let track = GeneratedTrack(
                    title: request.prompt,
                    prompt: request.prompt,
                    lyrics: request.lyrics,
                    tags: request.tags,
                    duration: request.duration,
                    variance: request.variance,
                    seed: request.seed,
                    generationID: jobID,
                    audioFilePath: status.audioPath
                )
                try historyService.insert(track)
                lastTrack = track
                currentJobID = nil
                return
            }

            try await Task.sleep(for: .milliseconds(500))
        }

        throw CancellationError()
    }
}
