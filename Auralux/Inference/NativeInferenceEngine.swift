import Foundation
import MLX
import MLXRandom
import Observation

// MARK: - Model State

enum ModelState: Equatable {
    case notDownloaded
    case downloading(progress: Double)
    case loading
    case ready
    case error(String)

    var isReady: Bool {
        if case .ready = self { return true }
        return false
    }

    var isLoading: Bool {
        switch self {
        case .loading, .downloading: return true
        default: return false
        }
    }
}

// MARK: - Generation Progress

enum GenerationProgress: Sendable {
    case preparing(message: String)
    case step(current: Int, total: Int)
    case saving
    case completed(audioURL: URL)
}

// MARK: - Errors

enum NativeEngineError: Error, LocalizedError {
    case modelsNotLoaded
    case weightsNotFound(URL)
    case generationFailed(String)

    var errorDescription: String? {
        switch self {
        case .modelsNotLoaded:
            return "Models are not loaded. Convert weights first using tools/convert_weights.py."
        case .weightsNotFound(let url):
            return "Model weights not found at \(url.path). Run tools/convert_weights.py first."
        case .generationFailed(let message):
            return "Generation failed: \(message)"
        }
    }
}

// MARK: - Engine

@MainActor
@Observable
final class NativeInferenceEngine {

    private(set) var modelState: ModelState = .notDownloaded
    private(set) var isGenerating: Bool = false
    var isOnboarding: Bool = false

    private var dit: ACEStepDiT?
    private var lm: ACEStepLMModel?
    private var vae: DCHiFiGANDecoder?
    private var tokenizer: BPETokenizer?
    private var generationTask: Task<Void, Never>?
    private var activeContinuation: AsyncThrowingStream<GenerationProgress, Error>.Continuation?
    private let log = AppLogger.shared

    // MARK: - Paths

    var mlxModelDirectory: URL {
        FileUtilities.modelDirectory.appendingPathComponent("ace-step-v1.5-mlx", isDirectory: true)
    }

    var weightsExist: Bool {
        let path = mlxModelDirectory
            .appendingPathComponent("dit")
            .appendingPathComponent("dit_weights.safetensors")
        return FileManager.default.fileExists(atPath: path.path)
    }

    // MARK: - App Lifecycle

    func checkStatus() async {
        if weightsExist {
            guard case .notDownloaded = modelState else { return }
            await loadModels()
        } else {
            modelState = .notDownloaded
            isOnboarding = true
        }
    }

    func shutdown() {
        cancelGeneration()
        dit = nil
        lm = nil
        vae = nil
        tokenizer = nil
    }

    // MARK: - Model Download + Load

    /// Downloads all weights from HuggingFace then loads models into memory.
    /// Throws on download failure; loading errors are reflected in `modelState`.
    func downloadAndLoad() async throws {
        guard !isGenerating else { return }
        modelState = .downloading(progress: 0)
        log.info("Downloading model weights from HuggingFace", category: .inference)

        try await ModelDownloader.shared.downloadAll(to: mlxModelDirectory) { [weak self] progress in
            Task { @MainActor [weak self] in
                self?.modelState = .downloading(progress: progress)
            }
        }

        log.info("Download complete — loading models", category: .inference)
        await loadModels()
    }

    // MARK: - Model Loading

    func loadModels() async {
        guard !isGenerating else { return }
        modelState = .loading
        log.info("Loading MLX models from \(mlxModelDirectory.path)", category: .inference)

        let baseDir = mlxModelDirectory

        do {
            let models = try await Task<LoadedModels, Error>.detached(priority: .userInitiated) {
                let dit = ACEStepDiT()
                try DiTWeightLoader.load(baseDir: baseDir, into: dit)
                let lm = ACEStepLMModel()
                try LMWeightLoader.load(baseDir: baseDir, into: lm)
                let vae = DCHiFiGANDecoder()
                try? VAEWeightLoader.load(baseDir: baseDir, into: vae)
                let tokenizer = try? BPETokenizer(
                    vocabURL: baseDir.appendingPathComponent("lm/lm_vocab.json"),
                    mergesURL: baseDir.appendingPathComponent("lm/lm_merges.txt")
                )
                return LoadedModels(dit: dit, lm: lm, vae: vae, tokenizer: tokenizer)
            }.value

            self.dit = models.dit
            self.lm = models.lm
            self.vae = models.vae
            self.tokenizer = models.tokenizer
            modelState = .ready
            log.info("MLX models loaded successfully", category: .inference)
        } catch {
            modelState = .error(error.localizedDescription)
            log.error("Failed to load MLX models: \(error.localizedDescription)", category: .inference)
        }
    }

    // MARK: - Generation

    func generate(request: GenerationParameters) -> AsyncThrowingStream<GenerationProgress, Error> {
        let (stream, continuation) = AsyncThrowingStream<GenerationProgress, Error>.makeStream()

        guard case .ready = modelState,
              let dit = dit, let lm = lm, let vae = vae else {
            continuation.finish(throwing: NativeEngineError.modelsNotLoaded)
            return stream
        }

        generationTask?.cancel()
        activeContinuation?.finish(throwing: CancellationError())
        activeContinuation = continuation
        isGenerating = true

        let localCont      = continuation
        let localDit       = dit
        let localLm        = lm
        let localVae       = vae
        let localTokenizer = tokenizer
        let generatedDir = FileUtilities.generatedAudioDirectory
        let duration     = request.duration

        generationTask = Task.detached(priority: .userInitiated) {
            do {
                try Task.checkCancellation()
                localCont.yield(.preparing(message: "Building latent noise..."))

                // 25 Hz latent frame rate, 64-dim acoustic latent
                let T = max(1, Int(duration * 25.0))
                let acousticDim = localDit.config.audioAcousticHiddenDim
                let contextDim  = localDit.config.inChannels - acousticDim

                let noise          = MLXRandom.normal([1, T, acousticDim])
                let contextLatents = MLXArray.zeros([1, T, contextDim])

                // Text conditioning: tokenize prompt+tags, run through LM + lyricEncoder
                let encH: MLXArray
                if let tok = localTokenizer, !request.prompt.isEmpty || !request.tags.isEmpty {
                    let condText = ([request.prompt] + request.tags + [request.lyrics])
                        .filter { !$0.isEmpty }
                        .joined(separator: "\n")
                    let tokenIds = tok.encode(condText)
                    if !tokenIds.isEmpty {
                        let inputIds  = MLXArray(tokenIds).reshaped([1, tokenIds.count])
                        let lmHidden  = localLm.hiddenStates(inputIds)   // [1, S, 1024]
                        eval(lmHidden)
                        encH = localDit.lyricEncoder(lmHidden)           // [1, S, 2048]
                        eval(encH)
                    } else {
                        encH = localDit.nullConditionEmb
                    }
                } else {
                    encH = localDit.nullConditionEmb
                }

                try Task.checkCancellation()

                let sampler = TurboSampler()
                let result  = sampler.sample(
                    noise:               noise,
                    contextLatents:      contextLatents,
                    encoderHiddenStates: encH,
                    model:               localDit.decoder
                ) { step, total in
                    localCont.yield(.step(current: step + 1, total: total))
                }

                try Task.checkCancellation()
                localCont.yield(.saving)

                let audio = localVae.decode(latent: result)
                eval(audio)

                let filename = "generated-\(UUID().uuidString).wav"
                let outputURL = generatedDir.appendingPathComponent(filename)
                try NativeInferenceEngine.writeWAV(samples: audio, to: outputURL, sampleRate: 48000)

                try Task.checkCancellation()
                localCont.yield(.completed(audioURL: outputURL))
                localCont.finish()
            } catch {
                localCont.finish(throwing: error)
            }
        }

        continuation.onTermination = { @Sendable _ in
            Task { @MainActor [weak self] in
                self?.isGenerating = false
                self?.activeContinuation = nil
            }
        }

        return stream
    }

    func cancelGeneration() {
        generationTask?.cancel()
        generationTask = nil
        activeContinuation?.finish(throwing: CancellationError())
        activeContinuation = nil
        isGenerating = false
    }

    // MARK: - WAV Export

    private nonisolated static func writeWAV(samples: MLXArray, to url: URL, sampleRate: Int) throws {
        let flat = samples.flattened()
        eval(flat)
        let floats = flat.asArray(Float.self)
        let dataSize = UInt32(floats.count * 2)

        var data = Data()

        func le32(_ v: UInt32) {
            var x = v.littleEndian
            withUnsafeBytes(of: &x) { data.append(contentsOf: $0) }
        }
        func le16(_ v: UInt16) {
            var x = v.littleEndian
            withUnsafeBytes(of: &x) { data.append(contentsOf: $0) }
        }

        data.append(contentsOf: "RIFF".utf8)
        le32(36 + dataSize)
        data.append(contentsOf: "WAVE".utf8)
        data.append(contentsOf: "fmt ".utf8)
        le32(16); le16(1); le16(1)                   // chunkSize, PCM, mono
        le32(UInt32(sampleRate))
        le32(UInt32(sampleRate * 2))                 // byteRate
        le16(2); le16(16)                            // blockAlign, bitsPerSample
        data.append(contentsOf: "data".utf8)
        le32(dataSize)

        for s in floats {
            let clamped = max(-1.0, min(1.0, s))
            le16(UInt16(bitPattern: Int16(clamped * 32767)))
        }

        try data.write(to: url, options: .atomic)
    }
}

// MARK: - Private Helpers

private struct LoadedModels: @unchecked Sendable {
    let dit: ACEStepDiT
    let lm: ACEStepLMModel
    let vae: DCHiFiGANDecoder
    let tokenizer: BPETokenizer?
}
