import MLX
import MLXNN
import Foundation

/// Loads converted VAE decoder weights from `<baseDir>/vae/vae_weights.safetensors`.
///
/// Silently no-ops when the file is absent so the app degrades gracefully
/// (DCHiFiGANDecoder returns silence) until weights are downloaded.
enum VAEWeightLoader {

    static func load(baseDir: URL, into model: DCHiFiGANDecoder) throws {
        let url = baseDir
            .appendingPathComponent("vae")
            .appendingPathComponent("vae_weights.safetensors")

        guard FileManager.default.fileExists(atPath: url.path) else { return }

        let flat   = try loadArrays(url: url)
        let nested = ModuleParameters.unflattened(flat)
        try model.update(parameters: nested, verify: .none)
        eval(model.parameters())
    }
}
