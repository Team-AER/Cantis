import CryptoKit
import Foundation

struct ModelArtifact: Identifiable, Codable, Hashable, Sendable {
    var id: String { name }
    var name: String
    var downloadURL: URL
    var sha256: String
    var sizeBytes: Int64
}

enum ModelManagerError: Error {
    case checksumMismatch
    case invalidSource
}

actor ModelManagerService {
    func ensureModelDirectory() {
        _ = FileUtilities.modelDirectory
    }

    func localPath(for artifact: ModelArtifact) -> URL {
        FileUtilities.modelDirectory.appendingPathComponent(artifact.name)
    }

    func isDownloaded(_ artifact: ModelArtifact) -> Bool {
        FileManager.default.fileExists(atPath: localPath(for: artifact).path)
    }

    func download(_ artifact: ModelArtifact) async throws -> URL {
        let (tempURL, _) = try await URLSession.shared.download(from: artifact.downloadURL)
        let destination = localPath(for: artifact)

        if FileManager.default.fileExists(atPath: destination.path) {
            try FileManager.default.removeItem(at: destination)
        }
        try FileManager.default.copyItem(at: tempURL, to: destination)

        let digest = try sha256(of: destination)
        guard digest.caseInsensitiveCompare(artifact.sha256) == .orderedSame else {
            try? FileManager.default.removeItem(at: destination)
            throw ModelManagerError.checksumMismatch
        }
        return destination
    }

    private func sha256(of fileURL: URL) throws -> String {
        let data = try Data(contentsOf: fileURL)
        return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}
