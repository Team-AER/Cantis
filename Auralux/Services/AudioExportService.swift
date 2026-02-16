import AVFoundation
import Foundation

enum AudioExportFormat: String, CaseIterable, Identifiable, Codable {
    case wav
    case flac
    case mp3
    case aac
    case alac

    var id: String { rawValue }
    var fileExtension: String { rawValue }
}

enum AudioExportError: Error {
    case invalidSource
    case unsupported
}

struct AudioExportConfiguration: Codable, Sendable {
    var format: AudioExportFormat
    var sampleRate: Double
    var title: String
    var tags: [String]
}

final class AudioExportService {
    func export(sourceURL: URL, destinationDirectory: URL, configuration: AudioExportConfiguration) throws -> URL {
        guard FileManager.default.fileExists(atPath: sourceURL.path) else {
            throw AudioExportError.invalidSource
        }

        let name = "\(configuration.title.replacingOccurrences(of: " ", with: "_"))_\(UUID().uuidString.prefix(8)).\(configuration.format.fileExtension)"
        let destination = destinationDirectory.appendingPathComponent(name)

        switch configuration.format {
        case .wav:
            try copyFile(from: sourceURL, to: destination)
            return destination
        case .flac, .mp3, .aac, .alac:
            // Phase 1 placeholder: writes source audio with target extension.
            // Dedicated transcoding is implemented in later phases.
            try copyFile(from: sourceURL, to: destination)
            return destination
        }
    }

    private func copyFile(from source: URL, to destination: URL) throws {
        if FileManager.default.fileExists(atPath: destination.path) {
            try FileManager.default.removeItem(at: destination)
        }
        try FileManager.default.copyItem(at: source, to: destination)
    }
}
