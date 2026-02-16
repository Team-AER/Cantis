import Foundation

enum FileUtilities {
    static var appSupportDirectory: URL {
        let baseURL = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let appURL = baseURL.appendingPathComponent(AppConstants.appName, isDirectory: true)
        if !FileManager.default.fileExists(atPath: appURL.path) {
            try? FileManager.default.createDirectory(at: appURL, withIntermediateDirectories: true)
        }
        return appURL
    }

    static var modelDirectory: URL {
        let url = appSupportDirectory.appendingPathComponent(AppConstants.modelDirectoryName, isDirectory: true)
        if !FileManager.default.fileExists(atPath: url.path) {
            try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        }
        return url
    }

    static var generatedAudioDirectory: URL {
        let url = appSupportDirectory.appendingPathComponent("Generated", isDirectory: true)
        if !FileManager.default.fileExists(atPath: url.path) {
            try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        }
        return url
    }

    static func copyToGeneratedDirectory(from sourceURL: URL, name: String) throws -> URL {
        let destination = generatedAudioDirectory.appendingPathComponent(name)
        if FileManager.default.fileExists(atPath: destination.path) {
            try FileManager.default.removeItem(at: destination)
        }
        try FileManager.default.copyItem(at: sourceURL, to: destination)
        return destination
    }
}
