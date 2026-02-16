import Foundation

struct GenerationRequest: Codable, Sendable {
    var prompt: String
    var lyrics: String
    var tags: [String]
    var duration: TimeInterval
    var variance: Double
    var seed: Int?
}

struct GenerationResponse: Codable, Sendable {
    var jobID: String
    var status: String
    var message: String?
}

struct GenerationStatusResponse: Codable, Sendable {
    var jobID: String
    var status: String
    var progress: Double
    var message: String?
    var audioPath: String?
}

enum InferenceError: Error {
    case serverScriptMissing
    case serverLaunchFailed
    case serverUnhealthy
    case invalidResponse
    case requestFailed(String)
}

actor InferenceService {
    private let baseURL: URL
    private var serverProcess: Process?

    init(baseURL: URL = AppConstants.inferenceBaseURL) {
        self.baseURL = baseURL
    }

    func startServerIfNeeded() async throws {
        if await isHealthy() { return }
        if serverProcess?.isRunning == true { return }

        let rootURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        let scriptURL = rootURL.appendingPathComponent("AuraluxEngine/start_api_server_macos.sh")

        guard FileManager.default.fileExists(atPath: scriptURL.path) else {
            throw InferenceError.serverScriptMissing
        }

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/zsh")
        process.arguments = [scriptURL.path]
        process.currentDirectoryURL = scriptURL.deletingLastPathComponent()
        process.standardOutput = Pipe()
        process.standardError = Pipe()

        do {
            try process.run()
            serverProcess = process
        } catch {
            throw InferenceError.serverLaunchFailed
        }

        for _ in 0..<20 {
            if await isHealthy() {
                return
            }
            try? await Task.sleep(for: .milliseconds(500))
        }
        throw InferenceError.serverUnhealthy
    }

    func stopServer() {
        guard let process = serverProcess else { return }
        if process.isRunning {
            process.terminate()
        }
        serverProcess = nil
    }

    func isHealthy() async -> Bool {
        var request = URLRequest(url: baseURL.appendingPathComponent("health"))
        request.timeoutInterval = 2

        do {
            let (_, response) = try await URLSession.shared.data(for: request)
            guard let http = response as? HTTPURLResponse else { return false }
            return (200...299).contains(http.statusCode)
        } catch {
            return false
        }
    }

    func generate(_ requestBody: GenerationRequest) async throws -> GenerationResponse {
        try await startServerIfNeeded()

        var request = URLRequest(url: baseURL.appendingPathComponent("generate"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONEncoder().encode(requestBody)

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else { throw InferenceError.invalidResponse }
        guard (200...299).contains(http.statusCode) else {
            throw InferenceError.requestFailed("generate failed: \(http.statusCode)")
        }

        return try JSONDecoder().decode(GenerationResponse.self, from: data)
    }

    func poll(jobID: String) async throws -> GenerationStatusResponse {
        let request = URLRequest(url: baseURL.appendingPathComponent("jobs/\(jobID)"))
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else { throw InferenceError.invalidResponse }
        guard (200...299).contains(http.statusCode) else {
            throw InferenceError.requestFailed("poll failed: \(http.statusCode)")
        }
        return try JSONDecoder().decode(GenerationStatusResponse.self, from: data)
    }

    func cancel(jobID: String) async throws {
        var request = URLRequest(url: baseURL.appendingPathComponent("jobs/\(jobID)/cancel"))
        request.httpMethod = "POST"
        let (_, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else { throw InferenceError.invalidResponse }
        guard (200...299).contains(http.statusCode) else {
            throw InferenceError.requestFailed("cancel failed: \(http.statusCode)")
        }
    }
}
