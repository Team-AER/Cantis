import Foundation

// MARK: - Request / Response types

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

struct HealthResponse: Codable, Sendable {
    var status: String
    var modelLoaded: Bool
    var llmLoaded: Bool?
    var modelError: String?
    var device: String?
    var engine: String?
    var ditModel: String?
    var llmModel: String?
}

// MARK: - Errors

enum InferenceError: Error, LocalizedError {
    case serverScriptMissing
    case serverLaunchFailed
    case serverUnhealthy
    case invalidResponse
    case requestFailed(String)
    case modelNotReady

    var errorDescription: String? {
        switch self {
        case .serverScriptMissing:
            return "Server launch script not found. Run setup_env.sh first."
        case .serverLaunchFailed:
            return "Failed to start the inference server process."
        case .serverUnhealthy:
            return "Inference server did not become healthy in time."
        case .invalidResponse:
            return "Received an invalid response from the server."
        case .requestFailed(let detail):
            return "Request failed: \(detail)"
        case .modelNotReady:
            return "Model is still downloading or not yet loaded."
        }
    }
}

// MARK: - Service

actor InferenceService {
    private let baseURL: URL
    private var serverProcess: Process?

    init(baseURL: URL = AppConstants.inferenceBaseURL) {
        self.baseURL = baseURL
    }

    // MARK: Server lifecycle

    func startServerIfNeeded() async throws {
        if await isHealthy() { return }
        if serverProcess?.isRunning == true { return }

        let scriptURL = Self.locateServerScript()

        guard let scriptURL, FileManager.default.fileExists(atPath: scriptURL.path) else {
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

        // ACE-Step 1.5 may need time to download models on first launch.
        // Allow up to 5 minutes for the server to become healthy.
        let maxAttempts = 600
        for _ in 0..<maxAttempts {
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

    // MARK: Health

    func isHealthy() async -> Bool {
        var request = URLRequest(url: baseURL.appendingPathComponent("health"))
        request.timeoutInterval = 5

        do {
            let (_, response) = try await URLSession.shared.data(for: request)
            guard let http = response as? HTTPURLResponse else { return false }
            return (200...299).contains(http.statusCode)
        } catch {
            return false
        }
    }

    func fetchHealth() async -> HealthResponse? {
        var request = URLRequest(url: baseURL.appendingPathComponent("health"))
        request.timeoutInterval = 5

        do {
            let (data, response) = try await URLSession.shared.data(for: request)
            guard let http = response as? HTTPURLResponse,
                  (200...299).contains(http.statusCode) else { return nil }
            return try JSONDecoder().decode(HealthResponse.self, from: data)
        } catch {
            return nil
        }
    }

    // MARK: Generation

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

    // MARK: Model management

    func triggerModelDownload() async throws {
        try await startServerIfNeeded()

        var request = URLRequest(url: baseURL.appendingPathComponent("models/download"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        let (_, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else { throw InferenceError.invalidResponse }
        guard (200...299).contains(http.statusCode) else {
            throw InferenceError.requestFailed("model download trigger failed: \(http.statusCode)")
        }
    }

    // MARK: Helpers

    private static func locateServerScript() -> URL? {
        let candidates = [
            URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
                .appendingPathComponent("AuraluxEngine/start_api_server_macos.sh"),
            Bundle.main.bundleURL
                .deletingLastPathComponent()
                .appendingPathComponent("AuraluxEngine/start_api_server_macos.sh"),
        ]

        for candidate in candidates {
            if FileManager.default.fileExists(atPath: candidate.path) {
                return candidate
            }
        }
        return nil
    }
}
