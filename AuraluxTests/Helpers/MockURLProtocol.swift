import Foundation

/// A `URLProtocol` subclass that intercepts all requests made through a
/// `URLSession` whose configuration includes it.  Tests register scripted
/// responses keyed by URL path; the mock returns them synchronously with
/// no network delay.
///
/// Supports **sequential responses** per path — the first call to a path
/// returns the first registered response, the second call returns the second,
/// and so on.  When the sequence is exhausted the last response is repeated.
final class MockURLProtocol: URLProtocol {

    /// Scripted response: HTTP status code + body data.
    struct Response: @unchecked Sendable {
        let statusCode: Int
        let data: Data

        init(statusCode: Int, data: Data) {
            self.statusCode = statusCode
            self.data = data
        }

        init(statusCode: Int, json: Any) {
            self.statusCode = statusCode
            self.data = (try? JSONSerialization.data(withJSONObject: json)) ?? Data()
        }

        init<T: Encodable>(statusCode: Int, body: T) {
            self.statusCode = statusCode
            self.data = (try? JSONEncoder().encode(body)) ?? Data()
        }
    }

    // MARK: - Static state

    private static let lock = NSLock()
    nonisolated(unsafe) private static var responses: [String: [Response]] = [:]
    nonisolated(unsafe) private static var callCounts: [String: Int] = [:]
    nonisolated(unsafe) private static var requestLog: [URLRequest] = []

    // MARK: - Configuration helpers

    /// Remove all registered responses and reset call counts.
    static func reset() {
        lock.withLock {
            responses.removeAll()
            callCounts.removeAll()
            requestLog.removeAll()
        }
    }

    /// Register one or more sequential responses for a URL path.
    /// The path should be the path component only (e.g. "/generate").
    static func register(path: String, responses: [Response]) {
        lock.withLock {
            self.responses[path] = responses
            self.callCounts[path] = 0
        }
    }

    /// Convenience: register a single response that repeats forever.
    static func register(path: String, response: Response) {
        register(path: path, responses: [response])
    }

    /// Returns all requests captured during the test.
    static func capturedRequests() -> [URLRequest] {
        lock.withLock { requestLog }
    }

    /// Returns a `URLSessionConfiguration` pre-configured with this mock.
    static func sessionConfiguration() -> URLSessionConfiguration {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [MockURLProtocol.self]
        return config
    }

    /// Convenience: create a `URLSession` that routes through this mock.
    static func urlSession() -> URLSession {
        URLSession(configuration: sessionConfiguration())
    }

    // MARK: - URLProtocol overrides

    override class func canInit(with request: URLRequest) -> Bool {
        true
    }

    override class func canonicalRequest(for request: URLRequest) -> URLRequest {
        request
    }

    override func startLoading() {
        Self.lock.withLock { Self.requestLog.append(request) }

        let path = request.url?.path ?? ""
        let (statusCode, data) = Self.lock.withLock { () -> (Int, Data) in
            guard let sequence = Self.responses[path], !sequence.isEmpty else {
                return (500, Data("{\"error\":\"no mock registered for path: \(path)\"}".utf8))
            }
            let idx = Self.callCounts[path] ?? 0
            let effectiveIdx = min(idx, sequence.count - 1)
            Self.callCounts[path] = idx + 1
            let entry = sequence[effectiveIdx]
            return (entry.statusCode, entry.data)
        }

        let url = request.url ?? URL(string: "http://mock")!
        let httpResponse = HTTPURLResponse(
            url: url,
            statusCode: statusCode,
            httpVersion: "HTTP/1.1",
            headerFields: ["Content-Type": "application/json"]
        )!

        client?.urlProtocol(self, didReceive: httpResponse, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: data)
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}
}
