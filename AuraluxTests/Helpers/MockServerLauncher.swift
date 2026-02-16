import Foundation
@testable import Auralux

/// A no-op `ServerLauncher` for tests.  Always reports as running so that
/// `InferenceService.startServerIfNeeded()` skips the real launch path and
/// goes straight to the (mocked) health check.
final class MockServerLauncher: ServerLauncher, @unchecked Sendable {
    private let lock = NSLock()
    private var _isRunning = true
    private var _launchCallCount = 0

    var isRunning: Bool {
        lock.withLock { _isRunning }
    }

    var launchCallCount: Int {
        lock.withLock { _launchCallCount }
    }

    func launch() async throws {
        lock.withLock {
            _launchCallCount += 1
            _isRunning = true
        }
    }

    func stop() async {
        lock.withLock { _isRunning = false }
    }
}
