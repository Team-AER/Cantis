import Foundation
import Sentry
import XCTest
@testable import Cantis

final class ErrorMonitoringTests: XCTestCase {
    func testNSErrorUserInfoAndMechanismAreRemoved() throws {
        let error = NSError(domain: "DiagnosticError", code: 1, userInfo: [NSLocalizedDescriptionKey: "PRIVATE_SENTINEL"])
        let event = Event(error: error)
        // Exercise the SDK's NSError conversion, including contexts and mechanism metadata.
        let data = try JSONSerialization.data(withJSONObject: ErrorMonitoring.scrub(event).serialize())
        XCTAssertFalse(String(decoding: data, as: UTF8.self).contains("PRIVATE_SENTINEL"))
    }
}
