import XCTest
@testable import Auralux

final class ModelTests: XCTestCase {
    func testGenerationParametersCodableRoundTrip() throws {
        let params = GenerationParameters(
            prompt: "ambient piano",
            lyrics: "[verse]",
            tags: ["ambient", "piano"],
            duration: 48,
            variance: 0.3,
            seed: 42
        )

        let data = try JSONEncoder().encode(params)
        let decoded = try JSONDecoder().decode(GenerationParameters.self, from: data)
        XCTAssertEqual(params, decoded)
    }
}
