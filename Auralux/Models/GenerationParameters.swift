import Foundation

struct GenerationParameters: Codable, Hashable, Sendable {
    var prompt: String
    var lyrics: String
    var tags: [String]
    var duration: TimeInterval
    var variance: Double
    var seed: Int?

    static let `default` = GenerationParameters(
        prompt: "",
        lyrics: "",
        tags: [],
        duration: 30,
        variance: 0.5,
        seed: nil
    )
}
