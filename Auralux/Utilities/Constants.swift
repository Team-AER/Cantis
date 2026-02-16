import Foundation

enum AppConstants {
    static let appName = "Auralux"
    static let minimumWindowWidth: Double = 1024
    static let minimumWindowHeight: Double = 768
    static let inferenceBaseURL = URL(string: "http://127.0.0.1:8765")!
    static let modelDirectoryName = "Models"

    static let suggestedTags = [
        "ambient", "cinematic", "lofi", "electronic", "jazz", "piano", "guitar", "synth",
        "uplifting", "melancholic", "driving", "intimate", "orchestral", "vocal", "instrumental"
    ]
}
