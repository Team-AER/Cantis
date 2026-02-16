import XCTest
@testable import Auralux

@MainActor
final class ViewModelTests: XCTestCase {
    func testGenerationViewModelTagMutations() {
        let viewModel = GenerationViewModel(inferenceService: InferenceService())

        viewModel.addTag("Ambient")
        viewModel.addTag("ambient")
        viewModel.addTag("piano")

        XCTAssertEqual(viewModel.tags, ["ambient", "piano"])

        viewModel.removeTag("ambient")
        XCTAssertEqual(viewModel.tags, ["piano"])
    }

    func testApplyingPresetUpdatesParameters() {
        let viewModel = GenerationViewModel(inferenceService: InferenceService())
        let preset = Preset(
            name: "Test",
            summary: "Summary",
            prompt: "Prompt",
            lyricTemplate: "Lyrics",
            tags: ["a", "b"],
            duration: 55,
            variance: 0.66
        )

        viewModel.applyPreset(preset)

        XCTAssertEqual(viewModel.prompt, "Prompt")
        XCTAssertEqual(viewModel.lyrics, "Lyrics")
        XCTAssertEqual(viewModel.tags, ["a", "b"])
        XCTAssertEqual(viewModel.duration, 55)
        XCTAssertEqual(viewModel.variance, 0.66)
    }
}
