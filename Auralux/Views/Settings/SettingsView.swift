import SwiftUI

struct SettingsView: View {
    @Environment(SettingsViewModel.self) private var viewModel

    var body: some View {
        ScrollView {
            HStack(alignment: .top, spacing: 24) {
                leftColumn
                rightColumn
            }
            .padding(28)
        }
    }

    // MARK: - Left Column

    private var leftColumn: some View {
        VStack(alignment: .leading, spacing: 16) {
            settingsCard("Performance", systemImage: "gauge.with.dots.needle.50percent") {
                VStack(spacing: 10) {
                    LabeledContent("Quantization") {
                        Picker("", selection: Bindable(viewModel).quantizationMode) {
                            ForEach(SettingsViewModel.QuantizationMode.allCases) { mode in
                                Text(mode.rawValue.uppercased()).tag(mode)
                            }
                        }
                        .labelsHidden()
                        .frame(width: 100)
                    }
                    Divider()
                    LabeledContent("Low-memory mode") {
                        Toggle("", isOn: Bindable(viewModel).lowMemoryMode)
                            .labelsHidden()
                    }
                    Divider()
                    LabeledContent("Max parallel jobs") {
                        Stepper(
                            "\(viewModel.maxConcurrentJobs)",
                            value: Bindable(viewModel).maxConcurrentJobs,
                            in: 1...4
                        )
                        .fixedSize()
                    }
                }
            }

            settingsCard("Inference", systemImage: "server.rack") {
                LabeledContent("Auto-start local server") {
                    Toggle("", isOn: Bindable(viewModel).autoStartServer)
                        .labelsHidden()
                }
            }

            settingsCard("Export", systemImage: "square.and.arrow.up") {
                LabeledContent("Default format") {
                    Picker("", selection: Bindable(viewModel).defaultExportFormat) {
                        ForEach(AudioExportFormat.allCases.filter(\.isAvailable)) { format in
                            Text(format.rawValue.uppercased()).tag(format)
                        }
                    }
                    .labelsHidden()
                    .frame(width: 100)
                }
            }
        }
        .frame(width: 300)
    }

    // MARK: - Right Column

    private var rightColumn: some View {
        VStack(alignment: .leading, spacing: 16) {
            settingsCard("Engine", systemImage: "cpu") {
                EngineControlPanel()
            }

            GroupBox {
                ModelSettingsView()
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    // MARK: - Card Builder

    @ViewBuilder
    private func settingsCard<Content: View>(
        _ title: String,
        systemImage: String,
        @ViewBuilder content: () -> Content
    ) -> some View {
        GroupBox(
            label: Label(title, systemImage: systemImage)
                .font(.headline)
                .padding(.bottom, 4)
        ) {
            content()
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}
