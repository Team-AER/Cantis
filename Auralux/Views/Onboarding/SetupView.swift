import SwiftUI

/// The first-run setup experience that guides users through engine configuration.
/// Shown automatically when the engine is not ready.
struct SetupView: View {
    @Environment(EngineService.self) private var engine

    @State private var currentStep: SetupStep = .welcome
    @State private var showLog = false
    @State private var isRunningSetup = false
    @State private var showDirectoryPicker = false

    enum SetupStep: Int, CaseIterable {
        case welcome
        case systemCheck
        case environmentSetup
        case serverStart
        case ready
    }

    var body: some View {
        VStack(spacing: 0) {
            // Header
            header

            Divider()

            // Content
            ScrollView {
                VStack(spacing: 24) {
                    stepContent
                }
                .padding(40)
                .frame(maxWidth: 600)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)

            Divider()

            // Footer
            footer
        }
        .frame(minWidth: 700, minHeight: 500)
        .task {
            await evaluateInitialStep()
        }
    }

    // MARK: - Header

    private var header: some View {
        VStack(spacing: 8) {
            Image(systemName: "wand.and.stars")
                .font(.system(size: 48))
                .foregroundStyle(.tint)
                .padding(.top, 24)

            Text("Auralux")
                .font(.largeTitle.bold())

            Text("AI Music Generation on Apple Silicon")
                .font(.title3)
                .foregroundStyle(.secondary)

            // Progress indicators
            HStack(spacing: 12) {
                ForEach(SetupStep.allCases, id: \.rawValue) { step in
                    stepIndicator(step)
                }
            }
            .padding(.top, 8)
            .padding(.bottom, 16)
        }
    }

    @ViewBuilder
    private func stepIndicator(_ step: SetupStep) -> some View {
        let isCurrent = step == currentStep
        let isCompleted = step.rawValue < currentStep.rawValue

        HStack(spacing: 4) {
            Circle()
                .fill(isCompleted ? Color.green : (isCurrent ? Color.accentColor : Color.secondary.opacity(0.3)))
                .frame(width: 8, height: 8)

            if isCurrent {
                Text(step.label)
                    .font(.caption.bold())
                    .foregroundStyle(.primary)
            }
        }
    }

    // MARK: - Step Content

    @ViewBuilder
    private var stepContent: some View {
        switch currentStep {
        case .welcome:
            welcomeContent
        case .systemCheck:
            systemCheckContent
        case .environmentSetup:
            environmentSetupContent
        case .serverStart:
            serverStartContent
        case .ready:
            readyContent
        }
    }

    // MARK: Welcome

    private var welcomeContent: some View {
        VStack(spacing: 20) {
            VStack(spacing: 12) {
                Text("Welcome to Auralux")
                    .font(.title2.bold())

                Text("Auralux generates music using ACE-Step v1.5, running entirely on your Mac. Let's get everything set up.")
                    .multilineTextAlignment(.center)
                    .foregroundStyle(.secondary)
            }

            VStack(alignment: .leading, spacing: 12) {
                featureRow(icon: "cpu", title: "On-Device Inference", description: "All processing happens locally on Apple Silicon")
                featureRow(icon: "waveform", title: "Text to Music", description: "Describe what you want and get audio in minutes")
                featureRow(icon: "music.note.list", title: "Lyrics Support", description: "Generate vocal tracks with your own lyrics")
                featureRow(icon: "square.and.arrow.down", title: "Export Anywhere", description: "Save as WAV, FLAC, MP3, AAC, or ALAC")
            }
            .padding(.vertical, 8)
        }
    }

    private func featureRow(icon: String, title: String, description: String) -> some View {
        HStack(spacing: 12) {
            Image(systemName: icon)
                .font(.title3)
                .foregroundStyle(.tint)
                .frame(width: 32)

            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.body.bold())
                Text(description)
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
        }
    }

    // MARK: System Check

    private var systemCheckContent: some View {
        VStack(spacing: 20) {
            Text("System Requirements")
                .font(.title2.bold())

            VStack(alignment: .leading, spacing: 16) {
                checkRow(
                    label: "macOS 15.0 (Sequoia) or later",
                    passed: ProcessInfo.processInfo.operatingSystemVersion.majorVersion >= 15
                )
                checkRow(
                    label: "Apple Silicon (M1 or later)",
                    passed: isAppleSilicon
                )
                checkRow(
                    label: "AuraluxEngine directory found",
                    passed: engine.engineDirectory != nil
                )
                checkRow(
                    label: "8 GB RAM recommended (16 GB ideal)",
                    passed: ProcessInfo.processInfo.physicalMemory >= 8 * 1024 * 1024 * 1024,
                    isWarning: true
                )
            }
            .padding(16)
            .background(.quaternary.opacity(0.3), in: RoundedRectangle(cornerRadius: 8))

            if engine.engineDirectory == nil {
                VStack(spacing: 10) {
                    Label("AuraluxEngine not found", systemImage: "exclamationmark.triangle.fill")
                        .foregroundStyle(.red)
                    Text("The AuraluxEngine folder wasn't detected automatically. Use the button below to locate it, or make sure it's in your project root.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)

                    Button("Browse for AuraluxEngine Folder...") {
                        showDirectoryPicker = true
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                    .fileImporter(
                        isPresented: $showDirectoryPicker,
                        allowedContentTypes: [.folder],
                        allowsMultipleSelection: false
                    ) { result in
                        if case .success(let urls) = result, let url = urls.first {
                            let serverPy = url.appendingPathComponent("server.py")
                            if FileManager.default.fileExists(atPath: serverPy.path) {
                                UserDefaults.standard.set(url.path, forKey: "engine.directoryOverride")
                                Task { await engine.checkStatus() }
                            }
                        }
                    }
                }
            }
        }
    }

    private func checkRow(label: String, passed: Bool, isWarning: Bool = false) -> some View {
        HStack(spacing: 12) {
            Image(systemName: passed ? "checkmark.circle.fill" : (isWarning ? "exclamationmark.triangle.fill" : "xmark.circle.fill"))
                .foregroundStyle(passed ? .green : (isWarning ? .orange : .red))
                .font(.title3)

            Text(label)
                .font(.body)

            Spacer()
        }
    }

    private var isAppleSilicon: Bool {
        #if arch(arm64)
        return true
        #else
        return false
        #endif
    }

    // MARK: Environment Setup

    private var environmentSetupContent: some View {
        VStack(spacing: 20) {
            Text("Environment Setup")
                .font(.title2.bold())

            if engine.isACEStepCloned && engine.isVenvReady {
                VStack(spacing: 12) {
                    Image(systemName: "checkmark.circle.fill")
                        .font(.system(size: 48))
                        .foregroundStyle(.green)

                    Text("Environment is already set up!")
                        .font(.headline)

                    Text("ACE-Step 1.5 is installed and the Python environment is ready.")
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                }
            } else if isRunningSetup {
                setupProgressView
            } else {
                VStack(spacing: 12) {
                    Text("This will:")
                        .font(.headline)

                    VStack(alignment: .leading, spacing: 8) {
                        Label("Clone ACE-Step 1.5 from GitHub (~100 MB)", systemImage: "arrow.down.circle")
                        Label("Install Python dependencies via uv (~2 GB)", systemImage: "shippingbox")
                        Label("Configure the inference environment", systemImage: "gearshape.2")
                    }
                    .font(.callout)
                    .foregroundStyle(.secondary)

                    Text("This may take 5-15 minutes depending on your internet connection.")
                        .font(.caption)
                        .foregroundStyle(.tertiary)
                        .padding(.top, 4)
                }
            }

            if showLog && !engine.setupLog.isEmpty {
                logView
            }
        }
    }

    private var setupProgressView: some View {
        VStack(spacing: 16) {
            ProgressView()
                .controlSize(.large)

            if case .settingUp(let progress) = engine.state {
                Text(progress)
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }

            Button(showLog ? "Hide Log" : "Show Log") {
                showLog.toggle()
            }
            .buttonStyle(.plain)
            .font(.caption)
            .foregroundStyle(.tint)
        }
    }

    private var logView: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 2) {
                    ForEach(Array(engine.setupLog.enumerated()), id: \.offset) { index, line in
                        Text(line)
                            .font(.system(.caption2, design: .monospaced))
                            .foregroundStyle(.secondary)
                            .id(index)
                    }
                }
                .padding(8)
            }
            .frame(maxHeight: 200)
            .background(.black.opacity(0.05), in: RoundedRectangle(cornerRadius: 6))
            .onChange(of: engine.setupLog.count) {
                if let last = engine.setupLog.indices.last {
                    proxy.scrollTo(last, anchor: .bottom)
                }
            }
        }
    }

    // MARK: Server Start

    private var serverStartContent: some View {
        VStack(spacing: 20) {
            Text("Starting Server")
                .font(.title2.bold())

            switch engine.state {
            case .starting:
                VStack(spacing: 16) {
                    ProgressView()
                        .controlSize(.large)
                    Text("Starting the inference server...")
                        .foregroundStyle(.secondary)
                    Text("The first launch downloads AI models (~4 GB). This may take several minutes.")
                        .font(.caption)
                        .foregroundStyle(.tertiary)
                        .multilineTextAlignment(.center)
                }

            case .running:
                VStack(spacing: 12) {
                    Image(systemName: "bolt.circle.fill")
                        .font(.system(size: 48))
                        .foregroundStyle(.yellow)
                    Text("Server is running")
                        .font(.headline)
                    Text("Waiting for models to load...")
                        .foregroundStyle(.secondary)
                }

            case .ready:
                VStack(spacing: 12) {
                    Image(systemName: "checkmark.circle.fill")
                        .font(.system(size: 48))
                        .foregroundStyle(.green)
                    Text("Server is ready!")
                        .font(.headline)
                }

            case .error(let message):
                VStack(spacing: 12) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .font(.system(size: 48))
                        .foregroundStyle(.red)
                    Text("Server Error")
                        .font(.headline)
                    Text(message)
                        .font(.callout)
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                }

            default:
                VStack(spacing: 12) {
                    Text("Ready to start the inference server.")
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    // MARK: Ready

    private var readyContent: some View {
        VStack(spacing: 20) {
            Image(systemName: "checkmark.seal.fill")
                .font(.system(size: 64))
                .foregroundStyle(.green)

            Text("All Set!")
                .font(.title.bold())

            Text("Auralux is ready to generate music. Write a prompt, add some tags, and hit Generate.")
                .multilineTextAlignment(.center)
                .foregroundStyle(.secondary)

            if engine.modelStatus.ditLoaded {
                HStack(spacing: 16) {
                    Label(engine.modelStatus.device, systemImage: "cpu")
                    Label(engine.modelStatus.engine, systemImage: "waveform")
                }
                .font(.caption)
                .foregroundStyle(.tertiary)
            }
        }
    }

    // MARK: - Footer

    private var footer: some View {
        HStack {
            if currentStep == .welcome {
                Button("Skip Setup") {
                    withAnimation { engine.isOnboarding = false }
                }
                .foregroundStyle(.secondary)
                .buttonStyle(.plain)
                .font(.callout)
            } else {
                Button("Back") {
                    withAnimation {
                        goBack()
                    }
                }
                .disabled(isRunningSetup)
            }

            Spacer()

            if case .error = engine.state {
                Button("Retry") {
                    Task { await retryCurrentStep() }
                }
                .buttonStyle(.bordered)
            }

            Button(currentStep == .ready ? "Get Started" : "Continue") {
                Task {
                    await advanceStep()
                }
            }
            .buttonStyle(.borderedProminent)
            .disabled(!canAdvance)
        }
        .padding(.horizontal, 40)
        .padding(.vertical, 16)
    }

    private var canAdvance: Bool {
        switch currentStep {
        case .welcome:
            return true
        case .systemCheck:
            return engine.engineDirectory != nil
        case .environmentSetup:
            if isRunningSetup { return false }
            return engine.isVenvReady
        case .serverStart:
            return engine.state.isReady || engine.state.isRunning
        case .ready:
            return true
        }
    }

    // MARK: - Navigation

    private func evaluateInitialStep() async {
        if engine.state.isReady || engine.state.isRunning {
            currentStep = .ready
            return
        }
        if engine.isACEStepCloned && engine.isVenvReady {
            currentStep = .serverStart
            return
        }
        currentStep = .welcome
    }

    private func advanceStep() async {
        switch currentStep {
        case .welcome:
            withAnimation { currentStep = .systemCheck }

        case .systemCheck:
            withAnimation { currentStep = .environmentSetup }
            if !engine.isVenvReady {
                isRunningSetup = true
                await engine.runSetup()
                isRunningSetup = false
            }

        case .environmentSetup:
            withAnimation { currentStep = .serverStart }
            if !engine.state.isRunning && !engine.state.isReady {
                await engine.startServer()
            }

        case .serverStart:
            withAnimation { currentStep = .ready }

        case .ready:
            withAnimation { engine.isOnboarding = false }
        }
    }

    private func goBack() {
        if let prev = SetupStep(rawValue: currentStep.rawValue - 1) {
            currentStep = prev
        }
    }

    private func retryCurrentStep() async {
        switch currentStep {
        case .environmentSetup:
            isRunningSetup = true
            await engine.runSetup()
            isRunningSetup = false
        case .serverStart:
            await engine.startServer()
        default:
            break
        }
    }
}

// MARK: - SetupStep Labels

extension SetupView.SetupStep {
    var label: String {
        switch self {
        case .welcome: return "Welcome"
        case .systemCheck: return "System"
        case .environmentSetup: return "Setup"
        case .serverStart: return "Server"
        case .ready: return "Ready"
        }
    }
}
