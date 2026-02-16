import AVFoundation
import Foundation
import Observation

@MainActor
@Observable
final class AudioPlayerService {
    var isPlaying = false
    var isLooping = false
    var currentTime: TimeInterval = 0
    var duration: TimeInterval = 0
    var volume: Float = 1 {
        didSet {
            if engineReady { engine.mainMixerNode.outputVolume = volume }
        }
    }

    /// Latest FFT magnitude data (updated ~20 times/sec while playing).
    var spectrumMagnitudes: [Float] = []

    private var engine = AVAudioEngine()
    private var playerNode = AVAudioPlayerNode()
    private var audioFile: AVAudioFile?
    private var progressTask: Task<Void, Never>?
    private let fftSize: Int = 1024
    private var engineReady = false

    init() {
        // Audio engine setup is deferred to first use to avoid
        // GCD queue assertion failures during app startup.
    }

    /// Ensures the audio engine is configured and running.
    /// Called lazily before any audio operation.
    private func ensureEngine() {
        guard !engineReady else { return }
        engine.attach(playerNode)
        engine.connect(playerNode, to: engine.mainMixerNode, format: nil)
        installFFTTap()

        do {
            try engine.start()
        } catch {
            NSLog("Audio engine failed to start: \(error)")
        }
        engine.mainMixerNode.outputVolume = volume
        engineReady = true
    }

    private func installFFTTap() {
        let mixerNode = engine.mainMixerNode
        let size = fftSize
        let bufferSize = AVAudioFrameCount(size)
        let format = mixerNode.outputFormat(forBus: 0)
        guard format.sampleRate > 0 else { return }

        mixerNode.installTap(onBus: 0, bufferSize: bufferSize, format: format) { [weak self] buffer, _ in
            guard let self else { return }
            guard let channelData = buffer.floatChannelData?[0] else { return }
            let frameLength = Int(buffer.frameLength)
            let samples = Array(UnsafeBufferPointer(start: channelData, count: frameLength))
            let mags = AudioFFT.magnitudes(samples: samples, fftSize: size)
            Task { @MainActor [weak self] in
                guard let self, self.isPlaying else { return }
                self.spectrumMagnitudes = mags
            }
        }
    }

    func load(url: URL) throws {
        ensureEngine()
        let file = try AVAudioFile(forReading: url)
        audioFile = file
        duration = Double(file.length) / file.fileFormat.sampleRate
        currentTime = 0
        scheduleFile()
    }

    func play() {
        ensureEngine()
        guard !isPlaying else { return }
        if !engine.isRunning {
            try? engine.start()
        }
        playerNode.play()
        isPlaying = true
        startProgressUpdates()
    }

    func pause() {
        playerNode.pause()
        isPlaying = false
        progressTask?.cancel()
        progressTask = nil
    }

    func stop() {
        playerNode.stop()
        isPlaying = false
        currentTime = 0
        progressTask?.cancel()
        progressTask = nil
        scheduleFile()
    }

    func seek(to time: TimeInterval) {
        ensureEngine()
        guard let audioFile else { return }
        let wasPlaying = isPlaying
        playerNode.stop()

        let sampleRate = audioFile.processingFormat.sampleRate
        let targetFrame = AVAudioFramePosition(time * sampleRate)
        let totalFrames = AVAudioFramePosition(audioFile.length)
        let clampedFrame = max(0, min(targetFrame, totalFrames))
        let remainingFrames = AVAudioFrameCount(totalFrames - clampedFrame)

        guard remainingFrames > 0 else {
            currentTime = duration
            isPlaying = false
            return
        }

        playerNode.scheduleSegment(audioFile, startingFrame: clampedFrame, frameCount: remainingFrames, at: nil) { [weak self] in
            guard let self else { return }
            Task { @MainActor in
                self.isPlaying = false
                self.currentTime = 0
                if self.isLooping {
                    self.scheduleFile()
                    self.play()
                }
            }
        }

        currentTime = time
        if wasPlaying {
            playerNode.play()
            isPlaying = true
            startProgressUpdates()
        }
    }

    private func scheduleFile() {
        guard let audioFile else { return }
        playerNode.stop()
        playerNode.scheduleFile(audioFile, at: nil) { [weak self] in
            guard let self else { return }
            Task { @MainActor in
                self.isPlaying = false
                self.currentTime = 0
                if self.isLooping {
                    self.scheduleFile()
                    self.play()
                }
            }
        }
    }

    private func updateProgress() {
        guard isPlaying,
              let nodeTime = playerNode.lastRenderTime,
              let playerTime = playerNode.playerTime(forNodeTime: nodeTime)
        else { return }

        let time = Double(playerTime.sampleTime) / playerTime.sampleRate
        currentTime = min(duration, time)
    }

    private func startProgressUpdates() {
        progressTask?.cancel()
        progressTask = Task {
            while !Task.isCancelled {
                updateProgress()
                try? await Task.sleep(for: .milliseconds(50))
            }
        }
    }
}
