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
            engine.mainMixerNode.outputVolume = volume
        }
    }

    private let engine = AVAudioEngine()
    private let playerNode = AVAudioPlayerNode()
    private var audioFile: AVAudioFile?
    private var progressTask: Task<Void, Never>?

    init() {
        engine.attach(playerNode)
        engine.connect(playerNode, to: engine.mainMixerNode, format: nil)

        do {
            try engine.start()
        } catch {
            NSLog("Audio engine failed to start: \(error)")
        }
    }

    func load(url: URL) throws {
        let file = try AVAudioFile(forReading: url)
        audioFile = file
        duration = Double(file.length) / file.fileFormat.sampleRate
        currentTime = 0
        scheduleFile()
    }

    func play() {
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
        progressTask = Task { [weak self] in
            guard let self else { return }
            while !Task.isCancelled {
                self.updateProgress()
                try? await Task.sleep(for: .milliseconds(50))
            }
        }
    }
}
