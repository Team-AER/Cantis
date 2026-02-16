import Foundation
import Observation

@MainActor
@Observable
final class PlayerViewModel {
    let playerService: AudioPlayerService
    var loadedPath: String?

    init(playerService: AudioPlayerService = AudioPlayerService()) {
        self.playerService = playerService
    }

    func load(path: String?) {
        guard let path else { return }
        do {
            try playerService.load(url: URL(fileURLWithPath: path))
            loadedPath = path
        } catch {
            NSLog("Failed to load audio: \(error)")
        }
    }

    func playPause() {
        if playerService.isPlaying {
            playerService.pause()
        } else {
            playerService.play()
        }
    }

    func stop() {
        playerService.stop()
    }
}
