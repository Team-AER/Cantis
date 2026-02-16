import AVFoundation
import Accelerate
import Foundation

struct AudioFFT {
    static func magnitudes(samples: [Float], fftSize: Int = 1024) -> [Float] {
        guard samples.count >= fftSize, fftSize.isPowerOfTwo else { return [] }
        let frame = Array(samples.prefix(fftSize))
        let bands = fftSize / 2
        let chunkSize = max(1, frame.count / bands)
        var output: [Float] = []
        output.reserveCapacity(bands)

        for index in 0..<bands {
            let start = index * chunkSize
            if start >= frame.count {
                output.append(0)
                continue
            }
            let end = min(frame.count, start + chunkSize)
            let slice = Array(frame[start..<end])
            var meanSquare: Float = 0
            vDSP_measqv(slice, 1, &meanSquare, vDSP_Length(slice.count))
            output.append(sqrt(meanSquare))
        }

        return output
    }
}

private extension Int {
    var isPowerOfTwo: Bool {
        self > 0 && (self & (self - 1)) == 0
    }
}
