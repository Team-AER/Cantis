import XCTest
@testable import Auralux

final class ServiceTests: XCTestCase {
    func testQueueRespectsPriorityOrdering() async {
        let queue = GenerationQueueService()

        await queue.enqueue(.init(parameters: .default, priority: .low))
        await queue.enqueue(.init(parameters: .default, priority: .high))
        await queue.enqueue(.init(parameters: .default, priority: .normal))

        let first = await queue.dequeue()
        let second = await queue.dequeue()
        let third = await queue.dequeue()

        XCTAssertEqual(first?.priority, .high)
        XCTAssertEqual(second?.priority, .normal)
        XCTAssertEqual(third?.priority, .low)
    }
}
