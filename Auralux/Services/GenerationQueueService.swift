import Collections
import Foundation

struct GenerationQueueItem: Identifiable, Equatable, Sendable {
    enum Priority: Int, Comparable, Sendable {
        case low
        case normal
        case high

        static func < (lhs: Priority, rhs: Priority) -> Bool {
            lhs.rawValue < rhs.rawValue
        }
    }

    var id: UUID
    var parameters: GenerationParameters
    var priority: Priority
    var createdAt: Date

    init(id: UUID = UUID(), parameters: GenerationParameters, priority: Priority = .normal, createdAt: Date = .now) {
        self.id = id
        self.parameters = parameters
        self.priority = priority
        self.createdAt = createdAt
    }
}

actor GenerationQueueService {
    private var queue = Deque<GenerationQueueItem>()

    func enqueue(_ item: GenerationQueueItem) {
        if let idx = queue.firstIndex(where: { $0.priority < item.priority }) {
            queue.insert(item, at: idx)
        } else {
            queue.append(item)
        }
    }

    func dequeue() -> GenerationQueueItem? {
        queue.popFirst()
    }

    func remove(id: UUID) {
        if let idx = queue.firstIndex(where: { $0.id == id }) {
            queue.remove(at: idx)
        }
    }

    func pendingItems() -> [GenerationQueueItem] {
        Array(queue)
    }

    func clear() {
        queue.removeAll()
    }
}
