import Foundation
import SwiftData

@MainActor
final class HistoryService {
    private let context: ModelContext

    init(context: ModelContext) {
        self.context = context
    }

    func insert(_ track: GeneratedTrack) throws {
        context.insert(track)
        try context.save()
    }

    func recent(limit: Int = 25) throws -> [GeneratedTrack] {
        var descriptor = FetchDescriptor<GeneratedTrack>(
            sortBy: [SortDescriptor(\GeneratedTrack.createdAt, order: .reverse)]
        )
        descriptor.fetchLimit = limit
        return try context.fetch(descriptor)
    }

    func search(query: String) throws -> [GeneratedTrack] {
        let descriptor = FetchDescriptor<GeneratedTrack>(
            predicate: #Predicate { track in
                track.title.localizedStandardContains(query)
                || track.prompt.localizedStandardContains(query)
            },
            sortBy: [SortDescriptor(\GeneratedTrack.createdAt, order: .reverse)]
        )
        return try context.fetch(descriptor)
    }

    func setFavorite(_ track: GeneratedTrack, isFavorite: Bool) throws {
        track.isFavorite = isFavorite
        try context.save()
    }
}
