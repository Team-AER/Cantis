import Foundation
import Sentry

enum ErrorMonitoring {
    static func start() {
        let environment = ProcessInfo.processInfo.environment
        let dsn = environment["GLITCHTIP_DSN"] ?? environment["SENTRY_DSN"]
            ?? Bundle.main.object(forInfoDictionaryKey: "GlitchTipDSN") as? String ?? ""
        guard !dsn.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        SentrySDK.start { options in
            options.dsn = dsn
            options.environment = environment["SENTRY_ENVIRONMENT"] ?? "production"
            options.releaseName = environment["SENTRY_RELEASE"] ?? "cantis@0.1.0"
            options.sendDefaultPii = false
            options.enableAutoSessionTracking = false
            options.enableAutoPerformanceTracing = false
            options.enableNetworkTracking = false
            options.enableAutoBreadcrumbTracking = false
            options.enableAppHangTracking = false
            options.enableSwizzling = false
            options.tracesSampleRate = 0
            options.beforeSend = { event in
                event.request = nil
                event.user = nil
                event.breadcrumbs = nil
                event.extra = nil
                for exception in event.exceptions ?? [] { exception.value = exception.type }
                return event
            }
        }
    }
}
