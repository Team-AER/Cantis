# Error monitoring

The official Sentry Cocoa SDK provides GlitchTip-compatible crash reporting. Set `GLITCHTIP_DSN` or `SENTRY_DSN` in the launch environment, or set `GlitchTipDSN` in the signed application Info.plist. An empty DSN disables reporting. `SENTRY_ENVIRONMENT` and `SENTRY_RELEASE` label the build. Rebuild and distribute the app to monitor installed releases; changing a portal project cannot modify an existing binary.

Tracing, sessions, network tracking, swizzling, automatic breadcrumbs and app-hang monitoring are disabled. Request context, users, breadcrumbs, extras and exception messages are removed before submission. For verification, use `SentrySDK.capture(message: "Team AER monitoring verification")` then `SentrySDK.flush(timeout: 10)` from an operator test build and confirm its event ID in GlitchTip.
