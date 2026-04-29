// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "Cantis",
    platforms: [
        .macOS(.v26)
    ],
    products: [
        .executable(name: "Cantis", targets: ["Cantis"])
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-collections.git", from: "1.1.0"),
        .package(url: "https://github.com/markiv/SwiftUI-Shimmer.git", from: "1.5.1"),
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.21.0")
    ],
    targets: [
        .executableTarget(
            name: "Cantis",
            dependencies: [
                .product(name: "Collections", package: "swift-collections"),
                .product(name: "Shimmer", package: "SwiftUI-Shimmer"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift")
            ],
            path: "Cantis",
            exclude: [
                "Info.plist",
                "Entitlements.plist"
            ],
            resources: [
                .process("Resources")
            ]
        ),
        .testTarget(
            name: "CantisTests",
            dependencies: [
                "Cantis",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift")
            ],
            path: "CantisTests"
        )
    ]
)
