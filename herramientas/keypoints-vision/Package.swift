// swift-tools-version:6.0
// Extractor de keypoints del corpus (TT-12 · SP-213 · contracts/extractor.md): adaptador fino de
// Apple Vision con las mismas peticiones, revisiones y llamada que `VisionKeypointDetector` de la app.
import PackageDescription

let package = Package(
    name: "keypoints-vision",
    platforms: [.macOS(.v15)],
    targets: [
        .executableTarget(
            name: "keypoints-vision",
            path: "Sources/keypoints-vision"
        )
    ]
)
