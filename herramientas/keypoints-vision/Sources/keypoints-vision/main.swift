//  keypoints-vision — extractor de keypoints del corpus (TT-12 · SP-213 · contracts/extractor.md).
//
//  Adaptador fino de Apple Vision: hace lo mismo que `VisionKeypointDetector` de la app
//  (signai-ios/…/Captura/Adaptadores) y nada más. Por cada fotograma conservado por la decimación:
//  una sola `ImageRequestHandler.perform` con `DetectHumanBodyPoseRequest(.revision2)` y
//  `DetectHumanHandPoseRequest(.revision1)`, dos manos como máximo, orientación `.up` (E-1).
//
//  La imagen que recibe Vision es la de la cámara de la app: 720×1280 vertical, SDR BT.709 de 8 bits,
//  sin espejo (research R-03). Se rechaza un vídeo con rotación en los metadatos o que no sea 9:16.
//
//  Escribe lo que ve el detector, sin interpretar: las 6 articulaciones de pose de E-2 y las 21 de cada
//  mano, en coordenadas de Vision (origen abajo), con el lado de cada mano y **sin confianza**. Elegir
//  persona y manos e invertir el eje es de `signai_ml.preproceso.seleccion`.
//
//  Uso: keypoints-vision <clip.mp4> <salida.json.gz> --cada <k> [--volcar <carpeta>]
//  Códigos: 0 bien · 1 error de entrada · 2 uso incorrecto.

import AVFoundation
import CoreImage
import CoreVideo
import CryptoKit
import Foundation
import ImageIO
import UniformTypeIdentifiers
import VideoToolbox
import Vision

let VERSION_EXTRACTOR = "signai-ml/keypoints/1"
let ANCHO = 720
let ALTO = 1280

func fallar(_ mensaje: String, codigo: Int32 = 1) -> Never {
    FileHandle.standardError.write("\(mensaje)\n".data(using: .utf8)!)
    exit(codigo)
}

func uso() -> Never {
    fallar("uso: keypoints-vision <clip.mp4> <salida.json.gz> --cada <k> [--volcar <carpeta>]", codigo: 2)
}

// MARK: - Argumentos (X-1)

let args = Array(CommandLine.arguments.dropFirst())
guard args.count == 4 || args.count == 6, args[2] == "--cada", let cada = Int(args[3]), cada >= 1 else { uso() }
let entrada = URL(fileURLWithPath: args[0])
let salida = URL(fileURLWithPath: args[1])
var volcar: URL? = nil
if args.count == 6 {
    guard args[4] == "--volcar" else { uso() }
    volcar = URL(fileURLWithPath: args[5], isDirectory: true)
}
guard FileManager.default.fileExists(atPath: entrada.path) else { fallar("no existe el vídeo: \(entrada.path)") }

// MARK: - Huella (la misma que etiquetas.csv)

func md5(_ url: URL) throws -> String {
    let lector = try FileHandle(forReadingFrom: url)
    defer { try? lector.close() }
    var resumen = Insecure.MD5()
    while let bloque = try lector.read(upToCount: 1 << 20), !bloque.isEmpty {
        resumen.update(data: bloque)
    }
    return resumen.finalize().map { String(format: "%02x", $0) }.joined()
}

// MARK: - gzip (RFC 1952) sobre el DEFLATE crudo de Foundation

func crc32(_ datos: Data) -> UInt32 {
    var tabla = [UInt32](repeating: 0, count: 256)
    for i in 0..<256 {
        var c = UInt32(i)
        for _ in 0..<8 { c = (c & 1) != 0 ? 0xEDB8_8320 ^ (c >> 1) : c >> 1 }
        tabla[i] = c
    }
    var crc: UInt32 = 0xFFFF_FFFF
    for byte in datos { crc = tabla[Int((crc ^ UInt32(byte)) & 0xFF)] ^ (crc >> 8) }
    return crc ^ 0xFFFF_FFFF
}

func gzip(_ datos: Data) throws -> Data {
    let deflate = try (datos as NSData).compressed(using: .zlib) as Data  // DEFLATE crudo (RFC 1951)
    var resultado = Data([0x1F, 0x8B, 0x08, 0x00, 0, 0, 0, 0, 0x00, 0x03])
    resultado.append(deflate)
    var crc = crc32(datos).littleEndian
    var tamano = UInt32(truncatingIfNeeded: datos.count).littleEndian
    resultado.append(Data(bytes: &crc, count: 4))
    resultado.append(Data(bytes: &tamano, count: 4))
    return resultado
}

// MARK: - Nombres de E-2 (los mismos que la app)

let nombresPose: [HumanBodyPoseObservation.JointName: String] = [
    .leftShoulder: "leftShoulder", .rightShoulder: "rightShoulder",
    .leftElbow: "leftElbow", .rightElbow: "rightElbow",
    .leftWrist: "leftWrist", .rightWrist: "rightWrist",
]

let nombresMano: [HumanHandPoseObservation.JointName: String] = [
    .wrist: "wrist",
    .thumbCMC: "thumbCMC", .thumbMP: "thumbMP", .thumbIP: "thumbIP", .thumbTip: "thumbTip",
    .indexMCP: "indexMCP", .indexPIP: "indexPIP", .indexDIP: "indexDIP", .indexTip: "indexTip",
    .middleMCP: "middleMCP", .middlePIP: "middlePIP", .middleDIP: "middleDIP", .middleTip: "middleTip",
    .ringMCP: "ringMCP", .ringPIP: "ringPIP", .ringDIP: "ringDIP", .ringTip: "ringTip",
    .littleMCP: "littleMCP", .littlePIP: "littlePIP", .littleDIP: "littleDIP", .littleTip: "littleTip",
]

func coordenada(_ joint: Vision.Joint) -> [Double] {
    [Double(joint.location.x), Double(joint.location.y)]
}

// MARK: - Lectura del vídeo

let huella: String
do { huella = try md5(entrada) } catch { fallar("no se pudo leer \(entrada.path): \(error)") }

let asset = AVURLAsset(url: entrada)
guard let pista = try? await asset.loadTracks(withMediaType: .video).first else {
    fallar("sin pista de vídeo: \(entrada.path)")
}
let (transform, tamano) = try await pista.load(.preferredTransform, .naturalSize)
guard transform.isIdentity else {
    fallar("el vídeo tiene rotación en sus metadatos (preferredTransform no es la identidad): \(entrada.path)")
}
guard tamano.width > 0, abs(tamano.height / tamano.width - 16.0 / 9.0) < 0.01 else {
    fallar("aspecto no admitido \(Int(tamano.width))×\(Int(tamano.height)); se espera vertical 9:16")
}

let lector: AVAssetReader
do { lector = try AVAssetReader(asset: asset) } catch { fallar("no se pudo abrir el vídeo: \(error)") }
let toma = AVAssetReaderTrackOutput(track: pista, outputSettings: [
    kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
    AVVideoColorPropertiesKey: [
        AVVideoColorPrimariesKey: AVVideoColorPrimaries_ITU_R_709_2,
        AVVideoTransferFunctionKey: AVVideoTransferFunction_ITU_R_709_2,
        AVVideoYCbCrMatrixKey: AVVideoYCbCrMatrix_ITU_R_709_2,
    ],
])
toma.alwaysCopiesSampleData = false
lector.add(toma)
guard lector.startReading() else { fallar("no se pudo leer el vídeo: \(String(describing: lector.error))") }

var transferencia: VTPixelTransferSession?
VTPixelTransferSessionCreate(allocator: nil, pixelTransferSessionOut: &transferencia)
guard let transferencia else { fallar("no se pudo crear la sesión de escalado") }

func reducir(_ origen: CVPixelBuffer) -> CVPixelBuffer? {
    var destino: CVPixelBuffer?
    let atributos = [kCVPixelBufferIOSurfacePropertiesKey as String: [:]] as CFDictionary
    CVPixelBufferCreate(kCFAllocatorDefault, ANCHO, ALTO, kCVPixelFormatType_32BGRA, atributos, &destino)
    guard let destino, VTPixelTransferSessionTransferImage(transferencia, from: origen, to: destino) == noErr else {
        return nil
    }
    return destino
}

let contexto = CIContext()
func volcarPNG(_ imagen: CVPixelBuffer, indice: Int, en carpeta: URL) {
    let ci = CIImage(cvPixelBuffer: imagen)
    guard let cg = contexto.createCGImage(ci, from: ci.extent),
          let destino = CGImageDestinationCreateWithURL(
              carpeta.appendingPathComponent(String(format: "%06d.png", indice)) as CFURL,
              UTType.png.identifier as CFString, 1, nil)
    else { return }
    CGImageDestinationAddImage(destino, cg, nil)
    CGImageDestinationFinalize(destino)
}

if let volcar {
    try? FileManager.default.createDirectory(at: volcar, withIntermediateDirectories: true)
}

// MARK: - Detección (X-3)

let peticionPose = DetectHumanBodyPoseRequest(.revision2)
var peticionManos = DetectHumanHandPoseRequest(.revision1)
peticionManos.maximumHandCount = 2

var fotogramas: [[String: Any]] = []
var indice = 0
while let muestra = toma.copyNextSampleBuffer() {
    defer { indice += 1 }
    guard indice % cada == 0 else { continue }
    guard let original = CMSampleBufferGetImageBuffer(muestra), let imagen = reducir(original) else {
        fotogramas.append(["origen": indice, "fallido": true, "cuerpos": [], "manos": []])
        continue
    }
    if let volcar { volcarPNG(imagen, indice: indice, en: volcar) }
    do {
        let manejador = ImageRequestHandler(imagen, orientation: .up)
        let (cuerpos, manos) = try await manejador.perform(peticionPose, peticionManos)
        let cuerposJSON: [[String: Any]] = cuerpos.map { obs in
            var puntos: [String: [Double]] = [:]
            for (nombre, joint) in obs.allJoints() {
                if let clave = nombresPose[nombre] { puntos[clave] = coordenada(joint) }
            }
            return ["puntos": puntos]
        }
        let manosJSON: [[String: Any]] = manos.map { obs in
            var puntos: [String: [Double]] = [:]
            for (nombre, joint) in obs.allJoints() {
                if let clave = nombresMano[nombre] { puntos[clave] = coordenada(joint) }
            }
            let lado: Any
            switch obs.chirality {
            case .left: lado = "left"
            case .right: lado = "right"
            default: lado = NSNull()
            }
            return ["lado": lado, "puntos": puntos]
        }
        fotogramas.append(["origen": indice, "fallido": false, "cuerpos": cuerposJSON, "manos": manosJSON])
    } catch {
        fotogramas.append(["origen": indice, "fallido": true, "cuerpos": [], "manos": []])
    }
}
guard lector.status == .completed else { fallar("la lectura del vídeo no terminó: \(String(describing: lector.error))") }

// MARK: - Salida (X-4)

let sistema = ProcessInfo.processInfo.operatingSystemVersion
let documento: [String: Any] = [
    "version_extractor": VERSION_EXTRACTOR,
    "detector": ["pose_revision": 2, "manos_revision": 1, "max_manos": 2],
    "macos": "\(sistema.majorVersion).\(sistema.minorVersion).\(sistema.patchVersion)",
    "md5": huella,
    "fotogramas_decodificados": indice,
    "cada": cada,
    "imagen": ["ancho": ANCHO, "alto": ALTO, "color": "bt709-sdr-8bit"],
    "fotogramas": fotogramas,
]
do {
    let json = try JSONSerialization.data(withJSONObject: documento, options: [.sortedKeys])
    let temporal = salida.appendingPathExtension("tmp")
    try FileManager.default.createDirectory(at: salida.deletingLastPathComponent(), withIntermediateDirectories: true)
    try gzip(json).write(to: temporal)
    _ = try FileManager.default.replaceItemAt(salida, withItemAt: temporal)
} catch {
    fallar("no se pudo escribir \(salida.path): \(error)")
}
