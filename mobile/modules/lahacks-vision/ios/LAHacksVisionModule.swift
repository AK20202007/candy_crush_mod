import AVFoundation
import CoreImage
import ExpoModulesCore
import UIKit
import Vision

/// `AVCaptureVideoDataOutputSampleBufferDelegate` requires `NSObject`; Expo `Module` does not inherit it, so we use a helper.
private final class VisionSampleBufferDelegate: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
  weak var module: LAHacksVisionModule?

  func captureOutput(
    _ output: AVCaptureOutput,
    didOutput sampleBuffer: CMSampleBuffer,
    from connection: AVCaptureConnection
  ) {
    module?.handleVideoSampleBuffer(sampleBuffer)
  }
}

public final class LAHacksVisionModule: Module {
  private var session: AVCaptureSession?
  private let outputQueue = DispatchQueue(label: "lahacks.vision.output")
  private let sessionQueue = DispatchQueue(label: "lahacks.vision.session")
  private let sampleBufferDelegate = VisionSampleBufferDelegate()
  private var lastInferenceAt: TimeInterval = 0
  private var lastWarningAt: TimeInterval = 0
  private var consecutivePersonHits: Int = 0
  private var minAreaRatio: CGFloat = 0.12
  private var personCenterRadius: CGFloat = 0.22
  private var warningCooldownS: Double = 2.5
  private var confirmFrames: Int = 2
  private var navApiBaseUrl: URL?
  private var cloudFrameIntervalS: Double = 0.75
  private var cloudJpegQuality: CGFloat = 0.55
  private var cloudRequestInFlight = false
  private let ciContext = CIContext()

  public func definition() -> ModuleDefinition {
    Name("LAHacksVision")

    Events("visionWarning")

    OnCreate {
      self.sampleBufferDelegate.module = self
    }

    AsyncFunction("start") { [weak self] (config: [String: Any], promise: Promise) in
      guard let self else {
        promise.reject("E_MODULE", "Module deallocated")
        return
      }
      self.applyConfig(config)
      self.startCapture(promise: promise)
    }

    AsyncFunction("stop") { [weak self] (promise: Promise) in
      guard let self else {
        promise.resolve(nil)
        return
      }
      self.stopCapture()
      promise.resolve(nil)
    }
  }

  private func applyConfig(_ config: [String: Any]) {
    if let value = config["obstacleAreaRatio"] as? CGFloat {
      minAreaRatio = max(0.01, min(0.9, value))
    }
    if let value = config["personCenterRadius"] as? CGFloat {
      personCenterRadius = max(0.05, min(0.7, value))
    }
    if let value = config["warningCooldownS"] as? Double {
      warningCooldownS = max(0.5, min(15, value))
    }
    if let value = config["confirmFrames"] as? Int {
      confirmFrames = max(1, min(8, value))
    }
    if let value = config["navApiBaseUrl"] as? String {
      let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
      if let url = URL(string: trimmed), ["http", "https"].contains(url.scheme?.lowercased() ?? "") {
        navApiBaseUrl = url
      } else {
        navApiBaseUrl = nil
      }
    } else {
      navApiBaseUrl = nil
    }
    if let value = config["cloudFrameIntervalS"] as? Double {
      cloudFrameIntervalS = max(0.25, min(5.0, value))
    }
    if let value = config["cloudJpegQuality"] as? Double {
      cloudJpegQuality = CGFloat(max(0.2, min(0.9, value)))
    }
  }

  private func startCapture(promise: Promise) {
    sessionQueue.async {
      AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
        guard let self else { return }
        guard granted else {
          promise.reject("E_CAMERA_PERMISSION", "Camera permission not granted")
          return
        }
        do {
          try self.setupAndRunSession()
          promise.resolve(nil)
        } catch {
          promise.reject("E_CAMERA_START", "Failed to start camera: \(error.localizedDescription)")
        }
      }
    }
  }

  private func setupAndRunSession() throws {
    stopCapture()
    let session = AVCaptureSession()
    session.beginConfiguration()
    session.sessionPreset = .vga640x480

    guard let camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
      throw NSError(domain: "LAHacksVision", code: 1, userInfo: [NSLocalizedDescriptionKey: "Back camera not found"])
    }

    let input = try AVCaptureDeviceInput(device: camera)
    guard session.canAddInput(input) else {
      throw NSError(domain: "LAHacksVision", code: 2, userInfo: [NSLocalizedDescriptionKey: "Cannot add camera input"])
    }
    session.addInput(input)

    let output = AVCaptureVideoDataOutput()
    output.alwaysDiscardsLateVideoFrames = true
    output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA)]
    output.setSampleBufferDelegate(sampleBufferDelegate, queue: outputQueue)
    guard session.canAddOutput(output) else {
      throw NSError(domain: "LAHacksVision", code: 3, userInfo: [NSLocalizedDescriptionKey: "Cannot add video output"])
    }
    session.addOutput(output)
    session.commitConfiguration()

    self.session = session
    self.lastInferenceAt = 0
    self.lastWarningAt = 0
    self.consecutivePersonHits = 0
    session.startRunning()
  }

  private func stopCapture() {
    if let s = session, s.isRunning {
      s.stopRunning()
    }
    session = nil
    consecutivePersonHits = 0
  }

  fileprivate func handleVideoSampleBuffer(_ sampleBuffer: CMSampleBuffer) {
    let now = CACurrentMediaTime()

    if let navApiBaseUrl {
      handleCloudVideoSampleBuffer(sampleBuffer, baseUrl: navApiBaseUrl, now: now)
      return
    }

    if now - lastInferenceAt < 0.16 {
      return
    }
    lastInferenceAt = now

    guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
      return
    }

    let request = VNDetectHumanRectanglesRequest { [weak self] request, error in
      guard let self else { return }
      if error != nil {
        return
      }
      guard let observations = request.results as? [VNHumanObservation], !observations.isEmpty else {
        self.consecutivePersonHits = 0
        return
      }

      let frameHasHazard = observations.contains { obs in
        let bb = obs.boundingBox
        let area = bb.width * bb.height
        let cx = bb.midX
        let cy = bb.midY
        let dx = cx - 0.5
        let dy = cy - 0.5
        let dist = sqrt(dx * dx + dy * dy)
        return area >= self.minAreaRatio || dist <= self.personCenterRadius
      }

      if frameHasHazard {
        self.consecutivePersonHits += 1
      } else {
        self.consecutivePersonHits = 0
      }
      if self.consecutivePersonHits < self.confirmFrames {
        return
      }
      if now - self.lastWarningAt < self.warningCooldownS {
        return
      }
      self.lastWarningAt = now
      self.sendEvent(
        "visionWarning",
        [
          "message": "Watch out, person ahead",
          "level": "urgent",
          "ts": Date().timeIntervalSince1970 * 1000
        ]
      )
      self.consecutivePersonHits = 0
    }
    request.upperBodyOnly = false
    let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up)
    try? handler.perform([request])
  }

  private func handleCloudVideoSampleBuffer(_ sampleBuffer: CMSampleBuffer, baseUrl: URL, now: TimeInterval) {
    if cloudRequestInFlight || now - lastInferenceAt < cloudFrameIntervalS {
      return
    }
    lastInferenceAt = now
    cloudRequestInFlight = true

    guard let imageBase64 = jpegBase64(from: sampleBuffer) else {
      cloudRequestInFlight = false
      return
    }

    var request = URLRequest(url: baseUrl.appendingPathComponent("api/vision/frame"))
    request.httpMethod = "POST"
    request.timeoutInterval = 8
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")

    let payload: [String: Any] = [
      "image_base64": imageBase64,
      "indoor_start": "yes",
      "scene": [
        "location_type": "room"
      ],
      "motion": [
        "is_moving": false,
        "speed_mps": 0
      ],
      "route": [
        "active": true,
        "exit_seeking": true,
        "mapping_state": "mapping",
        "next_instruction": "Leave the room first. Stand still, turn 360 degrees slowly, and scan for a door or exit sign."
      ]
    ]

    do {
      request.httpBody = try JSONSerialization.data(withJSONObject: payload)
    } catch {
      cloudRequestInFlight = false
      return
    }

    URLSession.shared.dataTask(with: request) { [weak self] data, response, error in
      guard let self else {
        return
      }
      defer {
        self.cloudRequestInFlight = false
      }

      guard error == nil else {
        return
      }
      guard
        let httpResponse = response as? HTTPURLResponse,
        (200..<300).contains(httpResponse.statusCode),
        let data
      else {
        return
      }
      if let eventPayload = self.warningEventPayload(from: data) {
        self.emitCloudWarning(eventPayload)
      }
    }.resume()
  }

  private func jpegBase64(from sampleBuffer: CMSampleBuffer) -> String? {
    guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
      return nil
    }
    let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
    guard let cgImage = ciContext.createCGImage(ciImage, from: ciImage.extent) else {
      return nil
    }
    guard let jpegData = UIImage(cgImage: cgImage).jpegData(compressionQuality: cloudJpegQuality) else {
      return nil
    }
    return jpegData.base64EncodedString()
  }

  private func warningEventPayload(from data: Data) -> [String: Any]? {
    guard
      let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
      let decision = root["decision"] as? [String: Any],
      (decision["should_speak"] as? Bool) ?? true,
      let rawMessage = decision["message"] as? String
    else {
      return nil
    }

    let message = rawMessage.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !message.isEmpty else {
      return nil
    }

    let priority = (decision["priority"] as? NSNumber)?.intValue ?? 50
    return [
      "message": message,
      "level": priority >= 80 ? "urgent" : "normal",
      "ts": Date().timeIntervalSince1970 * 1000
    ]
  }

  private func emitCloudWarning(_ payload: [String: Any]) {
    let now = CACurrentMediaTime()
    if now - lastWarningAt < warningCooldownS {
      return
    }
    lastWarningAt = now
    DispatchQueue.main.async {
      self.sendEvent("visionWarning", payload)
    }
  }
}
