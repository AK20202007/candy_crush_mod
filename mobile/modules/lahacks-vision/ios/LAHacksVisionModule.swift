import AVFoundation
import ExpoModulesCore
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
}
