# iOS Refactor (Expo + Native Module Ready)

This folder is an iOS-first mobile refactor path for the existing Python prototype.

It includes:

- React Native + Expo app shell
- Live route-step progression based on location updates
- Google Maps walking route fetch (destination text + API key input in app)
- Coordinate simulation input (`lat,lon`) for quick testing
- Camera-based visual alerts in Expo Go (barcode marker detection)
- Native vision module interface (`LAHacksVision`) so obstacle detection can run in Swift/Kotlin while JS handles route logic and TTS
- Startup voice prompt ("Where do you want to go?") with automatic route loading from spoken destination

## Run

```bash
cd mobile
npm install
npx expo start
```

For iOS simulator/device native builds:

```bash
npx expo run:ios
```

## What is already ported

- The "only speak next instruction when user reaches the maneuver" behavior from Python.
- Haversine distance check with configurable arrival radius and confirmation hits.
- A warning event bridge (`visionWarning`) with urgent/normal levels.
- Real GPS tracking via `expo-location`.
- Camera preview + prototype visual alert trigger via `expo-camera`.
- Spoken-destination capture via `expo-speech-recognition`.

## Native vision module contract

React Native expects a native module named `LAHacksVision` with:

- `start(config: object): Promise<void>`
- `stop(): Promise<void>`
- emits event `"visionWarning"` with payload:
  - `message: string`
  - `level: "urgent" | "normal"`
  - `ts: number`

This lets iOS inference run natively while keeping business logic in TypeScript.

### Current iOS native implementation

This repo now includes a local Expo module at `modules/lahacks-vision`:

- Uses iOS `AVCaptureSession` + `Vision` (`VNDetectHumanRectanglesRequest`)
- Emits `visionWarning` when a person is detected as a likely near/front hazard
- Applies confirmation frames + cooldown to reduce repeated alerts
- If a bundled CoreML detector exists (`LAHacksObjectDetector.mlmodelc`, `YOLOv8n.mlmodelc`, or `YOLOv8.mlmodelc`), the module switches to higher-precision box-based object detection for `person`, `chair`, `car`, and `door`.

To use native detection, run a development build (not plain Expo Go):

```bash
npx expo prebuild --clean
npx expo run:ios
```

### Enable high-precision CoreML object detection

The native module automatically switches to box-based object detection when it finds one of these compiled model bundle names in the iOS app:

- `LAHacksObjectDetector.mlmodelc`
- `YOLOv8n.mlmodelc`
- `YOLOv8.mlmodelc`

#### 1) Prepare a CoreML object detector model

- Preferred model type: `VNRecognizedObjectObservation` output (object detection, not plain image classification).
- Ensure model labels include these class names (or close synonyms):
  - `person`
  - `chair`
  - `car` (or `automobile` / `vehicle`)
  - `door`

#### 2) Add the model to iOS target

1. Open `mobile/ios/LAHacksNav.xcworkspace` in Xcode.
2. Drag your `.mlmodel` into the project navigator.
3. In the add dialog:
   - Check **Copy items if needed**
   - Select target **LAHacksNav**
4. Build once; Xcode compiles it into `.mlmodelc` in app resources.
5. If the model filename differs, rename it in Xcode to one of:
   - `LAHacksObjectDetector.mlmodel`
   - `YOLOv8n.mlmodel`
   - `YOLOv8.mlmodel`

#### 3) Rebuild dev client

```bash
cd mobile
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
npx expo run:ios --device
npx expo start --dev-client
```

#### 4) Verify behavior

- Person in view should trigger: `Watch out, person ahead`.
- Chair/car/door near and confidently detected should trigger:
  - `Obstacle ahead: chair`
  - `Obstacle ahead: car`
  - `Obstacle ahead: door`
- Alerts are still gated by confirm frames + cooldown.

#### Troubleshooting

- If only person fallback seems active, your model likely was not bundled under one of the expected names.
- If no obstacle classes fire:
  - confirm your model is object-detection (not image-classifier-only),
  - check label names/synonyms,
  - lower confidence by changing `minClassificationConfidence` in `src/vision/VisionModule.ts`.

## Notes

- This does not delete or replace the current Python prototype; it is the mobile migration track.
- For App Store release, you still need full iOS app hardening, privacy strings review, and native module implementation.
