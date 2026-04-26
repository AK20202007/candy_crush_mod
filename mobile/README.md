# iOS Refactor (Expo + Native Module Ready)

This folder is an iOS-first mobile refactor path for the existing Python prototype.

It includes:

- React Native + Expo app shell
- Live route-step progression based on location updates
- Google Maps walking route fetch (destination text + API key input in app)
- Coordinate simulation input (`lat,lon`) for quick testing
- Camera-based visual alerts in Expo Go (barcode marker detection)
- Native vision module interface (`LAHacksVision`) so obstacle detection can run in Swift/Kotlin while JS handles route logic and TTS

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

To use native detection, run a development build (not plain Expo Go):

```bash
npx expo prebuild --clean
npx expo run:ios
```

## Cloudflare vision API

The same app UI can use the local Python door-handle detector through a
Cloudflare Tunnel. Start the local API and tunnel from the repo root, then launch
the mobile app with the generated URL:

```bash
bash scripts/run_cloudflare_vision_api.sh
bash scripts/run_trycloudflare_tunnel.sh
EXPO_PUBLIC_NAV_API_BASE_URL="https://YOUR-SUBDOMAIN.trycloudflare.com" bash scripts/run_mobile_with_cloudflare_api.sh
```

With `EXPO_PUBLIC_NAV_API_BASE_URL` set, `LAHacksVision` posts camera frames to
`/api/vision/frame` and emits the returned `decision.message` as the existing
`visionWarning` event. Without that value, it uses the local iOS Vision fallback.

## Notes

- This does not delete or replace the current Python prototype; it is the mobile migration track.
- For App Store release, you still need full iOS app hardening, privacy strings review, and native module implementation.
