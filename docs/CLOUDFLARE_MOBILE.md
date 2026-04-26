# Cloudflare Mobile Setup

This branch can expose the local Python door-handle detector to the iOS app
through Cloudflare Tunnel. Cloudflare is the public HTTPS transport; the
YOLO/PyTorch model still runs on the local machine.

Cloudflare's quick tunnel mode generates a random `trycloudflare.com` URL and
proxies it to a localhost service.

## 1. Prepare the branch

```bash
cd "/Users/maheshk/Documents/New project 2/candy_crush_mod_main_live_test"
git switch door-handle-detected
git pull origin door-handle-detected
```

## 2. Install dependencies and weights

```bash
"/Users/maheshk/Documents/New project 2/candy_crush_mod_main_backup/.venv/bin/pip" install -r requirements.txt
test -d /tmp/joechencc_door_detection_review || git clone https://github.com/Joechencc/Door_detection /tmp/joechencc_door_detection_review
```

## 3. Start the local detector API

Terminal 1:

```bash
bash scripts/run_cloudflare_vision_api.sh
```

Health check:

```bash
curl http://localhost:8000/health
```

## 4. Start a TryCloudflare tunnel

Terminal 2:

```bash
bash scripts/run_trycloudflare_tunnel.sh
```

Copy the generated `https://...trycloudflare.com` URL. It should behave like:

```text
https://basis-lisa-acc-korean.trycloudflare.com
```

Verify:

```bash
curl https://YOUR-SUBDOMAIN.trycloudflare.com/health
```

## 5. Start the mobile app with that API URL

Terminal 3:

```bash
EXPO_PUBLIC_NAV_API_BASE_URL="https://YOUR-SUBDOMAIN.trycloudflare.com" bash scripts/run_mobile_with_cloudflare_api.sh
```

The app keeps the same Walking Route UI as the main worktree. When configured,
it shows a `cloud ...trycloudflare.com` status pill and displays the URL in the
Cloudflare API panel.

Use a development build for this path; plain Expo Go cannot load the local
`LAHacksVision` native module:

```bash
cd mobile
npx expo run:ios --device
```

## API contract

The local API exposes:

- `GET /health`
- `POST /api/vision/frame`

Frame request JSON:

```json
{
  "image_base64": "<jpeg-or-png-base64>",
  "indoor_start": "yes",
  "scene": { "location_type": "room" },
  "motion": { "is_moving": false, "speed_mps": 0 },
  "route": {
    "active": true,
    "exit_seeking": true,
    "mapping_state": "mapping",
    "next_instruction": "Leave the room first. Stand still, turn 360 degrees slowly, and scan for a door or exit sign."
  }
}
```

Response:

```json
{
  "ok": true,
  "decision": {
    "action": "guide",
    "priority": 92,
    "message": "Door handle detected right..."
  },
  "route": {
    "active": true,
    "exit_seeking": true
  }
}
```

## How camera frames flow

When `EXPO_PUBLIC_NAV_API_BASE_URL` is set, the React Native app passes that URL
to the iOS `LAHacksVision` module. The native module captures camera frames,
POSTs JPEG frames to `/api/vision/frame`, and emits the returned
`decision.message` through the existing `visionWarning` event. If no Cloudflare
URL is configured, the app falls back to the local iOS Vision person-warning
prototype.
