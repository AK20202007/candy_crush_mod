# Assistive Navigation Prototype

Minimal local demo: **webcam** + **YOLOv8** object detection + **spoken warnings** + **mock turn-by-turn directions** for visually impaired navigation research.

## What it does

- Opens your laptop webcam and runs **YOLOv8** (`yolov8n.pt`, downloaded automatically on first run).
- Watches for **person**, **chair**, **car**, and **door** if your model exposes a `door` class (standard COCO weights usually do **not** include door; the code still runs and tracks the other classes).
- **Person near the center** of the frame → *"Watch out, person ahead"* (urgent speech).
- **Chair / car / (door)** with a large bounding box (simulated closeness) → *"Obstacle ahead"* (urgent).
- You type a **destination**; the app speaks **mock directions** by default, or **real turn-by-turn steps** if maps are configured.
- Urgent warnings **skip queued** navigation lines so safety messages are not stuck behind a long backlog.

## Setup

Use Python 3.10+ recommended.

```bash
cd "/path/to/LA Hacks Project"
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Or install directly:

```bash
pip install opencv-python ultralytics pyttsx3
```

**Notes**

- First launch downloads `yolov8n.pt` (Ultralytics).
- **macOS**: `pyttsx3` uses system TTS; if you hit driver issues, install / update Xcode command line tools and try again.
- Grant **Camera** (and **Microphone** if prompted) permissions when your OS asks.

## Run

```bash
python main.py
```

Optional flags:

```bash
python main.py --camera 0 --nav-interval 5
```

- `--camera`: webcam index if you have multiple cameras.
- `--nav-interval`: seconds between navigation phrases.
- `--origin`: starting point as `longitude,latitude` (required for real map directions).
- `--maps-provider`: `auto` (default), `google`, or `ors`.

### Real map routing (Google Maps)

1) Enable **Directions API** + **Geocoding API** in your Google Cloud project.
2) Set your API key:

```bash
export GOOGLE_MAPS_API_KEY="your_google_key"
```

3) Run with an origin coordinate:

```bash
python3 main.py --origin=-122.1697,37.4275 --maps-provider=google
```

The app will geocode the destination you type, fetch walking directions, and speak each step.

### Live progress mode (type coordinates)

To advance each step only after you reach the maneuver location, enable live mode:

```bash
python3 main.py --origin=-122.1697,37.4275 --maps-provider=google --live-nav
```

Then type your current location in the terminal as:

```text
lat,lon
```

Example:

```text
37.427500,-122.169700
```

The app advances to the next instruction only after your typed location is close enough to the current maneuver endpoint (default `--arrival-radius-m 14`).

### Real map routing (OpenRouteService)

```bash
export OPENROUTESERVICE_API_KEY="your_ors_key"
python3 main.py --origin=-122.1697,37.4275 --maps-provider=ors
```

### Detection quality (Ultralytics)

The project [jjking00/YOLO-OD](https://github.com/jjking00/YOLO-OD) is an **MMYOLO**-style fork (OpenMMLab: MMCV, MMDetection, config-driven training). It is aimed at training and benchmarks, not a lightweight drop-in next to this demo’s `pip install ultralytics` stack. This repo instead tightens **Ultralytics** inference and warning logic:

- **Class filter**: only person / chair / car / (door) run through NMS, which cuts stray classes and work.
- **Defaults**: slightly higher `--conf` (0.35) and **2-frame confirmation** before speaking to reduce flicker.
- **Optional**: larger `--imgsz`, heavier `--model` (e.g. `yolov8s.pt`), `--augment` (TTA), FP16 on CUDA unless `--no-half`.

```bash
python main.py --model yolov8s.pt --imgsz 960 --confirm-frames 3
python main.py --conf 0.25 --confirm-frames 1   # more sensitive, chattier
```

**Controls**

- **Preview window**: press **`q`** to quit.
- **Terminal**: **Ctrl+C** to quit.

## Project layout

| File            | Role                                              |
|-----------------|---------------------------------------------------|
| `main.py`       | Starts speech worker, navigation thread, vision |
| `vision.py`     | Webcam + YOLO + overlays + warning callbacks      |
| `speech.py`     | `pyttsx3` wrapper, urgent vs normal priority      |
| `navigation.py` | Mock directions for a destination string        |

## Logs

The app prints timestamps of navigation lines, urgent warnings, and vision status to the console so you can follow behavior without relying on audio alone.

## Limitations (prototype)

- Real routing requires API keys (`GOOGLE_MAPS_API_KEY` or `OPENROUTESERVICE_API_KEY`) and a manual `--origin`.
- **Interrupting** speech mid-sentence depends on the `pyttsx3` backend; urgent messages are prioritized **between** phrases and pending navigation lines are dropped when a warning is pending.
- Detection thresholds are simple heuristics (bbox size / center distance), not true depth.
