# Agentic layer patch for `AK20202007/glasses`

Drop these files into the repo root:

```text
agentic_layer/
  __init__.py
  agents.py
  models.py
  router.py
  utils.py
main.py       # replacement
vision.py     # replacement
```

This keeps the existing `speech.py`, `navigation.py`, `routing.py`, `requirements.txt`, and YOLOv8 setup.

## What changed

The old flow was:

```text
vision detects hazard -> vision directly calls speak_urgent("Obstacle ahead")
navigation thread -> directly calls speak_normal(route step)
```

The new flow is:

```text
vision detects objects + estimates rough distance
existing hazards become WarningEvent objects
route thread updates route context instead of speaking directly
AgenticNavigationRouter chooses one action:
  safety warning, target finding, wayfinding, orientation, fallback, or silence
speech layer speaks urgent vs normal based on the chosen action
```

## Why the distance repo is incorporated this way

`paul-pias/Object-Detection-and-Distance-Measurement` uses YOLOv3/YOLOv4 plus a bounding-box based monocular distance heuristic. This patch keeps your repo's lighter Ultralytics YOLOv8 stack, but adds the same useful idea: every YOLO box becomes a structured `Detection` with:

```json
{
  "label": "chair",
  "confidence": 0.82,
  "direction": "slight_right",
  "distance_m": 1.3,
  "attributes": { "area_ratio": 0.16 }
}
```

The heuristic is intentionally treated as rough. Calibrate `--distance-scale` with real measured distances for your camera.

## Example runs

Basic:

```bash
python main.py
```

Look for a target:

```bash
python main.py --target chair --standing-still
python main.py --query "where is the door" --mode object_search
```

More sensitive detection:

```bash
python main.py --conf 0.25 --confirm-frames 1
```

Use a better YOLO model:

```bash
python main.py --model yolov8s.pt --imgsz 960 --confirm-frames 3
```

Distance calibration:

```bash
python main.py --distance-scale 1.5
```

If the app keeps saying things are too close, lower `--distance-scale`. If it misses close obstacles, raise it or lower `--obstacle_area_ratio` inside `VisionConfig`.

## Router behavior

Priority order:

1. Existing warning events, such as stairs/curb/vehicle, get top priority.
2. Close centered object-distance hazards interrupt route guidance.
3. User target queries like "where is the chair" get guided responses.
4. Route instructions are spoken only when safe enough.
5. Orientation descriptions happen when standing still or in orientation mode.

Street crossing policy is conservative: the router describes/intervenes, but does not claim it is safe to cross.

## Test the router without a webcam

```bash
python tests/test_agentic_router.py
```
