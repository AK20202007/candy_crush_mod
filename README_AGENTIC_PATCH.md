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
  safety warning, crossing signal description, target finding, wayfinding, orientation, fallback, or silence
speech layer speaks urgent vs normal based on the chosen action
```

Additional hardening in this version:

- Safety policy is centralized in `agentic_layer/policy.py`.
- Runtime profiles live in `config/agentic_profiles.json`.
- Decision traces and an in-memory event bus live in `agentic_layer/runtime.py`.
- Optional structured MongoDB telemetry lives in `agentic_layer/database.py`.
- Critical warnings use a lower, severity-aware confidence floor after frame confirmation.
- Repeated urgent speech is throttled separately from normal guidance to avoid TTS flooding.
- Warning events can expire via `observed_at_ms` + `expires_ms`.
- Street-crossing decisions are marked `requires_human=True`.
- Pedestrian signals, traffic lights, and stop signs have a dedicated descriptive-only agent.
- Sidewalk, road, curb, and crosswalk surface observations have a dedicated conservative agent.
- Target finding ignores very low-confidence detections and prefers the closer usable match.
- Warning confirmation keys include class/message, direction, and coarse distance bucket.

## Why the distance repo is incorporated this way

`paul-pias/Object-Detection-and-Distance-Measurement` uses YOLOv3/YOLOv4 plus a bounding-box based monocular distance heuristic. This patch keeps your repo's lighter Ultralytics YOLOv8 stack, but adds the same useful idea: every YOLO box becomes a structured `Detection` with:

```json
{
  "label": "chair",
  "confidence": 0.82,
  "direction": "slight_right",
  "distance_m": 1.3,
  "attributes": {
    "area_ratio": 0.16,
    "center_x_ratio": 0.64,
    "center_y_ratio": 0.58
  }
}
```

The heuristic is intentionally treated as rough. Calibrate `--distance-scale` with real measured distances for your camera.

## Example runs

From the repo root:

```bash
cd /path/to/glasses-agentic-build
```

If the repo has a local virtual environment, prefer it:

```bash
.venv/bin/python main.py --help
```

Basic:

```bash
.venv/bin/python main.py --profile balanced
```

The app now uses codeword-based destination capture by voice. Say:

```text
navigate Rieber Hall UCLA stop
```

For repeatable tests, skip voice capture with:

```bash
.venv/bin/python main.py --profile balanced --destination "Central Library"
```

The old terminal prompt is still available only when explicitly requested:

```bash
.venv/bin/python main.py --profile balanced --typed-destination
```

One-shot speech without codeword start/stop is available with:

```bash
.venv/bin/python main.py --profile balanced --no-codeword
```

Look for a target:

```bash
.venv/bin/python main.py --profile indoor_cautious --target chair --standing-still
.venv/bin/python main.py --profile indoor_cautious --query "where is the door" --mode object_search
```

Ask about crossing signals:

```bash
.venv/bin/python main.py --profile outdoor_sidewalk --location-type street_crossing --standing-still --query "what does the walk sign say"
```

Ask about sidewalk surface:

```bash
.venv/bin/python main.py --profile outdoor_sidewalk --standing-still --query "where is the sidewalk"
```

Enable optional MongoDB decision telemetry:

```bash
export MONGODB_URI="mongodb://localhost:27017"
export MONGODB_DB="glasses_agentic"
export MONGODB_COLLECTION="decision_events"
.venv/bin/python main.py --profile outdoor_sidewalk --destination "Rieber Hall"
```

More sensitive detection:

```bash
.venv/bin/python main.py --profile debug_fast --conf 0.25 --confirm-frames 1
```

Use a better YOLO model:

```bash
.venv/bin/python main.py --profile indoor_cautious --model yolov8s.pt --imgsz 960 --confirm-frames 3
```

Distance calibration:

```bash
.venv/bin/python main.py --profile balanced --distance-scale 1.5
```

If the app keeps saying things are too close, lower `--distance-scale`. If it misses close obstacles, raise it or lower `--obstacle_area_ratio` inside `VisionConfig`.

Speech cadence:

```bash
.venv/bin/python main.py --profile balanced --agent-repeat-ms 4500 --agent-urgent-repeat-ms 1000
```

`--agent-repeat-ms` controls repeated normal route/target/orientation speech.
`--agent-urgent-repeat-ms` controls repeated urgent safety speech for the same
message. Keep urgent repeats short enough to remain noticeable, but not so short
that the TTS queue floods.

## Router behavior

Priority order:

1. Existing warning events, such as stairs/curb/vehicle, get top priority.
2. Close centered object-distance hazards interrupt route guidance.
3. Road/curb surface observations can stop or caution before route guidance.
4. Crossing signal descriptions override generic crossing reminders.
5. User target queries like "where is the chair" get guided responses.
6. Route instructions are spoken only when safe enough.
7. Orientation descriptions happen when standing still or in orientation mode.

Street crossing policy is conservative: the router describes/intervenes, but does not claim it is safe to cross.

## Test the router without a webcam

The router tests use synthetic frame contexts only. They do not require a webcam,
YOLO model download, TTS, or route API key.

```bash
python tests/test_agentic_router.py
```

Run architecture contract tests:

```bash
python tests/test_architecture_contracts.py
python tests/test_voice_input.py
python tests/test_vision_signal_state.py
python tests/test_database.py
```

Run named JSON scenarios through the same router:

```bash
python tests/run_router_scenarios.py
```

If `pytest` is available locally, the same coverage can also be run with:

```bash
python -m pytest tests/test_agentic_router.py
```

The focused router coverage checks:

1. Safety warnings win over route guidance.
2. Repeated non-safety messages are suppressed by the anti-spam window.
3. Target finding chooses and guides toward the best matching detection.
4. Critical warnings can speak below the default YOLO threshold once confirmed.
5. Repeated urgent warnings use a shorter throttle.
6. Expired external warnings do not override route guidance.
7. Street crossing output stays conservative and does not claim it is safe to cross.
8. Low visual confidence while moving falls back to a slow-down prompt.
9. Profile loading, policy blocking, and runtime telemetry contracts remain valid.
10. Pedestrian signals, vehicle lights, and stop signs are described without crossing permission.
11. Spoken destination capture retries and reports failure cleanly.
12. Sidewalk/road/curb surface observations route through stop, caution, or orientation behavior.
13. MongoDB telemetry document shape excludes raw images and includes structured context.

## Architecture docs

Concrete architecture and operations docs:

```text
docs/TECHNICAL_ARCHITECTURE.md
docs/MANUAL_QA_PLAN.md
docs/DEPLOYMENT.md
```

## Local development notes

Recommended smoke-test loop:

```bash
python tests/test_agentic_router.py
python tests/test_architecture_contracts.py
python tests/test_voice_input.py
python tests/test_vision_signal_state.py
python tests/test_database.py
python tests/run_router_scenarios.py
python main.py --profile indoor_cautious --standing-still --target chair
```

For camera/manual testing, start with `--standing-still` so target finding and
orientation prompts can be evaluated before moving through a space. Keep
`--visual-confidence` fixed when reproducing low-confidence behavior:

```bash
python main.py --standing-still --target door --visual-confidence 0.25
```

## Subagent task assignment

This patch works well with small, focused subagent assignments because the router
has deterministic inputs and outputs. Suggested task boundaries:

1. Router policy tests: add or adjust synthetic `FrameContext` cases in
   `tests/test_agentic_router.py`.
2. Router scenario fixtures: add user-readable JSON cases under
   `tests/scenarios/` and run them with `tests/run_router_scenarios.py`.
3. Sidewalk perception: replace the current heuristic in `vision.py` with a
   segmentation/depth producer that emits `SurfaceObservation` objects.
4. MongoDB telemetry: build replay/dashboard tooling from `decision_events`
   without storing raw frames.
5. Manual QA notes: run the local commands above and record observed behavior in
   this README.
6. Integration review: inspect how `vision.py` builds detections, warnings, and
   route context before they reach `AgenticNavigationRouter`.
7. Documentation cleanup: keep example commands, expected behavior, and caveats
   current as the patch evolves.

When assigning subagents, keep each one scoped to one behavior such as safety
priority, target finding, anti-spam, crossing policy, or fallback. Ask for a
short report with changed files, commands run, and any assumptions about camera,
model, or route state.

Example subagent prompts:

```text
Audit agentic_layer/router.py and agentic_layer/agents.py for priority inversions.
Do not edit files. Return file/line findings and suggested tests.
```

```text
Own tests/test_agentic_router.py only. Add deterministic tests for one behavior:
urgent warning throttling. Do not modify runtime code.
```

```text
Own vision.py only. Improve warning confirmation tracking without changing speech
or routing policy. List manual camera cases that still need validation.
```
