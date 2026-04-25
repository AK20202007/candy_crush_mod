# Assistive Navigation Prototype

Local prototype for blind/low-vision navigation research:

```text
webcam + YOLOv8 + rough distance cues
    -> structured frame context
    -> deterministic agentic router
    -> safety-aware speech and haptic decisions
```

This is not an autonomous navigation or street-crossing system. It describes,
warns, and guides conservatively. It must not claim that it is safe to cross or
safe to move through a high-risk area.

## What It Does

- Runs YOLOv8 on the webcam.
- Converts detections into structured `Detection` and `WarningEvent` objects.
- Routes each frame through specialized agents:
  - safety
  - sidewalk / road / curb surface reasoning
  - crossing signal categorization
  - target finding
  - wayfinding
  - orientation
  - fallback
- Uses `SafetyPolicy` to block unsafe phrasing and high-risk guidance.
- Supports runtime profiles in `config/agentic_profiles.json`.
- Can optionally persist structured agent telemetry to MongoDB.
- Provides deterministic tests and JSON router scenarios that do not require a webcam.

## Setup

Use the project virtual environment:

```bash
cd /Users/maheshk/glasses
.venv/bin/python -B main.py --help
```

If creating a new environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Dependencies:

- `opencv-python`
- `ultralytics`
- `pyttsx3`
- `pymongo` for optional MongoDB telemetry

## Run

Basic:

```bash
.venv/bin/python main.py --profile balanced
```

By default, the app starts continuous destination capture after a codeword. When
prompted, say:

```text
navigate Rieber Hall UCLA stop
```

`navigate` starts capture. `stop` ends capture. For non-interactive tests, pass
the destination directly:

```bash
.venv/bin/python main.py --profile balanced --destination "Central Library"
```

Change the words if needed:

```bash
.venv/bin/python main.py --profile balanced --codeword "computer" --stop-word "done"
```

Use one-shot speech capture without codeword start/stop:

```bash
.venv/bin/python main.py --profile balanced --no-codeword
```

To restore the old terminal prompt explicitly:

```bash
.venv/bin/python main.py --profile balanced --typed-destination
```

Target finding:

```bash
.venv/bin/python main.py --profile indoor_cautious --standing-still --target chair
.venv/bin/python main.py --profile indoor_cautious --standing-still --query "where is the door" --mode object_search
```

Street-crossing policy simulation:

```bash
.venv/bin/python main.py --profile outdoor_sidewalk --location-type street_crossing --standing-still
```

Pedestrian-signal query:

```bash
.venv/bin/python main.py --profile outdoor_sidewalk --location-type street_crossing --standing-still --query "what does the walk sign say"
```

Sidewalk/curb query:

```bash
.venv/bin/python main.py --profile outdoor_sidewalk --standing-still --query "where is the sidewalk"
```

Debug:

```bash
.venv/bin/python main.py --profile debug_fast --target chair
```

Controls:

- Preview window: press `q` to quit.
- Terminal: press Ctrl+C to quit.

## Profiles

Profiles live in:

```text
config/agentic_profiles.json
```

Available profiles:

- `balanced`
- `indoor_cautious`
- `outdoor_sidewalk`
- `low_light`
- `debug_fast`

Profile values can be overridden with CLI flags such as:

```bash
.venv/bin/python main.py --profile indoor_cautious --conf 0.28 --confirm-frames 2 --agent-urgent-repeat-ms 900
```

## Verification

Run before manual testing:

```bash
.venv/bin/python -B tests/test_agentic_router.py
.venv/bin/python -B tests/test_architecture_contracts.py
.venv/bin/python -B tests/run_router_scenarios.py
.venv/bin/python -B tests/test_vision_signal_state.py
.venv/bin/python -B tests/test_voice_input.py
.venv/bin/python -B tests/test_database.py
.venv/bin/python -B main.py --help
```

These checks do not require a webcam, model download, TTS output, or maps API.

## Project Layout

| Path | Role |
|---|---|
| `main.py` | CLI, speech worker, route thread, profile wiring |
| `vision.py` | Webcam, YOLO, detection conversion, warning creation |
| `agentic_layer/models.py` | Data contracts |
| `agentic_layer/agents.py` | Safety, target, route, orientation, fallback agents |
| `agentic_layer/router.py` | Deterministic supervisory router |
| `agentic_layer/policy.py` | Safety thresholds and prohibited-phrase enforcement |
| `agentic_layer/database.py` | Optional MongoDB structured telemetry sink |
| `agentic_layer/runtime.py` | Event bus, shared runtime state, decision traces |
| `agentic_layer/config.py` | Runtime profile loader |
| `tests/scenarios/router_scenarios.json` | Human-readable router scenarios |
| `docs/TECHNICAL_ARCHITECTURE.md` | Architecture contract and milestones |
| `docs/MANUAL_QA_PLAN.md` | Manual safety testing plan |
| `docs/DEPLOYMENT.md` | Local deployment runbook |

## Crossing Signals

`CrossingSignalAgent` categorizes pedestrian and road signals when detections
provide enough information:

- pedestrian signal: `walk`, `don't walk`, `countdown`, or unreadable
- traffic light: `red`, `yellow`, `green`, or unreadable
- stop sign: road-sign information only

The current webcam bridge can add a rough color state for COCO `traffic light`
detections. Reliable pedestrian `walk` / `don't walk` classification still needs
a custom detector or OCR pipeline. The agent always describes signal state only;
it never grants permission to cross.

## Sidewalk / Road / Curb Surfaces

`SidewalkAgent` consumes `SurfaceObservation` objects for `sidewalk`, `road`,
`curb`, and `crosswalk`. The current webcam bridge includes a lightweight
lower-frame color/edge heuristic so you can test the router today:

- dark gray near-field surface can become a conservative `road` observation
- bright gray near-field surface can become a `sidewalk` observation
- strong horizontal lower-frame edges can become a possible `curb`

This is not a production sidewalk detector. For field use, replace or supplement
the heuristic with semantic segmentation plus depth/LiDAR. The router behavior
is already wired so a better perception module can emit the same
`SurfaceObservation` contract.

## MongoDB Telemetry

MongoDB is optional. If configured, the app stores structured frame context and
agent decisions, not raw camera frames:

```bash
export MONGODB_URI="mongodb://localhost:27017"
export MONGODB_DB="glasses_agentic"
export MONGODB_COLLECTION="decision_events"
.venv/bin/python main.py --profile outdoor_sidewalk --destination "Rieber Hall"
```

Use this to verify which agent won, what surfaces/signs were observed, and why a
message was spoken.

## Maps

Optional maps use OpenRouteService:

```bash
export OPENROUTESERVICE_API_KEY=...
.venv/bin/python main.py --origin "longitude,latitude" --profile outdoor_sidewalk
```

Without an API key or origin, the app falls back to mock route prompts.

## Limitations

- YOLO COCO weights do not reliably detect doors, stairs, curbs, elevators, or signs.
- YOLO COCO weights do not classify pedestrian walk/don't-walk signals.
- Sidewalk/road/curb perception is currently a conservative heuristic unless you plug in a segmentation/depth model.
- Distance is a rough bounding-box heuristic, not calibrated depth.
- Motion and scene state are still mostly CLI/profile hints.
- Spoken destination input uses macOS Speech through PyObjC and requires microphone/speech permissions.
- Manual walking tests require a sighted spotter.
- Street crossings are descriptive only and require independent confirmation.
