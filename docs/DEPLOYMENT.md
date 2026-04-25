# Deployment Runbook

This is a local prototype runbook. It is not a production medical or mobility
device deployment.

## Local Environment

Use the repo virtual environment:

```bash
cd /Users/maheshk/glasses
.venv/bin/python -B main.py --help
```

System `python3` may not have `pyttsx3`, OpenCV, or Ultralytics installed.

## Dependencies

Project dependencies are listed in `requirements.txt`:

```text
opencv-python
ultralytics
pyttsx3
pymongo
```

Install or refresh dependencies:

```bash
.venv/bin/pip install -r requirements.txt
```

## Model Files

The default profile uses:

```text
yolov8n.pt
```

Use a larger model for slower but stronger detection:

```bash
.venv/bin/python main.py --profile indoor_cautious --model yolov8s.pt
```

For production-like testing, pin:

- model filename
- model checksum
- class list
- profile name
- run command

## Runtime Profiles

Profiles live in:

```text
config/agentic_profiles.json
```

Common commands:

```bash
.venv/bin/python main.py --profile balanced
.venv/bin/python main.py --profile balanced --destination "Central Library"
.venv/bin/python main.py --profile indoor_cautious --standing-still --target chair
.venv/bin/python main.py --profile outdoor_sidewalk --location-type street_crossing --standing-still
.venv/bin/python main.py --profile outdoor_sidewalk --standing-still --query "where is the sidewalk"
.venv/bin/python main.py --profile low_light
.venv/bin/python main.py --profile debug_fast --target chair
```

By default, `main.py` captures the destination by voice using the macOS Speech
framework through PyObjC. The default phrase shape is:

```text
navigate Rieber Hall UCLA stop
```

`navigate` starts capture and `stop` ends capture. Use `--codeword` and
`--stop-word` to change those words. Use `--no-codeword` for one-shot speech,
`--destination` for automation, or `--typed-destination` to restore the terminal
prompt.

## Maps

Real route steps require:

```bash
export OPENROUTESERVICE_API_KEY=...
.venv/bin/python main.py --origin "longitude,latitude" --profile outdoor_sidewalk
```

If the key or origin is missing, the app falls back to mock navigation prompts.

## MongoDB Telemetry

MongoDB is optional. When enabled, the app writes structured decision events only:
no raw camera frames, no crops, and no image bytes.

```bash
export MONGODB_URI="mongodb://localhost:27017"
export MONGODB_DB="glasses_agentic"
export MONGODB_COLLECTION="decision_events"
.venv/bin/python main.py --profile outdoor_sidewalk --destination "Rieber Hall"
```

Each document contains:

- full normalized `FrameContext`
- chosen `AgentDecision`
- route, scene, user summary fields
- counts for detections, warnings, and surface observations

Use this to inspect whether `safety`, `sidewalk`, `crossing_signal`,
`target_finding`, or `wayfinding` won for each frame.

## Verification Before Demo

Run:

```bash
.venv/bin/python -B tests/test_agentic_router.py
.venv/bin/python -B tests/test_architecture_contracts.py
.venv/bin/python -B tests/run_router_scenarios.py
.venv/bin/python -B tests/test_vision_signal_state.py
.venv/bin/python -B tests/test_voice_input.py
.venv/bin/python -B tests/test_database.py
.venv/bin/python -B main.py --help
```

Expected:

- router tests pass
- architecture contract tests pass
- JSON scenarios pass
- vision signal/surface heuristic tests pass
- database document-shape tests pass
- CLI help prints available profiles and overrides
- voice destination capture unit tests pass

## Privacy

Do not persist raw camera frames by default.

If adding logs:

- store structured decisions, not images
- redact user queries where possible
- avoid addresses, bystander faces, screens, and documents
- make capture opt-in

## Deployment Targets

The architecture can move to:

- laptop prototype
- phone app
- wearable/glasses edge process
- phone plus wearable camera split

Before moving to mobile/wearable, replace CLI hints with platform signals:

- IMU motion
- camera orientation
- route provider
- microphone or button intent input
- destination voice capture permission flow
- haptic adapter
- pedestrian signal detector or OCR model
- sidewalk/road/crosswalk segmentation and depth model
- traffic-light/pedestrian-signal state confidence

## Release Gate

Do not demo walking use until all are true:

- synthetic router and scenario tests pass
- a sighted spotter has completed manual indoor QA
- urgent repeat cadence is understandable
- street-crossing wording never grants permission to cross
- camera confidence fallback triggers in poor visibility
- pedestrian signal output is descriptive only and requires independent confirmation
- sidewalk/road/curb warnings have been manually checked with a sighted spotter
- known limitations are read aloud or shown before use
