# Manual QA Plan

Manual testing must happen in controlled spaces first. Do not use this prototype
as an independent navigation aid.

## Preflight

Run:

```bash
cd /Users/maheshk/glasses
.venv/bin/python -B tests/test_agentic_router.py
.venv/bin/python -B tests/test_architecture_contracts.py
.venv/bin/python -B tests/run_router_scenarios.py
.venv/bin/python -B tests/test_vision_signal_state.py
.venv/bin/python -B tests/test_voice_input.py
.venv/bin/python -B tests/test_database.py
.venv/bin/python -B main.py --help
```

Confirm:

- webcam permission is granted
- microphone and Speech Recognition permissions are granted
- TTS speaks normally
- MongoDB telemetry is either intentionally disabled or configured
- preview window opens
- `q` exits preview
- Ctrl+C shuts down cleanly

## Controlled Indoor Tests

Use a hallway or room with a sighted spotter.

### Spoken Destination

Command:

```bash
.venv/bin/python main.py --profile balanced --standing-still
```

Expected:

- app asks for codeword, destination, and stop word
- user says "navigate Central Library stop"
- app says "Navigating to ..."
- terminal does not ask for destination text

For repeatable non-interactive tests, use:

```bash
.venv/bin/python main.py --profile balanced --destination "Central Library" --standing-still
```

To test one-shot speech without codeword start/stop:

```bash
.venv/bin/python main.py --profile balanced --standing-still --no-codeword
```

### Stationary Orientation

Command:

```bash
.venv/bin/python main.py --profile indoor_cautious --standing-still --mode orientation
```

Expected:

- short description
- clock-face directions
- no repeated speech flood

Record:

- missed salient objects
- incorrect directions
- confusing wording

### Target Finding

Command:

```bash
.venv/bin/python main.py --profile indoor_cautious --standing-still --target chair
```

Repeat with:

```bash
.venv/bin/python main.py --profile indoor_cautious --standing-still --query "where is the door" --mode object_search
```

Expected:

- if target is detected, output direction and rough distance
- if target is not detected, asks for slow scanning
- no route guidance should interrupt object search unless a hazard appears

### Close Obstacle

Command:

```bash
.venv/bin/python main.py --profile debug_fast --target chair
```

Place a chair or obstacle near the center path.

Expected:

- caution or stop message when centered and close/large
- urgent repeat cadence is noticeable but not continuous
- haptic field printed as `stop` or `caution`

## Stairs / Curb-Like Hazard Test

Only test near stairs with a spotter and no movement toward the hazard.

Expected:

- critical warning wins over route guidance
- wording uses "Stop" or "Caution"
- no claim that movement is safe

## Sidewalk / Road / Curb Surface Test

Use only a controlled sidewalk-like area, driveway, or taped indoor mockup with a
sighted spotter. Do not step into a street for this test.

Stationary surface query:

```bash
.venv/bin/python main.py --profile outdoor_sidewalk --standing-still --query "where is the sidewalk"
```

Expected:

- if a sidewalk-like bright gray surface is visible, the app gives a direction
- if no confident surface is visible, it asks the user to stop and scan
- no output says the path is safe or clear

Moving simulation:

```bash
.venv/bin/python main.py --profile outdoor_sidewalk --destination "test point"
```

Expected:

- a road-like dark near-field surface can trigger `Stop`
- a curb-like horizontal edge can trigger `Caution` or `Stop`
- route guidance is suppressed when the sidewalk agent raises a higher-priority warning

Known limitation:

- current surface perception is a color/edge heuristic; production testing needs
  semantic segmentation plus depth/LiDAR.

## Low Visual Confidence

Command:

```bash
.venv/bin/python main.py --profile low_light
```

Expected:

- user is asked to slow down or hold the camera steady
- no confident target guidance in poor visual conditions

## Street Crossing Policy Test

Do not test in live traffic. Use a simulated context first:

```bash
.venv/bin/python -B tests/run_router_scenarios.py
```

If manually testing near a curb or crossing area:

```bash
.venv/bin/python main.py --profile outdoor_sidewalk --location-type street_crossing --standing-still
```

Expected:

- requires human judgment
- says it can describe what it sees
- never says "safe to cross", "all clear", or "go now"

## Pedestrian Signal Categorization Test

Start with synthetic scenarios:

```bash
.venv/bin/python -B tests/run_router_scenarios.py
```

Manual crossing-signal testing must be stationary and away from live traffic:

```bash
.venv/bin/python main.py --profile outdoor_sidewalk --location-type street_crossing --standing-still --query "what does the walk sign say"
```

Expected:

- if a pedestrian signal is detected, it describes `walk`, `don't walk`,
  `countdown`, or unreadable state
- if only a traffic light is detected, it says vehicle lights may not match
  pedestrian signals
- if no signal is visible, it asks the user to point the camera toward the
  far-side signal and hold steady
- every output requires human judgment
- no output grants permission to cross

## Data To Record

For every manual session, record:

- date/time
- profile
- command
- environment
- model file
- observed false positives
- observed missed hazards
- repeated speech problems
- unsafe or confusing phrases
- approximate latency
- if MongoDB is enabled, the `_id` or timestamp range for the matching decision events

## Stop Conditions

Stop the test immediately if:

- warnings repeat too quickly to understand
- any message says or implies it is safe to cross
- target guidance continues in a high-risk area
- camera confidence is low while the user is moving
- a sighted spotter disagrees with a safety-relevant warning
