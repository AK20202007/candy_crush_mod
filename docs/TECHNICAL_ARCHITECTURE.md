# Technical Architecture

This project should be treated as a conservative assistive navigation prototype.
The core safety rule is:

```text
Describe, warn, and guide only within confidence limits. Never claim that a
high-risk action is safe.
```

## Runtime Shape

```text
camera / sensors / route / user intent
        |
        v
perception + surface state builders
        |
        v
FrameContext
        |
        v
AgenticNavigationRouter
        |
        v
SafetyPolicy enforcement
        |
        v
AgentDecision
        |
        v
speech / haptics / telemetry / human escalation
```

The router is deterministic. Free-form LLM output must not decide immediate
hazard or crossing safety.

## Crossing Signal Categorization

Pedestrian crossing signal categorization is handled by a dedicated
`CrossingSignalAgent`. It is intentionally descriptive-only:

```text
signal detector / OCR / traffic-light color heuristic
        |
        v
Detection(label="pedestrian signal", attributes={"signal_state": "dont_walk"})
        |
        v
CrossingSignalAgent
        |
        v
"Pedestrian signal shows don't walk at 12 o'clock. Use crossing aids or ask for
human assistance before moving."
```

Supported categories:

- `pedestrian_signal`
- `traffic_light`
- `stop_sign`

Supported states:

- pedestrian signal: `walk`, `dont_walk`, `countdown`, `unknown`
- traffic light: `red`, `yellow`, `green`, `unknown`
- stop sign: `stop_sign`

The agent must not convert any of these categories into crossing permission.
Even a detected `walk` state remains a description, not a command.

## Main Data Contracts

### Detection

`Detection` is one object or text-like observation from perception.

Required fields:

- `label`
- `confidence`
- `direction`

Recommended fields:

- `bbox`
- `distance_m`
- `attributes.area_ratio`
- `attributes.center_x_ratio`
- `attributes.center_y_ratio`
- `source`

Distance is currently a monocular heuristic. Treat it as a ranking signal unless
the camera/device has been calibrated.

### WarningEvent

`WarningEvent` is a safety-relevant observation that may interrupt guidance.

Important fields:

- `severity`: `critical`, `high`, `medium`, `low`, or `info`
- `confidence`
- `distance_m`
- `direction`
- `observed_at_ms`
- `expires_ms`

Warnings expire when `timestamp_ms - observed_at_ms > expires_ms`.

### SurfaceObservation

`SurfaceObservation` is a walking-surface observation separate from object
detections.

Supported kinds:

- `sidewalk`
- `road`
- `curb`
- `crosswalk`
- `unknown`

Important fields:

- `confidence`
- `direction`
- `near_field_ratio`
- `distance_m`
- `source`

The current webcam implementation produces conservative color/edge heuristic
observations. Production sidewalk navigation should use semantic segmentation
and depth/LiDAR, then emit the same contract.

### FrameContext

`FrameContext` is the full input to the router:

- detections
- warnings
- surfaces
- motion state
- route state
- scene state
- user state
- last spoken message

Synthetic `FrameContext` inputs are the preferred way to test policy behavior.

### AgentDecision

`AgentDecision` is the one output chosen for the frame.

Important fields:

- `action`: `warn`, `guide`, `orient`, `ask`, `answer`, `escalate`, `silent`
- `priority`
- `message`
- `haptic`
- `requires_human`
- `debug`

## Agents

### SafetyAgent

Owns immediate warnings and high-risk contexts.

Responsibilities:

- prioritize explicit `WarningEvent`s
- interrupt route guidance for close centered hazards
- enforce street-crossing/crossing-aid language
- mark high-risk decisions as requiring human judgment

### CrossingSignalAgent

Owns pedestrian crossing signal and road-sign categorization.

Responsibilities:

- categorize pedestrian signals as walk, don't walk, countdown, or unreadable
- categorize traffic lights by color when perception provides that state
- describe stop signs as road-sign information only
- ask the user to point at the far-side signal when the user asks but no signal is visible
- always mark output as requiring human judgment
- never say or imply permission to cross

### SidewalkAgent

Owns walking surface reasoning.

Responsibilities:

- stop when a road-like surface appears centered in the near field while moving
- warn on possible curb edges ahead
- describe sidewalk/crosswalk direction when asked or stationary
- never say the path is safe or clear

### TargetFindingAgent

Owns user requests like "where is the door?"

Responsibilities:

- normalize target names
- ignore weak detections below the policy target confidence floor
- prefer closer usable targets
- ask the user to scan when target confidence is low

### WayfindingAgent

Owns route prompts.

Responsibilities:

- speak route instructions only when higher-priority safety agents do not block
- ask the user to stop and scan when off route

### OrientationAgent

Owns "what is around me?" output.

Responsibilities:

- speak only when stationary, explicitly in orientation mode, or responding to a query
- keep descriptions short and directional

### FallbackAgent

Owns low-confidence camera state.

Responsibilities:

- ask user to slow down when visual confidence is low while moving

## Safety Policy

`agentic_layer/policy.py` centralizes auditable rules:

- warning confidence floors by severity
- target confidence floor
- prohibited safety phrases
- high-risk location types
- post-router decision enforcement

High-risk contexts include:

- `street_crossing`
- `intersection`
- `roadway`
- `curb_edge`
- `platform_edge`

Prohibited safety claims include phrases such as:

- "safe to cross"
- "all clear"
- "clear to cross"
- "go now"
- "cross now"

If a decision contains prohibited language, policy enforcement replaces it with
a conservative warning that requires human judgment.

## Config Profiles

Runtime profiles live in `config/agentic_profiles.json`.

Current profiles:

- `balanced`: default controlled indoor testing
- `indoor_cautious`: indoor hallway/lobby testing with larger model input
- `outdoor_sidewalk`: outdoor sidewalk testing
- `low_light`: low-light evaluation
- `debug_fast`: fast local debugging

The CLI accepts:

```bash
.venv/bin/python main.py --profile indoor_cautious
```

Every profile value can be overridden by explicit CLI flags.

## Runtime And Telemetry

`agentic_layer/runtime.py` provides:

- `AgenticRuntime`: wrapper around the router
- `InMemoryEventBus`: bounded local event store
- `DecisionTrace`: auditable decision summary
- `SharedNavigationState`: route/user/last-decision state

The current camera loop still calls the router directly. The runtime wrapper is
available for tests, replay tools, and future app integration.

`agentic_layer/database.py` provides optional MongoDB persistence. Enable it with:

```bash
export MONGODB_URI="mongodb://localhost:27017"
export MONGODB_DB="glasses_agentic"
export MONGODB_COLLECTION="decision_events"
```

MongoDB documents include structured `FrameContext`, `AgentDecision`, and summary
fields. They do not include raw frames or image crops.

## Evaluation Layers

Use three levels of testing:

1. Synthetic router tests: fast, deterministic, no camera.
2. JSON scenario tests: readable expected behavior traces.
3. Recorded video replay: future work for perception and timing.

Commands:

```bash
.venv/bin/python -B tests/test_agentic_router.py
.venv/bin/python -B tests/test_architecture_contracts.py
.venv/bin/python -B tests/run_router_scenarios.py
.venv/bin/python -B tests/test_vision_signal_state.py
.venv/bin/python -B tests/test_database.py
```

## Remaining Engineering Milestones

### Perception

- custom classes for doors, stairs, curbs, elevators, signs, crosswalks
- custom classes for pedestrian signal states: walk, don't walk, countdown
- OCR observations as structured detections or dedicated text events
- temporal tracking with object identity
- class-specific distance calibration
- sidewalk/road/crosswalk semantic segmentation
- depth/LiDAR integration when available

### Sensor Fusion

- IMU-derived moving/stationary state
- camera pitch/yaw/roll
- GPS/map confidence outdoors
- indoor positioning strategy
- off-route confidence

### Output

- interruptible speech queue
- haptic adapter
- repeat/stop/replay commands
- human assistance escalation adapter

### Deployment

- setup/bootstrap command
- pinned model versions
- profile-specific launch commands
- privacy-preserving logs
- MongoDB replay/dashboard tooling for structured agent telemetry
- crash diagnostics
