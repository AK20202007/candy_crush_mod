# Product Roadmap

This repository should move forward as a mobile-first assistive navigation
prototype. The Python webcam app is the reference implementation for perception
contracts, safety policy, and agent behavior. Mobile should converge on the same
contracts rather than growing a separate safety stack.

## Product Position

Current status: controlled prototype.

The product must describe, warn, and guide only within confidence limits. It must
not claim that a path, crossing, or traffic interaction is safe.

## Baseline Branch

Use `candy_crush_mod/main` as the baseline for product work. Keep experimental
traffic/crosswalk/vehicle-tracking changes on separate branches until they have
tests and manual validation.

## Milestone 1: Controlled Indoor MVP

Goal: a reliable demo in familiar indoor environments.

Required behavior:

- camera opens and runs continuously
- person/chair/table/couch obstacle warnings interrupt route prompts
- partial-frame objects produce cautious distance language
- route instructions flow through `RouteState.next_instruction`
- door-handle assist can describe side, height, and conservative use guidance
- repeated speech is throttled
- decisions are testable from synthetic `FrameContext` inputs

Exit criteria:

- Python CI passes
- mobile TypeScript CI passes
- no generated artifacts are tracked
- manual QA passes for controlled indoor scenes
- README and UI copy say controlled prototype, not production-ready

## Milestone 2: Mobile Reference Build

Goal: make the phone app the primary user surface.

Required behavior:

- restore the functional Expo app as the default screen
- keep camera preview, GPS progression, speech, and native warning subscription
- native iOS vision emits events that can be mapped to the shared perception contract
- mobile route progression and Python route state use compatible data shapes

Do not duplicate safety policy in ad hoc UI logic. The mobile app should either
port the deterministic router or emit equivalent `FrameContext` data to a shared
router implementation.

## Milestone 3: Dataset And Perception Upgrade

Goal: replace fragile heuristics with measured perception.

Collect labeled indoor and outdoor clips for:

- door handles: lever, knob, vertical pull, push plate, partially occluded
- obstacles: chairs, tables, poles, signs, walls, low-contrast objects
- walking surfaces: sidewalk, road, curb, crosswalk, stairs
- partial-frame hazards at left/right/bottom edges

Track precision, recall, false warnings per minute, missed-critical hazards, and
latency. Door assist should stay feature-gated until it clears manual validation.

## Milestone 4: Outdoor And Traffic Research

Outdoor traffic/crossing support is high risk. Keep it descriptive only:

- describe traffic lights, stop signs, pedestrian signals, and vehicles
- never say "safe to cross", "all clear", "go now", or equivalent language
- require crossing aids or human confirmation for high-risk contexts

Vehicle motion, crosswalk, and traffic-agent experiments should not merge into
the product baseline without tests, annotated clips, and a manual QA report.

## Engineering Rules

- Safety-critical decisions stay deterministic.
- Free-form LLM output must not decide immediate hazards or crossing safety.
- Route guidance always loses to safety warnings.
- Monocular distance is a rough ranking signal until calibrated.
- Generated files, model weights, crash logs, and local keys stay out of Git.
