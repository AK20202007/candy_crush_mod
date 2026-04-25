from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentic_layer import (  # noqa: E402
    AgentAction,
    AgentDecision,
    AgenticNavigationRouter,
    AgenticRuntime,
    Detection,
    Direction,
    FrameContext,
    HapticPattern,
    MotionState,
    SafetyPolicy,
    SceneState,
    SurfaceKind,
    SurfaceObservation,
    UserState,
    load_profile,
    load_profiles,
)


BASE_TS_MS = 1714060300000


def _ctx(**overrides) -> FrameContext:
    values = {
        "timestamp_ms": BASE_TS_MS,
        "frame_id": "architecture-contract",
        "motion": MotionState(is_moving=False),
        "scene": SceneState(location_type="hallway", visual_confidence=0.9),
        "user": UserState(),
    }
    values.update(overrides)
    return FrameContext(**values)


def test_profiles_load_and_expose_expected_runtime_knobs() -> None:
    profiles = load_profiles()

    assert "balanced" in profiles
    assert "indoor_cautious" in profiles
    assert "outdoor_sidewalk" in profiles

    balanced = load_profile("balanced")
    assert balanced.model_path
    assert 0.0 < balanced.conf <= 1.0
    assert balanced.confirm_frames >= 1
    assert balanced.agent_urgent_repeat_ms <= balanced.agent_repeat_ms


def test_safety_policy_blocks_prohibited_crossing_claims() -> None:
    policy = SafetyPolicy()
    ctx = _ctx(scene=SceneState(location_type="street_crossing", visual_confidence=0.9))
    unsafe = AgentDecision(
        action=AgentAction.ANSWER,
        priority=40,
        message="It is safe to cross now.",
        haptic=HapticPattern.NONE,
        agents_consulted=["test_agent"],
    )

    decision = policy.enforce_decision(ctx, unsafe)

    assert decision.action == AgentAction.WARN
    assert decision.priority == 90
    assert decision.requires_human is True
    assert "cannot make that safety call" in decision.message
    assert decision.debug["policy"] == "prohibited safety claim"


def test_router_policy_blocks_guidance_in_high_risk_context() -> None:
    router = AgenticNavigationRouter()
    ctx = _ctx(
        scene=SceneState(location_type="street_crossing", visual_confidence=0.9),
        user=UserState(target="chair"),
        detections=[
            Detection(label="chair", confidence=0.9, distance_m=1.2, direction=Direction.CENTER),
        ],
    )

    decision = router.decide(ctx)

    assert decision.action == AgentAction.WARN
    assert decision.requires_human is True
    assert "crossing aids" in decision.message or "High-risk area" in decision.message
    assert "safe to cross" not in decision.message.lower()


def test_runtime_records_frame_and_decision_events() -> None:
    runtime = AgenticRuntime()
    ctx = _ctx(
        user=UserState(target="chair"),
        detections=[
            Detection(label="chair", confidence=0.9, distance_m=1.2, direction=Direction.SLIGHT_RIGHT),
        ],
    )

    decision = runtime.decide(ctx)
    decision_events = runtime.event_bus.recent("agent_decision")
    frame_events = runtime.event_bus.recent("frame_context")

    assert decision.action == AgentAction.GUIDE
    assert len(frame_events) == 1
    assert len(decision_events) == 1
    assert runtime.state.last_decision is not None
    assert runtime.state.last_decision.action == "guide"
    assert decision_events[0].payload["decision"]["frame_id"] == "architecture-contract"


def test_frame_context_round_trips_surface_observations() -> None:
    ctx = _ctx(
        surfaces=[
            SurfaceObservation(
                kind=SurfaceKind.SIDEWALK,
                confidence=0.62,
                direction=Direction.CENTER,
                near_field_ratio=0.71,
                distance_m=1.0,
                source="test",
            )
        ]
    )

    rebuilt = FrameContext.from_dict(ctx.model_dump())

    assert len(rebuilt.surfaces) == 1
    assert rebuilt.surfaces[0].kind == SurfaceKind.SIDEWALK
    assert rebuilt.surfaces[0].direction == Direction.CENTER
    assert rebuilt.surfaces[0].near_field_ratio == 0.71


def main() -> None:
    test_profiles_load_and_expose_expected_runtime_knobs()
    test_safety_policy_blocks_prohibited_crossing_claims()
    test_router_policy_blocks_guidance_in_high_risk_context()
    test_runtime_records_frame_and_decision_events()
    test_frame_context_round_trips_surface_observations()
    print("architecture contract tests passed")


if __name__ == "__main__":
    main()
