from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentic_layer import (  # noqa: E402
    AgentAction,
    AgenticNavigationRouter,
    Detection,
    Direction,
    FrameContext,
    HapticPattern,
    MotionState,
    RouteState,
    SceneState,
    UserState,
    WarningEvent,
)


BASE_TS_MS = 1714060200000


def _ctx(**overrides) -> FrameContext:
    values = {
        "timestamp_ms": BASE_TS_MS,
        "frame_id": "test-frame",
        "motion": MotionState(is_moving=False, speed_mps=0.0),
        "route": RouteState(),
        "scene": SceneState(location_type="hallway", visual_confidence=0.9),
        "user": UserState(),
    }
    values.update(overrides)
    return FrameContext(**values)


def test_safety_warning_takes_priority_over_route_guidance() -> None:
    router = AgenticNavigationRouter()
    ctx = _ctx(
        warnings=[
            WarningEvent(
                kind="stairs",
                message="flight of stairs ahead",
                severity="critical",
                confidence=0.9,
                distance_m=1.0,
                direction=Direction.CENTER,
            )
        ],
        motion=MotionState(is_moving=True, speed_mps=0.8),
        route=RouteState(active=True, destination="lobby", next_instruction="Walk straight"),
    )

    decision = router.decide(ctx)

    assert decision.action == AgentAction.WARN
    assert decision.priority == 100
    assert decision.haptic == HapticPattern.STOP
    assert decision.message.startswith("Stop.")
    assert any(c["action"] == AgentAction.GUIDE.value for c in decision.debug["candidates"])


def test_anti_spam_suppresses_repeated_non_safety_message() -> None:
    router = AgenticNavigationRouter(min_repeat_interval_ms=2000)
    first = _ctx(
        user=UserState(target="chair"),
        detections=[
            Detection(
                label="chair",
                confidence=0.88,
                distance_m=1.2,
                direction=Direction.SLIGHT_RIGHT,
            )
        ],
    )
    second = _ctx(
        timestamp_ms=BASE_TS_MS + 500,
        user=UserState(target="chair"),
        detections=[
            Detection(
                label="chair",
                confidence=0.88,
                distance_m=1.2,
                direction=Direction.SLIGHT_RIGHT,
            )
        ],
    )

    first_decision = router.decide(first)
    second_decision = router.decide(second)

    assert first_decision.action == AgentAction.GUIDE
    assert first_decision.should_speak is True
    assert second_decision.action == AgentAction.SILENT
    assert second_decision.should_speak is False
    assert second_decision.debug["reason"] == "repeat suppressed"


def test_target_finding_guides_to_best_matching_detection() -> None:
    router = AgenticNavigationRouter()
    ctx = _ctx(
        user=UserState(query="where is the chair"),
        detections=[
            Detection(label="chair", confidence=0.70, distance_m=0.9, direction=Direction.LEFT),
            Detection(label="chair", confidence=0.93, distance_m=2.0, direction=Direction.RIGHT),
            Detection(label="door", confidence=0.99, distance_m=1.0, direction=Direction.CENTER),
        ],
    )

    decision = router.decide(ctx)

    assert decision.action == AgentAction.GUIDE
    assert decision.priority == 70
    assert decision.haptic == HapticPattern.LEFT
    assert decision.message == "chair found at 9 o'clock, less than 3 feet away."
    assert decision.debug["target"] == "chair"


def test_confirmed_critical_warning_can_speak_below_default_detection_confidence() -> None:
    router = AgenticNavigationRouter()
    ctx = _ctx(
        warnings=[
            WarningEvent(
                kind="curb",
                message="possible curb ahead",
                severity="critical",
                confidence=0.42,
                distance_m=0.9,
                direction=Direction.CENTER,
            )
        ],
        motion=MotionState(is_moving=True, speed_mps=0.6),
        route=RouteState(active=True, destination="corner", next_instruction="Continue forward"),
    )

    decision = router.decide(ctx)

    assert decision.action == AgentAction.WARN
    assert decision.priority == 100
    assert decision.message.startswith("Stop. Possible curb ahead")
    assert "at 12 o'clock" in decision.message


def test_repeated_urgent_warning_uses_short_throttle() -> None:
    router = AgenticNavigationRouter(min_urgent_repeat_interval_ms=1000)
    warning = WarningEvent(
        kind="stairs",
        message="stairs ahead",
        severity="critical",
        confidence=0.9,
        distance_m=0.8,
        direction=Direction.CENTER,
    )

    first = router.decide(_ctx(warnings=[warning], motion=MotionState(is_moving=True), timestamp_ms=BASE_TS_MS))
    second = router.decide(
        _ctx(warnings=[warning], motion=MotionState(is_moving=True), timestamp_ms=BASE_TS_MS + 500)
    )
    third = router.decide(
        _ctx(warnings=[warning], motion=MotionState(is_moving=True), timestamp_ms=BASE_TS_MS + 1100)
    )

    assert first.action == AgentAction.WARN
    assert second.action == AgentAction.SILENT
    assert second.debug["reason"] == "repeat suppressed"
    assert second.debug["suppressed_decision"]["priority"] == 100
    assert third.action == AgentAction.WARN


def test_expired_warning_does_not_override_route_guidance() -> None:
    router = AgenticNavigationRouter()
    ctx = _ctx(
        timestamp_ms=BASE_TS_MS + 5000,
        warnings=[
            WarningEvent(
                kind="stairs",
                message="stairs ahead",
                severity="critical",
                confidence=0.9,
                distance_m=0.8,
                direction=Direction.CENTER,
                observed_at_ms=BASE_TS_MS,
                expires_ms=1000,
            )
        ],
        motion=MotionState(is_moving=True),
        route=RouteState(active=True, destination="lobby", next_instruction="Turn right"),
    )

    decision = router.decide(ctx)

    assert decision.action == AgentAction.GUIDE
    assert decision.message == "Turn right."


def test_street_crossing_policy_warns_without_safe_to_cross_claim() -> None:
    router = AgenticNavigationRouter()
    ctx = _ctx(
        motion=MotionState(is_moving=True, speed_mps=0.7),
        route=RouteState(active=True, destination="market", next_instruction="Continue north"),
        scene=SceneState(location_type="street_crossing", visual_confidence=0.95),
    )

    decision = router.decide(ctx)

    assert decision.action == AgentAction.WARN
    assert decision.priority == 90
    assert decision.haptic == HapticPattern.CAUTION
    assert decision.requires_human is True
    assert "do not rely on me to decide when it is safe to cross" in decision.message
    assert "it is safe to cross now" not in decision.message.lower()
    assert "safe to cross." != decision.message.lower()
    assert decision.debug["policy"] == "crossing-description-only"


def test_low_confidence_fallback_asks_user_to_slow_down() -> None:
    router = AgenticNavigationRouter()
    ctx = _ctx(
        motion=MotionState(is_moving=True, speed_mps=0.8),
        scene=SceneState(location_type="hallway", visual_confidence=0.2),
    )

    decision = router.decide(ctx)

    assert decision.action == AgentAction.ASK
    assert decision.priority == 75
    assert decision.haptic == HapticPattern.CAUTION
    assert decision.message == "My camera confidence is low. Slow down and hold the phone steady."
    assert decision.debug["visual_confidence"] == 0.2


def main() -> None:
    test_safety_warning_takes_priority_over_route_guidance()
    test_anti_spam_suppresses_repeated_non_safety_message()
    test_target_finding_guides_to_best_matching_detection()
    test_confirmed_critical_warning_can_speak_below_default_detection_confidence()
    test_repeated_urgent_warning_uses_short_throttle()
    test_expired_warning_does_not_override_route_guidance()
    test_street_crossing_policy_warns_without_safe_to_cross_claim()
    test_low_confidence_fallback_asks_user_to_slow_down()
    print("agentic router tests passed")


if __name__ == "__main__":
    main()
