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
    SurfaceKind,
    SurfaceObservation,
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
    assert decision.message == "chair found left, less than 3 feet away."
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
    assert "ahead" in decision.message


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
    assert "use crossing aids or ask for human assistance" in decision.message
    assert "it is safe to cross now" not in decision.message.lower()
    assert "safe to cross" not in decision.message.lower()
    assert "safe to cross." != decision.message.lower()
    assert decision.debug["policy"] == "crossing-description-only"


def test_crossing_signal_agent_categorizes_pedestrian_dont_walk_signal() -> None:
    router = AgenticNavigationRouter()
    ctx = _ctx(
        detections=[
            Detection(
                label="pedestrian signal",
                confidence=0.92,
                distance_m=6.0,
                direction=Direction.CENTER,
                attributes={"signal_category": "pedestrian_signal", "signal_state": "dont_walk"},
            )
        ],
        motion=MotionState(is_moving=False),
        scene=SceneState(location_type="street_crossing", visual_confidence=0.9),
    )

    decision = router.decide(ctx)

    assert decision.action == AgentAction.WARN
    assert decision.priority == 94
    assert decision.requires_human is True
    assert decision.haptic == HapticPattern.STOP
    assert "Pedestrian signal shows don't walk" in decision.message
    assert "crossing aids" in decision.message
    assert "safe to cross" not in decision.message.lower()
    assert decision.debug["signal_state"] == "dont_walk"


def test_crossing_signal_agent_describes_walk_signal_without_permission() -> None:
    router = AgenticNavigationRouter()
    ctx = _ctx(
        detections=[
            Detection(
                label="walk sign",
                confidence=0.88,
                distance_m=5.0,
                direction=Direction.SLIGHT_RIGHT,
                attributes={"signal_category": "pedestrian_signal", "signal_state": "walk"},
            )
        ],
        motion=MotionState(is_moving=False),
        scene=SceneState(location_type="street_crossing", visual_confidence=0.9),
    )

    decision = router.decide(ctx)

    assert decision.action == AgentAction.WARN
    assert decision.priority == 92
    assert "Pedestrian signal shows walk right" in decision.message
    assert "before moving" in decision.message
    assert "safe to cross" not in decision.message.lower()


def test_crossing_signal_agent_categorizes_traffic_light_state() -> None:
    router = AgenticNavigationRouter()
    ctx = _ctx(
        detections=[
            Detection(
                label="traffic light",
                confidence=0.8,
                distance_m=8.0,
                direction=Direction.LEFT,
                attributes={"signal_category": "traffic_light", "signal_state": "green"},
            )
        ],
        motion=MotionState(is_moving=False),
        scene=SceneState(location_type="street_crossing", visual_confidence=0.9),
    )

    decision = router.decide(ctx)

    assert decision.action == AgentAction.WARN
    assert decision.priority == 92
    assert "Traffic light appears green left" in decision.message
    assert "Vehicle lights may not match pedestrian signals" in decision.message
    assert "safe to cross" not in decision.message.lower()


def test_crossing_signal_agent_treats_stop_sign_as_stop_warning() -> None:
    router = AgenticNavigationRouter()
    ctx = _ctx(
        detections=[
            Detection(
                label="stop sign",
                confidence=0.91,
                distance_m=4.0,
                direction=Direction.CENTER,
                attributes={"signal_category": "stop_sign", "signal_state": "stop_sign"},
            )
        ],
        motion=MotionState(is_moving=True),
        scene=SceneState(location_type="street_crossing", visual_confidence=0.9),
    )

    decision = router.decide(ctx)

    assert decision.action == AgentAction.WARN
    assert decision.priority == 94
    assert decision.haptic == HapticPattern.STOP
    assert decision.requires_human is True
    assert "Stop sign detected ahead" in decision.message
    assert "Stop and confirm surroundings before moving" in decision.message
    assert "safe to cross" not in decision.message.lower()


def test_crossing_signal_agent_prompts_when_user_asks_but_signal_not_visible() -> None:
    router = AgenticNavigationRouter()
    ctx = _ctx(
        user=UserState(query="what does the walk sign say"),
        motion=MotionState(is_moving=False),
        scene=SceneState(location_type="street_crossing", visual_confidence=0.9),
    )

    decision = router.decide(ctx)

    assert decision.action == AgentAction.ASK
    assert decision.priority == 91
    assert decision.requires_human is True
    assert "cannot read a pedestrian signal yet" in decision.message


def test_sidewalk_agent_stops_for_road_surface_ahead() -> None:
    router = AgenticNavigationRouter()
    ctx = _ctx(
        surfaces=[
            SurfaceObservation(
                kind=SurfaceKind.ROAD,
                confidence=0.64,
                direction=Direction.CENTER,
                near_field_ratio=0.56,
                distance_m=0.8,
                source="test",
            )
        ],
        motion=MotionState(is_moving=True, speed_mps=0.8),
        route=RouteState(active=True, destination="Rieber Hall", next_instruction="Continue forward"),
        scene=SceneState(location_type="sidewalk", visual_confidence=0.9),
    )

    decision = router.decide(ctx)

    assert decision.action == AgentAction.WARN
    assert decision.priority == 96
    assert decision.haptic == HapticPattern.STOP
    assert decision.requires_human is True
    assert "Road surface appears ahead" in decision.message
    assert "safe" not in decision.message.lower()


def test_sidewalk_agent_warns_for_curb_edge_ahead() -> None:
    router = AgenticNavigationRouter()
    ctx = _ctx(
        surfaces=[
            SurfaceObservation(
                kind=SurfaceKind.CURB,
                confidence=0.70,
                direction=Direction.CENTER,
                near_field_ratio=0.66,
                distance_m=0.7,
                source="test",
            )
        ],
        motion=MotionState(is_moving=True, speed_mps=0.5),
        scene=SceneState(location_type="sidewalk", visual_confidence=0.9),
    )

    decision = router.decide(ctx)

    assert decision.action == AgentAction.WARN
    assert decision.priority == 95
    assert decision.haptic == HapticPattern.STOP
    assert "possible curb edge ahead" in decision.message


def test_sidewalk_agent_orients_to_visible_sidewalk_when_asked() -> None:
    router = AgenticNavigationRouter()
    ctx = _ctx(
        surfaces=[
            SurfaceObservation(
                kind=SurfaceKind.SIDEWALK,
                confidence=0.61,
                direction=Direction.SLIGHT_LEFT,
                near_field_ratio=0.72,
                distance_m=1.2,
                source="test",
            )
        ],
        user=UserState(query="where is the sidewalk"),
        motion=MotionState(is_moving=False),
        scene=SceneState(location_type="sidewalk", visual_confidence=0.9),
    )

    decision = router.decide(ctx)

    assert decision.action == AgentAction.ORIENT
    assert decision.priority == 52
    assert decision.haptic == HapticPattern.LEFT
    assert "Sidewalk surface appears to continue left" in decision.message
    assert "safe" not in decision.message.lower()


def test_far_person_and_stop_sign_stop_sign_wins() -> None:
    """Person next to a stop sign at the far end of the street.

    Both detections are distant (small bbox, ~4-5 m).  The vision layer
    will NOT promote either into a WarningEvent (not close/large enough).
    The CrossingSignalAgent should pick up the stop sign (priority 94)
    and the person should be ignored entirely.
    """
    router = AgenticNavigationRouter()
    ctx = _ctx(
        detections=[
            Detection(
                label="person",
                confidence=0.72,
                distance_m=5.0,
                direction=Direction.CENTER,
                attributes={"area_ratio": 0.01, "center_x_ratio": 0.50},
            ),
            Detection(
                label="stop sign",
                confidence=0.85,
                distance_m=4.0,
                direction=Direction.SLIGHT_RIGHT,
                attributes={"signal_category": "stop_sign", "signal_state": "stop_sign"},
            ),
        ],
        warnings=[],  # no warnings — both objects are far away
        motion=MotionState(is_moving=True, speed_mps=0.8),
        scene=SceneState(location_type="sidewalk", visual_confidence=0.85),
    )

    decision = router.decide(ctx)

    # CrossingSignalAgent should win
    assert decision.action == AgentAction.WARN
    assert decision.priority == 94
    assert decision.haptic == HapticPattern.STOP
    assert decision.requires_human is True
    assert "Stop sign detected right" in decision.message
    assert "safe to cross" not in decision.message.lower()


def test_far_person_and_stop_sign_with_target_query_stop_sign_still_wins() -> None:
    """Same far-away scene, but user asked 'where is the person?'

    TargetFindingAgent (priority 70) should produce a candidate, but
    CrossingSignalAgent (priority 94) still outranks it.
    """
    router = AgenticNavigationRouter()
    ctx = _ctx(
        detections=[
            Detection(
                label="person",
                confidence=0.72,
                distance_m=5.0,
                direction=Direction.CENTER,
                attributes={"area_ratio": 0.01, "center_x_ratio": 0.50},
            ),
            Detection(
                label="stop sign",
                confidence=0.85,
                distance_m=4.0,
                direction=Direction.SLIGHT_RIGHT,
                attributes={"signal_category": "stop_sign", "signal_state": "stop_sign"},
            ),
        ],
        warnings=[],
        user=UserState(query="where is the person"),
        motion=MotionState(is_moving=True, speed_mps=0.8),
        scene=SceneState(location_type="sidewalk", visual_confidence=0.85),
    )

    decision = router.decide(ctx)

    # Stop sign still wins on priority
    assert decision.action == AgentAction.WARN
    assert decision.priority == 94
    assert "Stop sign" in decision.message

    # But the TargetFinding candidate should appear in debug
    candidates = decision.debug.get("candidates", [])
    target_candidates = [c for c in candidates if "target_finding" in c.get("agents", [])]
    assert len(target_candidates) >= 1, "TargetFindingAgent should have produced a candidate"
    assert target_candidates[0]["priority"] == 70
    assert "person" in target_candidates[0]["message"].lower()


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


def test_universal_proximity_warning_stops_for_any_close_object() -> None:
    router = AgenticNavigationRouter()
    ctx = _ctx(
        motion=MotionState(is_moving=True, speed_mps=0.5),
        detections=[
            Detection(
                label="laptop",  # Not in HAZARD_LABELS
                confidence=0.75,
                distance_m=0.8,
                direction=Direction.CENTER,
                attributes={"area_ratio": 0.03},
            )
        ],
    )

    decision = router.decide(ctx)

    assert decision.action == AgentAction.WARN
    assert decision.priority == 95
    assert decision.haptic == HapticPattern.STOP
    assert decision.message == "Stop: laptop ahead. Move left or right about one step."
    assert decision.debug["reason"] == "universal-proximity-immediate"


def test_partial_edge_hazard_warns_when_object_is_cut_off_by_camera() -> None:
    router = AgenticNavigationRouter()
    ctx = _ctx(
        motion=MotionState(is_moving=True, speed_mps=0.5),
        detections=[
            Detection(
                label="chair",
                confidence=0.46,
                distance_m=1.3,
                direction=Direction.RIGHT,
                attributes={
                    "area_ratio": 0.024,
                    "bottom_y_ratio": 0.86,
                    "edge_contact": ["right"],
                    "edge_truncated": True,
                    "partial_visibility": "frame_edge",
                    "distance_reliability": "low",
                },
            )
        ],
    )

    decision = router.decide(ctx)

    assert decision.action == AgentAction.WARN
    assert decision.priority == 82
    assert decision.haptic == HapticPattern.RIGHT
    assert decision.agents_consulted == ["safety", "partial_edge"]
    assert "partially visible chair right" in decision.message
    assert "distance is uncertain" in decision.message
    assert decision.debug["reason"] == "partial-edge object-distance hazard"


def test_door_handle_guidance_instructs_hand_and_action() -> None:
    router = AgenticNavigationRouter()
    ctx = _ctx(
        motion=MotionState(is_moving=True, speed_mps=0.3),
        surfaces=[
            SurfaceObservation(
                kind=SurfaceKind.DOOR,
                confidence=0.82,
                direction=Direction.SLIGHT_RIGHT,
                distance_m=1.2,
                source="vision-door-handle",
                attributes={
                    "handle_detected": True,
                    "handle_side": "right",
                    "recommended_hand": "right",
                    "handle_height_zone": "waist height",
                    "handle_orientation": "lever_horizontal",
                    "handle_action": "press the lever down, then gently test whether the door pushes or pulls",
                },
            )
        ],
    )

    decision = router.decide(ctx)

    assert decision.action == AgentAction.GUIDE
    assert decision.priority == 76
    assert decision.haptic == HapticPattern.RIGHT
    assert "Door handle on the right" in decision.message
    assert "Use your right" in decision.message
    assert "press the lever down" in decision.message
    assert decision.debug["reason"] == "door-handle-guidance"


def main() -> None:
    test_safety_warning_takes_priority_over_route_guidance()
    test_anti_spam_suppresses_repeated_non_safety_message()
    test_target_finding_guides_to_best_matching_detection()
    test_confirmed_critical_warning_can_speak_below_default_detection_confidence()
    test_repeated_urgent_warning_uses_short_throttle()
    test_expired_warning_does_not_override_route_guidance()
    test_street_crossing_policy_warns_without_safe_to_cross_claim()
    test_crossing_signal_agent_categorizes_pedestrian_dont_walk_signal()
    test_crossing_signal_agent_describes_walk_signal_without_permission()
    test_crossing_signal_agent_categorizes_traffic_light_state()
    test_crossing_signal_agent_treats_stop_sign_as_stop_warning()
    test_crossing_signal_agent_prompts_when_user_asks_but_signal_not_visible()
    test_sidewalk_agent_stops_for_road_surface_ahead()
    test_sidewalk_agent_warns_for_curb_edge_ahead()
    test_sidewalk_agent_orients_to_visible_sidewalk_when_asked()
    test_low_confidence_fallback_asks_user_to_slow_down()
    test_universal_proximity_warning_stops_for_any_close_object()
    test_partial_edge_hazard_warns_when_object_is_cut_off_by_camera()
    test_door_handle_guidance_instructs_hand_and_action()
    test_far_person_and_stop_sign_stop_sign_wins()
    test_far_person_and_stop_sign_with_target_query_stop_sign_still_wins()
    print("agentic router tests passed")


if __name__ == "__main__":
    main()
