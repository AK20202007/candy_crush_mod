#!/usr/bin/env python3
"""
Demo run-through of different situations handled by different agents.

Each scenario simulates a real-world situation and shows which agent
fires, what it says, and why it wins over other candidates.

Run:  python tests/demo_agent_scenarios.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentic_layer import (
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

BASE_TS = 1714060200000


def _ctx(**overrides) -> FrameContext:
    values = {
        "timestamp_ms": BASE_TS,
        "frame_id": "demo",
        "motion": MotionState(is_moving=True, speed_mps=0.8),
        "route": RouteState(),
        "scene": SceneState(location_type="hallway", visual_confidence=0.9),
        "user": UserState(),
    }
    values.update(overrides)
    return FrameContext(**values)


def _print_decision(title: str, description: str, decision):
    bar = "=" * 70
    print(f"\n{bar}")
    print(f"  SCENARIO: {title}")
    print(f"  {description}")
    print(bar)
    print(f"  Winner Agent : {', '.join(decision.agents_consulted)}")
    print(f"  Action       : {decision.action.value}")
    print(f"  Priority     : {decision.priority}")
    print(f"  Haptic       : {decision.haptic.value}")
    print(f"  Requires Human: {decision.requires_human}")
    print(f"  Message      : \"{decision.message}\"")

    candidates = decision.debug.get("candidates", [])
    if candidates:
        print(f"\n  All candidates ({len(candidates)}):")
        for i, c in enumerate(candidates, 1):
            print(f"    {i}. [{c['action']}] priority={c['priority']} by {c['agents']}")
            print(f"       \"{c['message'][:80]}\"")
    print()


def main():
    print("\n" + "#" * 70)
    print("#  AGENTIC LAYER DEMO: 10 Scenarios, 7 Agents")
    print("#" * 70)

    # ----------------------------------------------------------------
    # 1. SafetyAgent: Person walking straight toward you
    # ----------------------------------------------------------------
    router = AgenticNavigationRouter()
    decision = router.decide(_ctx(
        warnings=[
            WarningEvent(
                kind="person",
                message="person directly ahead",
                severity="critical",
                confidence=0.92,
                distance_m=0.8,
                direction=Direction.CENTER,
            )
        ],
        route=RouteState(active=True, destination="lobby", next_instruction="Walk straight"),
    ))
    _print_decision(
        "1. SAFETY AGENT — Person directly ahead",
        "You're walking toward the lobby. A person is 0.8m ahead. Safety overrides route.",
        decision,
    )

    # ----------------------------------------------------------------
    # 2. SafetyAgent: Car approaching from the right
    # ----------------------------------------------------------------
    router = AgenticNavigationRouter()
    decision = router.decide(_ctx(
        warnings=[
            WarningEvent(
                kind="car",
                message="car approaching from right",
                severity="critical",
                confidence=0.88,
                distance_m=1.2,
                direction=Direction.RIGHT,
            )
        ],
    ))
    _print_decision(
        "2. SAFETY AGENT — Car approaching from the right",
        "A car is detected 1.2m to your right while walking on a street.",
        decision,
    )

    # ----------------------------------------------------------------
    # 3. SidewalkAgent: Road surface detected ahead
    # ----------------------------------------------------------------
    router = AgenticNavigationRouter()
    decision = router.decide(_ctx(
        surfaces=[
            SurfaceObservation(
                kind=SurfaceKind.ROAD,
                confidence=0.65,
                direction=Direction.CENTER,
                near_field_ratio=0.55,
                distance_m=0.9,
                source="vision-surface-heuristic",
            )
        ],
        scene=SceneState(location_type="sidewalk", visual_confidence=0.9),
    ))
    _print_decision(
        "3. SIDEWALK AGENT — Road surface ahead",
        "You're on a sidewalk. The camera detects road surface ahead — you may be stepping off the curb.",
        decision,
    )

    # ----------------------------------------------------------------
    # 4. SidewalkAgent: Curb edge detected
    # ----------------------------------------------------------------
    router = AgenticNavigationRouter()
    decision = router.decide(_ctx(
        surfaces=[
            SurfaceObservation(
                kind=SurfaceKind.CURB,
                confidence=0.70,
                direction=Direction.CENTER,
                near_field_ratio=0.66,
                distance_m=0.7,
                source="vision-curb-edge-heuristic",
            )
        ],
        scene=SceneState(location_type="sidewalk", visual_confidence=0.9),
    ))
    _print_decision(
        "4. SIDEWALK AGENT — Curb edge detected",
        "You're walking on a sidewalk and approaching a curb edge.",
        decision,
    )

    # ----------------------------------------------------------------
    # 5. CrossingSignalAgent: Stop sign at an intersection
    # ----------------------------------------------------------------
    router = AgenticNavigationRouter()
    decision = router.decide(_ctx(
        detections=[
            Detection(
                label="stop sign",
                confidence=0.91,
                distance_m=4.0,
                direction=Direction.CENTER,
                attributes={"signal_category": "stop_sign", "signal_state": "stop_sign"},
            )
        ],
        scene=SceneState(location_type="street_crossing", visual_confidence=0.9),
    ))
    _print_decision(
        "5. CROSSING SIGNAL AGENT — Stop sign at intersection",
        "You approach an intersection. Camera sees a stop sign 4m ahead.",
        decision,
    )

    # ----------------------------------------------------------------
    # 6. CrossingSignalAgent: Pedestrian signal says DON'T WALK
    # ----------------------------------------------------------------
    router = AgenticNavigationRouter()
    decision = router.decide(_ctx(
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
    ))
    _print_decision(
        "6. CROSSING SIGNAL AGENT — Pedestrian signal: DON'T WALK",
        "You're at a crosswalk. The pedestrian signal shows don't walk.",
        decision,
    )

    # ----------------------------------------------------------------
    # 7. TargetFindingAgent: User asked "where is the door?"
    # ----------------------------------------------------------------
    router = AgenticNavigationRouter()
    decision = router.decide(_ctx(
        user=UserState(query="where is the door"),
        detections=[
            Detection(label="door", confidence=0.85, distance_m=3.0, direction=Direction.LEFT),
            Detection(label="chair", confidence=0.90, distance_m=1.5, direction=Direction.RIGHT),
        ],
    ))
    _print_decision(
        "7. TARGET FINDING AGENT — User asks 'where is the door?'",
        "User verbally asked for the door. Camera sees a door to the left and a chair to the right.",
        decision,
    )

    # ----------------------------------------------------------------
    # 8. WayfindingAgent: Active route with next instruction
    # ----------------------------------------------------------------
    router = AgenticNavigationRouter()
    decision = router.decide(_ctx(
        route=RouteState(active=True, destination="cafeteria", next_instruction="Turn left at the next corridor"),
    ))
    _print_decision(
        "8. WAYFINDING AGENT — Active route guidance",
        "Navigation is active. No hazards or objects. The route says 'turn left at the next corridor'.",
        decision,
    )

    # ----------------------------------------------------------------
    # 9. OrientationAgent: User standing still, objects visible
    # ----------------------------------------------------------------
    router = AgenticNavigationRouter()
    decision = router.decide(_ctx(
        motion=MotionState(is_moving=False, speed_mps=0.0),
        user=UserState(mode="orientation"),
        detections=[
            Detection(label="door", confidence=0.88, distance_m=2.5, direction=Direction.LEFT),
            Detection(label="chair", confidence=0.75, distance_m=1.0, direction=Direction.RIGHT),
            Detection(label="person", confidence=0.92, distance_m=4.0, direction=Direction.CENTER),
        ],
    ))
    _print_decision(
        "9. ORIENTATION AGENT — Standing still, describe scene",
        "User is standing still in orientation mode. Camera sees door, chair, and person.",
        decision,
    )

    # ----------------------------------------------------------------
    # 10. FallbackAgent: Camera confidence is very low
    # ----------------------------------------------------------------
    router = AgenticNavigationRouter()
    decision = router.decide(_ctx(
        scene=SceneState(location_type="hallway", visual_confidence=0.15),
    ))
    _print_decision(
        "10. FALLBACK AGENT — Low camera confidence",
        "Camera is obstructed or in poor lighting. Visual confidence is only 15%.",
        decision,
    )

    # ----------------------------------------------------------------
    # BONUS: Multi-agent competition — person + stop sign + route
    # ----------------------------------------------------------------
    router = AgenticNavigationRouter()
    decision = router.decide(_ctx(
        warnings=[
            WarningEvent(
                kind="person",
                message="person ahead",
                severity="critical",
                confidence=0.90,
                distance_m=1.0,
                direction=Direction.CENTER,
            )
        ],
        detections=[
            Detection(
                label="stop sign",
                confidence=0.85,
                distance_m=4.0,
                direction=Direction.SLIGHT_RIGHT,
                attributes={"signal_category": "stop_sign", "signal_state": "stop_sign"},
            ),
            Detection(
                label="person",
                confidence=0.90,
                distance_m=1.0,
                direction=Direction.CENTER,
            ),
        ],
        route=RouteState(active=True, destination="cafeteria", next_instruction="Walk straight"),
        scene=SceneState(location_type="street_crossing", visual_confidence=0.85),
    ))
    _print_decision(
        "BONUS: MULTI-AGENT COMPETITION",
        "Person 1m ahead + stop sign 4m away + active route. Who wins?",
        decision,
    )

    print("=" * 70)
    print("  DEMO COMPLETE — All 7 agents demonstrated")
    print("=" * 70)


if __name__ == "__main__":
    main()
