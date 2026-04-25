#!/usr/bin/env python3
"""Test script to verify universal proximity warning catches unknown close objects."""

import sys
sys.path.append('.')

from agentic_layer.router import AgenticNavigationRouter
from agentic_layer.models import Detection, Direction, MotionState, AgentAction, HapticPattern, FrameContext, RouteState, SceneState, UserState


def _ctx(**overrides):
    values = {
        "timestamp_ms": 1714060200000,
        "frame_id": "test-frame",
        "motion": MotionState(is_moving=False, speed_mps=0.0),
        "route": RouteState(),
        "scene": SceneState(location_type="hallway", visual_confidence=0.9),
        "user": UserState(),
    }
    values.update(overrides)
    return FrameContext(**values)


def test_universal_proximity():
    router = AgenticNavigationRouter()
    
    # Test 1: Unknown object (laptop) at 0.8m should trigger STOP
    print("Test 1: Unknown object (laptop) at 0.8m")
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
    print(f"  Action: {decision.action.value}")
    print(f"  Priority: {decision.priority}")
    print(f"  Message: {decision.message}")
    print(f"  Reason: {decision.debug.get('reason', 'N/A')}")
    print(f"  Expected: STOP, Priority 95, 'universal-proximity-immediate'")
    assert decision.action == AgentAction.WARN
    assert decision.priority == 95
    assert decision.debug["reason"] == "universal-proximity-immediate"
    print("  ✓ PASS\n")
    
    # Test 2: Unknown object at 1.5m should NOT trigger (too far)
    print("Test 2: Unknown object (backpack) at 1.5m")
    ctx = _ctx(
        motion=MotionState(is_moving=True, speed_mps=0.5),
        detections=[
            Detection(
                label="backpack",
                confidence=0.75,
                distance_m=1.5,
                direction=Direction.CENTER,
                attributes={"area_ratio": 0.03},
            )
        ],
    )
    
    decision = router.decide(ctx)
    print(f"  Action: {decision.action.value}")
    print(f"  Priority: {decision.priority}")
    print(f"  Expected: SILENT (too far)")
    assert decision.action.value == "silent"
    print("  ✓ PASS\n")
    
    # Test 3: Known hazard (chair) at 0.8m should trigger via hazard logic, not universal
    print("Test 3: Known hazard (chair) at 0.8m")
    ctx = _ctx(
        motion=MotionState(is_moving=True, speed_mps=0.5),
        detections=[
            Detection(
                label="chair",
                confidence=0.75,
                distance_m=0.8,
                direction=Direction.CENTER,
                attributes={"area_ratio": 0.03},
            )
        ],
    )
    
    decision = router.decide(ctx)
    print(f"  Action: {decision.action.value}")
    print(f"  Priority: {decision.priority}")
    print(f"  Reason: {decision.debug.get('reason', 'N/A')}")
    print(f"  Expected: STOP, Priority 95, 'object-distance hazard' (not universal)")
    assert decision.action == AgentAction.WARN
    assert decision.priority == 95
    assert decision.debug["reason"] == "object-distance hazard"
    print("  ✓ PASS\n")
    
    # Test 4: Unknown object with low confidence should NOT trigger
    print("Test 4: Unknown object (cup) at 0.8m with low confidence")
    ctx = _ctx(
        motion=MotionState(is_moving=True, speed_mps=0.5),
        detections=[
            Detection(
                label="cup",
                confidence=0.45,  # Below 0.60 threshold
                distance_m=0.8,
                direction=Direction.CENTER,
                attributes={"area_ratio": 0.03},
            )
        ],
    )
    
    decision = router.decide(ctx)
    print(f"  Action: {decision.action.value}")
    print(f"  Priority: {decision.priority}")
    print(f"  Expected: SILENT (confidence too low)")
    assert decision.action.value == "silent"
    print("  ✓ PASS\n")
    
    # Test 5: Unknown object off-center should NOT trigger
    print("Test 5: Unknown object (book) at 0.8m but far right")
    ctx = _ctx(
        motion=MotionState(is_moving=True, speed_mps=0.5),
        detections=[
            Detection(
                label="book",
                confidence=0.75,
                distance_m=0.8,
                direction=Direction.RIGHT,  # Not ahead
                attributes={"area_ratio": 0.03},
            )
        ],
    )
    
    decision = router.decide(ctx)
    print(f"  Action: {decision.action.value}")
    print(f"  Priority: {decision.priority}")
    print(f"  Expected: SILENT (not ahead)")
    assert decision.action.value == "silent"
    print("  ✓ PASS\n")
    
    print("All universal proximity tests passed! ✅")


if __name__ == "__main__":
    test_universal_proximity()
