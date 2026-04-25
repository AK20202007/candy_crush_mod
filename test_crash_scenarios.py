#!/usr/bin/env python3
"""Test script to identify crash scenarios in the updated system."""

import sys
import traceback
import numpy as np
sys.path.append('.')

from agentic_layer.router import AgenticNavigationRouter
from agentic_layer.models import Detection, Direction, MotionState, FrameContext, RouteState, SceneState, UserState, SurfaceKind, SurfaceObservation


def _ctx(**overrides):
    values = {
        "timestamp_ms": 1714060200000,
        "frame_id": "test-frame",
        "motion": MotionState(is_moving=False, speed_mps=0.0),
        "route": RouteState(),
        "scene": SceneState(location_type="hallway", visual_confidence=0.9),
        "user": UserState(),
        "surfaces": [],
    }
    values.update(overrides)
    return FrameContext(**values)


def test_edge_density_crashes():
    """Test edge density surface observations that might crash."""
    router = AgenticNavigationRouter()
    
    print("Testing edge density crash scenarios...")
    
    # Test 1: Surface with None distance
    print("\n1. Surface with None distance:")
    try:
        ctx = _ctx(
            motion=MotionState(is_moving=True, speed_mps=0.5),
            surfaces=[
                SurfaceObservation(
                    kind=SurfaceKind.OBSTACLE_EDGE,
                    confidence=0.6,
                    direction=Direction.CENTER,
                    distance_m=None,  # This could crash
                    source="test",
                )
            ],
        )
        decision = router.decide(ctx)
        print(f"   ✅ No crash, decision: {decision.action.value}")
    except Exception as e:
        print(f"   ❌ Crash: {e}")
        traceback.print_exc()
    
    # Test 2: Surface with very small confidence
    print("\n2. Surface with very small confidence:")
    try:
        ctx = _ctx(
            motion=MotionState(is_moving=True, speed_mps=0.5),
            surfaces=[
                SurfaceObservation(
                    kind=SurfaceKind.OBSTACLE_EDGE,
                    confidence=0.0001,
                    direction=Direction.CENTER,
                    distance_m=0.5,
                    source="test",
                )
            ],
        )
        decision = router.decide(ctx)
        print(f"   ✅ No crash, decision: {decision.action.value}")
    except Exception as e:
        print(f"   ❌ Crash: {e}")
        traceback.print_exc()
    
    # Test 3: Multiple surfaces including edge
    print("\n3. Multiple surfaces including edge:")
    try:
        ctx = _ctx(
            motion=MotionState(is_moving=True, speed_mps=0.5),
            surfaces=[
                SurfaceObservation(
                    kind=SurfaceKind.SIDEWALK,
                    confidence=0.7,
                    direction=Direction.CENTER,
                    distance_m=1.0,
                    source="test",
                ),
                SurfaceObservation(
                    kind=SurfaceKind.OBSTACLE_EDGE,
                    confidence=0.8,
                    direction=Direction.CENTER,
                    distance_m=0.7,
                    source="test",
                ),
            ],
        )
        decision = router.decide(ctx)
        print(f"   ✅ No crash, decision: {decision.action.value}")
    except Exception as e:
        print(f"   ❌ Crash: {e}")
        traceback.print_exc()


def test_partial_detection_crashes():
    """Test partial detection scenarios that might crash."""
    router = AgenticNavigationRouter()
    
    print("\n\nTesting partial detection crash scenarios...")
    
    # Test 1: Detection with None distance
    print("\n1. Detection with None distance:")
    try:
        ctx = _ctx(
            motion=MotionState(is_moving=True, speed_mps=0.5),
            detections=[
                Detection(
                    label="chair",
                    confidence=0.4,  # Low confidence for partial
                    distance_m=None,  # This could crash
                    direction=Direction.CENTER,
                    attributes={"area_ratio": 0.03},
                )
            ],
        )
        decision = router.decide(ctx)
        print(f"   ✅ No crash, decision: {decision.action.value}")
    except Exception as e:
        print(f"   ❌ Crash: {e}")
        traceback.print_exc()
    
    # Test 2: Detection with zero area ratio
    print("\n2. Detection with zero area ratio:")
    try:
        ctx = _ctx(
            motion=MotionState(is_moving=True, speed_mps=0.5),
            detections=[
                Detection(
                    label="table",
                    confidence=0.35,
                    distance_m=0.5,
                    direction=Direction.CENTER,
                    attributes={"area_ratio": 0.0},  # Could cause division by zero
                )
            ],
        )
        decision = router.decide(ctx)
        print(f"   ✅ No crash, decision: {decision.action.value}")
    except Exception as e:
        print(f"   ❌ Crash: {e}")
        traceback.print_exc()
    
    # Test 3: Detection with negative distance (shouldn't happen but test anyway)
    print("\n3. Detection with negative distance:")
    try:
        ctx = _ctx(
            motion=MotionState(is_moving=True, speed_mps=0.5),
            detections=[
                Detection(
                    label="laptop",
                    confidence=0.8,
                    distance_m=-0.5,  # Invalid distance
                    direction=Direction.CENTER,
                    attributes={"area_ratio": 0.03},
                )
            ],
        )
        decision = router.decide(ctx)
        print(f"   ✅ No crash, decision: {decision.action.value}")
    except Exception as e:
        print(f"   ❌ Crash: {e}")
        traceback.print_exc()


def test_vision_system_crashes():
    """Test vision system edge cases."""
    from vision import VisionSystem, VisionConfig
    
    print("\n\nTesting vision system crash scenarios...")
    
    cfg = VisionConfig(camera_mount='head')
    
    # Test with various problematic frames
    test_frames = [
        ("Empty frame", np.zeros((0, 0, 3), dtype=np.uint8)),
        ("Single pixel", np.zeros((1, 1, 3), dtype=np.uint8)),
        ("All white", np.full((480, 640, 3), 255, dtype=np.uint8)),
        ("All black", np.zeros((480, 640, 3), dtype=np.uint8)),
        ("Very noisy", np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)),
    ]
    
    for name, frame in test_frames:
        print(f"\n{name}:")
        try:
            if frame.size == 0:
                print(f"   ⚠️  Skipping empty frame")
                continue
            surfaces = VisionSystem._detect_edge_density_obstacle(
                VisionSystem(config=cfg, router=None, on_decision=None),
                frame, frame.shape[1], frame.shape[0]
            )
            print(f"   ✅ No crash, detected: {surfaces is not None}")
        except Exception as e:
            print(f"   ❌ Crash: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("CRASH SCENARIO TESTING")
    print("=" * 60)
    
    test_edge_density_crashes()
    test_partial_detection_crashes()
    test_vision_system_crashes()
    
    print("\n" + "=" * 60)
    print("Testing complete!")
    print("=" * 60)
