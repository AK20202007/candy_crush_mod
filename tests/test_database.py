from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentic_layer import (  # noqa: E402
    AgentAction,
    AgentDecision,
    Direction,
    FrameContext,
    HapticPattern,
    MotionState,
    SceneState,
    SurfaceKind,
    SurfaceObservation,
    UserState,
)
from agentic_layer.database import MongoTelemetryStore  # noqa: E402


def _ctx() -> FrameContext:
    return FrameContext(
        timestamp_ms=1714060500000,
        frame_id="mongo-test-frame",
        surfaces=[
            SurfaceObservation(
                kind=SurfaceKind.ROAD,
                confidence=0.63,
                direction=Direction.CENTER,
                near_field_ratio=0.52,
                distance_m=0.8,
                source="test",
            )
        ],
        motion=MotionState(is_moving=True),
        scene=SceneState(location_type="sidewalk", visual_confidence=0.9),
        user=UserState(query="directions to Rieber Hall"),
    )


def test_from_env_is_disabled_without_uri() -> None:
    store = MongoTelemetryStore.from_env({})

    assert store.enabled is False
    assert store.unavailable_reason == "MONGODB_URI is unset"


def test_build_document_contains_structured_context_without_images() -> None:
    decision = AgentDecision(
        action=AgentAction.WARN,
        priority=96,
        message="Stop. Road surface appears at 12 o'clock in the near field.",
        haptic=HapticPattern.STOP,
        agents_consulted=["sidewalk"],
        requires_human=True,
    )

    doc = MongoTelemetryStore.build_document(_ctx(), decision)

    assert doc["frame_id"] == "mongo-test-frame"
    assert doc["summary"]["surface_count"] == 1
    assert doc["summary"]["action"] == "warn"
    assert doc["decision"]["haptic"] == "stop"
    assert doc["context"]["surfaces"][0]["kind"] == "road"
    assert "frame" not in doc
    assert "image" not in doc


def main() -> None:
    test_from_env_is_disabled_without_uri()
    test_build_document_contains_structured_context_without_images()
    print("database tests passed")


if __name__ == "__main__":
    main()
