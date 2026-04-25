from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentic_layer import AgenticNavigationRouter, FrameContext  # noqa: E402


SITUATIONS = [
    {
        "name": "Safety warning overrides route",
        "context": {
            "timestamp_ms": 1714060200000,
            "frame_id": "safety-route",
            "warnings": [
                {
                    "kind": "person",
                    "message": "person ahead",
                    "severity": "high",
                    "confidence": 0.92,
                    "distance_m": 0.6,
                    "direction": "center",
                }
            ],
            "motion": {"is_moving": True},
            "route": {"active": True, "destination": "library", "next_instruction": "Continue straight"},
            "scene": {"location_type": "sidewalk", "visual_confidence": 0.9},
        },
    },
    {
        "name": "Object search wins over route",
        "context": {
            "timestamp_ms": 1714060201000,
            "frame_id": "target-chair",
            "detections": [
                {"label": "chair", "confidence": 0.86, "distance_m": 1.1, "direction": "slight_right"}
            ],
            "motion": {"is_moving": False},
            "route": {"active": True, "destination": "lobby", "next_instruction": "Walk forward"},
            "scene": {"location_type": "room", "visual_confidence": 0.9},
            "user": {"target": "chair", "mode": "object_search"},
        },
    },
    {
        "name": "Route guidance when path is clear",
        "context": {
            "timestamp_ms": 1714060202000,
            "frame_id": "route-clear",
            "motion": {"is_moving": True},
            "route": {"active": True, "destination": "lobby", "next_instruction": "Turn left"},
            "scene": {"location_type": "hallway", "visual_confidence": 0.9},
        },
    },
    {
        "name": "Orientation while standing still",
        "context": {
            "timestamp_ms": 1714060203000,
            "frame_id": "orientation",
            "detections": [
                {"label": "bench", "confidence": 0.78, "distance_m": 2.0, "direction": "left"},
                {"label": "door", "confidence": 0.91, "distance_m": 3.2, "direction": "center"},
            ],
            "motion": {"is_moving": False},
            "scene": {"location_type": "hallway", "visual_confidence": 0.9},
            "user": {"mode": "orientation"},
        },
    },
    {
        "name": "Low visual confidence fallback",
        "context": {
            "timestamp_ms": 1714060204000,
            "frame_id": "fallback",
            "motion": {"is_moving": True},
            "route": {"active": True, "destination": "lobby", "next_instruction": "Continue forward"},
            "scene": {"location_type": "sidewalk", "visual_confidence": 0.2},
        },
    },
]


def main() -> None:
    router = AgenticNavigationRouter()
    for situation in SITUATIONS:
        ctx = FrameContext.from_dict(situation["context"])
        decision = router.decide(ctx)
        agents = ", ".join(decision.agents_consulted) or "none"
        print(f"{situation['name']}")
        print(f"  action={decision.action.value} priority={decision.priority} haptic={decision.haptic.value}")
        print(f"  agents={agents}")
        print(f"  message={decision.message or '<silent>'}")


if __name__ == "__main__":
    main()
