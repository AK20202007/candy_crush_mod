from agentic_layer import (
    AgenticNavigationRouter,
    Direction,
    FrameContext,
    MotionState,
    RouteState,
    SceneState,
    WarningEvent,
)


def main() -> None:
    router = AgenticNavigationRouter()
    ctx = FrameContext(
        timestamp_ms=1714060200000,
        frame_id="test",
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
        scene=SceneState(location_type="hallway", visual_confidence=0.9),
    )
    decision = router.decide(ctx)
    print(decision.model_dump())
    assert decision.action.value == "warn"
    assert "Stop" in decision.message


if __name__ == "__main__":
    main()
