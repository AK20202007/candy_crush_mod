from __future__ import annotations

import re
from typing import Iterable, List, Optional

from .models import AgentAction, AgentDecision, Detection, Direction, FrameContext, HapticPattern, INDOOR_LOCATION_TYPES, OUTDOOR_LOCATION_TYPES, SurfaceKind, SurfaceObservation
from .policy import SafetyPolicy

HAZARD_LABELS = {
    "stairs",
    "stair",
    "staircase",
    "curb",
    "hole",
    "dropoff",
    "drop-off",
    "vehicle",
    "car",
    "bus",
    "truck",
    "bicycle",
    "motorcycle",
    "person",
    "chair",
    "bench",
    "dining table",
    "couch",
    "traffic cone",
    "pothole",
    "door",
}

INDOOR_SCAN_CONTEXT_LABELS = {
    "person",
    "chair",
    "bench",
    "dining table",
    "table",
    "couch",
    "door",
    "stairs",
    "stair",
    "staircase",
}

STOP_HAZARDS = {
    "stairs",
    "stair",
    "staircase",
    "curb",
    "hole",
    "dropoff",
    "drop-off",
    "vehicle",
    "car",
    "bus",
    "truck",
}

TARGET_KEYWORDS = {
    "door": ["door", "entrance", "exit"],
    "stairs": ["stairs", "stair", "steps", "staircase"],
    "elevator": ["elevator", "lift"],
    "restroom": ["restroom", "bathroom", "toilet"],
    "chair": ["chair", "seat", "bench"],
    "person": ["person", "people", "human"],
    "sign": ["sign", "text", "label", "stop sign"],
    "traffic light": ["traffic light", "signal"],
}

SEVERITY_SCORE = {"info": 1, "low": 2, "medium": 3, "high": 4, "critical": 5}
WALKING_STEP_M = 0.75
CROSSING_SIGNAL_LABELS = {
    "pedestrian signal",
    "pedestrian crossing signal",
    "crosswalk signal",
    "walk signal",
    "walk sign",
    "dont walk sign",
    "don't walk sign",
    "do not walk sign",
    "countdown signal",
    "traffic light",
    "traffic signal",
    "stop sign",
}
SIGNAL_QUERY_TERMS = {
    "signal",
    "walk sign",
    "walking sign",
    "pedestrian sign",
    "pedestrian signal",
    "crosswalk signal",
    "traffic light",
    "stop sign",
    "light",
}
SIDEWALK_QUERY_TERMS = {
    "sidewalk",
    "walkway",
    "path",
    "curb",
    "road",
    "street",
    "crosswalk",
    "walking surface",
}


def _distance_phrase(distance_m: Optional[float]) -> str:
    if distance_m is None:
        return "unknown distance"
    if distance_m < 0.5:
        return "less than 2 feet away"
    elif distance_m < 1.0:
        return "less than 3 feet away"
    elif distance_m < 1.5:
        return "less than 5 feet away"
    elif distance_m < 2.5:
        return "about 6 feet away"
    elif distance_m < 4.0:
        return "about 10 feet away"
    elif distance_m < 6.0:
        return "about 15 feet away"
    elif distance_m < 10.0:
        return "about 20 feet away"
    else:
        return "more than 30 feet away"


def _walking_steps_phrase(distance_m: Optional[float]) -> str:
    if distance_m is None:
        return ""
    steps = max(1, int(round(distance_m / WALKING_STEP_M)))
    unit = "step" if steps == 1 else "steps"
    return f"in about {steps} {unit}"


def _route_instruction_message(ctx: FrameContext) -> str:
    route = ctx.route
    instruction = route.next_instruction or "Continue toward your destination"
    distance = ""
    if route.next_turn_distance_m is not None:
        distance = f" {_walking_steps_phrase(route.next_turn_distance_m)}"
    message = f"{instruction}{distance}."
    if getattr(route, "exit_seeking", False) and getattr(route, "pending_outdoor_instruction", None):
        outdoor_distance = ""
        pending_distance = getattr(route, "pending_outdoor_distance_m", None)
        if pending_distance is not None:
            outdoor_distance = f" {_walking_steps_phrase(pending_distance)}"
        message += f" After you are outside, first outdoor direction: {route.pending_outdoor_instruction}{outdoor_distance}."
    return message


def _speed_phrase(speed_mps: float) -> str:
    """Convert speed in m/s to human-readable phrase."""
    if speed_mps < 1.0:
        return "walking speed"
    elif speed_mps < 3.0:
        return "slow speed"
    elif speed_mps < 8.0:
        return "moderate speed"
    elif speed_mps < 15.0:
        return "fast speed"
    else:
        return "very high speed"


def _direction_phrase(direction: Direction) -> str:
    return {
        Direction.LEFT: "left",
        Direction.SLIGHT_LEFT: "left",
        Direction.CENTER: "ahead",
        Direction.SLIGHT_RIGHT: "right",
        Direction.RIGHT: "right",
        Direction.UNKNOWN: "ahead",
    }[direction]


def _avoidance_phrase(direction: Direction, distance_m: Optional[float] = None, label: Optional[str] = None) -> str:
    """Return a concrete phrase telling the user how to avoid the obstacle.

    Uses body-oriented language (extend hand, trail wall, step around) and
    includes approximate step counts so the user can judge how far to move.
    """
    avoid_dir = {
        Direction.LEFT: "right",
        Direction.SLIGHT_LEFT: "right",
        Direction.CENTER: "right",
        Direction.SLIGHT_RIGHT: "left",
        Direction.RIGHT: "left",
        Direction.UNKNOWN: "right",
    }[direction]

    hand = "left" if avoid_dir == "right" else "right"

    if distance_m is not None and distance_m <= 0.6:
        return f"Stop. Extend your {hand} hand to feel the obstacle, then sidestep {avoid_dir}."
    if distance_m is not None and distance_m <= 1.0:
        return f"Slow down. Step one pace to the {avoid_dir} to clear it."

    steps = _estimated_clearance_steps(distance_m, label)
    if steps and steps > 1:
        return f"Move {steps} steps to the {avoid_dir} to go around."
    return f"Step to the {avoid_dir} to go around."


def _estimated_clearance_steps(distance_m: Optional[float], label: Optional[str] = None) -> Optional[int]:
    """Estimate how many steps the user needs to clear the obstacle."""
    if label:
        width_hint = {
            "dining table": 3, "couch": 3, "bench": 2, "chair": 2, "car": 4,
            "bus": 5, "truck": 4, "bicycle": 2, "motorcycle": 2,
        }.get(label.lower())
        if width_hint is not None:
            return width_hint
    if distance_m is not None and distance_m > 0:
        return max(1, int(round(distance_m / 0.75)))
    return None


def _haptic_for_direction(direction: Direction, urgent: bool = False) -> HapticPattern:
    if urgent:
        return HapticPattern.STOP
    if direction in {Direction.LEFT, Direction.SLIGHT_LEFT}:
        return HapticPattern.LEFT
    if direction in {Direction.RIGHT, Direction.SLIGHT_RIGHT}:
        return HapticPattern.RIGHT
    return HapticPattern.CAUTION


class BaseAgent:
    name = "base"

    def __init__(self, policy: Optional[SafetyPolicy] = None) -> None:
        self.policy = policy or SafetyPolicy()

    def handle(self, ctx: FrameContext) -> Optional[AgentDecision]:
        raise NotImplementedError


class SafetyAgent(BaseAgent):
    """Highest-priority agent. Existing explicit warnings win over everything else."""

    name = "safety"

    def handle(self, ctx: FrameContext) -> Optional[AgentDecision]:
        valid_warnings = [warning for warning in ctx.warnings if not warning.is_expired(ctx.timestamp_ms)]
        if valid_warnings:
            high = sorted(
                valid_warnings,
                key=lambda w: (
                    1 if w.is_immediate() else 0,
                    SEVERITY_SCORE.get(w.severity, 3),
                    w.confidence,
                    -(w.distance_m if w.distance_m is not None else 99.0),
                ),
                reverse=True,
            )[0]
            if self.policy.warning_confidence_ok(high.severity, high.confidence) and (ctx.motion.is_moving or high.is_immediate()):
                prefix = "Stop." if high.is_immediate() or high.severity == "critical" else "Caution."
                avoidance = _avoidance_phrase(high.direction, high.distance_m, high.kind) if high.direction else ""
                return AgentDecision(
                    action=AgentAction.WARN,
                    priority=100 if high.is_immediate() else 85,
                    message=f"{prefix} {_warning_message(high.message or high.kind, high.direction, high.distance_m)} {avoidance}".strip(),
                    haptic=HapticPattern.STOP if high.is_immediate() else _haptic_for_direction(high.direction),
                    agents_consulted=[self.name],
                    debug={"warning": high.model_dump(), "reason": "existing warning prioritized"},
                )

        hazards: List[Detection] = []
        for det in ctx.detections:
            label = det.label.lower()
            partial_edge = _is_partial_edge_detection(det)
            min_confidence = 0.42 if partial_edge else 0.50
            if label not in HAZARD_LABELS or det.confidence < min_confidence:
                continue
            close = det.distance_m is not None and det.distance_m <= 1.4
            large = float(det.attributes.get("area_ratio", 0.0) or 0.0) >= 0.12
            centered = det.direction in {
                Direction.CENTER,
                Direction.SLIGHT_LEFT,
                Direction.SLIGHT_RIGHT,
                Direction.UNKNOWN,
            }
            if (centered and (close or large)) or _partial_edge_hazard(det):
                hazards.append(det)

        if hazards and ctx.motion.is_moving:
            det = sorted(hazards, key=_hazard_sort_key)[0]
            partial_edge = _is_partial_edge_detection(det)
            is_stop = (
                det.label.lower() in STOP_HAZARDS
                or (det.distance_m is not None and det.distance_m <= 0.8)
                or (det.label.lower() == "chair" and det.distance_m is not None and det.distance_m <= 0.6)
                or _partial_edge_requires_stop(det)
            )
            subject = f"partially visible {det.label}" if partial_edge else det.label
            avoidance = _avoidance_phrase(det.direction, det.distance_m, det.label)
            # Omit distance for immediate obstacles (≤1.0m) for brevity
            distance_str = "" if (det.distance_m is not None and det.distance_m <= 1.0) else f", {_detection_distance_phrase(det)}"
            return AgentDecision(
                action=AgentAction.WARN,
                priority=95 if is_stop else 82 if partial_edge else 80,
                message=(
                    f"{'Stop' if is_stop else 'Caution'}: {subject} "
                    f"{_direction_phrase(det.direction)}{distance_str}. {avoidance}"
                ),
                haptic=_haptic_for_direction(det.direction, urgent=is_stop),
                agents_consulted=[self.name, "partial_edge" if partial_edge else "object_distance"],
                debug={
                    "detection": det.model_dump(),
                    "reason": "partial-edge object-distance hazard" if partial_edge else "object-distance hazard",
                },
            )

        # Universal proximity warning: ANY object within immediate range triggers a stop
        # This catches unknown obstacles (pillars, walls, luggage) not in HAZARD_LABELS
        # Uses lower confidence for very close objects to catch partial detections
        if ctx.motion.is_moving:
            immediate: List[Detection] = []
            for det in ctx.detections:
                # Skip known hazards (already handled above) to avoid duplicate processing
                if det.label.lower() in HAZARD_LABELS:
                    continue
                partial_edge = _is_partial_edge_detection(det)
                # Only consider objects directly ahead (center/slight-left/slight-right)
                centered = det.direction in {Direction.CENTER, Direction.SLIGHT_LEFT, Direction.SLIGHT_RIGHT, Direction.UNKNOWN}
                if not centered and not _partial_edge_hazard(det):
                    continue
                # Use configurable proximity threshold (default 1.0m)
                threshold_m = 1.4 if partial_edge else 1.0  # TODO: Pass VisionConfig to agents for configurable threshold
                close = det.distance_m is not None and det.distance_m <= threshold_m
                if not close and not _partial_edge_hazard(det):
                    continue
                # Exclude tiny detections (likely false positives)
                area_ratio = float(det.attributes.get("area_ratio", 0.0) or 0.0)
                not_tiny = area_ratio >= (0.015 if partial_edge else 0.02)
                if not not_tiny:
                    continue
                # Tiered confidence: lower threshold for very close objects (partial detections)
                # Very close (≤0.6m): accept confidence ≥0.30 (catches partial chairs, table edges)
                # Close (0.6-1.0m): require confidence ≥0.60 (reduces false positives)
                very_close = det.distance_m is not None and det.distance_m <= 0.6
                min_confidence = 0.30 if very_close else 0.40 if partial_edge else 0.60
                if det.confidence < min_confidence:
                    continue
                immediate.append(det)

            if immediate:
                # Sort by distance (closest first)
                det = sorted(immediate, key=lambda d: d.distance_m if d.distance_m is not None else 99)[0]
                partial_edge = _is_partial_edge_detection(det)
                subject = f"partially visible {det.label}" if partial_edge else det.label
                avoidance = _avoidance_phrase(det.direction, det.distance_m, det.label)
                # Omit distance for immediate obstacles (≤1.0m) for brevity
                distance_str = "" if (det.distance_m is not None and det.distance_m <= 1.0) else f", {_detection_distance_phrase(det)}"
                return AgentDecision(
                    action=AgentAction.WARN,
                    priority=95,  # Always stop for immediate unknown obstacles
                    message=(
                        f"Stop: {subject} {_direction_phrase(det.direction)}{distance_str}. {avoidance}"
                    ),
                    haptic=_haptic_for_direction(det.direction, urgent=True),
                    agents_consulted=[self.name, "partial_edge" if partial_edge else "universal_proximity"],
                    debug={
                        "detection": det.model_dump(),
                        "reason": "partial-edge-universal-proximity" if partial_edge else "universal-proximity-immediate",
                    },
                )

        # Check for edge-based obstacle detection (walls, pillars, boxes YOLO missed)
        if ctx.motion.is_moving:
            for surface in ctx.surfaces:
                if surface.kind in {SurfaceKind.OBSTACLE_EDGE, SurfaceKind.WALL}:
                    # Only trigger if confident and close enough
                    min_confidence = 0.58 if surface.kind == SurfaceKind.WALL else 0.5
                    max_distance = 1.5 if surface.kind == SurfaceKind.WALL else 1.2
                    if surface.confidence >= min_confidence and surface.distance_m is not None and surface.distance_m <= max_distance:
                        priority = 95 if surface.distance_m <= 0.8 else 85
                        avoidance = _avoidance_phrase(surface.direction, surface.distance_m)
                        # Omit distance for immediate obstacles (≤1.0m) for brevity
                        distance_str = "" if surface.distance_m <= 1.0 else f" {_distance_phrase(surface.distance_m)}"
                        if surface.kind == SurfaceKind.WALL:
                            message = f"Stop: wall ahead{distance_str}. {avoidance}"
                            reason = "wall-plane-obstacle"
                            consulted = [self.name, "wall_plane"]
                        else:
                            message = (
                                f"Stop: partially visible obstacle {_direction_phrase(surface.direction)}{distance_str}. {avoidance}"
                            )
                            reason = "edge-density-obstacle"
                            consulted = [self.name, "edge_density"]
                        return AgentDecision(
                            action=AgentAction.WARN,
                            priority=priority,
                            message=message,
                            haptic=HapticPattern.STOP if priority >= 95 else HapticPattern.CAUTION,
                            agents_consulted=consulted,
                            debug={
                                "surface": surface.model_dump(),
                                "reason": reason,
                            },
                        )

        # Door-handle surfaces are handled as guidance by the door/exit agents
        # so they do not masquerade as safety warnings.

        if ctx.scene.location_type == "street_crossing":
            return AgentDecision(
                action=AgentAction.WARN,
                priority=90,
                message="Intersection area detected. I can describe what I see, but use crossing aids or ask for human assistance before moving.",
                haptic=HapticPattern.CAUTION,
                agents_consulted=[self.name],
                requires_human=True,
                debug={"policy": "crossing-description-only"},
            )
        return None


class SidewalkAgent(BaseAgent):
    name = "sidewalk"

    def handle(self, ctx: FrameContext) -> Optional[AgentDecision]:
        # Suppress outdoor-only surface warnings when indoors — indoor floors
        # (tile, concrete) frequently trigger false road/sidewalk detections.
        if ctx.scene.is_indoor and not _user_is_asking_about_sidewalk(ctx):
            return None

        if not ctx.surfaces:
            if _user_is_asking_about_sidewalk(ctx):
                return AgentDecision(
                    action=AgentAction.ASK,
                    priority=74,
                    message="I cannot identify the sidewalk yet. Stop, hold the camera forward, and slowly pan left and right.",
                    haptic=HapticPattern.CAUTION,
                    agents_consulted=[self.name],
                    debug={"reason": "sidewalk requested but no surface observations"},
                )
            return None

        road = _best_surface(ctx.surfaces, SurfaceKind.ROAD)
        if road and _surface_ahead(road) and road.confidence >= 0.50 and road.near_field_ratio >= 0.35 and ctx.motion.is_moving:
            return AgentDecision(
                action=AgentAction.WARN,
                priority=96,
                message=(
                    f"Stop. Road surface appears {_direction_phrase(road.direction)} in the near field. "
                    "Confirm the sidewalk edge before moving."
                ),
                haptic=HapticPattern.STOP,
                agents_consulted=[self.name],
                requires_human=True,
                debug={"surface": road.model_dump(), "reason": "road surface near walking path"},
            )

        curb = _best_surface(ctx.surfaces, SurfaceKind.CURB)
        if curb and _surface_ahead(curb) and curb.confidence >= 0.45:
            very_near = curb.distance_m is not None and curb.distance_m <= 0.9
            return AgentDecision(
                action=AgentAction.WARN,
                priority=95 if very_near and ctx.motion.is_moving else 82,
                message=(
                    f"{'Stop' if very_near and ctx.motion.is_moving else 'Caution'}: "
                    f"possible curb edge {_direction_phrase(curb.direction)}, {_distance_phrase(curb.distance_m)}."
                ),
                haptic=HapticPattern.STOP if very_near and ctx.motion.is_moving else HapticPattern.CAUTION,
                agents_consulted=[self.name],
                debug={"surface": curb.model_dump(), "reason": "curb edge surface observation"},
            )

        walkable = _best_walkable_surface(ctx.surfaces)
        if walkable and (_user_is_asking_about_sidewalk(ctx) or ctx.user.mode == "orientation" or not ctx.motion.is_moving):
            if walkable.confidence >= 0.45:
                kind = "crosswalk" if walkable.kind == SurfaceKind.CROSSWALK else "sidewalk"
                return AgentDecision(
                    action=AgentAction.ORIENT,
                    priority=52,
                    message=(
                        f"{kind.title()} surface appears to continue {_direction_phrase(walkable.direction)}. "
                        "Keep scanning for curbs, signs, and vehicles."
                    ),
                    haptic=_haptic_for_direction(walkable.direction),
                    agents_consulted=[self.name],
                    debug={"surface": walkable.model_dump(), "reason": "walkable surface orientation"},
                )

        if _user_is_asking_about_sidewalk(ctx):
            return AgentDecision(
                action=AgentAction.ASK,
                priority=73,
                message="I do not have a confident sidewalk read. Stop and angle the camera slightly down toward the walking surface.",
                haptic=HapticPattern.CAUTION,
                agents_consulted=[self.name],
                debug={"surfaces": [s.model_dump() for s in ctx.surfaces], "reason": "low-confidence surface observations"},
            )

        return None


class TargetFindingAgent(BaseAgent):
    name = "target_finding"

    def handle(self, ctx: FrameContext) -> Optional[AgentDecision]:
        target = _target_from_context(ctx)
        if not target:
            return None

        if target == "door":
            handle_surface = _best_clear_door_handle_surface(ctx)
            if handle_surface is not None:
                return AgentDecision(
                    action=AgentAction.GUIDE,
                    priority=72,
                    message=_door_handle_guidance_message(handle_surface),
                    haptic=_door_haptic(handle_surface),
                    agents_consulted=[self.name, "door_handle"],
                    debug={"surface": handle_surface.model_dump(), "reason": "target-door-handle-found"},
                )

        matches = _matching_detections(ctx.detections, target, self.policy.target_confidence_floor)
        if matches:
            best = sorted(
                matches,
                key=lambda d: (
                    d.distance_m is None,
                    d.distance_m if d.distance_m is not None else 99.0,
                    -d.confidence,
                ),
            )[0]
            return AgentDecision(
                action=AgentAction.GUIDE,
                priority=70,
                message=f"{best.label} found {_direction_phrase(best.direction)}, {_distance_phrase(best.distance_m)}.",
                haptic=_haptic_for_direction(best.direction),
                agents_consulted=[self.name],
                debug={"target": target, "detection": best.model_dump()},
            )

        if ctx.scene.visual_confidence < 0.45:
            return AgentDecision(
                action=AgentAction.ASK,
                priority=55,
                message=f"I cannot confidently find {target}. Slowly pan the phone left and right.",
                haptic=HapticPattern.CAUTION,
                agents_consulted=[self.name],
                debug={"target": target, "reason": "low visual confidence"},
            )

        return AgentDecision(
            action=AgentAction.ASK,
            priority=45,
            message=f"I do not see {target} yet. Point the camera slightly higher and slowly scan.",
            haptic=HapticPattern.NONE,
            agents_consulted=[self.name],
            debug={"target": target, "reason": "not found"},
        )


class TrafficAgent(BaseAgent):
    """Detects moving vehicles and provides warnings based on speed and direction.
    
    Prioritizes vehicles moving toward the user or at high speeds.
    """
    name = "traffic"

    def handle(self, ctx: FrameContext) -> Optional[AgentDecision]:
        if ctx.scene.is_indoor:
            return None

        # Check for moving vehicles
        moving_vehicles = []
        for det in ctx.detections:
            is_moving = det.attributes.get("is_moving", False)
            if not is_moving:
                continue
            
            speed_mps = det.attributes.get("speed_mps", 0.0)
            movement_direction = det.attributes.get("movement_direction", "unknown")
            
            # Filter for vehicle labels
            if det.label.lower() not in {"car", "bus", "truck", "motorcycle", "bicycle"}:
                continue
            
            # Prioritize vehicles moving toward user (down in frame = getting closer)
            if movement_direction == "down" and speed_mps > 1.0:
                moving_vehicles.append((det, speed_mps, movement_direction))
            # Also warn about fast-moving vehicles regardless of direction
            elif speed_mps > 5.0:  # > 18 km/h
                moving_vehicles.append((det, speed_mps, movement_direction))

        if not moving_vehicles:
            return None

        # Sort by speed (fastest first)
        moving_vehicles.sort(key=lambda x: x[1], reverse=True)
        det, speed_mps, movement_direction = moving_vehicles[0]

        # Determine urgency based on speed and direction
        is_approaching = movement_direction == "down"
        is_fast = speed_mps > 5.0
        is_close = det.distance_m is not None and det.distance_m < 5.0

        avoidance = _avoidance_phrase(det.direction, det.distance_m, det.label)
        if is_approaching and (is_fast or is_close):
            priority = 100  # Critical
            message = f"Stop! {det.label} approaching. {avoidance}"
            haptic = HapticPattern.STOP
        elif is_approaching:
            priority = 95
            message = f"Caution: {det.label} moving toward you. {avoidance}"
            haptic = HapticPattern.STOP
        elif is_fast:
            priority = 90
            message = f"Fast {det.label} {_direction_phrase(det.direction)}. {avoidance}"
            haptic = HapticPattern.CAUTION
        else:
            priority = 85
            message = f"{det.label} {_direction_phrase(det.direction)}. {avoidance}"
            haptic = HapticPattern.CAUTION

        return AgentDecision(
            action=AgentAction.WARN,
            priority=priority,
            message=message,
            haptic=haptic,
            agents_consulted=[self.name],
            debug={
                "detection": det.model_dump(),
                "speed_mps": speed_mps,
                "movement_direction": movement_direction,
                "reason": "moving-vehicle-warning",
            },
        )


class CrossingSignalAgent(BaseAgent):
    name = "crossing_signal"

    def handle(self, ctx: FrameContext) -> Optional[AgentDecision]:
        # No crossing signals indoors — skip to avoid false positives from
        # indoor signage or colored lights.
        if ctx.scene.is_indoor:
            return None

        signal = _best_crossing_signal(ctx.detections)
        if signal:
            category = _signal_category(signal)
            state = _signal_state(signal)
            message = _signal_message(signal, category, state)
            stop_semantics = category == "stop_sign" or state == "dont_walk"
            return AgentDecision(
                action=AgentAction.WARN,
                priority=94 if stop_semantics else 92,
                message=message,
                haptic=HapticPattern.STOP if stop_semantics else HapticPattern.CAUTION,
                agents_consulted=[self.name],
                requires_human=True,
                debug={
                    "signal_category": category,
                    "signal_state": state,
                    "detection": signal.model_dump(),
                    "policy": "crossing-signal-description-only",
                },
            )

        if _user_is_asking_about_crossing_signal(ctx):
            return AgentDecision(
                action=AgentAction.ASK,
                priority=91,
                message="I cannot read a pedestrian signal yet. Point the camera toward the far-side signal and hold steady.",
                haptic=HapticPattern.CAUTION,
                agents_consulted=[self.name],
                requires_human=True,
                debug={"reason": "crossing signal requested but not detected"},
            )

        return None


class WayfindingAgent(BaseAgent):
    name = "wayfinding"

    # Navigation instructions must periodically break through obstacle/surface
    # agents so the user hears where to go. Every N frames we boost priority.
    _frame_counter: int = 0
    _NAV_BOOST_EVERY: int = 30  # Boost every ~30 frames (~1-2 seconds at typical FPS)

    def handle(self, ctx: FrameContext) -> Optional[AgentDecision]:
        if not ctx.route.active:
            return None
        
        WayfindingAgent._frame_counter += 1
        
        if ctx.route.off_route:
            return AgentDecision(
                action=AgentAction.ASK,
                priority=75,
                message="You may be off route. Stop and slowly scan so I can re-orient you.",
                haptic=HapticPattern.CAUTION,
                agents_consulted=[self.name],
                debug={"route": ctx.route.model_dump()},
            )
        if ctx.route.next_instruction:
            # Periodically boost priority so navigation breaks through
            # low-value scan/orientation prompts. Safety (>=90) still overrides.
            is_boost_frame = (WayfindingAgent._frame_counter % self._NAV_BOOST_EVERY == 0)
            if getattr(ctx.route, "exit_seeking", False):
                priority = 76 if is_boost_frame else 72
            else:
                priority = 78 if is_boost_frame else 70
            
            return AgentDecision(
                action=AgentAction.GUIDE,
                priority=priority,
                message=_route_instruction_message(ctx),
                haptic=HapticPattern.NONE,
                agents_consulted=[self.name],
                debug={
                    "route": ctx.route.model_dump(),
                    "boosted": is_boost_frame,
                    "exit_seeking": getattr(ctx.route, "exit_seeking", False),
                },
            )
        return None


class OrientationAgent(BaseAgent):
    name = "orientation"

    def handle(self, ctx: FrameContext) -> Optional[AgentDecision]:
        if ctx.motion.is_moving and not ctx.user.query and ctx.user.mode != "orientation":
            return None

        wall = _best_wall_surface(ctx)
        if wall is not None and (ctx.user.mode == "orientation" or not ctx.motion.is_moving or getattr(ctx.route, "exit_seeking", False)):
            return AgentDecision(
                action=AgentAction.ORIENT,
                priority=42,
                message=_wall_observation_message(wall),
                haptic=HapticPattern.CAUTION,
                agents_consulted=[self.name, "wall_plane"],
                debug={"surface": wall.model_dump(), "reason": "wall-visible-orientation"},
            )

        salient = _salient_detections(ctx.detections)
        if not salient:
            if ctx.scene.visual_confidence < 0.45:
                return AgentDecision(
                    action=AgentAction.ASK,
                    priority=35,
                    message="I do not have a clear view. Hold the phone steady and point it forward.",
                    haptic=HapticPattern.NONE,
                    agents_consulted=[self.name],
                    debug={"visual_confidence": ctx.scene.visual_confidence},
                )
            return AgentDecision.silent("no salient detections")

        phrases = [f"{det.label} {_direction_phrase(det.direction)}" for det in salient[:3]]
        place = ""
        if ctx.scene.location_type not in {"unknown", "room"}:
            place = f"You appear to be in a {ctx.scene.location_type}. "
        return AgentDecision(
            action=AgentAction.ORIENT,
            priority=35,
            message=place + "I see " + ", ".join(phrases) + ".",
            haptic=HapticPattern.NONE,
            agents_consulted=[self.name],
            debug={"salient": [d.model_dump() for d in salient[:3]]},
        )


class IndoorNavigationAgent(BaseAgent):
    """Combines indoor routing with obstacle detection to provide rerouting guidance.

    When an obstacle (chair, table, person, etc.) is detected on the current
    path direction, this agent suggests avoidance maneuvers or alternate routes.
    Priority 68 — below safety (80-100) but close to wayfinding (60).
    """

    name = "indoor_navigation"

    # Labels to IGNORE for indoor avoidance (not physical obstacles)
    NON_OBSTACLE_LABELS = {"traffic light", "stop sign", "fire hydrant", "parking meter", "backpack", "handbag", "suitcase", "tie", "umbrella"}

    def handle(self, ctx: FrameContext) -> Optional[AgentDecision]:
        if ctx.scene.location_type not in INDOOR_LOCATION_TYPES and ctx.scene.location_type != "unknown":
            return None

        if ctx.user.mode == "orientation" or not ctx.motion.is_moving:
            return None

        # Defer to TargetFindingAgent when the user is actively searching
        if ctx.user.target or ctx.user.query:
            return None

        # Do not announce heuristic door handles as physical facts. Indoor
        # navigation should prioritize route steps and confirmed obstacles.

        blocking: List[Detection] = []
        for det in ctx.detections:
            if det.label.lower() in self.NON_OBSTACLE_LABELS:
                continue
            if det.confidence < 0.50:
                continue
            ahead = det.direction in {Direction.CENTER, Direction.SLIGHT_LEFT, Direction.SLIGHT_RIGHT}
            close = det.distance_m is not None and det.distance_m <= 3.0
            if ahead and close:
                blocking.append(det)

        if not blocking:
            return None

        blocking.sort(key=lambda d: (d.distance_m if d.distance_m is not None else 99, -d.confidence))
        nearest = blocking[0]

        # Cluster nearby obstacles to describe groups instead of just the nearest
        cluster_labels = _cluster_obstacle_labels(blocking)

        # Build obstacle-type-aware avoidance instruction
        instruction = _indoor_obstacle_instruction(nearest, blocking, cluster_labels)

        if nearest.distance_m is not None and nearest.distance_m <= 1.0:
            return AgentDecision(
                action=AgentAction.WARN,
                priority=78,
                message=instruction,
                haptic=_obstacle_avoidance_haptic(nearest),
                agents_consulted=[self.name],
                debug={
                    "blocking_objects": [d.model_dump() for d in blocking],
                    "cluster_labels": cluster_labels,
                    "reason": "indoor-obstacle-close",
                },
            )

        return AgentDecision(
            action=AgentAction.GUIDE,
            priority=68,
            message=instruction,
            haptic=_obstacle_avoidance_haptic(nearest),
            agents_consulted=[self.name],
            debug={
                "blocking_objects": [d.model_dump() for d in blocking],
                "cluster_labels": cluster_labels,
                "reason": "indoor-obstacle-avoidance",
            },
        )


def _obstacle_avoidance_haptic(det: Detection) -> HapticPattern:
    if det.direction in {Direction.SLIGHT_LEFT, Direction.LEFT, Direction.CENTER}:
        return HapticPattern.RIGHT
    return HapticPattern.LEFT


def _cluster_obstacle_labels(blocking: List[Detection]) -> str:
    """Describe a group of blocking objects concisely."""
    labels = sorted({d.label.lower() for d in blocking})
    if len(labels) == 1:
        count = len(blocking)
        label = labels[0]
        if count > 1:
            return f"{count} {label}s" if not label.endswith("s") else f"{count} {label}"
        return label
    if len(labels) == 2:
        return f"{labels[0]} and {labels[1]}"
    return f"{labels[0]}, {labels[1]}, and {len(labels) - 2} more"


def _indoor_obstacle_instruction(nearest: Detection, blocking: List[Detection], cluster_labels: str) -> str:
    """Build obstacle-type-aware avoidance instructions for indoor navigation."""
    label = nearest.label.lower()
    direction = _direction_phrase(nearest.direction)
    distance = _distance_phrase(nearest.distance_m)
    avoidance = _avoidance_phrase(nearest.direction, nearest.distance_m, label)

    # Obstacle-specific instructions with concrete body guidance
    if label in {"chair", "bench"}:
        if nearest.distance_m is not None and nearest.distance_m <= 1.0:
            return f"{label.capitalize()} {direction}. {avoidance}"
        return f"{label.capitalize()} ahead {direction}, {distance}. {avoidance}"
    if label in {"dining table", "table"}:
        if nearest.distance_m is not None and nearest.distance_m <= 1.0:
            return f"Table edge {direction}. Trail your hand along it and step around."
        return f"Table ahead {direction}, {distance}. {avoidance}"
    if label == "person":
        if nearest.distance_m is not None and nearest.distance_m <= 1.5:
            return f"Person {direction}, {distance}. {avoidance}"
        return f"Person ahead {direction}, {distance}. {avoidance}"
    if label == "couch":
        return f"Couch {direction}, {distance}. {avoidance}"
    if label == "door":
        return f"Door {direction}, {distance}. Reach forward to find the handle."

    # Multiple grouped obstacles
    if len(blocking) > 1:
        return f"{cluster_labels} ahead {direction}, nearest {distance}. {avoidance}"

    # Generic fallback
    if nearest.distance_m is not None and nearest.distance_m <= 1.0:
        return f"Obstacle {direction}: {label}. {avoidance}"
    return f"{label.capitalize()} ahead {direction}, {distance}. {avoidance}"


def _best_visible_door(ctx: FrameContext):
    doors = [
        d for d in ctx.detections
        if d.label.lower() == "door" and d.confidence >= 0.65
    ]
    if doors:
        return sorted(doors, key=lambda d: d.distance_m if d.distance_m is not None else 99.0)[0]

    return None


def _best_possible_door_surface(ctx: FrameContext) -> Optional[SurfaceObservation]:
    candidates = [
        s for s in ctx.surfaces
        if (
            s.kind == SurfaceKind.DOOR
            and (
                (
                    s.source == "vision-wall-handle-candidate"
                    and s.confidence >= 0.72
                )
                or (
                    s.source == "vision-door-handle"
                    and bool(s.attributes.get("handle_detected"))
                    and s.confidence >= 0.76
                )
            )
        )
    ]
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda s: (
            0 if s.source == "vision-door-handle" else 1,
            s.distance_m if s.distance_m is not None else 99.0,
            -s.confidence,
        ),
    )[0]


def _best_clear_door_handle_surface(ctx: FrameContext) -> Optional[SurfaceObservation]:
    candidates = [
        s for s in ctx.surfaces
        if (
            s.kind == SurfaceKind.DOOR
            and bool(s.attributes.get("handle_detected"))
            and bool(s.attributes.get("clear_handle", True))
            and s.source in {"vision-door-handle", "vision-wall-handle-candidate", "joechencc-door-parts-handle"}
            and (
                (
                    s.source == "vision-door-handle"
                    and s.confidence >= 0.80
                    and float(s.attributes.get("handle_confidence", s.confidence) or 0.0) >= 0.74
                )
                or (
                    s.source == "joechencc-door-parts-handle"
                    and str(s.attributes.get("model_label", "")).lower() == "handle"
                    and s.confidence >= 0.40
                    and float(s.attributes.get("handle_confidence", s.confidence) or 0.0) >= 0.40
                )
                or (
                    s.source == "vision-wall-handle-candidate"
                    and s.confidence >= 0.76
                    and float(s.attributes.get("handle_confidence", s.confidence) or 0.0) >= 0.72
                )
            )
        )
    ]
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda s: (
            {"joechencc-door-parts-handle": 0, "vision-door-handle": 1, "vision-wall-handle-candidate": 2}.get(s.source, 3),
            s.distance_m if s.distance_m is not None else 99.0,
            -s.confidence,
        ),
    )[0]


def _best_wall_surface(ctx: FrameContext) -> Optional[SurfaceObservation]:
    candidates = [
        s for s in ctx.surfaces
        if s.kind == SurfaceKind.WALL and s.confidence >= 0.58
    ]
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda s: (
            s.distance_m is None,
            s.distance_m if s.distance_m is not None else 99.0,
            -s.confidence,
        ),
    )[0]


def _possible_wall_handle_message(surface: SurfaceObservation) -> str:
    distance = _distance_phrase(surface.distance_m)
    direction = _direction_phrase(surface.direction)
    return f"Possible doorway {direction}, {distance}: wall-like surface with a handle-like feature. Confirm by touch before using it."


def _door_handle_guidance_message(surface: SurfaceObservation) -> str:
    direction = _direction_phrase(surface.direction)
    distance = _walking_steps_phrase(surface.distance_m) or _distance_phrase(surface.distance_m)
    approach = _door_handle_approach_phrase(surface)
    hand = str(surface.attributes.get("recommended_hand", "")).strip().lower()
    if hand not in {"left", "right"}:
        hand_phrase = "either hand"
    else:
        hand_phrase = f"your {hand} hand"
    height = str(surface.attributes.get("handle_height_zone", "")).strip().lower()
    height_phrase = f" at {height}" if height else ""
    action = str(surface.attributes.get("handle_action", "")).strip()
    if not action:
        action = "find it by touch, then gently test whether the door pushes or pulls"
    return (
        f"Door handle detected {direction}, {distance}{height_phrase}. "
        f"{approach}. Reach with {hand_phrase}, {action}. Confirm by touch before opening."
    )


def _careful_steps_phrase(distance_m: Optional[float]) -> str:
    if distance_m is None:
        return "carefully"
    steps = max(1, int(round(distance_m / WALKING_STEP_M)))
    unit = "step" if steps == 1 else "steps"
    return f"about {steps} careful {unit}"


def _door_handle_approach_phrase(surface: SurfaceObservation) -> str:
    direction = _direction_phrase(surface.direction)
    steps = _careful_steps_phrase(surface.distance_m)
    if direction == "ahead":
        return f"Move straight ahead {steps} toward it"
    return f"Turn slightly {direction} and move {steps} toward it"


def _door_handle_route_instruction(surface: SurfaceObservation) -> str:
    direction = _direction_phrase(surface.direction)
    direction_phrase = "straight ahead" if direction == "ahead" else f"to your {direction}"
    distance = _walking_steps_phrase(surface.distance_m) or _distance_phrase(surface.distance_m)
    return (
        f"Move toward the detected door handle {direction_phrase}, {distance}, "
        "then find it by touch and pass through to exit"
    )


def _wall_observation_message(surface: SurfaceObservation) -> str:
    direction = _direction_phrase(surface.direction)
    distance = _distance_phrase(surface.distance_m)
    if surface.direction in {Direction.CENTER, Direction.UNKNOWN}:
        return f"Wall detected ahead, {distance}. Stop before it and scan left or right for an opening or door handle."
    return f"Wall detected to the {direction}, {distance}. Keep it to your {direction} and continue scanning for an opening or door handle."


def _indoor_scan_context_message(ctx: FrameContext) -> Optional[str]:
    phrases: List[str] = []
    wall = _best_wall_surface(ctx)
    if wall is not None:
        phrases.append(f"wall {_direction_phrase(wall.direction)}, {_distance_phrase(wall.distance_m)}")

    detections = [
        det for det in ctx.detections
        if (
            det.label.lower() in INDOOR_SCAN_CONTEXT_LABELS
            and det.label.lower() != "door"
            and det.confidence >= 0.50
            and (det.distance_m is None or det.distance_m <= 6.0)
        )
    ]
    detections = sorted(
        detections,
        key=lambda det: (
            det.distance_m is None,
            det.distance_m if det.distance_m is not None else 99.0,
            -det.confidence,
        ),
    )
    for det in detections[:3]:
        phrases.append(f"{det.label} {_direction_phrase(det.direction)}, {_distance_phrase(det.distance_m)}")

    if not phrases:
        return None
    return (
        "I see " + "; ".join(phrases[:4]) + ". "
        "Keep scanning slowly for a door handle or exit sign before moving."
    )


def _indoor_scan_context_haptic(ctx: FrameContext) -> HapticPattern:
    wall = _best_wall_surface(ctx)
    if wall is not None:
        return _haptic_for_direction(wall.direction)
    detections = [
        det for det in ctx.detections
        if det.label.lower() in INDOOR_SCAN_CONTEXT_LABELS and det.confidence >= 0.50
    ]
    if not detections:
        return HapticPattern.CAUTION
    nearest = sorted(detections, key=lambda d: d.distance_m if d.distance_m is not None else 99.0)[0]
    return _haptic_for_direction(nearest.direction)


class EnvironmentMappingAgent(BaseAgent):
    """Handles the initial 360-degree environment mapping to determine indoor vs. outdoor."""

    name = "environment_mapping"
    _start_time_ms: int = 0
    _indoor_votes: int = 0
    _outdoor_votes: int = 0

    def handle(self, ctx: FrameContext) -> Optional[AgentDecision]:
        if not ctx.route.active or getattr(ctx.route, "mapping_state", "done") == "done":
            return None

        state = getattr(ctx.route, "mapping_state", "pending")
        
        if state == "pending":
            self._start_time_ms = ctx.timestamp_ms
            self._indoor_votes = 0
            self._outdoor_votes = 0
            ctx.route.mapping_state = "mapping"
            return AgentDecision(
                action=AgentAction.GUIDE,
                priority=90,
                message=(
                    "You appear to be indoors. Before the outdoor route, leave this room first. "
                    "Stand still, turn 360 degrees slowly, and scan for a door or exit sign."
                ),
                haptic=HapticPattern.NONE,
                agents_consulted=[self.name],
                debug={"reason": "mapping-started"}
            )
            
        if state == "mapping":
            if ctx.scene.is_indoor:
                self._indoor_votes += 1
            elif ctx.scene.is_outdoor:
                self._outdoor_votes += 1

            visible_door = _best_visible_door(ctx)
            if visible_door is not None and ctx.scene.is_indoor:
                ctx.route.mapping_state = "done"
                ctx.route.exit_seeking = True
                if isinstance(visible_door, Detection):
                    return AgentDecision(
                        action=AgentAction.GUIDE,
                        priority=91,
                        message=(
                            f"Door detected {_direction_phrase(visible_door.direction)}, "
                            f"{_walking_steps_phrase(visible_door.distance_m) or _distance_phrase(visible_door.distance_m)}. "
                            "Head toward it, find the handle, and pass through to exit."
                        ),
                        haptic=_haptic_for_direction(visible_door.direction),
                        agents_consulted=[self.name, "door_scan"],
                        debug={"door_detection": visible_door.model_dump(), "reason": "door-found-during-360-scan"},
                    )
                return AgentDecision(
                    action=AgentAction.GUIDE,
                    priority=91,
                    message="Possible doorway shape ahead. Confirm by touch before treating it as an exit.",
                    haptic=_door_haptic(visible_door),
                    agents_consulted=[self.name, "door_scan"],
                    debug={"surface": visible_door.model_dump(), "reason": "door-surface-found-during-360-scan"},
                )

            handle_surface = _best_clear_door_handle_surface(ctx)
            if handle_surface is not None and ctx.scene.is_indoor:
                ctx.route.mapping_state = "done"
                ctx.route.exit_seeking = True
                ctx.route.next_instruction = _door_handle_route_instruction(handle_surface)
                ctx.route.next_turn_distance_m = None
                return AgentDecision(
                    action=AgentAction.GUIDE,
                    priority=92,
                    message=_door_handle_guidance_message(handle_surface),
                    haptic=_door_haptic(handle_surface),
                    agents_consulted=[self.name, "door_handle_scan"],
                    debug={"surface": handle_surface.model_dump(), "reason": "clear-door-handle-during-360-scan"},
                )

            possible_door = _best_possible_door_surface(ctx)
            if possible_door is not None and ctx.scene.is_indoor:
                ctx.route.mapping_state = "done"
                ctx.route.exit_seeking = True
                return AgentDecision(
                    action=AgentAction.GUIDE,
                    priority=82,
                    message=_possible_wall_handle_message(possible_door),
                    haptic=_door_haptic(possible_door),
                    agents_consulted=[self.name, "wall_handle_scan"],
                    debug={"surface": possible_door.model_dump(), "reason": "wall-handle-possible-door"},
                )
                
            elapsed = ctx.timestamp_ms - self._start_time_ms
            if elapsed > 10000:  # 10 seconds to turn
                ctx.route.mapping_state = "done"
                if self._indoor_votes >= self._outdoor_votes:
                    ctx.route.exit_seeking = True
                    return AgentDecision(
                        action=AgentAction.GUIDE,
                        priority=90,
                        message=(
                            "Mapping complete. You are indoors. I will guide you out of this room first, "
                            "then resume the route to your destination."
                        ),
                        haptic=HapticPattern.CAUTION,
                        agents_consulted=[self.name],
                        debug={"reason": "mapping-done-indoors"}
                    )
                else:
                    ctx.route.exit_seeking = False
                    return AgentDecision(
                        action=AgentAction.GUIDE,
                        priority=90,
                        message="Mapping complete. You are outdoors. Initializing route.",
                        haptic=HapticPattern.SUCCESS,
                        agents_consulted=[self.name],
                        debug={"reason": "mapping-done-outdoors"}
                    )
                    
        return None


class ExitSeekingAgent(BaseAgent):
    """Guides the user to find a door/exit when leaving a building.

    Activated when the route indicates the user needs to exit (the route
    context contains ``exit_seeking=True``) and we are indoors without
    indoor map data.  The agent watches for door detections in the frame
    and tells the user which direction to turn to reach the door.
    """

    name = "exit_seeking"

    def handle(self, ctx: FrameContext) -> Optional[AgentDecision]:
        if not ctx.route.active:
            return None
        # Only engage when route metadata signals exit-seeking phase.
        if not getattr(ctx.route, "exit_seeking", False):
            return None
            
        # If the visual scene classifier detects we are outdoors, we have successfully exited!
        if ctx.scene.is_outdoor:
            ctx.route.exit_seeking = False
            if ctx.route.destination:
                ctx.route.next_instruction = f"Head toward {ctx.route.destination}."
            return AgentDecision(
                action=AgentAction.GUIDE,
                priority=75,
                message="You have exited the room or building. Switching to the route toward your destination.",
                haptic=HapticPattern.SUCCESS,
                agents_consulted=[self.name],
                debug={"reason": "transitioned-outdoors"}
            )

        if not ctx.scene.is_indoor:
            return None

        # Look for door detections in the frame
        doors = [
            d for d in ctx.detections
            if d.label.lower() == "door" and d.confidence >= 0.65
        ]

        if doors:
            nearest = sorted(doors, key=lambda d: d.distance_m if d.distance_m is not None else 99)[0]
            direction = _direction_phrase(nearest.direction)
            distance = _distance_phrase(nearest.distance_m)
            return AgentDecision(
                action=AgentAction.GUIDE,
                priority=73,
                message=f"Door detected {direction}, {_walking_steps_phrase(nearest.distance_m) or distance}. Head toward it, find the handle, and pass through to exit.",
                haptic=_haptic_for_direction(nearest.direction),
                agents_consulted=[self.name],
                debug={"door_detection": nearest.model_dump(), "reason": "exit-door-detected"},
            )

        handle_surface = _best_clear_door_handle_surface(ctx)
        if handle_surface is not None:
            ctx.route.next_instruction = _door_handle_route_instruction(handle_surface)
            ctx.route.next_turn_distance_m = None
            return AgentDecision(
                action=AgentAction.GUIDE,
                priority=90,
                message=_door_handle_guidance_message(handle_surface),
                haptic=_door_haptic(handle_surface),
                agents_consulted=[self.name, "door_handle_scan"],
                debug={"surface": handle_surface.model_dump(), "reason": "clear-door-handle-possible-exit"},
            )

        possible_door = _best_possible_door_surface(ctx)
        if possible_door is not None:
            return AgentDecision(
                action=AgentAction.GUIDE,
                priority=74,
                message=_possible_wall_handle_message(possible_door),
                haptic=_door_haptic(possible_door),
                agents_consulted=[self.name, "wall_handle_scan"],
                debug={"surface": possible_door.model_dump(), "reason": "wall-handle-possible-exit"},
            )

        context_message = _indoor_scan_context_message(ctx)
        if context_message is not None and not ctx.motion.is_moving:
            return AgentDecision(
                action=AgentAction.ORIENT,
                priority=74,
                message=context_message,
                haptic=_indoor_scan_context_haptic(ctx),
                agents_consulted=[self.name, "indoor_scan_context"],
                debug={
                    "detections": [d.model_dump() for d in ctx.detections],
                    "surfaces": [s.model_dump() for s in ctx.surfaces],
                    "reason": "exit-scan-context-visible",
                },
            )

        # No door visible. Do not block live route directions every frame; the
        # initial 360-degree mapping prompt already told the user to scan.
        if not ctx.motion.is_moving:
            return AgentDecision(
                action=AgentAction.ASK,
                priority=58,
                message="Looking for an exit. Stand still, slowly turn another 360 degrees, and scan for a door or exit sign.",
                haptic=HapticPattern.CAUTION,
                agents_consulted=[self.name],
                debug={"reason": "exit-scanning"},
            )

        # Moving but no door yet. Keep this below route guidance so directions
        # remain audible when there is no immediate safety issue.
        return AgentDecision(
            action=AgentAction.GUIDE,
            priority=55,
            message="No exit visible yet. Move to the nearest wall, trail your hand along it, and keep scanning for a door.",
            haptic=HapticPattern.CAUTION,
            agents_consulted=[self.name],
            debug={"reason": "exit-wall-trailing"},
        )


class FallbackAgent(BaseAgent):
    name = "fallback"

    def handle(self, ctx: FrameContext) -> Optional[AgentDecision]:
        if ctx.scene.visual_confidence < 0.30 and ctx.motion.is_moving:
            return AgentDecision(
                action=AgentAction.ASK,
                priority=75,
                message="My camera confidence is low. Slow down and hold the phone steady.",
                haptic=HapticPattern.CAUTION,
                agents_consulted=[self.name],
                debug={"visual_confidence": ctx.scene.visual_confidence},
            )
        return AgentDecision.silent("fallback")


def _target_from_context(ctx: FrameContext) -> Optional[str]:
    if ctx.user.target:
        return ctx.user.target.lower().strip()
    if not ctx.user.query:
        return None
    q = ctx.user.query.lower()
    for canonical, variants in TARGET_KEYWORDS.items():
        if any(v in q for v in variants):
            return canonical
    m = re.search(r"(?:find|where(?: is|'s)?|locate|look for)\s+(?:the\s+)?([a-z ]{2,30})", q)
    if m:
        return m.group(1).strip()
    return None


def _matching_detections(detections: Iterable[Detection], target: str, min_confidence: float) -> List[Detection]:
    variants = set(TARGET_KEYWORDS.get(target, [target]))
    variants.add(target)
    out = []
    for det in detections:
        if det.confidence < min_confidence:
            continue
        label = det.label.lower()
        if label in variants or any(_contains_phrase(label, v) for v in variants):
            out.append(det)
    return out


def _salient_detections(detections: Iterable[Detection]) -> List[Detection]:
    keep = []
    for det in detections:
        if det.confidence < 0.50:
            continue
        if det.distance_m is not None and det.distance_m > 6.0:
            continue
        label = det.label.lower()
        if label in HAZARD_LABELS or label in {"door", "sign", "elevator", "traffic light", "stop sign"}:
            keep.append(det)
    return sorted(keep, key=lambda d: (d.distance_m if d.distance_m is not None else 99, -d.confidence))


def _best_crossing_signal(detections: Iterable[Detection]) -> Optional[Detection]:
    candidates = []
    for det in detections:
        if det.confidence < 0.30:
            continue
        label = det.label.lower()
        category = str(det.attributes.get("signal_category", "")).lower()
        if label in CROSSING_SIGNAL_LABELS or category in {"pedestrian_signal", "traffic_light", "stop_sign"}:
            candidates.append(det)
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda d: (
            _signal_priority(d),
            -(d.distance_m if d.distance_m is not None else 99.0),
            d.confidence,
        ),
        reverse=True,
    )[0]


def _signal_priority(det: Detection) -> int:
    category = _signal_category(det)
    if category == "pedestrian_signal":
        return 3
    if category == "traffic_light":
        return 2
    if category == "stop_sign":
        return 1
    return 0


def _signal_category(det: Detection) -> str:
    category = str(det.attributes.get("signal_category", "")).lower().strip()
    if category:
        return category
    label = det.label.lower()
    if label in {"traffic light", "traffic signal"}:
        return "traffic_light"
    if label == "stop sign":
        return "stop_sign"
    if any(term in label for term in {"walk", "pedestrian", "crosswalk", "countdown"}):
        return "pedestrian_signal"
    return "unknown"


def _signal_state(det: Detection) -> str:
    for key in ("signal_state", "pedestrian_signal_state", "state", "color"):
        raw = det.attributes.get(key)
        if raw is not None:
            normalized = _normalize_signal_state(str(raw))
            if normalized != "unknown":
                return normalized

    text = str(det.attributes.get("text", "") or det.attributes.get("ocr_text", "")).lower()
    if text:
        return _normalize_signal_state(text)
    return _normalize_signal_state(det.label)


def _normalize_signal_state(value: str) -> str:
    text = value.lower().replace("_", " ").replace("-", " ").strip()
    if re.search(r"\b\d{1,2}\b", text) or "countdown" in text:
        return "countdown"
    if any(term in text for term in {"dont walk", "don't walk", "do not walk", "red hand", "raised hand", "stop hand", "wait"}):
        return "dont_walk"
    if any(term in text for term in {"walk", "walking person", "white person", "green person"}):
        return "walk"
    if "green" in text:
        return "green"
    if "yellow" in text or "amber" in text:
        return "yellow"
    if "red" in text:
        return "red"
    if "stop sign" in text or text == "stop":
        return "stop_sign"
    return "unknown"


def _signal_message(det: Detection, category: str, state: str) -> str:
    direction = _direction_phrase(det.direction)
    caveat = "Use crossing aids or ask for human assistance before moving."
    if category == "pedestrian_signal":
        readable = {
            "walk": "walk",
            "dont_walk": "don't walk",
            "countdown": "countdown",
            "unknown": "an unreadable state",
        }.get(state, state.replace("_", " "))
        return f"Pedestrian signal shows {readable} {direction}. {caveat}"
    if category == "traffic_light":
        readable = {
            "red": "red",
            "yellow": "yellow",
            "green": "green",
            "unknown": "an unreadable color",
        }.get(state, state.replace("_", " "))
        return f"Traffic light appears {readable} {direction}. Vehicle lights may not match pedestrian signals. {caveat}"
    if category == "stop_sign":
        return f"Stop sign detected {direction}. Stop and confirm surroundings before moving. This is road-sign information only."
    return f"Crossing signal detected {direction}, but I cannot read its state. {caveat}"


def _user_is_asking_about_crossing_signal(ctx: FrameContext) -> bool:
    text = " ".join(filter(None, [ctx.user.query or "", ctx.user.target or "", ctx.user.mode or ""])).lower()
    return any(term in text for term in SIGNAL_QUERY_TERMS)


def _user_is_asking_about_sidewalk(ctx: FrameContext) -> bool:
    text = " ".join(filter(None, [ctx.user.query or "", ctx.user.target or "", ctx.user.mode or ""])).lower()
    return any(term in text for term in SIDEWALK_QUERY_TERMS)


def _best_surface(surfaces: Iterable[SurfaceObservation], kind: SurfaceKind) -> Optional[SurfaceObservation]:
    candidates = [surface for surface in surfaces if surface.kind == kind]
    if not candidates:
        return None
    return sorted(candidates, key=lambda s: (s.confidence, s.near_field_ratio), reverse=True)[0]


def _best_walkable_surface(surfaces: Iterable[SurfaceObservation]) -> Optional[SurfaceObservation]:
    candidates = [surface for surface in surfaces if surface.kind in {SurfaceKind.SIDEWALK, SurfaceKind.CROSSWALK}]
    if not candidates:
        return None
    return sorted(candidates, key=lambda s: (s.confidence, s.near_field_ratio), reverse=True)[0]


def _surface_ahead(surface: SurfaceObservation) -> bool:
    return surface.direction in {Direction.CENTER, Direction.SLIGHT_LEFT, Direction.SLIGHT_RIGHT, Direction.UNKNOWN}


def _door_haptic(surface: SurfaceObservation) -> HapticPattern:
    hand = str(surface.attributes.get("recommended_hand", "")).lower()
    side = str(surface.attributes.get("handle_side", "")).lower()
    cue = hand if hand in {"left", "right"} else side
    if cue == "left":
        return HapticPattern.LEFT
    if cue == "right":
        return HapticPattern.RIGHT
    return HapticPattern.CAUTION


def _door_approach_message(surface: SurfaceObservation) -> str:
    if surface.attributes.get("handle_detected"):
        return (
            f"Possible doorway {_direction_phrase(surface.direction)}, "
            f"{_distance_phrase(surface.distance_m)}. Confirm it by touch before using it."
        )
    return f"Possible doorway {_direction_phrase(surface.direction)} {_distance_phrase(surface.distance_m)}. Confirm it by touch."


def _door_guidance_message(surface: SurfaceObservation) -> str:
    distance = _distance_phrase(surface.distance_m)
    if not surface.attributes.get("handle_detected"):
        return f"Possible doorway ahead {distance}. Confirm it by touch before treating it as an exit."

    return f"Possible doorway ahead {distance}. I see a handle-like shape, but confirm it by touch."


def _edge_contact_sides(det: Detection) -> set[str]:
    raw = det.attributes.get("edge_contact") or []
    if isinstance(raw, str):
        raw = [part.strip() for part in raw.split(",")]
    try:
        return {str(side).lower() for side in raw if side}
    except TypeError:
        return set()


def _is_partial_edge_detection(det: Detection) -> bool:
    if bool(det.attributes.get("edge_truncated")):
        return True
    return str(det.attributes.get("partial_visibility", "")).lower() == "frame_edge"


def _partial_edge_hazard(det: Detection) -> bool:
    if not _is_partial_edge_detection(det):
        return False

    sides = _edge_contact_sides(det)
    if not sides.intersection({"left", "right", "bottom"}):
        return False

    area_ratio = float(det.attributes.get("area_ratio", 0.0) or 0.0)
    bottom_y_ratio = float(det.attributes.get("bottom_y_ratio", 0.0) or 0.0)
    close = det.distance_m is not None and det.distance_m <= 1.6
    large_partial = area_ratio >= 0.035
    low_in_frame = bottom_y_ratio >= 0.82 and area_ratio >= 0.015
    side_intrusion = bool(sides.intersection({"left", "right"})) and area_ratio >= 0.020
    bottom_intrusion = "bottom" in sides and area_ratio >= 0.018
    return (close and (large_partial or low_in_frame or side_intrusion)) or bottom_intrusion


def _partial_edge_requires_stop(det: Detection) -> bool:
    if not _partial_edge_hazard(det):
        return False
    sides = _edge_contact_sides(det)
    area_ratio = float(det.attributes.get("area_ratio", 0.0) or 0.0)
    bottom_y_ratio = float(det.attributes.get("bottom_y_ratio", 0.0) or 0.0)
    close = det.distance_m is not None and det.distance_m <= 0.9
    return close or ("bottom" in sides and area_ratio >= 0.025) or (bottom_y_ratio >= 0.90 and area_ratio >= 0.025)


def _detection_distance_phrase(det: Detection) -> str:
    if not _is_partial_edge_detection(det):
        return _distance_phrase(det.distance_m)
    if det.distance_m is not None and det.distance_m <= 1.4:
        return "likely nearby; distance is uncertain"
    return "distance is uncertain"


def _warning_message(subject: str, direction: Direction, distance_m: Optional[float]) -> str:
    base = subject.strip() or "hazard"
    base = base[0].upper() + base[1:] if base else "Hazard"
    details = []
    if direction != Direction.UNKNOWN:
        details.append(_direction_phrase(direction))
    if distance_m is not None:
        details.append(_distance_phrase(distance_m))
    if details:
        base = f"{base}, {', '.join(details)}"
    if not base.endswith("."):
        base += "."
    return base


def _hazard_sort_key(det: Detection) -> tuple[float, float, float]:
    area_ratio = float(det.attributes.get("area_ratio", 0.0) or 0.0)
    distance = det.distance_m if det.distance_m is not None else max(0.5, 6.0 - area_ratio * 24.0)
    stop_bias = 0.0 if det.label.lower() in STOP_HAZARDS else 0.4
    return (distance + stop_bias, -area_ratio, -det.confidence)


def _contains_phrase(text: str, phrase: str) -> bool:
    phrase = phrase.strip()
    if not phrase:
        return False
    return bool(re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text))
