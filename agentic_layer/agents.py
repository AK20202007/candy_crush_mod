from __future__ import annotations

import re
from typing import Iterable, List, Optional

from .models import AgentAction, AgentDecision, Detection, Direction, FrameContext, HapticPattern, SurfaceKind, SurfaceObservation
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


def _avoidance_phrase(direction: Direction, distance_m: Optional[float] = None) -> str:
    """Return a phrase telling the user which way to move to avoid the obstacle."""
    base_phrase = {
        Direction.LEFT: "Move right",
        Direction.SLIGHT_LEFT: "Move right",
        Direction.CENTER: "Move left or right",
        Direction.SLIGHT_RIGHT: "Move left",
        Direction.RIGHT: "Move left",
        Direction.UNKNOWN: "Move left or right",
    }[direction]
    
    if distance_m is not None and distance_m > 0:
        # Approximate steps (1 step ≈ 0.75m)
        steps = max(1, int(round(distance_m / 0.75)))
        if steps == 1:
            return f"{base_phrase} about one step."
        elif steps <= 3:
            return f"{base_phrase} about {steps} steps."
        else:
            return f"{base_phrase} about {steps} steps."
    
    return base_phrase + "."


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
                avoidance = _avoidance_phrase(high.direction, high.distance_m) if high.direction else ""
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
                or _partial_edge_requires_stop(det)
            )
            subject = f"partially visible {det.label}" if partial_edge else det.label
            avoidance = _avoidance_phrase(det.direction, det.distance_m)
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
                avoidance = _avoidance_phrase(det.direction, det.distance_m)
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
                if surface.kind == SurfaceKind.OBSTACLE_EDGE:
                    # Only trigger if confident and close enough
                    if surface.confidence >= 0.5 and surface.distance_m is not None and surface.distance_m <= 1.2:
                        priority = 95 if surface.distance_m <= 0.8 else 85
                        avoidance = _avoidance_phrase(surface.direction, surface.distance_m)
                        # Omit distance for immediate obstacles (≤1.0m) for brevity
                        distance_str = "" if surface.distance_m <= 1.0 else f" {_distance_phrase(surface.distance_m)}"
                        message = (
                            f"Stop: partially visible obstacle {_direction_phrase(surface.direction)}{distance_str}. {avoidance}"
                        )
                        return AgentDecision(
                            action=AgentAction.WARN,
                            priority=priority,
                            message=message,
                            haptic=HapticPattern.STOP if priority >= 95 else HapticPattern.CAUTION,
                            agents_consulted=[self.name, "edge_density"],
                            debug={
                                "surface": surface.model_dump(),
                                "reason": "edge-density-obstacle",
                            },
                        )

        # Door detection and specific instructions
        if ctx.motion.is_moving:
            for surface in ctx.surfaces:
                if surface.kind == SurfaceKind.DOOR:
                    if surface.confidence >= 0.6 and surface.distance_m is not None:
                        handle_side = str(surface.attributes.get("handle_side", "unknown"))
                        if surface.distance_m <= 1.5:
                            return AgentDecision(
                                action=AgentAction.GUIDE,
                                priority=76 if surface.attributes.get("handle_detected") else 72,
                                message=_door_guidance_message(surface),
                                haptic=_door_haptic(surface),
                                agents_consulted=[self.name, "door_detection"],
                                debug={
                                    "surface": surface.model_dump(),
                                    "reason": "door-handle-guidance" if surface.attributes.get("handle_detected") else "door-frame-guidance",
                                },
                            )
                        elif surface.distance_m <= 3.0:
                            # Farther door - just announce it
                            return AgentDecision(
                                action=AgentAction.GUIDE,
                                priority=60,
                                message=_door_approach_message(surface),
                                haptic=_door_haptic(surface) if handle_side in {"left", "right"} else HapticPattern.NONE,
                                agents_consulted=[self.name, "door_detection"],
                                debug={
                                    "surface": surface.model_dump(),
                                    "reason": "door-detection-announce",
                                },
                            )

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
        if ctx.scene.location_type not in {"sidewalk", "street", "street_crossing", "outdoor", "unknown"}:
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

        avoidance = _avoidance_phrase(det.direction, det.distance_m)
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
            distance = ""
            if ctx.route.next_turn_distance_m is not None:
                distance = f" in about {round(ctx.route.next_turn_distance_m * 3.28084)} feet"
            
            # Periodically boost priority so navigation breaks through
            # surface/obstacle agents. Safety (>=90) still overrides.
            is_boost_frame = (WayfindingAgent._frame_counter % self._NAV_BOOST_EVERY == 0)
            priority = 78 if is_boost_frame else 60
            
            return AgentDecision(
                action=AgentAction.GUIDE,
                priority=priority,
                message=f"{ctx.route.next_instruction}{distance}.",
                haptic=HapticPattern.NONE,
                agents_consulted=[self.name],
                debug={"route": ctx.route.model_dump(), "boosted": is_boost_frame},
            )
        return None


class OrientationAgent(BaseAgent):
    name = "orientation"

    def handle(self, ctx: FrameContext) -> Optional[AgentDecision]:
        if ctx.motion.is_moving and not ctx.user.query and ctx.user.mode != "orientation":
            return None

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
        if ctx.scene.location_type not in {"hallway", "room", "corridor", "building", "indoor", "unknown"}:
            return None

        if ctx.user.mode == "orientation" or not ctx.motion.is_moving:
            return None

        # Defer to TargetFindingAgent when the user is actively searching
        if ctx.user.target or ctx.user.query:
            return None

        # Check for doors and provide hand guidance
        for surface in ctx.surfaces:
            if surface.kind == SurfaceKind.DOOR and surface.confidence >= 0.6:
                if surface.distance_m is not None and surface.distance_m <= 2.0:
                    return AgentDecision(
                        action=AgentAction.GUIDE,
                        priority=72,
                        message=_door_guidance_message(surface),
                        haptic=_door_haptic(surface),
                        agents_consulted=[self.name, "door_guidance"],
                        debug={
                            "surface": surface.model_dump(),
                            "reason": "door-handle-guidance" if surface.attributes.get("handle_detected") else "door-frame-guidance",
                        },
                    )

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

        # Suggest avoidance direction
        if nearest.direction == Direction.SLIGHT_LEFT or nearest.direction == Direction.CENTER:
            avoid_dir = "right"
            avoid_haptic = HapticPattern.RIGHT
        else:
            avoid_dir = "left"
            avoid_haptic = HapticPattern.LEFT

        obstacle_list = ", ".join(sorted({d.label.lower() for d in blocking}))
        distance = _distance_phrase(nearest.distance_m)

        if nearest.distance_m is not None and nearest.distance_m <= 1.0:
            return AgentDecision(
                action=AgentAction.WARN,
                priority=78,
                message=f"Indoor obstacle: {nearest.label} {_direction_phrase(nearest.direction)}, {distance}. Move {avoid_dir} to go around.",
                haptic=avoid_haptic,
                agents_consulted=[self.name],
                debug={
                    "blocking_objects": [d.model_dump() for d in blocking],
                    "avoidance_direction": avoid_dir,
                    "reason": "indoor-obstacle-close",
                },
            )

        return AgentDecision(
            action=AgentAction.GUIDE,
            priority=68,
            message=f"{nearest.label.capitalize()} ahead {_direction_phrase(nearest.direction)}, {distance}. Veer {avoid_dir} to avoid.",
            haptic=avoid_haptic,
            agents_consulted=[self.name],
            debug={
                "blocking_objects": [d.model_dump() for d in blocking],
                "avoidance_direction": avoid_dir,
                "reason": "indoor-obstacle-avoidance",
            },
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
        side = str(surface.attributes.get("handle_side", "ahead")).replace("_", " ")
        height = str(surface.attributes.get("handle_height_zone", "hand height"))
        return (
            f"Door handle {side} {_direction_phrase(surface.direction)}, {height}, "
            f"{_distance_phrase(surface.distance_m)}."
        )
    return f"Door {_direction_phrase(surface.direction)} {_distance_phrase(surface.distance_m)}. I do not see the handle yet."


def _door_guidance_message(surface: SurfaceObservation) -> str:
    distance = _distance_phrase(surface.distance_m)
    if not surface.attributes.get("handle_detected"):
        return f"Door ahead {distance}. I do not see the handle; stop, scan the door edge, then find the handle by touch."

    side = str(surface.attributes.get("handle_side", "ahead")).replace("_", " ")
    hand = str(surface.attributes.get("recommended_hand", "either hand")).replace("_", " ")
    height = str(surface.attributes.get("handle_height_zone", "hand height"))
    action = str(
        surface.attributes.get(
            "handle_action",
            "use the handle, then gently test whether the door pushes or pulls",
        )
    )
    return f"Door handle on the {side}, {height}, {distance}. Use your {hand}; {action}."


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
