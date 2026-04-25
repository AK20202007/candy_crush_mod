from __future__ import annotations

import re
from typing import Iterable, List, Optional

from .models import AgentAction, AgentDecision, Detection, Direction, FrameContext, HapticPattern

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


def _distance_phrase(distance_m: Optional[float]) -> str:
    if distance_m is None:
        return "nearby"
    if distance_m < 0.35:
        return "within arm's reach"
    if distance_m < 1.0:
        return "less than 3 feet away"
    feet = round(distance_m * 3.28084)
    return f"about {feet} feet away"


def _direction_phrase(direction: Direction) -> str:
    return {
        Direction.LEFT: "on your left",
        Direction.SLIGHT_LEFT: "slightly left",
        Direction.CENTER: "straight ahead",
        Direction.SLIGHT_RIGHT: "slightly right",
        Direction.RIGHT: "on your right",
        Direction.UNKNOWN: "ahead",
    }[direction]


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

    def handle(self, ctx: FrameContext) -> Optional[AgentDecision]:
        raise NotImplementedError


class SafetyAgent(BaseAgent):
    """Highest-priority agent. Existing explicit warnings win over everything else."""

    name = "safety"

    def handle(self, ctx: FrameContext) -> Optional[AgentDecision]:
        if ctx.warnings:
            high = sorted(
                ctx.warnings,
                key=lambda w: (
                    SEVERITY_SCORE.get(w.severity, 3),
                    w.confidence,
                    -(w.distance_m if w.distance_m is not None else 99.0),
                ),
                reverse=True,
            )[0]
            if high.confidence >= 0.50 and (ctx.motion.is_moving or high.is_immediate()):
                prefix = "Stop." if high.is_immediate() or high.severity == "critical" else "Caution."
                base = high.message.strip()
                if base:
                    base = base[0].upper() + base[1:]
                if high.direction != Direction.UNKNOWN and "ahead" not in base.lower():
                    base = f"{base} {_direction_phrase(high.direction)}"
                if base and not base.endswith("."):
                    base += "."
                return AgentDecision(
                    action=AgentAction.WARN,
                    priority=100 if high.is_immediate() else 85,
                    message=f"{prefix} {base}",
                    haptic=HapticPattern.STOP if high.is_immediate() else _haptic_for_direction(high.direction),
                    agents_consulted=[self.name],
                    debug={"warning": high.model_dump(), "reason": "existing warning prioritized"},
                )

        hazards: List[Detection] = []
        for det in ctx.detections:
            label = det.label.lower()
            if label not in HAZARD_LABELS or det.confidence < 0.50:
                continue
            close = det.distance_m is not None and det.distance_m <= 1.4
            large = float(det.attributes.get("area_ratio", 0.0) or 0.0) >= 0.12
            centered = det.direction in {
                Direction.CENTER,
                Direction.SLIGHT_LEFT,
                Direction.SLIGHT_RIGHT,
                Direction.UNKNOWN,
            }
            if centered and (close or large):
                hazards.append(det)

        if hazards and ctx.motion.is_moving:
            det = sorted(hazards, key=lambda d: (d.distance_m if d.distance_m is not None else 9.9))[0]
            is_stop = det.label.lower() in STOP_HAZARDS or (det.distance_m is not None and det.distance_m <= 0.8)
            return AgentDecision(
                action=AgentAction.WARN,
                priority=95 if is_stop else 80,
                message=(
                    f"{'Stop' if is_stop else 'Caution'}: {det.label} "
                    f"{_direction_phrase(det.direction)}, {_distance_phrase(det.distance_m)}."
                ),
                haptic=_haptic_for_direction(det.direction, urgent=is_stop),
                agents_consulted=[self.name, "object_distance"],
                debug={"detection": det.model_dump(), "reason": "object-distance hazard"},
            )

        if ctx.scene.location_type == "street_crossing":
            return AgentDecision(
                action=AgentAction.WARN,
                priority=90,
                message="Intersection area detected. I can describe what I see, but do not rely on me to decide when it is safe to cross.",
                haptic=HapticPattern.CAUTION,
                agents_consulted=[self.name],
                debug={"policy": "crossing-description-only"},
            )
        return None


class TargetFindingAgent(BaseAgent):
    name = "target_finding"

    def handle(self, ctx: FrameContext) -> Optional[AgentDecision]:
        target = _target_from_context(ctx)
        if not target:
            return None

        matches = _matching_detections(ctx.detections, target)
        if matches:
            best = sorted(matches, key=lambda d: (-d.confidence, d.distance_m if d.distance_m is not None else 99))[0]
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


class WayfindingAgent(BaseAgent):
    name = "wayfinding"

    def handle(self, ctx: FrameContext) -> Optional[AgentDecision]:
        if not ctx.route.active:
            return None
        if ctx.route.off_route:
            return AgentDecision(
                action=AgentAction.ASK,
                priority=65,
                message="You may be off route. Stop and slowly scan so I can re-orient you.",
                haptic=HapticPattern.CAUTION,
                agents_consulted=[self.name],
                debug={"route": ctx.route.model_dump()},
            )
        if ctx.route.next_instruction:
            distance = ""
            if ctx.route.next_turn_distance_m is not None:
                distance = f" in about {round(ctx.route.next_turn_distance_m * 3.28084)} feet"
            return AgentDecision(
                action=AgentAction.GUIDE,
                priority=60,
                message=f"{ctx.route.next_instruction}{distance}.",
                haptic=HapticPattern.NONE,
                agents_consulted=[self.name],
                debug={"route": ctx.route.model_dump()},
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


def _matching_detections(detections: Iterable[Detection], target: str) -> List[Detection]:
    variants = set(TARGET_KEYWORDS.get(target, [target]))
    variants.add(target)
    out = []
    for det in detections:
        label = det.label.lower()
        if label in variants or any(v in label for v in variants):
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
