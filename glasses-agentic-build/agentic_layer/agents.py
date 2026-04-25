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
MIN_WARNING_CONFIDENCE = {
    "critical": 0.35,
    "high": 0.45,
    "medium": 0.50,
    "low": 0.60,
    "info": 0.70,
}
MIN_TARGET_CONFIDENCE = 0.35


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
        Direction.LEFT: "at 9 o'clock",
        Direction.SLIGHT_LEFT: "at 10 o'clock",
        Direction.CENTER: "at 12 o'clock",
        Direction.SLIGHT_RIGHT: "at 2 o'clock",
        Direction.RIGHT: "at 3 o'clock",
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
            if _warning_confidence_ok(high.severity, high.confidence) and (ctx.motion.is_moving or high.is_immediate()):
                prefix = "Stop." if high.is_immediate() or high.severity == "critical" else "Caution."
                return AgentDecision(
                    action=AgentAction.WARN,
                    priority=100 if high.is_immediate() else 85,
                    message=f"{prefix} {_warning_message(high.message or high.kind, high.direction, high.distance_m)}",
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
            det = sorted(hazards, key=_hazard_sort_key)[0]
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
                requires_human=True,
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
        if det.confidence < MIN_TARGET_CONFIDENCE:
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


def _warning_confidence_ok(severity: str, confidence: float) -> bool:
    return confidence >= MIN_WARNING_CONFIDENCE.get(severity, 0.50)


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
