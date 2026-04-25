from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Type, TypeVar


class Direction(str, Enum):
    LEFT = "left"
    SLIGHT_LEFT = "slight_left"
    CENTER = "center"
    SLIGHT_RIGHT = "slight_right"
    RIGHT = "right"
    UNKNOWN = "unknown"


class AgentAction(str, Enum):
    SILENT = "silent"
    WARN = "warn"
    GUIDE = "guide"
    ORIENT = "orient"
    ANSWER = "answer"
    ASK = "ask"
    ESCALATE = "escalate"


class HapticPattern(str, Enum):
    NONE = "none"
    STOP = "stop"
    CAUTION = "caution"
    LEFT = "left"
    RIGHT = "right"
    SUCCESS = "success"


Severity = Literal["info", "low", "medium", "high", "critical"]

E = TypeVar("E", bound=Enum)


def _enum(enum_type: Type[E], value: Any, default: E) -> E:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(value)
    except Exception:
        return default


def _bounded_float(value: Any, default: float, low: float, high: float) -> float:
    try:
        out = float(value)
    except Exception:
        out = default
    return max(low, min(high, out))


@dataclass
class Serializable:
    def to_dict(self) -> Dict[str, Any]:
        def clean(v: Any) -> Any:
            if isinstance(v, Enum):
                return v.value
            if isinstance(v, list):
                return [clean(i) for i in v]
            if isinstance(v, dict):
                return {k: clean(val) for k, val in v.items()}
            if hasattr(v, "to_dict"):
                return v.to_dict()
            return v

        return clean(asdict(self))

    # Compatibility with Pydantic-style examples.
    def model_dump(self) -> Dict[str, Any]:
        return self.to_dict()


@dataclass
class BBox(Serializable):
    """Pixel-space bounding box. x1/y1 upper-left, x2/y2 lower-right."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> Optional["BBox"]:
        if not data:
            return None
        return cls(
            x1=float(data.get("x1", 0.0)),
            y1=float(data.get("y1", 0.0)),
            x2=float(data.get("x2", 0.0)),
            y2=float(data.get("y2", 0.0)),
        )


@dataclass
class Detection(Serializable):
    label: str
    confidence: float
    bbox: Optional[BBox] = None
    distance_m: Optional[float] = None
    direction: Direction = Direction.UNKNOWN
    source: str = "vision"
    attributes: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.label = self.label.lower().strip()
        self.confidence = _bounded_float(self.confidence, 0.0, 0.0, 1.0)
        self.direction = _enum(Direction, self.direction, Direction.UNKNOWN)
        if self.distance_m is not None:
            self.distance_m = max(0.0, float(self.distance_m))

    def clock_direction(self) -> str:
        return {
            Direction.LEFT: "9 o'clock",
            Direction.SLIGHT_LEFT: "10 o'clock",
            Direction.CENTER: "12 o'clock",
            Direction.SLIGHT_RIGHT: "2 o'clock",
            Direction.RIGHT: "3 o'clock",
            Direction.UNKNOWN: "ahead",
        }[self.direction]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Detection":
        return cls(
            label=str(data.get("label", "unknown")),
            confidence=float(data.get("confidence", 0.0)),
            bbox=BBox.from_dict(data.get("bbox")),
            distance_m=data.get("distance_m"),
            direction=_enum(Direction, data.get("direction", "unknown"), Direction.UNKNOWN),
            source=str(data.get("source", "vision")),
            attributes=dict(data.get("attributes", {})),
        )


@dataclass
class WarningEvent(Serializable):
    kind: str
    message: str
    severity: Severity = "medium"
    confidence: float = 0.0
    distance_m: Optional[float] = None
    direction: Direction = Direction.UNKNOWN
    source: str = "existing-warning-system"
    expires_ms: int = 1500
    observed_at_ms: Optional[int] = None

    def __post_init__(self) -> None:
        self.kind = self.kind.lower().strip()
        self.message = self.message.strip()
        if self.severity not in {"info", "low", "medium", "high", "critical"}:
            self.severity = "medium"
        self.confidence = _bounded_float(self.confidence, 0.0, 0.0, 1.0)
        self.direction = _enum(Direction, self.direction, Direction.UNKNOWN)
        if self.distance_m is not None:
            self.distance_m = max(0.0, float(self.distance_m))

    def is_immediate(self) -> bool:
        if self.severity == "critical":
            return True
        if self.distance_m is not None and self.distance_m <= 1.2 and self.severity in {"high", "medium"}:
            return True
        return False

    def is_expired(self, now_ms: int) -> bool:
        if self.observed_at_ms is None or self.expires_ms <= 0:
            return False
        return now_ms - self.observed_at_ms > self.expires_ms

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WarningEvent":
        return cls(
            kind=str(data.get("kind", "warning")),
            message=str(data.get("message", "hazard ahead")),
            severity=data.get("severity", "medium"),
            confidence=float(data.get("confidence", 0.0)),
            distance_m=data.get("distance_m"),
            direction=_enum(Direction, data.get("direction", "unknown"), Direction.UNKNOWN),
            source=str(data.get("source", "existing-warning-system")),
            expires_ms=int(data.get("expires_ms", 1500)),
            observed_at_ms=data.get("observed_at_ms"),
        )


@dataclass
class MotionState(Serializable):
    is_moving: bool = False
    speed_mps: float = 0.0
    phone_pitch_deg: Optional[float] = None
    phone_yaw_deg: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "MotionState":
        data = data or {}
        return cls(
            is_moving=bool(data.get("is_moving", False)),
            speed_mps=max(0.0, float(data.get("speed_mps", 0.0) or 0.0)),
            phone_pitch_deg=data.get("phone_pitch_deg"),
            phone_yaw_deg=data.get("phone_yaw_deg"),
        )


@dataclass
class RouteState(Serializable):
    active: bool = False
    destination: Optional[str] = None
    next_instruction: Optional[str] = None
    next_turn_distance_m: Optional[float] = None
    off_route: bool = False

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "RouteState":
        data = data or {}
        return cls(
            active=bool(data.get("active", False)),
            destination=data.get("destination"),
            next_instruction=data.get("next_instruction"),
            next_turn_distance_m=data.get("next_turn_distance_m"),
            off_route=bool(data.get("off_route", False)),
        )


@dataclass
class SceneState(Serializable):
    location_type: str = "unknown"
    lighting: str = "unknown"
    visual_confidence: float = 0.7
    gps_available: bool = False
    map_confidence: float = 0.0

    def __post_init__(self) -> None:
        self.visual_confidence = _bounded_float(self.visual_confidence, 0.7, 0.0, 1.0)
        self.map_confidence = _bounded_float(self.map_confidence, 0.0, 0.0, 1.0)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "SceneState":
        data = data or {}
        return cls(
            location_type=str(data.get("location_type", "unknown")),
            lighting=str(data.get("lighting", "unknown")),
            visual_confidence=float(data.get("visual_confidence", 0.7)),
            gps_available=bool(data.get("gps_available", False)),
            map_confidence=float(data.get("map_confidence", 0.0)),
        )


@dataclass
class UserState(Serializable):
    query: Optional[str] = None
    verbosity: str = "minimal"
    mode: str = "auto"
    target: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "UserState":
        data = data or {}
        return cls(
            query=data.get("query"),
            verbosity=str(data.get("verbosity", "minimal")),
            mode=str(data.get("mode", "auto")),
            target=data.get("target"),
        )


@dataclass
class FrameContext(Serializable):
    timestamp_ms: int
    frame_id: str
    detections: List[Detection] = field(default_factory=list)
    warnings: List[WarningEvent] = field(default_factory=list)
    motion: MotionState = field(default_factory=MotionState)
    route: RouteState = field(default_factory=RouteState)
    scene: SceneState = field(default_factory=SceneState)
    user: UserState = field(default_factory=UserState)
    last_spoken: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FrameContext":
        return cls(
            timestamp_ms=int(data.get("timestamp_ms", 0)),
            frame_id=str(data.get("frame_id", "frame")),
            detections=[Detection.from_dict(d) for d in data.get("detections", [])],
            warnings=[WarningEvent.from_dict(w) for w in data.get("warnings", [])],
            motion=MotionState.from_dict(data.get("motion")),
            route=RouteState.from_dict(data.get("route")),
            scene=SceneState.from_dict(data.get("scene")),
            user=UserState.from_dict(data.get("user")),
            last_spoken=data.get("last_spoken"),
        )


@dataclass
class AgentDecision(Serializable):
    action: AgentAction
    priority: int = 0
    message: str = ""
    haptic: HapticPattern = HapticPattern.NONE
    agents_consulted: List[str] = field(default_factory=list)
    should_speak: bool = True
    requires_human: bool = False
    debug: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.action = _enum(AgentAction, self.action, AgentAction.SILENT)
        self.haptic = _enum(HapticPattern, self.haptic, HapticPattern.NONE)
        self.priority = int(max(0, min(100, self.priority)))

    @classmethod
    def silent(cls, reason: str = "nothing important") -> "AgentDecision":
        return cls(
            action=AgentAction.SILENT,
            priority=0,
            message="",
            haptic=HapticPattern.NONE,
            should_speak=False,
            debug={"reason": reason},
        )
