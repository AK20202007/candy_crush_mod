from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_PROFILE_NAME = "balanced"
DEFAULT_PROFILE_PATH = Path(__file__).resolve().parents[1] / "config" / "agentic_profiles.json"


@dataclass(frozen=True)
class AgenticProfile:
    name: str
    description: str
    model_path: str = "yolov8n.pt"
    conf: float = 0.35
    iou: float = 0.5
    imgsz: int = 640
    confirm_frames: int = 2
    obstacle_area_ratio: float = 0.12
    person_center_radius: float = 0.18
    warning_cooldown_s: float = 2.5
    distance_scale: float = 1.0
    visual_confidence: float = 0.75
    location_type: str = "unknown"
    agent_repeat_ms: int = 4500
    agent_urgent_repeat_ms: int = 1000

    @classmethod
    def from_dict(cls, name: str, data: Dict[str, Any]) -> "AgenticProfile":
        return cls(
            name=name,
            description=str(data.get("description", "")),
            model_path=str(data.get("model_path", cls.model_path)),
            conf=float(data.get("conf", cls.conf)),
            iou=float(data.get("iou", cls.iou)),
            imgsz=int(data.get("imgsz", cls.imgsz)),
            confirm_frames=int(data.get("confirm_frames", cls.confirm_frames)),
            obstacle_area_ratio=float(data.get("obstacle_area_ratio", cls.obstacle_area_ratio)),
            person_center_radius=float(data.get("person_center_radius", cls.person_center_radius)),
            warning_cooldown_s=float(data.get("warning_cooldown_s", cls.warning_cooldown_s)),
            distance_scale=float(data.get("distance_scale", cls.distance_scale)),
            visual_confidence=float(data.get("visual_confidence", cls.visual_confidence)),
            location_type=str(data.get("location_type", cls.location_type)),
            agent_repeat_ms=int(data.get("agent_repeat_ms", cls.agent_repeat_ms)),
            agent_urgent_repeat_ms=int(data.get("agent_urgent_repeat_ms", cls.agent_urgent_repeat_ms)),
        )


def load_profiles(path: Optional[Path] = None) -> Dict[str, AgenticProfile]:
    profile_path = path or DEFAULT_PROFILE_PATH
    with profile_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"{profile_path} must contain a JSON object keyed by profile name")
    return {name: AgenticProfile.from_dict(name, data) for name, data in raw.items()}


def load_profile(name: str = DEFAULT_PROFILE_NAME, path: Optional[Path] = None) -> AgenticProfile:
    profiles = load_profiles(path)
    if name not in profiles:
        available = ", ".join(sorted(profiles))
        raise ValueError(f"Unknown profile {name!r}. Available profiles: {available}")
    return profiles[name]


def choose(value: Any, fallback: Any) -> Any:
    return fallback if value is None else value
