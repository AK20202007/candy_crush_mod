from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, Set

from .models import AgentAction, AgentDecision, FrameContext, HapticPattern


DEFAULT_WARNING_CONFIDENCE_FLOORS: Dict[str, float] = {
    "critical": 0.35,
    "high": 0.45,
    "medium": 0.50,
    "low": 0.60,
    "info": 0.70,
}

DEFAULT_PROHIBITED_PHRASES = {
    "safe to cross",
    "safe to cross now",
    "it is safe",
    "all clear",
    "clear to cross",
    "go now",
    "cross now",
    "proceed into traffic",
}

DEFAULT_HIGH_RISK_LOCATIONS = {
    "street_crossing",
    "intersection",
    "roadway",
    "curb_edge",
    "platform_edge",
}


@dataclass(frozen=True)
class SafetyPolicy:
    """
    Auditable product policy for the agentic layer.

    The router and agents should stay deterministic for safety-relevant output.
    This policy centralizes thresholds and language constraints so the behavior
    can be reviewed without digging through each agent implementation.
    """

    warning_confidence_floors: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WARNING_CONFIDENCE_FLOORS))
    target_confidence_floor: float = 0.35
    high_risk_locations: Set[str] = field(default_factory=lambda: set(DEFAULT_HIGH_RISK_LOCATIONS))
    prohibited_phrases: Set[str] = field(default_factory=lambda: set(DEFAULT_PROHIBITED_PHRASES))

    def warning_confidence_ok(self, severity: str, confidence: float) -> bool:
        return confidence >= self.warning_confidence_floors.get(severity, 0.50)

    def confidence_prefix(self, severity: str, confidence: float) -> str:
        floor = self.warning_confidence_floors.get(severity, 0.50)
        if confidence < floor + 0.10:
            return "possible"
        if confidence < floor + 0.25:
            return "likely"
        return ""

    def is_high_risk_context(self, ctx: FrameContext) -> bool:
        return ctx.scene.location_type in self.high_risk_locations

    def contains_prohibited_safety_claim(self, message: str) -> bool:
        normalized = _normalize(message)
        return any(_contains_phrase(normalized, phrase) for phrase in self.prohibited_phrases)

    def enforce_decision(self, ctx: FrameContext, decision: AgentDecision) -> AgentDecision:
        if not decision.should_speak or decision.action == AgentAction.SILENT:
            return decision

        if self.contains_prohibited_safety_claim(decision.message):
            return self._blocked_decision(
                reason="prohibited safety claim",
                original=decision,
                message="I cannot make that safety call. Use crossing aids or ask for human assistance.",
            )

        if self.is_high_risk_context(ctx) and decision.action in {AgentAction.GUIDE, AgentAction.ANSWER}:
            return self._blocked_decision(
                reason="high-risk context blocks guidance",
                original=decision,
                message="High-risk area detected. I can describe what I see, but confirm with crossing aids or a person before moving.",
            )

        if self.is_high_risk_context(ctx):
            decision.requires_human = True
            decision.debug.setdefault("policy", "high-risk-human-judgment")
        return decision

    def _blocked_decision(self, reason: str, original: AgentDecision, message: str) -> AgentDecision:
        return AgentDecision(
            action=AgentAction.WARN,
            priority=max(90, original.priority),
            message=message,
            haptic=HapticPattern.CAUTION,
            agents_consulted=list(dict.fromkeys(original.agents_consulted + ["safety_policy"])),
            should_speak=True,
            requires_human=True,
            debug={
                "policy": reason,
                "blocked_decision": original.model_dump(),
            },
        )


def _normalize(message: str) -> str:
    return re.sub(r"\s+", " ", message.lower()).strip()


def _contains_phrase(text: str, phrase: str) -> bool:
    phrase = _normalize(phrase)
    return bool(re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text))


def safe_join_agents(agents: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(a for a in agents if a))
