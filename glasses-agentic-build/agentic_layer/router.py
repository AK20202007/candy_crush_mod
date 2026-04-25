from __future__ import annotations

import time
from typing import Optional

from .agents import FallbackAgent, OrientationAgent, SafetyAgent, TargetFindingAgent, WayfindingAgent
from .models import AgentAction, AgentDecision, FrameContext


class AgenticNavigationRouter:
    """
    Deterministic supervisory router for blind/low-vision navigation.

    This is intentionally not a free-form LLM for immediate hazards. The existing
    warning detector remains authoritative; this layer decides what should be said
    right now given safety, user intent, route state, and object-distance cues.
    """

    def __init__(self, min_repeat_interval_ms: int = 1800):
        self.agents = [
            SafetyAgent(),
            TargetFindingAgent(),
            WayfindingAgent(),
            OrientationAgent(),
            FallbackAgent(),
        ]
        self.min_repeat_interval_ms = min_repeat_interval_ms
        self._last_message: Optional[str] = None
        self._last_message_at_ms: int = 0

    def decide(self, ctx: FrameContext) -> AgentDecision:
        candidates = []
        for agent in self.agents:
            decision = agent.handle(ctx)
            if decision is not None:
                candidates.append(decision)

        if not candidates:
            return AgentDecision.silent("no candidates")

        decision = sorted(candidates, key=lambda d: d.priority, reverse=True)[0]
        decision.debug["candidates"] = [
            {
                "action": c.action.value if hasattr(c.action, "value") else c.action,
                "priority": c.priority,
                "message": c.message,
                "agents": c.agents_consulted,
            }
            for c in candidates
        ]
        return self._apply_anti_spam(ctx, decision)

    def _apply_anti_spam(self, ctx: FrameContext, decision: AgentDecision) -> AgentDecision:
        if not decision.should_speak or decision.action == AgentAction.SILENT:
            return decision

        # Never suppress immediate safety warnings.
        if decision.priority >= 90:
            self._remember(decision, ctx.timestamp_ms)
            return decision

        now = ctx.timestamp_ms or int(time.time() * 1000)
        same_as_last = decision.message == self._last_message or decision.message == ctx.last_spoken
        if same_as_last and now - self._last_message_at_ms < self.min_repeat_interval_ms:
            return AgentDecision.silent("repeat suppressed")

        self._remember(decision, now)
        return decision

    def _remember(self, decision: AgentDecision, timestamp_ms: int) -> None:
        if decision.message:
            self._last_message = decision.message
            self._last_message_at_ms = timestamp_ms
