from __future__ import annotations

import time
from typing import Optional

from .agents import CrossingSignalAgent, FallbackAgent, IndoorNavigationAgent, OrientationAgent, SafetyAgent, SidewalkAgent, TargetFindingAgent, WayfindingAgent
from .models import AgentAction, AgentDecision, FrameContext
from .policy import SafetyPolicy


class AgenticNavigationRouter:
    """
    Deterministic supervisory router for blind/low-vision navigation.

    This is intentionally not a free-form LLM for immediate hazards. The existing
    warning detector remains authoritative; this layer decides what should be said
    right now given safety, user intent, route state, and object-distance cues.
    """

    def __init__(
        self,
        min_repeat_interval_ms: int = 1800,
        min_urgent_repeat_interval_ms: int = 1000,
        policy: Optional[SafetyPolicy] = None,
    ):
        self.policy = policy or SafetyPolicy()
        self.agents = [
            SafetyAgent(self.policy),
            SidewalkAgent(self.policy),
            CrossingSignalAgent(self.policy),
            TargetFindingAgent(self.policy),
            WayfindingAgent(self.policy),
            IndoorNavigationAgent(self.policy),
            OrientationAgent(self.policy),
            FallbackAgent(self.policy),
        ]
        self.min_repeat_interval_ms = min_repeat_interval_ms
        self.min_urgent_repeat_interval_ms = min_urgent_repeat_interval_ms
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
        decision = self.policy.enforce_decision(ctx, decision)
        return self._apply_anti_spam(ctx, decision)

    def _apply_anti_spam(self, ctx: FrameContext, decision: AgentDecision) -> AgentDecision:
        if not decision.should_speak or decision.action == AgentAction.SILENT:
            return decision

        now = ctx.timestamp_ms or int(time.time() * 1000)
        same_as_last = decision.message == self._last_message or decision.message == ctx.last_spoken
        # Use shorter interval for urgent, longer for normal, and allow GUIDE
        # actions (navigation prompts) through more often so destination
        # instructions are not silenced by anti-spam.
        if decision.priority >= 90:
            interval = self.min_urgent_repeat_interval_ms
        elif decision.action == AgentAction.GUIDE:
            interval = max(3000, self.min_repeat_interval_ms)  # At least 3s but let nav through
        else:
            interval = self.min_repeat_interval_ms
        if same_as_last and now - self._last_message_at_ms < interval:
            suppressed = AgentDecision.silent("repeat suppressed")
            suppressed.debug["suppressed_decision"] = {
                "action": decision.action.value if hasattr(decision.action, "value") else decision.action,
                "priority": decision.priority,
                "message": decision.message,
                "haptic": decision.haptic.value if hasattr(decision.haptic, "value") else decision.haptic,
                "requires_human": decision.requires_human,
            }
            return suppressed

        self._remember(decision, now)
        return decision

    def _remember(self, decision: AgentDecision, timestamp_ms: int) -> None:
        if decision.message:
            self._last_message = decision.message
            self._last_message_at_ms = timestamp_ms
