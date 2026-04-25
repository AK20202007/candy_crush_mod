from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .models import AgentDecision, FrameContext, RouteState, UserState
from .router import AgenticNavigationRouter


@dataclass
class RuntimeEvent:
    event_type: str
    timestamp_ms: int
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionTrace:
    frame_id: str
    timestamp_ms: int
    action: str
    priority: int
    message: str
    agents_consulted: List[str]
    requires_human: bool
    candidate_count: int
    debug: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_decision(cls, ctx: FrameContext, decision: AgentDecision) -> "DecisionTrace":
        candidates = decision.debug.get("candidates", [])
        return cls(
            frame_id=ctx.frame_id,
            timestamp_ms=ctx.timestamp_ms,
            action=decision.action.value,
            priority=decision.priority,
            message=decision.message,
            agents_consulted=list(decision.agents_consulted),
            requires_human=decision.requires_human,
            candidate_count=len(candidates) if isinstance(candidates, list) else 0,
            debug=dict(decision.debug),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "timestamp_ms": self.timestamp_ms,
            "action": self.action,
            "priority": self.priority,
            "message": self.message,
            "agents_consulted": self.agents_consulted,
            "requires_human": self.requires_human,
            "candidate_count": self.candidate_count,
            "debug": self.debug,
        }


class InMemoryEventBus:
    def __init__(self, max_events: int = 500) -> None:
        self.max_events = max(1, max_events)
        self._events: List[RuntimeEvent] = []

    def publish(self, event_type: str, payload: Optional[Dict[str, Any]] = None, timestamp_ms: Optional[int] = None) -> None:
        event = RuntimeEvent(
            event_type=event_type,
            timestamp_ms=timestamp_ms or int(time.time() * 1000),
            payload=payload or {},
        )
        self._events.append(event)
        if len(self._events) > self.max_events:
            del self._events[: len(self._events) - self.max_events]

    def recent(self, event_type: Optional[str] = None, limit: int = 50) -> List[RuntimeEvent]:
        events = self._events
        if event_type is not None:
            events = [event for event in events if event.event_type == event_type]
        return events[-max(1, limit) :]


class SharedNavigationState:
    """
    Small state container for non-camera inputs.

    This is intentionally simple. Mobile/wearable implementations can replace it
    with platform state, but tests and local prototypes need the same contract.
    """

    def __init__(self) -> None:
        self.route = RouteState(active=False)
        self.user = UserState()
        self.last_decision: Optional[DecisionTrace] = None

    def update_route(self, route: RouteState) -> None:
        self.route = route

    def update_user(self, user: UserState) -> None:
        self.user = user


class AgenticRuntime:
    def __init__(
        self,
        router: Optional[AgenticNavigationRouter] = None,
        event_bus: Optional[InMemoryEventBus] = None,
        state: Optional[SharedNavigationState] = None,
    ) -> None:
        self.router = router or AgenticNavigationRouter()
        self.event_bus = event_bus or InMemoryEventBus()
        self.state = state or SharedNavigationState()

    def decide(self, ctx: FrameContext) -> AgentDecision:
        self.event_bus.publish("frame_context", {"context": ctx.model_dump()}, timestamp_ms=ctx.timestamp_ms)
        decision = self.router.decide(ctx)
        trace = DecisionTrace.from_decision(ctx, decision)
        self.state.last_decision = trace
        self.event_bus.publish("agent_decision", {"decision": trace.to_dict()}, timestamp_ms=ctx.timestamp_ms)
        return decision
