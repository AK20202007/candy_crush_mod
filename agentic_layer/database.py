from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional

from .models import AgentDecision, FrameContext


class MongoTelemetryStore:
    """Optional MongoDB sink for structured agent decisions.

    The store intentionally records no raw frames or image crops. It persists
    the normalized FrameContext, AgentDecision, and a few indexed convenience
    fields for replay/debug dashboards.
    """

    def __init__(
        self,
        collection: Any = None,
        enabled: bool = False,
        unavailable_reason: Optional[str] = None,
    ) -> None:
        self._collection = collection
        self.enabled = bool(enabled and collection is not None)
        self.unavailable_reason = unavailable_reason

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "MongoTelemetryStore":
        env = env or os.environ
        uri = (env.get("MONGODB_URI") or "").strip()
        if not uri:
            return cls(enabled=False, unavailable_reason="MONGODB_URI is unset")

        db_name = (env.get("MONGODB_DB") or "glasses_agentic").strip()
        collection_name = (env.get("MONGODB_COLLECTION") or "decision_events").strip()
        try:
            from pymongo import MongoClient
        except Exception as exc:
            return cls(enabled=False, unavailable_reason=f"pymongo unavailable: {exc}")

        try:
            client = MongoClient(uri, serverSelectionTimeoutMS=1500)
            client.admin.command("ping")
            collection = client[db_name][collection_name]
            collection.create_index([("created_at", -1)])
            collection.create_index([("frame_id", 1)])
            collection.create_index([("decision.action", 1), ("decision.priority", -1)])
            return cls(collection=collection, enabled=True)
        except Exception as exc:
            return cls(enabled=False, unavailable_reason=f"MongoDB connection failed: {exc}")

    @staticmethod
    def build_document(ctx: FrameContext, decision: AgentDecision) -> Dict[str, Any]:
        ctx_doc = ctx.model_dump()
        decision_doc = decision.model_dump()
        return {
            "created_at": datetime.now(timezone.utc),
            "timestamp_ms": ctx.timestamp_ms,
            "frame_id": ctx.frame_id,
            "context": ctx_doc,
            "decision": decision_doc,
            "route": ctx_doc.get("route", {}),
            "scene": ctx_doc.get("scene", {}),
            "user": ctx_doc.get("user", {}),
            "summary": {
                "detection_count": len(ctx.detections),
                "warning_count": len(ctx.warnings),
                "surface_count": len(ctx.surfaces),
                "action": decision_doc.get("action"),
                "priority": decision.priority,
                "haptic": decision_doc.get("haptic"),
                "requires_human": decision.requires_human,
            },
        }

    def record_decision(self, ctx: FrameContext, decision: AgentDecision) -> bool:
        if not self.enabled or self._collection is None:
            return False
        self._collection.insert_one(self.build_document(ctx, decision))
        return True
