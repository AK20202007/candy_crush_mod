from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentic_layer import AgenticNavigationRouter, FrameContext  # noqa: E402


DEFAULT_SCENARIO_FILE = Path(__file__).resolve().parent / "scenarios" / "router_scenarios.json"


def _load(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise AssertionError(f"{path} must contain a JSON list of scenarios")
    return data


def _assert_expected(name: str, step_index: int, decision: Any, expected: dict[str, Any]) -> None:
    prefix = f"{name} step {step_index}"

    if "action" in expected:
        assert decision.action.value == expected["action"], f"{prefix}: action {decision.action.value!r}"
    if "priority" in expected:
        assert decision.priority == expected["priority"], f"{prefix}: priority {decision.priority!r}"
    if "haptic" in expected:
        assert decision.haptic.value == expected["haptic"], f"{prefix}: haptic {decision.haptic.value!r}"
    if "should_speak" in expected:
        assert decision.should_speak is expected["should_speak"], f"{prefix}: should_speak {decision.should_speak!r}"
    if "requires_human" in expected:
        assert decision.requires_human is expected["requires_human"], f"{prefix}: requires_human {decision.requires_human!r}"

    message = decision.message.lower()
    for text in expected.get("message_contains", []):
        assert text.lower() in message, f"{prefix}: message missing {text!r}: {decision.message!r}"
    for text in expected.get("message_not_contains", []):
        assert text.lower() not in message, f"{prefix}: message unexpectedly contains {text!r}: {decision.message!r}"

    if "debug_reason" in expected:
        assert decision.debug.get("reason") == expected["debug_reason"], f"{prefix}: debug reason {decision.debug!r}"


def run_scenarios(path: Path) -> None:
    scenarios = _load(path)
    for scenario in scenarios:
        name = str(scenario.get("name", "unnamed"))
        router = AgenticNavigationRouter(**scenario.get("router", {}))
        steps = scenario.get("steps")
        if not isinstance(steps, list) or not steps:
            raise AssertionError(f"{name}: scenario must define at least one step")

        for index, step in enumerate(steps, start=1):
            ctx = FrameContext.from_dict(step["context"])
            decision = router.decide(ctx)
            _assert_expected(name, index, decision, step.get("expect", {}))


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SCENARIO_FILE
    run_scenarios(path)
    print(f"router scenarios passed: {path}")


if __name__ == "__main__":
    main()
