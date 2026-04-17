# Licensed under the LGPL: https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html
# For details: https://github.com/pylint-dev/astroid/blob/main/LICENSE
# Copyright (c) https://github.com/pylint-dev/astroid/blob/main/CONTRIBUTORS.txt

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from astroid.ai.bridge import AIInferenceRequest
from astroid.manager import AstroidManager

_MIN_CONFIDENCE = 0.5


class AIInferenceObserver(Protocol):
    def record(self, event: str, **payload: Any) -> None: ...  # pragma: no cover


@dataclass(frozen=True)
class AIInferencePolicy:
    enabled: bool
    timeout_ms: int
    max_candidates: int
    budget_per_module: int
    allowed_scenarios: frozenset[str]
    min_confidence: float = _MIN_CONFIDENCE

    def allows(self, request: AIInferenceRequest) -> bool:
        return self.enabled and request.scenario in self.allowed_scenarios


def policy_from_manager(manager: AstroidManager) -> AIInferencePolicy:
    return AIInferencePolicy(
        enabled=manager.ai_inference_enabled,
        timeout_ms=manager.ai_timeout_ms,
        max_candidates=manager.ai_max_candidates,
        budget_per_module=manager.ai_budget_per_module,
        allowed_scenarios=manager.ai_allowed_scenarios,
    )


def consume_budget(manager: AstroidManager, request: AIInferenceRequest) -> bool:
    budget = manager.ai_budget_per_module
    if budget <= 0:
        return False
    used = manager.ai_budget_counts.get(request.module_name, 0)
    if used >= budget:
        return False
    manager.ai_budget_counts[request.module_name] = used + 1
    return True


def observe(manager: AstroidManager, event: str, **payload: Any) -> None:
    observer = manager.ai_observer
    if observer is None:
        return
    if hasattr(observer, "record"):
        observer.record(event, **payload)
        return
    if callable(observer):
        callable_observer = observer
        callable_observer(event, **payload)
