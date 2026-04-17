# Licensed under the LGPL: https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html
# For details: https://github.com/pylint-dev/astroid/blob/main/LICENSE
# Copyright (c) https://github.com/pylint-dev/astroid/blob/main/CONTRIBUTORS.txt

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Protocol

from astroid.ai.bridge import AIInferenceRequest
from astroid.ai.schema import AIInferenceResponse


class AIInferenceProvider(Protocol):
    def infer(
        self, request: AIInferenceRequest, *, timeout_ms: int
    ) -> AIInferenceResponse: ...  # pragma: no cover


class NullAIInferenceProvider:
    """Default provider that never returns AI candidates."""

    def infer(self, request: AIInferenceRequest, *, timeout_ms: int) -> AIInferenceResponse:
        del request
        del timeout_ms
        return AIInferenceResponse()


class MockAIInferenceProvider:
    """Deterministic provider used by tests."""

    def __init__(
        self,
        responses: Mapping[
            str | tuple[str, str], AIInferenceResponse | Exception
        ] | None = None,
        callback: Callable[[AIInferenceRequest, int], AIInferenceResponse] | None = None,
        default_response: AIInferenceResponse | None = None,
    ) -> None:
        self._responses = dict(responses or ())
        self._callback = callback
        self._default_response = default_response or AIInferenceResponse()

    def infer(self, request: AIInferenceRequest, *, timeout_ms: int) -> AIInferenceResponse:
        if self._callback is not None:
            return self._callback(request, timeout_ms)

        lookup_keys = [
            (request.scenario, request.annotation_repr or request.call_repr or ""),
            request.scenario,
        ]
        for key in lookup_keys:
            if key not in self._responses:
                continue
            response = self._responses[key]
            if isinstance(response, Exception):
                raise response
            return response
        return self._default_response
