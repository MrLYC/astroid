# Licensed under the LGPL: https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html
# For details: https://github.com/pylint-dev/astroid/blob/main/LICENSE
# Copyright (c) https://github.com/pylint-dev/astroid/blob/main/CONTRIBUTORS.txt

from __future__ import annotations

from astroid.ai.bridge import AIInferenceRequest
from astroid.ai.schema import AIInferenceResponse


class AIInferenceCache:
    """Dedicated cache for opt-in AI inference responses."""

    def __init__(self) -> None:
        self._cache: dict[tuple[object, ...], AIInferenceResponse] = {}

    def clear(self) -> None:
        self._cache.clear()

    def get(self, request: AIInferenceRequest) -> AIInferenceResponse | None:
        return self._cache.get(self.make_key(request))

    def set(self, request: AIInferenceRequest, response: AIInferenceResponse) -> None:
        self._cache[self.make_key(request)] = response

    def make_key(self, request: AIInferenceRequest) -> tuple[object, ...]:
        return (
            request.scenario,
            request.node_kind,
            request.source_snippet,
            request.call_repr,
            request.annotation_repr,
            request.local_context,
            request.expected_output_kinds,
        )

    def __len__(self) -> int:
        return len(self._cache)
