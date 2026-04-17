# Licensed under the LGPL: https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html
# For details: https://github.com/pylint-dev/astroid/blob/main/LICENSE
# Copyright (c) https://github.com/pylint-dev/astroid/blob/main/CONTRIBUTORS.txt

from __future__ import annotations

import textwrap
import time

import pytest

import astroid
from astroid import nodes, test_utils
from astroid.ai.bridge import AIInferenceRequest, build_typing_cast_request
from astroid.ai.cache import AIInferenceCache
from astroid.ai.exceptions import (
    AIInferenceLowConfidenceError,
    AIInferenceSchemaError,
    AIInferenceTimeoutError,
)
from astroid.ai.policy import consume_budget, policy_from_manager
from astroid.ai.provider import MockAIInferenceProvider
from astroid.ai.schema import AIInferenceCandidate, AIInferenceResponse, response_to_nodes
from astroid.brain.helpers import register_ai_brains, register_all_brains
from astroid.exceptions import InferenceError
from astroid.manager import AstroidManager


class RecordingObserver:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def record(self, event: str, **payload: object) -> None:
        self.events.append((event, payload))


class CountingProvider:
    def __init__(self, response: AIInferenceResponse) -> None:
        self.calls = 0
        self._response = response

    def infer(self, request: AIInferenceRequest, *, timeout_ms: int) -> AIInferenceResponse:
        self.calls += 1
        del request
        del timeout_ms
        return self._response


@pytest.fixture(autouse=True)
def preserve_ai_manager_state():
    manager = astroid.MANAGER
    saved_provider = manager.ai_provider
    saved_enabled = manager.ai_inference_enabled
    saved_timeout = manager.ai_timeout_ms
    saved_max_candidates = manager.ai_max_candidates
    saved_budget = manager.ai_budget_per_module
    saved_allowed = manager.ai_allowed_scenarios
    saved_observer = manager.ai_observer
    saved_brains_registered = manager.ai_brains_registered
    yield
    manager.clear_cache()
    AstroidManager.brain["ai_provider"] = saved_provider
    AstroidManager.brain["ai_inference_enabled"] = saved_enabled
    AstroidManager.brain["ai_timeout_ms"] = saved_timeout
    AstroidManager.brain["ai_max_candidates"] = saved_max_candidates
    AstroidManager.brain["ai_budget_per_module"] = saved_budget
    AstroidManager.brain["ai_allowed_scenarios"] = saved_allowed
    AstroidManager.brain["ai_observer"] = saved_observer
    AstroidManager.brain["ai_brains_registered"] = saved_brains_registered


def _build_ai_manager(
    provider: object, observer: RecordingObserver | None = None
) -> AstroidManager:
    manager = test_utils.brainless_manager()
    register_all_brains(manager)
    manager.bootstrap()
    manager.ai_inference_enabled = True
    manager.ai_provider = provider
    manager.ai_timeout_ms = 10
    manager.ai_max_candidates = 2
    manager.ai_budget_per_module = 2
    manager.ai_allowed_scenarios = frozenset({"typing.cast", "dataclasses.annotation"})
    manager.ai_observer = observer
    register_ai_brains(manager)
    return manager


def test_build_typing_cast_request_is_stable() -> None:
    call = astroid.extract_node(
        """
        from typing import cast
        cast(str, missing)
        """
    )
    request = build_typing_cast_request(call)

    assert request.scenario == "typing.cast"
    assert request.annotation_repr == "str"
    assert request.call_repr == "cast(str, missing)"
    assert request.expected_output_kinds == ("instance", "class", "uninferable")
    assert request.local_context


def test_ai_cache_uses_request_content() -> None:
    request = AIInferenceRequest(
        scenario="typing.cast",
        node_kind="Call",
        source_snippet="cast(str, missing)",
        module_name="mod",
        annotation_repr="str",
        expected_output_kinds=("instance",),
    )
    cache = AIInferenceCache()
    response = AIInferenceResponse((AIInferenceCandidate(kind="uninferable"),))

    cache.set(request, response)

    assert cache.get(request) == response
    assert len(cache) == 1


def test_policy_tracks_budget() -> None:
    manager = test_utils.brainless_manager()
    manager.ai_inference_enabled = True
    manager.ai_budget_per_module = 1
    manager.ai_allowed_scenarios = frozenset({"typing.cast"})
    request = AIInferenceRequest(
        scenario="typing.cast",
        node_kind="Call",
        source_snippet="cast(str, missing)",
        module_name="mod",
        expected_output_kinds=("instance",),
    )

    policy = policy_from_manager(manager)

    assert policy.allows(request)
    assert consume_budget(manager, request)
    assert not consume_budget(manager, request)


def test_response_to_nodes_rejects_low_confidence() -> None:
    manager = test_utils.brainless_manager()
    manager.bootstrap()
    module = manager.ast_from_string("x = 1", modname="test_schema")
    node = module.locals["x"][0]
    response = AIInferenceResponse(
        (AIInferenceCandidate(kind="instance", module="builtins", name="str", confidence=0.2),)
    )

    with pytest.raises(AIInferenceLowConfidenceError):
        response_to_nodes(
            response,
            manager=manager,
            node=node,
            max_candidates=1,
            min_confidence=0.5,
        )


def test_response_to_nodes_rejects_invalid_schema() -> None:
    manager = test_utils.brainless_manager()
    manager.bootstrap()
    module = manager.ast_from_string("x = 1", modname="test_schema_invalid")
    node = module.locals["x"][0]
    response = AIInferenceResponse((AIInferenceCandidate(kind="instance"),))

    with pytest.raises(AIInferenceSchemaError):
        response_to_nodes(
            response,
            manager=manager,
            node=node,
            max_candidates=1,
            min_confidence=0.5,
        )


def test_mock_provider_matches_by_scenario_and_annotation() -> None:
    provider = MockAIInferenceProvider(
        responses={
            ("typing.cast", "str"): AIInferenceResponse(
                (AIInferenceCandidate(kind="instance", module="builtins", name="str"),)
            )
        }
    )
    request = AIInferenceRequest(
        scenario="typing.cast",
        node_kind="Call",
        source_snippet="cast(str, missing)",
        module_name="mod",
        annotation_repr="str",
        expected_output_kinds=("instance",),
    )

    response = provider.infer(request, timeout_ms=10)

    assert response.candidates[0].name == "str"


def test_ai_typing_cast_requires_explicit_registration() -> None:
    manager = test_utils.brainless_manager()
    register_all_brains(manager)
    manager.bootstrap()
    manager.ai_inference_enabled = True
    manager.ai_provider = MockAIInferenceProvider(
        responses={
            ("typing.cast", "str"): AIInferenceResponse(
                (AIInferenceCandidate(kind="instance", module="builtins", name="str"),)
            )
        }
    )
    manager.ai_budget_per_module = 1
    manager.ai_allowed_scenarios = frozenset({"typing.cast"})

    module = manager.ast_from_string(
        textwrap.dedent(
            """
            from typing import cast
            result = cast(str, missing)
            """
        ),
        modname="typing_no_ai",
    )

    with pytest.raises(InferenceError):
        module.locals["result"][0].inferred()


def test_ai_typing_cast_fallback_and_cache_hit() -> None:
    response = AIInferenceResponse(
        (AIInferenceCandidate(kind="instance", module="builtins", name="str"),)
    )
    observer = RecordingObserver()
    provider = CountingProvider(response)
    manager = _build_ai_manager(provider, observer)

    first_module = manager.ast_from_string(
        textwrap.dedent(
            """
            from typing import cast
            result = cast(str, missing)
            """
        ),
        modname="typing_first",
    )
    second_module = manager.ast_from_string(
        textwrap.dedent(
            """
            from typing import cast
            result = cast(str, missing)
            """
        ),
        modname="typing_second",
    )

    first_inferred = first_module.locals["result"][0].inferred()
    second_inferred = second_module.locals["result"][0].inferred()

    assert isinstance(first_inferred[0], astroid.bases.Instance)
    assert first_inferred[0].name == "str"
    assert isinstance(second_inferred[0], astroid.bases.Instance)
    assert second_inferred[0].name == "str"
    assert provider.calls == 1
    assert any(event == "ai_cache_hit" for event, _payload in observer.events)


def test_ai_typing_cast_timeout_fallback() -> None:
    class SlowProvider:
        def infer(self, request: AIInferenceRequest, *, timeout_ms: int) -> AIInferenceResponse:
            del request
            time.sleep((timeout_ms / 1000) + 0.05)
            return AIInferenceResponse(
                (AIInferenceCandidate(kind="instance", module="builtins", name="str"),)
            )

    observer = RecordingObserver()
    manager = _build_ai_manager(SlowProvider(), observer)
    manager.ai_timeout_ms = 1
    module = manager.ast_from_string(
        textwrap.dedent(
            """
            from typing import cast
            result = cast(str, missing)
            """
        ),
        modname="typing_timeout",
    )

    with pytest.raises(InferenceError):
        module.locals["result"][0].inferred()

    assert (
        "ai_fallback",
        {"reason": "AIInferenceTimeoutError", "scenario": "typing.cast"},
    ) in observer.events


def test_register_ai_brains_is_idempotent() -> None:
    manager = test_utils.brainless_manager()
    register_all_brains(manager)
    manager.bootstrap()

    register_ai_brains(manager)
    call_transforms = len(manager._transform.transforms[nodes.Call])
    unknown_transforms = len(manager._transform.transforms[nodes.Unknown])

    register_ai_brains(manager)

    assert manager.ai_brains_registered
    assert len(manager._transform.transforms[nodes.Call]) == call_transforms
    assert len(manager._transform.transforms[nodes.Unknown]) == unknown_transforms


def test_ai_dataclass_annotation_fallback() -> None:
    manager = _build_ai_manager(
        MockAIInferenceProvider(
            responses={
                ("dataclasses.annotation", "UnknownType"): AIInferenceResponse(
                    (AIInferenceCandidate(kind="instance", module="builtins", name="str"),)
                )
            }
        )
    )
    module = manager.ast_from_string(
        textwrap.dedent(
            """
            from dataclasses import dataclass

            @dataclass
            class Example:
                name: UnknownType

            result = Example(1).name
            """
        ),
        modname="dataclass_ai_success",
    )

    inferred = module.locals["result"][0].inferred()

    assert len(inferred) == 1
    assert isinstance(inferred[0], astroid.bases.Instance)
    assert inferred[0].name == "str"


@pytest.mark.parametrize(
    "provider",
    [
        MockAIInferenceProvider(
            responses={
                ("dataclasses.annotation", "UnknownType"): AIInferenceResponse(
                    (AIInferenceCandidate(kind="instance", confidence=0.2, module="builtins", name="str"),)
                )
            }
        ),
        MockAIInferenceProvider(
            responses={
                ("dataclasses.annotation", "UnknownType"): AIInferenceResponse(
                    (AIInferenceCandidate(kind="instance"),)
                )
            }
        ),
        MockAIInferenceProvider(
            responses={
                ("dataclasses.annotation", "UnknownType"): AIInferenceTimeoutError("timeout")
            }
        ),
    ],
)
def test_ai_dataclass_annotation_failures_fallback_safely(provider: object) -> None:
    manager = _build_ai_manager(provider)
    module = manager.ast_from_string(
        textwrap.dedent(
            """
            from dataclasses import dataclass

            @dataclass
            class Example:
                name: UnknownType

            result = Example(1).name
            """
        ),
        modname="dataclass_ai_fallback",
    )

    inferred = module.locals["result"][0].inferred()

    assert inferred == [astroid.util.Uninferable]
