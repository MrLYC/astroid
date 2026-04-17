# Licensed under the LGPL: https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html
# For details: https://github.com/pylint-dev/astroid/blob/main/LICENSE
# Copyright (c) https://github.com/pylint-dev/astroid/blob/main/CONTRIBUTORS.txt

from __future__ import annotations

import traceback
from collections.abc import Iterator
from threading import Thread
from types import BuiltinFunctionType, FunctionType, MethodType

from astroid import bases, context, nodes
from astroid.ai.bridge import (
    AIInferenceRequest,
    build_dataclass_attribute_request,
    build_typing_cast_request,
)
from astroid.ai.exceptions import (
    AIInferenceError,
    AIInferenceTimeoutError,
    AIProviderError,
    AIProviderUnavailableError,
)
from astroid.ai.policy import consume_budget, observe, policy_from_manager
from astroid.ai.provider import NullAIInferenceProvider
from astroid.ai.schema import AIInferenceResponse, response_to_nodes
from astroid.brain import brain_dataclasses, brain_typing
from astroid.exceptions import InferenceError, UseInferenceDefault
from astroid.inference_tip import inference_tip
from astroid.manager import AstroidManager
from astroid.typing import InferenceResult
from astroid.util import Uninferable, UninferableBase


def register(manager: AstroidManager) -> None:
    if manager.ai_brains_registered:
        return

    manager.register_transform(
        nodes.Call,
        inference_tip(_infer_typing_cast_with_ai(manager)),
        brain_typing._looks_like_typing_cast,
    )
    manager.register_transform(
        nodes.Unknown,
        inference_tip(_infer_dataclass_attribute_with_ai(manager)),
        brain_dataclasses._looks_like_dataclass_attribute,
    )
    manager.ai_brains_registered = True


def _infer_typing_cast_with_ai(
    manager: AstroidManager,
):
    def infer(
        node: nodes.Call, ctx: context.InferenceContext | None = None
    ) -> Iterator[InferenceResult]:
        original_exception: Exception | None = None
        results: tuple[InferenceResult, ...] = ()
        try:
            results = tuple(brain_typing.infer_typing_cast(node, ctx=ctx))
        except (InferenceError, StopIteration, UseInferenceDefault) as exc:
            original_exception = exc

        if _has_inferable_results(results):
            return iter(results)

        ai_results = _maybe_infer_with_ai(
            manager,
            request=build_typing_cast_request(node),
            node=node,
        )
        if ai_results is not None:
            return iter(ai_results)
        if results:
            return iter(results)
        if isinstance(original_exception, UseInferenceDefault):
            raise original_exception
        if original_exception is not None:
            raise original_exception
        raise UseInferenceDefault

    return infer


def _infer_dataclass_attribute_with_ai(
    manager: AstroidManager,
):
    def infer(
        node: nodes.Unknown, ctx: context.InferenceContext | None = None
    ) -> Iterator[InferenceResult]:
        assign = node.parent
        if not isinstance(assign, nodes.AnnAssign):
            return iter((Uninferable,))

        results: list[InferenceResult] = []
        if assign.value is not None:
            try:
                value_results = tuple(assign.value.infer(context=ctx))
            except (InferenceError, StopIteration):
                value_results = (Uninferable,)
            results.extend(value_results)
            if _has_inferable_results(value_results):
                annotation_results = tuple(
                    brain_dataclasses._infer_instance_from_annotation(assign.annotation, ctx=ctx)
                )
                results.extend(annotation_results)
                return iter(results)

        if assign.annotation is None:
            return iter(results or (Uninferable,))

        annotation_results = tuple(
            brain_dataclasses._infer_instance_from_annotation(assign.annotation, ctx=ctx)
        )
        if _has_inferable_results(annotation_results):
            results.extend(annotation_results)
            return iter(results)

        ai_results = _maybe_infer_with_ai(
            manager,
            request=build_dataclass_attribute_request(node),
            node=node,
        )
        if ai_results is not None:
            results.extend(ai_results)
        else:
            results.extend(annotation_results)
        return iter(results or (Uninferable,))

    return infer


def _maybe_infer_with_ai(
    manager: AstroidManager,
    *,
    request: AIInferenceRequest,
    node: nodes.NodeNG,
) -> tuple[InferenceResult, ...] | None:
    policy = policy_from_manager(manager)
    if not policy.allows(request):
        observe(manager, "ai_skipped", reason="disabled", scenario=request.scenario)
        return None

    cached = manager.ai_cache.get(request)
    if cached is not None:
        observe(manager, "ai_cache_hit", scenario=request.scenario)
        return _response_to_nodes(manager, request, node, cached)

    if not consume_budget(manager, request):
        observe(manager, "ai_fallback", reason="budget", scenario=request.scenario)
        return None

    try:
        provider = _resolve_provider(manager)
        response = _infer_with_timeout(provider, request, timeout_ms=policy.timeout_ms)
    except AIInferenceError as exc:
        observe(manager, "ai_fallback", reason=type(exc).__name__, scenario=request.scenario)
        return None
    except Exception as exc:  # pylint: disable=broad-except
        observe(
            manager,
            "ai_fallback",
            reason=type(exc).__name__,
            scenario=request.scenario,
            unexpected=True,
            traceback=traceback.format_exc(),
        )
        return None

    manager.ai_cache.set(request, response)
    observe(manager, "ai_provider_hit", scenario=request.scenario)
    return _response_to_nodes(manager, request, node, response)


def _response_to_nodes(
    manager: AstroidManager,
    request: AIInferenceRequest,
    node: nodes.NodeNG,
    response: AIInferenceResponse,
) -> tuple[InferenceResult, ...] | None:
    try:
        mapped = response_to_nodes(
            response,
            manager=manager,
            node=node,
            max_candidates=manager.ai_max_candidates,
            min_confidence=policy_from_manager(manager).min_confidence,
        )
    except AIInferenceError as exc:
        observe(manager, "ai_fallback", reason=type(exc).__name__, scenario=request.scenario)
        return None
    if not _has_inferable_results(mapped):
        observe(manager, "ai_fallback", reason="empty", scenario=request.scenario)
        return None
    observe(manager, "ai_hit", scenario=request.scenario)
    return mapped


def _resolve_provider(manager: AstroidManager):
    provider_or_factory = manager.ai_provider
    if _is_provider_factory(provider_or_factory):
        provider = provider_or_factory()
    else:
        provider = provider_or_factory
    if provider is None:
        raise AIProviderUnavailableError("no AI inference provider configured")
    if not hasattr(provider, "infer"):
        raise AIProviderError("configured AI inference provider is invalid")
    if isinstance(provider, NullAIInferenceProvider):
        raise AIProviderUnavailableError("null AI inference provider configured")
    return provider


def _is_provider_factory(provider_or_factory) -> bool:
    return isinstance(
        provider_or_factory, (BuiltinFunctionType, FunctionType, MethodType, type)
    )


def _infer_with_timeout(provider, request: AIInferenceRequest, *, timeout_ms: int) -> AIInferenceResponse:
    """Call a provider with best-effort timeout enforcement.

    Providers are still expected to honor ``timeout_ms`` themselves. The thread-based
    wrapper exists to preserve fallback semantics when a provider ignores the contract.
    """
    if timeout_ms <= 0:
        return provider.infer(request, timeout_ms=timeout_ms)

    response: AIInferenceResponse | None = None
    error: Exception | None = None
    error_traceback: str | None = None

    def invoke_provider() -> None:
        nonlocal response, error, error_traceback
        try:
            response = provider.infer(request, timeout_ms=timeout_ms)
        except Exception as exc:  # pylint: disable=broad-except
            error = exc
            error_traceback = traceback.format_exc()

    thread = Thread(target=invoke_provider, daemon=True)
    thread.start()
    thread.join(timeout_ms / 1000)
    if thread.is_alive():
        raise AIInferenceTimeoutError(
            f"AI inference timed out after {timeout_ms}ms for {request.scenario}"
        )
    if error is not None:
        if isinstance(error, AIInferenceError):
            raise error
        error_message = f"AI inference provider raised {type(error).__name__}: {error}"
        if error_traceback is not None:
            error_message = f"{error_message}\n{error_traceback}"
        raise AIProviderError(error_message) from error
    if response is None:
        raise AIProviderError("AI inference provider returned no response")
    return response


def _has_inferable_results(results: tuple[InferenceResult, ...] | list[InferenceResult]) -> bool:
    return any(
        not isinstance(result, UninferableBase) and result is not Uninferable
        for result in results
    )
