# Licensed under the LGPL: https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html
# For details: https://github.com/pylint-dev/astroid/blob/main/LICENSE
# Copyright (c) https://github.com/pylint-dev/astroid/blob/main/CONTRIBUTORS.txt

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from astroid import nodes
from astroid.ai.exceptions import AIInferenceLowConfidenceError, AIInferenceSchemaError
from astroid.exceptions import AttributeInferenceError, InferenceError
from astroid.typing import InferenceResult
from astroid.util import Uninferable

if TYPE_CHECKING:
    from astroid.manager import AstroidManager

AIInferenceKind = Literal["instance", "class", "const", "uninferable"]
_VALID_KINDS = frozenset({"instance", "class", "const", "uninferable"})


@dataclass(frozen=True)
class AIInferenceCandidate:
    kind: AIInferenceKind
    confidence: float = 1.0
    module: str | None = None
    name: str | None = None
    value: object | None = None


@dataclass(frozen=True)
class AIInferenceResponse:
    candidates: tuple[AIInferenceCandidate, ...] = ()


def response_to_nodes(
    response: AIInferenceResponse,
    *,
    manager: AstroidManager,
    node: nodes.NodeNG,
    max_candidates: int,
    min_confidence: float,
) -> tuple[InferenceResult, ...]:
    if len(response.candidates) > max_candidates:
        raise AIInferenceSchemaError("too many AI inference candidates")

    inferred: list[InferenceResult] = []
    for candidate in response.candidates:
        if candidate.kind not in _VALID_KINDS:
            raise AIInferenceSchemaError(f"unsupported AI inference kind: {candidate.kind}")
        if candidate.confidence < min_confidence:
            raise AIInferenceLowConfidenceError("AI inference confidence too low")
        inferred.append(_candidate_to_node(candidate, manager=manager, node=node))
    return tuple(inferred)


def _candidate_to_node(
    candidate: AIInferenceCandidate, *, manager: AstroidManager, node: nodes.NodeNG
) -> InferenceResult:
    if candidate.kind == "uninferable":
        return Uninferable
    if candidate.kind == "const":
        return nodes.Const(candidate.value, parent=node)
    if candidate.module is None or candidate.name is None:
        raise AIInferenceSchemaError("AI inference class candidates must define module and name")

    resolved = _resolve_candidate(candidate, manager=manager)
    if not isinstance(resolved, nodes.ClassDef):
        raise AIInferenceSchemaError("AI inference candidate did not resolve to a class")
    if candidate.kind == "instance":
        return resolved.instantiate_class()
    return resolved


def _resolve_candidate(
    candidate: AIInferenceCandidate, *, manager: AstroidManager
) -> InferenceResult:
    assert candidate.module is not None
    assert candidate.name is not None

    current: InferenceResult = manager.ast_from_module_name(candidate.module)
    for part in candidate.name.split("."):
        if not isinstance(current, (nodes.Module, nodes.ClassDef)):
            raise AIInferenceSchemaError("AI inference candidate cannot be resolved")
        try:
            current = next(current.igetattr(part))
        except (AttributeInferenceError, InferenceError, StopIteration) as exc:
            raise AIInferenceSchemaError("AI inference candidate cannot be resolved") from exc
    return current
