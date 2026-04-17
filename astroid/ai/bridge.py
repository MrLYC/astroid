# Licensed under the LGPL: https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html
# For details: https://github.com/pylint-dev/astroid/blob/main/LICENSE
# Copyright (c) https://github.com/pylint-dev/astroid/blob/main/CONTRIBUTORS.txt

from __future__ import annotations

from dataclasses import dataclass

from astroid import nodes
from astroid.context import InferenceContext

_MAX_SNIPPET_SIZE = 240
_MAX_LOCAL_CONTEXT_ITEMS = 8


@dataclass(frozen=True)
class AIInferenceRequest:
    scenario: str
    node_kind: str
    source_snippet: str
    module_name: str
    call_repr: str | None = None
    annotation_repr: str | None = None
    local_context: tuple[tuple[str, str], ...] = ()
    expected_output_kinds: tuple[str, ...] = ()


def build_request(
    node: nodes.NodeNG,
    *,
    scenario: str,
    module_name: str,
    expected_output_kinds: tuple[str, ...],
    call_repr: str | None = None,
    annotation_repr: str | None = None,
) -> AIInferenceRequest:
    return AIInferenceRequest(
        scenario=scenario,
        node_kind=type(node).__name__,
        source_snippet=_truncate(_safe_as_string(node.statement())),
        module_name=module_name,
        call_repr=_truncate(call_repr),
        annotation_repr=_truncate(annotation_repr),
        local_context=_build_local_context(node),
        expected_output_kinds=expected_output_kinds,
    )


def build_typing_cast_request(node: nodes.Call) -> AIInferenceRequest:
    annotation_repr = node.args[0].as_string() if node.args else None
    return build_request(
        node,
        scenario="typing.cast",
        module_name=node.root().name,
        expected_output_kinds=("instance", "class", "uninferable"),
        call_repr=_safe_as_string(node),
        annotation_repr=annotation_repr,
    )


def build_dataclass_attribute_request(node: nodes.Unknown) -> AIInferenceRequest:
    annotation_repr = None
    if isinstance(node.parent, nodes.AnnAssign) and node.parent.annotation is not None:
        annotation_repr = node.parent.annotation.as_string()
    return build_request(
        node,
        scenario="dataclasses.annotation",
        module_name=node.root().name,
        expected_output_kinds=("instance", "class", "const", "uninferable"),
        annotation_repr=annotation_repr,
    )


def _build_local_context(node: nodes.NodeNG) -> tuple[tuple[str, str], ...]:
    scope = node.scope()
    items: list[tuple[str, str]] = []
    for name in sorted(scope.locals)[:_MAX_LOCAL_CONTEXT_ITEMS]:
        values = scope.locals[name]
        if values:
            items.append((name, type(values[0]).__name__))
    return tuple(items)


def _safe_as_string(node: nodes.NodeNG | None) -> str:
    if node is None:
        return ""
    try:
        return node.as_string()
    except AttributeError:
        return ""


def _truncate(value: str | None) -> str | None:
    if value is None or len(value) <= _MAX_SNIPPET_SIZE:
        return value
    return value[: _MAX_SNIPPET_SIZE - 3] + "..."
