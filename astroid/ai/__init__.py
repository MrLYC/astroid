# Licensed under the LGPL: https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html
# For details: https://github.com/pylint-dev/astroid/blob/main/LICENSE
# Copyright (c) https://github.com/pylint-dev/astroid/blob/main/CONTRIBUTORS.txt

from __future__ import annotations

from importlib import import_module

__all__ = [
    "AIInferenceCache",
    "AIInferenceCandidate",
    "AIInferenceError",
    "AIInferenceLowConfidenceError",
    "AIInferenceObserver",
    "AIInferencePolicy",
    "AIInferenceProvider",
    "AIInferenceRequest",
    "AIInferenceResponse",
    "AIInferenceSchemaError",
    "AIInferenceTimeoutError",
    "AIProviderError",
    "AIProviderUnavailableError",
    "MockAIInferenceProvider",
    "NullAIInferenceProvider",
    "register",
]

_EXPORTS = {
    "AIInferenceCache": ("astroid.ai.cache", "AIInferenceCache"),
    "AIInferenceCandidate": ("astroid.ai.schema", "AIInferenceCandidate"),
    "AIInferenceError": ("astroid.ai.exceptions", "AIInferenceError"),
    "AIInferenceLowConfidenceError": (
        "astroid.ai.exceptions",
        "AIInferenceLowConfidenceError",
    ),
    "AIInferenceObserver": ("astroid.ai.policy", "AIInferenceObserver"),
    "AIInferencePolicy": ("astroid.ai.policy", "AIInferencePolicy"),
    "AIInferenceProvider": ("astroid.ai.provider", "AIInferenceProvider"),
    "AIInferenceRequest": ("astroid.ai.bridge", "AIInferenceRequest"),
    "AIInferenceResponse": ("astroid.ai.schema", "AIInferenceResponse"),
    "AIInferenceSchemaError": ("astroid.ai.exceptions", "AIInferenceSchemaError"),
    "AIInferenceTimeoutError": ("astroid.ai.exceptions", "AIInferenceTimeoutError"),
    "AIProviderError": ("astroid.ai.exceptions", "AIProviderError"),
    "AIProviderUnavailableError": (
        "astroid.ai.exceptions",
        "AIProviderUnavailableError",
    ),
    "MockAIInferenceProvider": ("astroid.ai.provider", "MockAIInferenceProvider"),
    "NullAIInferenceProvider": ("astroid.ai.provider", "NullAIInferenceProvider"),
    "register": ("astroid.ai.register", "register"),
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        msg = f"module '{__name__}' has no attribute '{name}'"
        raise AttributeError(msg)

    module_name, attribute = _EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attribute)
    globals()[name] = value
    return value
