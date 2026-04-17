# Licensed under the LGPL: https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html
# For details: https://github.com/pylint-dev/astroid/blob/main/LICENSE
# Copyright (c) https://github.com/pylint-dev/astroid/blob/main/CONTRIBUTORS.txt

from __future__ import annotations

from astroid.ai.bridge import AIInferenceRequest
from astroid.ai.cache import AIInferenceCache
from astroid.ai.exceptions import (
    AIInferenceError,
    AIInferenceLowConfidenceError,
    AIInferenceSchemaError,
    AIInferenceTimeoutError,
    AIProviderError,
    AIProviderUnavailableError,
)
from astroid.ai.policy import AIInferenceObserver, AIInferencePolicy
from astroid.ai.provider import AIInferenceProvider, MockAIInferenceProvider, NullAIInferenceProvider
from astroid.ai.register import register
from astroid.ai.schema import AIInferenceCandidate, AIInferenceResponse

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
