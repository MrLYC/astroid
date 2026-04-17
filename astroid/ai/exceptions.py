# Licensed under the LGPL: https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html
# For details: https://github.com/pylint-dev/astroid/blob/main/LICENSE
# Copyright (c) https://github.com/pylint-dev/astroid/blob/main/CONTRIBUTORS.txt

from __future__ import annotations


class AIInferenceError(Exception):
    """Base class for opt-in AI inference failures."""


class AIInferenceTimeoutError(AIInferenceError):
    """Raised when an AI inference request times out."""


class AIProviderUnavailableError(AIInferenceError):
    """Raised when no usable AI provider is available."""


class AIProviderError(AIInferenceError):
    """Raised when the configured AI provider fails."""


class AIInferenceLowConfidenceError(AIInferenceError):
    """Raised when AI candidates do not meet the confidence threshold."""


class AIInferenceSchemaError(AIInferenceError):
    """Raised when a provider response cannot be mapped safely."""
