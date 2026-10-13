# Copyright The OpenTelemetry Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Extended metrics recorder for GenAI invocations.

This module provides LoongSuite GenAI metrics recording.
It supports multiple GenAI invocation types: chat, generate_content, embedding, execute_tool, invoke_agent,
create_agent, retrieval, and rerank.

This is a community implementation that covers every invocation type declared by
``record_extended``; deployments that need different behaviour can override the
method.
"""

from __future__ import annotations

import logging
import timeit
from typing import Dict, Optional, Union

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.semconv.attributes import (
    error_attributes,
    server_attributes,
)
from opentelemetry.trace import Span, set_span_in_context
from opentelemetry.util.genai.extended_memory.memory_types import (
    MemoryInvocation,
)
from opentelemetry.util.genai.extended_semconv.gen_ai_extended_attributes import (
    GenAiExtendedOperationNameValues,
)
from opentelemetry.util.genai.extended_types import (
    CreateAgentInvocation,
    EmbeddingInvocation,
    EntryInvocation,
    ExecuteToolInvocation,
    InvokeAgentInvocation,
    ReactStepInvocation,
    RerankInvocation,
    RetrievalInvocation,
)
from opentelemetry.util.genai.metrics import InvocationMetricsRecorder
from opentelemetry.util.genai.types import LLMInvocation
from opentelemetry.util.types import AttributeValue

_logger = logging.getLogger(__name__)

# ``extended_memory`` names the memory operation ``memory_operation``; the
# metrics pillar reuses the same value as the spans do.
_MEMORY_OPERATION_NAME = "memory_operation"

# ``gen_ai.operation.name`` per invocation type, matching the span attributes
# emitted for the same invocations by ``extended_span_utils`` and
# ``extended_memory.memory_utils``.
_OPERATION_NAME_BY_INVOCATION_TYPE: Dict[type, str] = {
    EmbeddingInvocation: GenAI.GenAiOperationNameValues.EMBEDDINGS.value,
    ExecuteToolInvocation: GenAI.GenAiOperationNameValues.EXECUTE_TOOL.value,
    InvokeAgentInvocation: GenAI.GenAiOperationNameValues.INVOKE_AGENT.value,
    CreateAgentInvocation: GenAI.GenAiOperationNameValues.CREATE_AGENT.value,
    RetrievalInvocation: GenAiExtendedOperationNameValues.RETRIEVAL.value,
    RerankInvocation: GenAiExtendedOperationNameValues.RERANK_DOCUMENTS.value,
    MemoryInvocation: _MEMORY_OPERATION_NAME,
    EntryInvocation: GenAiExtendedOperationNameValues.ENTER.value,
    ReactStepInvocation: GenAiExtendedOperationNameValues.REACT.value,
}


def _operation_name_for(invocation: object) -> Optional[str]:
    """Return the ``gen_ai.operation.name`` of an extended invocation."""
    for invocation_type, operation_name in (
        _OPERATION_NAME_BY_INVOCATION_TYPE.items()
    ):
        if isinstance(invocation, invocation_type):
            return operation_name
    return None


def _token_counts_for(invocation: object) -> list[tuple[int, str]]:
    """Return the ``(count, gen_ai.token.type)`` pairs an invocation carries.

    Only the invocation types whose specification defines token usage expose
    these fields; every other type contributes duration only.
    """
    token_counts: list[tuple[int, str]] = []
    input_tokens = getattr(invocation, "input_tokens", None)
    if input_tokens is not None:
        token_counts.append(
            (input_tokens, GenAI.GenAiTokenTypeValues.INPUT.value)
        )
    output_tokens = getattr(invocation, "output_tokens", None)
    if output_tokens is not None:
        token_counts.append(
            (output_tokens, GenAI.GenAiTokenTypeValues.OUTPUT.value)
        )
    return token_counts


class ExtendedInvocationMetricsRecorder(InvocationMetricsRecorder):
    """
    Extended metrics recorder that supports multiple GenAI invocation types.

    This class provides LoongSuite GenAI metrics recording.
    It supports:
    - Chat/Generate content operations
    - Embedding operations
    - Execute tool operations
    - Invoke agent operations
    - Create agent operations
    - Retrieval operations
    - Rerank documents operations
    - Entry operations
    - ReAct Step operations
    """

    def record_extended(
        self,
        span: Optional[Span],
        invocation: Union[
            LLMInvocation,
            EmbeddingInvocation,
            ExecuteToolInvocation,
            InvokeAgentInvocation,
            CreateAgentInvocation,
            RetrievalInvocation,
            RerankInvocation,
            MemoryInvocation,
            EntryInvocation,
            ReactStepInvocation,
        ],
        *,
        error_type: Optional[str] = None,
    ) -> None:
        """
        Record duration and token metrics for any GenAI invocation type.

        This method automatically routes to the appropriate handler based on
        the invocation type.

        Every declared invocation type records the operation duration histogram
        under its own ``gen_ai.operation.name``, plus the request model,
        provider, response model and server endpoint the invocation carries.
        Invocation types whose specification defines token usage (embeddings,
        agent invocations) also record the token histogram. Span-only
        attributes (``invocation.attributes``) are deliberately not recorded on
        the metrics pillar.

        Args:
            span: The span associated with this invocation
            invocation: The invocation object (any supported type)
            error_type: Optional error type if the invocation failed
        """
        if isinstance(invocation, LLMInvocation):
            self.record(span, invocation, error_type=error_type)
            return

        if span is None:
            return

        operation_name = _operation_name_for(invocation)
        if operation_name is None:
            _logger.debug(
                "No extended metrics recorder for invocation type %s",
                type(invocation).__name__,
            )
            return

        attributes: Dict[str, AttributeValue] = {
            GenAI.GEN_AI_OPERATION_NAME: operation_name
        }
        request_model = getattr(invocation, "request_model", None)
        if request_model:
            attributes[GenAI.GEN_AI_REQUEST_MODEL] = request_model
        provider = getattr(invocation, "provider", None)
        if provider:
            attributes[GenAI.GEN_AI_PROVIDER_NAME] = provider
        response_model_name = getattr(invocation, "response_model_name", None)
        if response_model_name:
            attributes[GenAI.GEN_AI_RESPONSE_MODEL] = response_model_name
        server_address = getattr(invocation, "server_address", None)
        if server_address:
            attributes[server_attributes.SERVER_ADDRESS] = server_address
        server_port = getattr(invocation, "server_port", None)
        if server_port is not None:
            attributes[server_attributes.SERVER_PORT] = server_port
        if error_type:
            attributes[error_attributes.ERROR_TYPE] = error_type

        span_context = set_span_in_context(span)

        monotonic_start_s = getattr(invocation, "monotonic_start_s", None)
        if monotonic_start_s is not None:
            self._duration_histogram.record(
                max(
                    timeit.default_timer() - monotonic_start_s,
                    0.0,
                ),
                attributes=attributes,
                context=span_context,
            )

        for token_count, token_type in _token_counts_for(invocation):
            self._token_histogram.record(
                token_count,
                attributes={
                    **attributes,
                    GenAI.GEN_AI_TOKEN_TYPE: token_type,
                },  # LoongSuite Extension: For Python 3.8 Compatibility
                context=span_context,
            )


__all__ = ["ExtendedInvocationMetricsRecorder"]
