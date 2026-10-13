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

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
from unittest import TestCase
from unittest.mock import patch

from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.util.genai.extended_handler import ExtendedTelemetryHandler
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
from opentelemetry.util.genai.types import Error, LLMInvocation

DURATION_METRIC = "gen_ai.client.operation.duration"
TOKEN_METRIC = "gen_ai.client.token.usage"


@dataclass
class _UnsupportedInvocation:
    """An invocation type the extended recorder does not declare support for."""

    monotonic_start_s: float | None = None


# (invocation, expected gen_ai.operation.name, start method, stop method)
_EXTENDED_CASES: List[Tuple[Any, str, str, str]] = [
    (
        CreateAgentInvocation(provider="prov", agent_name="agent"),
        GenAI.GenAiOperationNameValues.CREATE_AGENT.value,
        "start_create_agent",
        "stop_create_agent",
    ),
    (
        EmbeddingInvocation(request_model="embed-model"),
        GenAI.GenAiOperationNameValues.EMBEDDINGS.value,
        "start_embedding",
        "stop_embedding",
    ),
    (
        ExecuteToolInvocation(tool_name="search"),
        GenAI.GenAiOperationNameValues.EXECUTE_TOOL.value,
        "start_execute_tool",
        "stop_execute_tool",
    ),
    (
        InvokeAgentInvocation(provider="prov", agent_name="agent"),
        GenAI.GenAiOperationNameValues.INVOKE_AGENT.value,
        "start_invoke_agent",
        "stop_invoke_agent",
    ),
    (
        RetrievalInvocation(query="q"),
        GenAiExtendedOperationNameValues.RETRIEVAL.value,
        "start_retrieval",
        "stop_retrieval",
    ),
    (
        RerankInvocation(provider="prov", request_model="rerank-model"),
        GenAiExtendedOperationNameValues.RERANK_DOCUMENTS.value,
        "start_rerank",
        "stop_rerank",
    ),
    (
        MemoryInvocation(operation="search"),
        "memory_operation",
        "start_memory",
        "stop_memory",
    ),
    (
        EntryInvocation(session_id="session"),
        GenAiExtendedOperationNameValues.ENTER.value,
        "start_entry",
        "stop_entry",
    ),
    (
        ReactStepInvocation(round=1),
        GenAiExtendedOperationNameValues.REACT.value,
        "start_react_step",
        "stop_react_step",
    ),
]


class ExtendedInvocationMetricsTest(TestCase):
    def setUp(self) -> None:
        self.metric_reader = InMemoryMetricReader()
        self.meter_provider = MeterProvider(
            metric_readers=[self.metric_reader]
        )
        self.span_exporter = InMemorySpanExporter()
        self.tracer_provider = TracerProvider()
        self.tracer_provider.add_span_processor(
            SimpleSpanProcessor(self.span_exporter)
        )
        self.handler = ExtendedTelemetryHandler(
            tracer_provider=self.tracer_provider,
            meter_provider=self.meter_provider,
        )

    def test_stop_records_duration_for_every_extended_invocation_type(
        self,
    ) -> None:
        for invocation, _, start_name, stop_name in _EXTENDED_CASES:
            with patch("timeit.default_timer", return_value=1000.0):
                getattr(self.handler, start_name)(invocation)
            with patch("timeit.default_timer", return_value=1002.0):
                getattr(self.handler, stop_name)(invocation)

        metrics = self._harvest_metrics()
        self.assertIn(DURATION_METRIC, metrics)
        by_operation = {
            point.attributes[GenAI.GEN_AI_OPERATION_NAME]: point
            for point in metrics[DURATION_METRIC]
        }
        for invocation, operation_name, _, _ in _EXTENDED_CASES:
            with self.subTest(invocation=type(invocation).__name__):
                self.assertIn(operation_name, by_operation)
                self.assertAlmostEqual(
                    by_operation[operation_name].sum, 2.0, places=3
                )
        # The spans for these invocations were already emitted before this
        # change; the metrics pillar is what was missing.
        self.assertEqual(
            len(self.span_exporter.get_finished_spans()), len(_EXTENDED_CASES)
        )

    def test_stop_records_duration_and_token_usage_for_embedding(self) -> None:
        invocation = EmbeddingInvocation(
            request_model="embed-model",
            provider="prov",
            server_address="embed.server.com",
            server_port=42,
        )
        invocation.input_tokens = 3
        invocation.output_tokens = 4
        self._run(invocation, "start_embedding", "stop_embedding")

        metrics = self._harvest_metrics()
        duration_point = self._single_point(metrics, DURATION_METRIC)
        attributes = duration_point.attributes
        self.assertEqual(
            attributes[GenAI.GEN_AI_OPERATION_NAME],
            GenAI.GenAiOperationNameValues.EMBEDDINGS.value,
        )
        self.assertEqual(
            attributes[GenAI.GEN_AI_REQUEST_MODEL], "embed-model"
        )
        self.assertEqual(attributes[GenAI.GEN_AI_PROVIDER_NAME], "prov")
        self.assertEqual(attributes["server.address"], "embed.server.com")
        self.assertEqual(attributes["server.port"], 42)
        self.assertAlmostEqual(duration_point.sum, 2.0, places=3)

        token_by_type = {
            point.attributes[GenAI.GEN_AI_TOKEN_TYPE]: point
            for point in metrics[TOKEN_METRIC]
        }
        self.assertEqual(
            set(token_by_type),
            {
                GenAI.GenAiTokenTypeValues.INPUT.value,
                GenAI.GenAiTokenTypeValues.COMPLETION.value,
            },
        )
        input_point = token_by_type[
            GenAI.GenAiTokenTypeValues.INPUT.value
        ]
        output_point = token_by_type[
            GenAI.GenAiTokenTypeValues.COMPLETION.value
        ]
        self.assertAlmostEqual(input_point.sum, 3.0, places=3)
        self.assertAlmostEqual(output_point.sum, 4.0, places=3)
        self.assertEqual(
            input_point.attributes[GenAI.GEN_AI_OPERATION_NAME],
            GenAI.GenAiOperationNameValues.EMBEDDINGS.value,
        )

    def test_stop_records_token_usage_for_invoke_agent(self) -> None:
        invocation = InvokeAgentInvocation(
            provider="prov", agent_name="agent"
        )
        invocation.input_tokens = 5
        invocation.output_tokens = 7
        self._run(invocation, "start_invoke_agent", "stop_invoke_agent")

        metrics = self._harvest_metrics()
        token_by_type = {
            point.attributes[GenAI.GEN_AI_TOKEN_TYPE]: point
            for point in metrics[TOKEN_METRIC]
        }
        self.assertAlmostEqual(
            token_by_type[
                GenAI.GenAiTokenTypeValues.INPUT.value
            ].sum,
            5.0,
            places=3,
        )
        self.assertAlmostEqual(
            token_by_type[
                GenAI.GenAiTokenTypeValues.COMPLETION.value
            ].sum,
            7.0,
            places=3,
        )
        for point in metrics[TOKEN_METRIC]:
            self.assertEqual(
                point.attributes[GenAI.GEN_AI_OPERATION_NAME],
                GenAI.GenAiOperationNameValues.INVOKE_AGENT.value,
            )

    def test_fail_records_error_type_for_extended_invocation(self) -> None:
        invocation = ExecuteToolInvocation(tool_name="search")
        with patch("timeit.default_timer", return_value=2000.0):
            self.handler.start_execute_tool(invocation)
        with patch("timeit.default_timer", return_value=2001.0):
            self.handler.fail_execute_tool(
                invocation, Error(message="boom", type=ValueError)
            )

        metrics = self._harvest_metrics()
        duration_point = self._single_point(metrics, DURATION_METRIC)
        self.assertEqual(
            duration_point.attributes[GenAI.GEN_AI_OPERATION_NAME],
            GenAI.GenAiOperationNameValues.EXECUTE_TOOL.value,
        )
        self.assertEqual(
            duration_point.attributes["error.type"], "ValueError"
        )
        self.assertAlmostEqual(duration_point.sum, 1.0, places=3)
        self.assertNotIn(TOKEN_METRIC, metrics)

    def test_span_only_attributes_are_not_recorded_on_metrics(self) -> None:
        invocation = RetrievalInvocation(query="q")
        invocation.attributes = {"span.only": "value"}
        self._run(invocation, "start_retrieval", "stop_retrieval")

        metrics = self._harvest_metrics()
        duration_point = self._single_point(metrics, DURATION_METRIC)
        self.assertIsNone(
            duration_point.attributes.get("span.only")
        )
        self.assertEqual(
            duration_point.attributes[GenAI.GEN_AI_OPERATION_NAME],
            GenAiExtendedOperationNameValues.RETRIEVAL.value,
        )

    def test_llm_invocation_still_routes_to_the_base_recorder(self) -> None:
        invocation = LLMInvocation(request_model="model", provider="prov")
        invocation.input_tokens = 5
        with patch("timeit.default_timer", return_value=1000.0):
            self.handler.start_llm(invocation)
        with patch("timeit.default_timer", return_value=1002.0):
            self.handler.stop_llm(invocation)

        metrics = self._harvest_metrics()
        duration_point = self._single_point(metrics, DURATION_METRIC)
        self.assertEqual(
            duration_point.attributes[GenAI.GEN_AI_OPERATION_NAME],
            GenAI.GenAiOperationNameValues.CHAT.value,
        )
        self.assertAlmostEqual(
            self._single_point(metrics, TOKEN_METRIC).sum, 5.0, places=3
        )

    def test_unsupported_invocation_type_records_nothing(self) -> None:
        invocation = _UnsupportedInvocation()
        self.handler._metrics_recorder.record_extended(None, invocation)

        metrics = self._harvest_metrics()
        self.assertNotIn(DURATION_METRIC, metrics)
        self.assertNotIn(TOKEN_METRIC, metrics)

    def _run(
        self, invocation: Any, start_name: str, stop_name: str
    ) -> None:
        with patch("timeit.default_timer", return_value=1000.0):
            getattr(self.handler, start_name)(invocation)
        with patch("timeit.default_timer", return_value=1002.0):
            getattr(self.handler, stop_name)(invocation)

    def _single_point(self, metrics: Dict[str, List[Any]], name: str) -> Any:
        self.assertIn(name, metrics)
        points = metrics[name]
        self.assertEqual(len(points), 1)
        return points[0]

    def _harvest_metrics(self) -> Dict[str, List[Any]]:
        try:
            self.meter_provider.force_flush()
        except Exception:  # pylint: disable=broad-except
            pass
        self.metric_reader.collect()
        metrics_by_name: Dict[str, List[Any]] = {}
        data = self.metric_reader.get_metrics_data()
        for resource_metric in (data and data.resource_metrics) or []:
            for scope_metric in resource_metric.scope_metrics or []:
                for metric in scope_metric.metrics or []:
                    points = metric.data.data_points or []
                    metrics_by_name.setdefault(metric.name, []).extend(points)
        return metrics_by_name
