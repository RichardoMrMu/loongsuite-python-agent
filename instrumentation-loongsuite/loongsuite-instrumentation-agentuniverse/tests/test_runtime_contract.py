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

import asyncio
import json

import pytest
from agentuniverse.agent.agent import Agent
from agentuniverse.base.annotation import trace as trace_module

from opentelemetry.instrumentation.agentuniverse import (
    AgentUniverseInstrumentor,
)

from .baseline import collect_metrics, counter_value
from .conftest import NATIVE_WRAPPER_GLOBALS
from .runtime_app import Workload
from .runtime_provider import MODEL


def test_missing_round_hook_install_is_atomic(monkeypatch):
    original = {
        name: getattr(trace_module, name) for name in NATIVE_WRAPPER_GLOBALS
    }
    monkeypatch.delattr(Agent, "on_before_llm_round")
    instrumentor = AgentUniverseInstrumentor()
    instrumentor.instrument()
    assert not instrumentor.is_instrumented_by_opentelemetry
    assert all(
        getattr(trace_module, name) is value
        for name, value in original.items()
    )


@pytest.mark.parametrize("mode", ["SPAN_ONLY", "NO_CONTENT"])
def test_native_parallel_reasoning_conversations(
    mode, monkeypatch, configured_providers, session_exporter
):
    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", mode
    )
    provider, meter_provider, reader = configured_providers
    instrumentor = AgentUniverseInstrumentor()
    instrumentor.instrument(
        tracer_provider=provider, meter_provider=meter_provider
    )
    assert instrumentor.is_instrumented_by_opentelemetry, (
        "Supported framework must enable without skip_dep_check"
    )
    workload = Workload()
    result = asyncio.run(workload.run(provider))
    assert result["request_count"] == 18
    assert all(item["reasoning"] for item in result["scenarios"])
    spans = session_exporter.get_finished_spans()
    llms = [
        span
        for span in spans
        if span.attributes.get("gen_ai.span.kind") == "LLM"
    ]
    agents = [
        span
        for span in spans
        if span.attributes.get("gen_ai.span.kind") == "AGENT"
    ]
    tools = [
        span
        for span in spans
        if span.attributes.get("gen_ai.span.kind") == "TOOL"
    ]
    assert (len(llms), len(agents), len(tools)) == (18, 6, 18)
    expected_ids = {
        call["id"]
        for response in workload.backend.responses
        for call in response["tool_calls"]
    }
    assert {
        span.attributes.get("gen_ai.tool.call.id") for span in tools
    } == expected_ids
    for span in spans:
        if span.attributes.get("gen_ai.span.kind"):
            assert span.attributes.get("gen_ai.session.id") in (
                "auSync",
                "auStreamA",
                "auStreamB",
            )
    by_id = {span.context.span_id: span for span in spans}
    for span in llms + tools + agents:
        assert span.parent is not None
        assert span.parent.span_id in by_id
        assert (
            by_id[span.parent.span_id].context.trace_id
            == span.context.trace_id
        )
    plans = {
        call["id"]: call
        for response in workload.backend.responses
        for call in response["tool_calls"]
    }
    for span in tools:
        attrs = span.attributes
        identifier = attrs["gen_ai.tool.call.id"]
        worker = identifier.split("-t")[0]
        assert attrs["gen_ai.session.id"] == worker
        if mode == "SPAN_ONLY":
            assert json.loads(
                attrs["gen_ai.tool.call.arguments"]
            ) == json.loads(plans[identifier]["function"]["arguments"])
    for span in llms:
        attrs = span.attributes
        assert attrs["gen_ai.usage.input_tokens"] == 20
        assert attrs["gen_ai.usage.output_tokens"] == 10
        assert attrs["au.llm.usage.total_tokens"] == 30
        assert attrs["gen_ai.provider.name"] == "openai"
        assert attrs["gen_ai.request.model"] == MODEL
        assert attrs["gen_ai.request.temperature"] == 0
        assert attrs["gen_ai.usage.cache_read.input_tokens"] == 5
        assert attrs["gen_ai.response.id"].startswith("synthetic-")
        if mode == "SPAN_ONLY":
            inputs = json.loads(attrs["gen_ai.input.messages"])
            outputs = json.loads(attrs["gen_ai.output.messages"])
            assert inputs[0]["role"] == "user"
            assert inputs[0]["parts"][0]["content"].startswith("worker=")
            serialized = json.dumps(inputs)
            worker = attrs["gen_ai.session.id"]
            assert worker in serialized
            for other in {"auSync", "auStreamA", "auStreamB"} - {worker}:
                assert other not in serialized
            for message in inputs:
                if message["role"] == "tool":
                    assert message["parts"][0]["type"] == "tool_call_response"
                    assert message["parts"][0]["id"] in expected_ids
            assert any(
                part["type"] == "reasoning" and part["content"]
                for message in outputs
                for part in message["parts"]
            )
    for span in agents:
        assert span.attributes["gen_ai.usage.input_tokens"] == 60
        assert span.attributes["gen_ai.usage.output_tokens"] == 30
        if mode == "SPAN_ONLY":
            outputs = json.loads(span.attributes["gen_ai.output.messages"])
            assert len(outputs) == 1 and len(outputs[0]["parts"]) == 1
            assert outputs[0]["parts"][0]["type"] == "text"
            assert outputs[0]["parts"][0]["content"].startswith("[")
    metrics = collect_metrics(reader)
    assert counter_value(metrics, "llm_total_tokens") == 540
    assert counter_value(metrics, "agent_total_tokens") == 540
    if mode == "NO_CONTENT":
        carriers = (
            "gen_ai.input.messages",
            "gen_ai.output.messages",
            "gen_ai.tool.call.arguments",
            "gen_ai.tool.call.result",
        )
        for span in spans:
            assert not any(key in span.attributes for key in carriers)
            assert not any(
                key.endswith((".input", ".output"))
                for key in span.attributes
                if key.startswith("au.")
            )
