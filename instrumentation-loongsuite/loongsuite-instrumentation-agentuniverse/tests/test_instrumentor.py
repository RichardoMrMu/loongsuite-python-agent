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

"""Tests for the agentUniverse instrumentation bridge.

Everything here runs against a real ``agentUniverse`` install, with no
stand-in for the framework. The package bridges three layers -- the agent, the
``@trace_llm`` call and the tool -- and in every setup each layer must produce
exactly one span:

* native Agent/LLM/Tool instrumentors only -- the ``au.*`` spans, no
  ``gen_ai.*`` attributes;
* LoongSuite only -- those same native spans plus the ``gen_ai.*`` attributes;
* both enabled -- still one span per layer, carrying both namespaces.

Asserting the single-span property on every path is what catches a regression
back to wrapping ``Agent.run``, the LLM call or the tool call as a duplicated
span, rather than as a silent extra span in production.

Attribute names are spelled out as literals on purpose. They are the wire
contract, so the tests should fail if the instrumentation renames one.
"""

from __future__ import annotations

import json
import queue
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from agentuniverse.base.annotation import trace as trace_module
from agentuniverse.base.tracing.otel.instrumentation.agent.agent_instrumentor import (  # noqa: E501
    AgentInstrumentor,
    AgentSpanAttributesSetter,
)
from agentuniverse.base.tracing.otel.instrumentation.llm.llm_instrumentor import (  # noqa: E501
    LLMInstrumentor,
    LLMSpanAttributesSetter,
)
from agentuniverse.base.tracing.otel.instrumentation.tool.tool_instrumentor import (  # noqa: E501
    ToolInstrumentor,
    ToolSpanAttributesSetter,
)
from agentuniverse.llm.llm_output import TokenUsage

from opentelemetry import propagate, trace
from opentelemetry.instrumentation.agentuniverse import (
    AgentUniverseInstrumentor,
)
from opentelemetry.sdk.metrics.export import (
    InMemoryMetricReader,
)
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import StatusCode

from .conftest import (
    AGENT_LAYER,
    LLM_LAYER,
    NATIVE_LAYER_GLOBALS,
    NATIVE_WRAPPER_GLOBALS,
    TOOL_LAYER,
    AsyncLLM,
    AsyncRichAgent,
    FailingAgent,
    FailingLLM,
    FailingTool,
    LLMAgent,
    MessagesLLM,
    RichAgent,
    StreamingAgent,
    StreamingLLMAgent,
    StubLLM,
    StubTool,
    ToolAgent,
    active_native_instrumentor,
    build_agent,
)

AU_AGENT_SPAN_PREFIX = "au.agent."
AU_LLM_SPAN_PREFIX = "au.llm."
AU_TOOL_SPAN_PREFIX = "au.tool."
GEN_AI_PREFIX = "gen_ai."
SESSION_ATTR = "au.trace.session.id"
_CONTENT_ENV = "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"
_USER_MESSAGE = [
    {"role": "user", "parts": [{"content": "hello", "type": "text"}]}
]

# The metric names each native instrumentor is expected to keep emitting.
# These are a wire contract too: a rename here means dashboards go blank.
_METRIC_NAMES = {
    AGENT_LAYER: (
        "agent_calls_total",
        "agent_errors_total",
        "agent_call_duration",
        "agent_first_token_duration",
        "agent_total_tokens",
        "agent_prompt_tokens",
        "agent_completion_tokens",
        "agent_cached_tokens",
        "agent_reasoning_tokens",
    ),
    LLM_LAYER: (
        "llm_calls_total",
        "llm_errors_total",
        "llm_call_duration",
        "llm_first_token_duration",
        "llm_total_tokens",
        "llm_prompt_tokens",
        "llm_completion_tokens",
        "llm_cached_tokens",
        "llm_reasoning_tokens",
    ),
    TOOL_LAYER: (
        "tool_calls_total",
        "tool_errors_total",
        "tool_call_duration",
        "tool_total_tokens",
        "tool_prompt_tokens",
        "tool_completion_tokens",
        "tool_cached_tokens",
        "tool_reasoning_tokens",
    ),
}

_ALL_LAYERS = (AGENT_LAYER, LLM_LAYER, TOOL_LAYER)
_SPAN_PREFIXES = {
    AGENT_LAYER: AU_AGENT_SPAN_PREFIX,
    LLM_LAYER: AU_LLM_SPAN_PREFIX,
    TOOL_LAYER: AU_TOOL_SPAN_PREFIX,
}
_NATIVE_CLASSES = {
    AGENT_LAYER: AgentInstrumentor,
    LLM_LAYER: LLMInstrumentor,
    TOOL_LAYER: ToolInstrumentor,
}
_SETTER_CLASSES = {
    AGENT_LAYER: AgentSpanAttributesSetter,
    LLM_LAYER: LLMSpanAttributesSetter,
    TOOL_LAYER: ToolSpanAttributesSetter,
}
_BRIDGED_METHODS = {
    AGENT_LAYER: (
        "set_input_attributes",
        "set_success_attributes",
        "set_error_attributes",
    ),
    LLM_LAYER: (
        "set_input_attributes",
        "set_success_attributes",
        "set_error_attributes",
        "set_first_token_attributes",
    ),
    TOOL_LAYER: (
        "set_input_attributes",
        "set_success_attributes",
        "set_error_attributes",
    ),
}
_USAGE_KEYS = (
    "au.{layer}.usage.total_tokens",
    "au.{layer}.usage.prompt_tokens",
    "au.{layer}.usage.completion_tokens",
    "au.{layer}.usage.detail_tokens",
)
_CONTENT_KEYS = {
    AGENT_LAYER: ("au.agent.input", "au.agent.output"),
    LLM_LAYER: ("au.llm.input", "au.llm.output"),
    TOOL_LAYER: ("au.tool.input", "au.tool.output"),
}
_GEN_AI_CONTENT_KEYS = (
    "gen_ai.input.messages",
    "gen_ai.output.messages",
    "gen_ai.tool.call.arguments",
    "gen_ai.tool.call.result",
)

_SESSION_ID = "session-abc123"
_CARRIER_SESSION = "session-from-carrier"
_SESSION_CHILD = Path(__file__).with_name("session_probe_child.py")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _attributes(span: ReadableSpan) -> dict[str, Any]:
    return dict(span.attributes or {})


def _spans(exporter: InMemorySpanExporter) -> list[ReadableSpan]:
    return list(exporter.get_finished_spans())


def _single_span(exporter: InMemorySpanExporter) -> ReadableSpan:
    spans = _spans(exporter)
    assert len(spans) == 1, (
        f"expected exactly one span, got {len(spans)}: "
        f"{[span.name for span in spans]}"
    )
    return spans[0]


def _layer_spans(
    exporter: InMemorySpanExporter, layer: str
) -> list[ReadableSpan]:
    prefix = _SPAN_PREFIXES[layer]
    return [span for span in _spans(exporter) if span.name.startswith(prefix)]


def _single_layer_span(
    exporter: InMemorySpanExporter, layer: str
) -> ReadableSpan:
    spans = _layer_spans(exporter, layer)
    assert len(spans) == 1, (
        f"expected exactly one {_SPAN_PREFIXES[layer]}* span, got "
        f"{len(spans)}: {[span.name for span in spans]}"
    )
    return spans[0]


def _assert_one_span_per_layer(exporter: InMemorySpanExporter) -> None:
    """The three-layer call tree is exactly one span per layer."""
    names = sorted(span.name for span in _spans(exporter))
    assert names == [
        "au.agent.rich_agent",
        "au.llm.stub_llm",
        "au.tool.stub_tool",
    ], f"unexpected span tree: {names}"


def _has_gen_ai_attributes(span: ReadableSpan) -> bool:
    return any(key.startswith(GEN_AI_PREFIX) for key in _attributes(span))


def _instrument_native(
    layer: str, tracer_provider: Any, meter_provider: Any = None
) -> Any:
    """Enable one native layer the way an application would."""
    instrumentor = _NATIVE_CLASSES[layer]()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
        skip_dep_check=True,
    )
    return instrumentor


def _instrument_all_natives(
    tracer_provider: Any, meter_provider: Any = None
) -> list[Any]:
    return [
        _instrument_native(layer, tracer_provider, meter_provider)
        for layer in _ALL_LAYERS
    ]


def _instrument_bridge(
    tracer_provider: Any, meter_provider: Any = None
) -> AgentUniverseInstrumentor:
    instrumentor = AgentUniverseInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
        skip_dep_check=True,
    )
    return instrumentor


def _collect_metrics(
    reader: InMemoryMetricReader,
) -> dict[str, list[Any]]:
    """Every recorded data point, keyed by metric name."""
    collected: dict[str, list[Any]] = {}
    for resource_metrics in reader.get_metrics_data().resource_metrics:
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                points = getattr(metric.data, "data_points", [])
                collected.setdefault(metric.name, []).extend(points)
    return collected


def _matching_points(
    reader: InMemoryMetricReader, name: str, labels: dict
) -> list[Any]:
    matched = []
    for point in _collect_metrics(reader).get(name, []):
        attributes = dict(point.attributes)
        if all(attributes.get(key) == value for key, value in labels.items()):
            matched.append(point)
    return matched


def _only_point(reader: InMemoryMetricReader, name: str, **labels: Any) -> Any:
    matched = _matching_points(reader, name, labels)
    assert len(matched) == 1, (
        f"{name}: expected one data point matching {labels}, got "
        f"{len(matched)}"
    )
    return matched[0]


def _counter(reader: InMemoryMetricReader, name: str, **labels: Any) -> Any:
    return _only_point(reader, name, **labels).value


def _histogram_sum(
    reader: InMemoryMetricReader, name: str, **labels: Any
) -> Any:
    return _only_point(reader, name, **labels).sum


def _assert_au_agent_span(
    span: ReadableSpan,
    agent_name: str,
    status: str = "success",
    content: bool = True,
) -> dict[str, Any]:
    """Assert the native ``au.*`` agent span shape the bridge must preserve.

    ``content=False`` asserts the privacy-filtered shape instead: the agent
    identity, caller, timing, status, pairing and token usage stay, while the
    two attributes that carry user content are gone.
    """
    attributes = _attributes(span)
    assert span.name == f"{AU_AGENT_SPAN_PREFIX}{agent_name}"
    assert attributes["au.span.kind"] == "agent"
    assert attributes["au.agent.name"] == agent_name
    assert attributes["au.agent.status"] == status
    assert attributes["au.agent.pair_id"]
    assert "au.agent.duration" in attributes
    assert attributes["au.trace.caller_type"] == "user"
    assert "au.trace.caller_name" in attributes
    for key in _USAGE_KEYS:
        assert key.format(layer=AGENT_LAYER) in attributes
    assert ("au.agent.input" in attributes) is content
    if status == "success":
        assert ("au.agent.output" in attributes) is content
    return attributes


def _assert_gen_ai_agent_span(
    span: ReadableSpan, agent_name: str, content: bool = False
) -> dict[str, Any]:
    attributes = _attributes(span)
    assert attributes["gen_ai.span.kind"] == "AGENT"
    assert attributes["gen_ai.operation.name"] == "invoke_agent"
    assert attributes["gen_ai.framework"] == "agentuniverse"
    assert attributes["gen_ai.agent.name"] == agent_name
    assert ("gen_ai.input.messages" in attributes) is content
    return attributes


def _assert_au_llm_span(
    span: ReadableSpan,
    llm_name: str,
    status: str = "success",
    content: bool = True,
) -> dict[str, Any]:
    attributes = _attributes(span)
    assert span.name == f"{AU_LLM_SPAN_PREFIX}{llm_name}"
    assert attributes["au.span.kind"] == "llm"
    assert attributes["au.llm.name"] == llm_name
    assert attributes["au.llm.channel_name"] == "test_channel"
    assert attributes["au.llm.status"] == status
    assert "au.llm.duration" in attributes
    assert "au.llm.llm_params" in attributes
    assert attributes["au.trace.caller_type"] in ("agent", "user")
    assert ("au.llm.input" in attributes) is content
    if status == "success":
        assert attributes["au.llm.streaming"] in (True, False)
        assert "au.llm.first_token.duration" in attributes
        for key in _USAGE_KEYS:
            assert key.format(layer=LLM_LAYER) in attributes
        assert ("au.llm.output" in attributes) is content
    return attributes


def _assert_gen_ai_llm_span(
    span: ReadableSpan, content: bool = False
) -> dict[str, Any]:
    attributes = _attributes(span)
    assert attributes["gen_ai.span.kind"] == "LLM"
    assert attributes["gen_ai.operation.name"] == "chat"
    assert attributes["gen_ai.framework"] == "agentuniverse"
    assert ("gen_ai.input.messages" in attributes) is content
    assert ("gen_ai.output.messages" in attributes) is content
    return attributes


def _assert_au_tool_span(
    span: ReadableSpan,
    tool_name: str,
    status: str = "success",
    content: bool = True,
) -> dict[str, Any]:
    attributes = _attributes(span)
    assert span.name == f"{AU_TOOL_SPAN_PREFIX}{tool_name}"
    assert attributes["au.span.kind"] == "tool"
    assert attributes["au.tool.name"] == tool_name
    assert attributes["au.tool.status"] == status
    assert attributes["au.tool.pair_id"]
    assert "au.tool.duration" in attributes
    assert attributes["au.trace.caller_type"] in ("agent", "user")
    for key in _USAGE_KEYS:
        assert key.format(layer=TOOL_LAYER) in attributes
    assert ("au.tool.input" in attributes) is content
    if status == "success":
        assert ("au.tool.output" in attributes) is content
    return attributes


def _assert_gen_ai_tool_span(
    span: ReadableSpan, tool_name: str, content: bool = False
) -> dict[str, Any]:
    attributes = _attributes(span)
    assert attributes["gen_ai.span.kind"] == "TOOL"
    assert attributes["gen_ai.operation.name"] == "execute_tool"
    assert attributes["gen_ai.framework"] == "agentuniverse"
    assert attributes["gen_ai.tool.name"] == tool_name
    assert attributes["gen_ai.tool.type"] == "function"
    assert attributes["gen_ai.tool.call.id"] == attributes["au.tool.pair_id"]
    assert ("gen_ai.tool.call.arguments" in attributes) is content
    assert ("gen_ai.tool.call.result" in attributes) is content
    return attributes


def _assert_no_content_on_the_layer(layer: str, span: ReadableSpan) -> None:
    attributes = _attributes(span)
    for key in _CONTENT_KEYS[layer]:
        assert key not in attributes, f"{key} must not be exported"


def _run_session_child() -> dict:
    completed = subprocess.run(
        [sys.executable, str(_SESSION_CHILD)],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout.strip().splitlines()[-1])


# ---------------------------------------------------------------------------
# Three-layer assembly: ownership and the single-span property
# ---------------------------------------------------------------------------


class TestThreeLayerAssembly:
    def test_bridge_owns_all_three_layers_when_none_is_active(
        self, tracer_provider, meter_provider
    ):
        bridge = _instrument_bridge(tracer_provider, meter_provider)
        try:
            assert [layer.key for layer in bridge._layers] == list(_ALL_LAYERS)
            assert all(layer.owned for layer in bridge._layers)
        finally:
            bridge.uninstrument()

    def test_one_span_per_layer_with_both_namespaces(
        self, tracer_provider, span_exporter
    ):
        bridge = _instrument_bridge(tracer_provider)
        try:
            result = build_agent("rich_agent", RichAgent).run(input="hello")
        finally:
            bridge.uninstrument()

        assert result.get_data("output") == "llm:hello|tool-output:hello"
        _assert_one_span_per_layer(span_exporter)
        agent_span = _single_layer_span(span_exporter, AGENT_LAYER)
        llm_span = _single_layer_span(span_exporter, LLM_LAYER)
        tool_span = _single_layer_span(span_exporter, TOOL_LAYER)
        _assert_au_agent_span(agent_span, "rich_agent", content=False)
        _assert_gen_ai_agent_span(agent_span, "rich_agent")
        _assert_au_llm_span(llm_span, "stub_llm", content=False)
        _assert_gen_ai_llm_span(llm_span)
        _assert_au_tool_span(tool_span, "stub_tool", content=False)
        _assert_gen_ai_tool_span(tool_span, "stub_tool")

    def test_llm_and_tool_spans_nest_under_the_agent_span(
        self, tracer_provider, span_exporter
    ):
        bridge = _instrument_bridge(tracer_provider)
        try:
            build_agent("rich_agent", RichAgent).run(input="hello")
        finally:
            bridge.uninstrument()

        agent_span = _single_layer_span(span_exporter, AGENT_LAYER)
        assert agent_span.parent is None
        for layer in (LLM_LAYER, TOOL_LAYER):
            span = _single_layer_span(span_exporter, layer)
            assert span.parent is not None
            assert span.parent.span_id == agent_span.context.span_id

    def test_native_only_creates_one_span_per_layer_without_gen_ai(
        self, tracer_provider, span_exporter
    ):
        natives = _instrument_all_natives(tracer_provider)
        try:
            build_agent("rich_agent", RichAgent).run(input="hello")
        finally:
            for native in natives:
                native.uninstrument()

        _assert_one_span_per_layer(span_exporter)
        # The native layers write their content unconditionally; the bridge is
        # not installed here, so nothing filters or augments them.
        _assert_au_agent_span(
            _single_layer_span(span_exporter, AGENT_LAYER), "rich_agent"
        )
        _assert_au_llm_span(
            _single_layer_span(span_exporter, LLM_LAYER), "stub_llm"
        )
        _assert_au_tool_span(
            _single_layer_span(span_exporter, TOOL_LAYER), "stub_tool"
        )
        for span in _spans(span_exporter):
            assert not _has_gen_ai_attributes(span), span.name

    def test_both_enabled_keeps_one_span_per_layer(
        self, tracer_provider, span_exporter, meter_provider, metric_reader
    ):
        natives = _instrument_all_natives(tracer_provider, meter_provider)
        bridge = _instrument_bridge(tracer_provider, meter_provider)
        try:
            # Reuse, never a second construction: a native instrumentor's
            # __init__ resets the wrapper originals and the metric recorder it
            # is already serving calls with.
            for layer, native in zip(bridge._layers, natives):
                assert layer.owned is False
                assert layer.native is native
            build_agent("rich_agent", RichAgent).run(input="hello")
        finally:
            bridge.uninstrument()
            for native in natives:
                native.uninstrument()

        _assert_one_span_per_layer(span_exporter)
        _assert_gen_ai_agent_span(
            _single_layer_span(span_exporter, AGENT_LAYER), "rich_agent"
        )
        _assert_gen_ai_llm_span(_single_layer_span(span_exporter, LLM_LAYER))
        _assert_gen_ai_tool_span(
            _single_layer_span(span_exporter, TOOL_LAYER), "stub_tool"
        )
        # One call per layer: a second native instrumentor would double these.
        assert (
            _counter(
                metric_reader, "agent_calls_total", au_agent_name="rich_agent"
            )
            == 1
        )
        assert (
            _counter(metric_reader, "llm_calls_total", au_llm_name="stub_llm")
            == 1
        )
        assert (
            _counter(
                metric_reader, "tool_calls_total", au_tool_name="stub_tool"
            )
            == 1
        )

    def test_partial_ownership_reuses_only_the_agent_layer(
        self, tracer_provider, span_exporter
    ):
        native_agent = _instrument_native(AGENT_LAYER, tracer_provider)
        bridge = _instrument_bridge(tracer_provider)
        try:
            ownership = {layer.key: layer.owned for layer in bridge._layers}
            assert ownership == {
                AGENT_LAYER: False,
                LLM_LAYER: True,
                TOOL_LAYER: True,
            }
            assert bridge._layers[0].native is native_agent
            build_agent("rich_agent", RichAgent).run(input="hello")
        finally:
            bridge.uninstrument()
            native_agent.uninstrument()

        _assert_one_span_per_layer(span_exporter)
        for layer in _ALL_LAYERS:
            assert _has_gen_ai_attributes(
                _single_layer_span(span_exporter, layer)
            )

    def test_partial_ownership_reuses_only_the_llm_layer(
        self, tracer_provider, span_exporter
    ):
        native_llm = _instrument_native(LLM_LAYER, tracer_provider)
        bridge = _instrument_bridge(tracer_provider)
        try:
            ownership = {layer.key: layer.owned for layer in bridge._layers}
            assert ownership == {
                AGENT_LAYER: True,
                LLM_LAYER: False,
                TOOL_LAYER: True,
            }
            assert bridge._layers[1].native is native_llm
            build_agent("rich_agent", RichAgent).run(input="hello")
        finally:
            bridge.uninstrument()
            native_llm.uninstrument()

        _assert_one_span_per_layer(span_exporter)
        for layer in _ALL_LAYERS:
            assert _has_gen_ai_attributes(
                _single_layer_span(span_exporter, layer)
            )


# ---------------------------------------------------------------------------
# The bridge delegates: the native wrappers stay the span owners
# ---------------------------------------------------------------------------


def test_bridge_does_not_wrap_framework_entry_points(tracer_provider):
    from agentuniverse.agent.action.tool.tool import Tool
    from agentuniverse.agent.agent import Agent

    run_before = Agent.run
    async_run_before = Agent.async_run
    tool_run_before = Tool.run
    tool_async_run_before = Tool.async_run

    bridge = _instrument_bridge(tracer_provider)
    try:
        assert Agent.run is run_before
        assert Agent.async_run is async_run_before
        assert Tool.run is tool_run_before
        assert Tool.async_run is tool_async_run_before
        # Every layer's span comes from a native instrumentor, not from here.
        for native_cls in NATIVE_LAYER_GLOBALS:
            assert active_native_instrumentor(native_cls) is not None
    finally:
        bridge.uninstrument()


def test_loongsuite_only_agent_span_carries_both_namespaces(
    tracer_provider, span_exporter, make_agent
):
    bridge = _instrument_bridge(tracer_provider)
    try:
        make_agent().run(input="hello")
    finally:
        bridge.uninstrument()

    span = _single_span(span_exporter)
    _assert_au_agent_span(span, "test_agent", content=False)
    _assert_gen_ai_agent_span(span, "test_agent")


# ---------------------------------------------------------------------------
# The agent layer
# ---------------------------------------------------------------------------


class TestAgentLayer:
    def test_error_span_keeps_the_native_error_attributes(
        self, tracer_provider, span_exporter, make_agent
    ):
        agent = make_agent("boom_agent", FailingAgent)
        bridge = _instrument_bridge(tracer_provider)
        try:
            with pytest.raises(RuntimeError, match="agent exploded"):
                agent.run(input="hello")
        finally:
            bridge.uninstrument()

        span = _single_span(span_exporter)
        assert span.status.status_code == StatusCode.ERROR
        attributes = _assert_au_agent_span(
            span, "boom_agent", status="error", content=False
        )
        assert attributes["au.agent.error.type"] == "RuntimeError"
        assert "agent exploded" in attributes["au.agent.error.message"]
        assert attributes["error.type"] == "RuntimeError"
        _assert_gen_ai_agent_span(span, "boom_agent")

    def test_error_is_counted_once(
        self, tracer_provider, meter_provider, metric_reader, make_agent
    ):
        agent = make_agent("boom_agent", FailingAgent)
        bridge = _instrument_bridge(tracer_provider, meter_provider)
        try:
            with pytest.raises(RuntimeError):
                agent.run(input="hello")
        finally:
            bridge.uninstrument()

        assert (
            _counter(
                metric_reader,
                "agent_errors_total",
                au_agent_name="boom_agent",
            )
            == 1
        )

    def test_streaming_records_the_first_token(
        self,
        tracer_provider,
        meter_provider,
        metric_reader,
        span_exporter,
        make_agent,
    ):
        agent = make_agent("stream_agent", StreamingAgent)
        bridge = _instrument_bridge(tracer_provider, meter_provider)
        try:
            result = agent.run(input="hello", output_stream=queue.Queue())
        finally:
            bridge.uninstrument()

        assert result.get_data("output") == "streamed"
        span = _single_span(span_exporter)
        attributes = _assert_au_agent_span(span, "stream_agent", content=False)
        assert attributes["au.agent.streaming"] is True
        assert attributes["au.agent.first_token.duration"] > 0
        _assert_gen_ai_agent_span(span, "stream_agent")

        # On the streaming path the native wrapper records the first token from
        # the queue callback, so this data point can only come from there.
        point = _only_point(
            metric_reader,
            "agent_first_token_duration",
            au_agent_name="stream_agent",
        )
        assert point.count == 1
        assert point.attributes["au_agent_streaming"] is True
        assert point.sum > 0

    def test_token_usage_attributes_stay_on_the_span(
        self, tracer_provider, span_exporter, make_agent
    ):
        bridge = _instrument_bridge(tracer_provider)
        try:
            make_agent().run(input="hello")
        finally:
            bridge.uninstrument()

        attributes = _attributes(_single_span(span_exporter))
        # A stub agent makes no LLM call, so the native totals are zero -- the
        # point is that the attributes are there and keep their native shape.
        assert attributes["au.agent.usage.total_tokens"] == 0
        assert attributes["au.agent.usage.prompt_tokens"] == 0
        assert attributes["au.agent.usage.completion_tokens"] == 0
        assert set(json.loads(attributes["au.agent.usage.detail_tokens"])) == {
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
        }

    def test_metrics_family_is_complete_and_recorded_once(
        self, tracer_provider, meter_provider, metric_reader, make_agent
    ):
        bridge = _instrument_bridge(tracer_provider, meter_provider)
        try:
            make_agent().run(input="hello")
        finally:
            bridge.uninstrument()

        collected = _collect_metrics(metric_reader)
        for name in _METRIC_NAMES[AGENT_LAYER]:
            if name == "agent_errors_total":
                continue  # only an error produces this one
            assert name in collected, f"{name} was never recorded"
        assert (
            _counter(
                metric_reader, "agent_calls_total", au_agent_name="test_agent"
            )
            == 1
        )
        # The duration is recorded once. Its value is not asserted positive:
        # ``time.time()`` on Windows only resolves to about 15 ms and this
        # agent body does no work.
        assert (
            _only_point(
                metric_reader,
                "agent_call_duration",
                au_agent_name="test_agent",
            ).count
            == 1
        )
        assert (
            _histogram_sum(
                metric_reader, "agent_total_tokens", au_agent_name="test_agent"
            )
            == 0
        )


@pytest.mark.asyncio
async def test_async_run_produces_one_span_per_layer(
    tracer_provider, span_exporter
):
    bridge = _instrument_bridge(tracer_provider)
    try:
        result = await build_agent("async_agent", AsyncRichAgent).async_run(
            input="hello async"
        )
    finally:
        bridge.uninstrument()

    assert (
        result.get_data("output")
        == "async:hello async|tool-output:hello async"
    )
    assert sorted(span.name for span in _spans(span_exporter)) == [
        "au.agent.async_agent",
        "au.llm.async_llm",
        "au.tool.stub_tool",
    ]
    _assert_au_agent_span(
        _single_layer_span(span_exporter, AGENT_LAYER),
        "async_agent",
        content=False,
    )
    _assert_gen_ai_agent_span(
        _single_layer_span(span_exporter, AGENT_LAYER), "async_agent"
    )
    _assert_gen_ai_llm_span(_single_layer_span(span_exporter, LLM_LAYER))
    _assert_gen_ai_tool_span(
        _single_layer_span(span_exporter, TOOL_LAYER), "stub_tool"
    )


# ---------------------------------------------------------------------------
# The LLM layer
# ---------------------------------------------------------------------------


class TestLLMLayer:
    def test_native_only_llm_span_has_no_gen_ai(
        self, tracer_provider, span_exporter
    ):
        native = _instrument_native(LLM_LAYER, tracer_provider)
        try:
            StubLLM().call(prompt="hello")
        finally:
            native.uninstrument()

        span = _single_span(span_exporter)
        attributes = _assert_au_llm_span(span, "stub_llm")
        assert attributes["au.llm.output"] == "llm:hello"
        assert not _has_gen_ai_attributes(span)

    def test_llm_span_carries_both_namespaces(
        self, tracer_provider, span_exporter
    ):
        bridge = _instrument_bridge(tracer_provider)
        try:
            StubLLM().call(prompt="hello")
        finally:
            bridge.uninstrument()

        span = _single_span(span_exporter)
        _assert_au_llm_span(span, "stub_llm", content=False)
        attributes = _assert_gen_ai_llm_span(span)
        # Token usage is mirrored from the native counters, so the two
        # namespaces cannot disagree.
        assert attributes["gen_ai.usage.input_tokens"] == 3
        assert attributes["gen_ai.usage.output_tokens"] == 5
        assert attributes["gen_ai.usage.total_tokens"] == 8

    def test_messages_input_becomes_a_gen_ai_input_message(
        self, tracer_provider, span_exporter, monkeypatch
    ):
        monkeypatch.setenv(_CONTENT_ENV, "SPAN_ONLY")
        bridge = _instrument_bridge(tracer_provider)
        try:
            MessagesLLM().call(messages=[{"role": "user", "content": "hi"}])
        finally:
            bridge.uninstrument()

        attributes = _attributes(_single_span(span_exporter))
        assert json.loads(attributes["gen_ai.input.messages"]) == [
            {"role": "user", "parts": [{"type": "text", "content": "hi"}]}
        ]
        # The native carrier keeps the caller's arguments, plus the empty
        # ``kwargs`` that ``inspect.signature().bind()`` fills in.
        assert json.loads(attributes["au.llm.input"])["messages"] == [
            {"role": "user", "content": "hi"}
        ]

    def test_prompt_input_becomes_a_gen_ai_input_message(
        self, tracer_provider, span_exporter, monkeypatch
    ):
        monkeypatch.setenv(_CONTENT_ENV, "SPAN_ONLY")
        bridge = _instrument_bridge(tracer_provider)
        try:
            StubLLM().call(prompt="hello")
        finally:
            bridge.uninstrument()

        attributes = _attributes(_single_span(span_exporter))
        assert json.loads(attributes["gen_ai.input.messages"]) == _USER_MESSAGE
        assert json.loads(attributes["gen_ai.output.messages"])[0][
            "parts"
        ] == [{"type": "text", "content": "llm:hello"}]

    def test_temperature_is_mirrored_when_the_caller_sets_it(
        self, tracer_provider, span_exporter
    ):
        bridge = _instrument_bridge(tracer_provider)
        try:
            StubLLM().call(prompt="hello", temperature=0.5)
        finally:
            bridge.uninstrument()

        attributes = _attributes(_single_span(span_exporter))
        assert attributes["gen_ai.request.temperature"] == 0.5
        assert json.loads(attributes["au.llm.llm_params"]) == {
            "temperature": 0.5
        }

    def test_the_native_temperature_sentinel_is_not_exported(
        self, tracer_provider, span_exporter
    ):
        # ``_get_llm_info`` overwrites a channel-model temperature with -1
        # unless the caller passes one, and -1 means "unknown" -- it must not
        # reach the GenAI attribute.
        bridge = _instrument_bridge(tracer_provider)
        try:
            StubLLM().call(prompt="hello")
        finally:
            bridge.uninstrument()

        attributes = _attributes(_single_span(span_exporter))
        assert json.loads(attributes["au.llm.llm_params"]) == {
            "temperature": -1
        }
        assert "gen_ai.request.temperature" not in attributes

    def test_streaming_first_token_is_positive_on_both_namespaces(
        self, tracer_provider, span_exporter, meter_provider, metric_reader
    ):
        agent = build_agent("stream_llm_agent", StreamingLLMAgent)
        bridge = _instrument_bridge(tracer_provider, meter_provider)
        try:
            agent.run(input="hello")
        finally:
            bridge.uninstrument()

        span = _single_layer_span(span_exporter, LLM_LAYER)
        attributes = _assert_au_llm_span(span, "stream_llm", content=False)
        assert attributes["au.llm.streaming"] is True
        assert attributes["au.llm.first_token.duration"] > 0
        assert attributes["gen_ai.response.time_to_first_token"] > 0
        _assert_gen_ai_llm_span(span)

        point = _only_point(
            metric_reader, "llm_first_token_duration", au_llm_name="stream_llm"
        )
        assert point.attributes["au_llm_streaming"] is True
        assert point.sum > 0

    def test_error_span_keeps_the_native_error_attributes(
        self, tracer_provider, span_exporter, meter_provider, metric_reader
    ):
        bridge = _instrument_bridge(tracer_provider, meter_provider)
        try:
            with pytest.raises(RuntimeError, match="llm exploded"):
                FailingLLM().call(prompt="hello")
        finally:
            bridge.uninstrument()

        span = _single_span(span_exporter)
        assert span.status.status_code == StatusCode.ERROR
        attributes = _assert_au_llm_span(
            span, "failing_llm", status="error", content=False
        )
        assert attributes["au.llm.error.type"] == "RuntimeError"
        assert "llm exploded" in attributes["au.llm.error.message"]
        assert attributes["error.type"] == "RuntimeError"
        _assert_gen_ai_llm_span(span)
        assert (
            _counter(
                metric_reader, "llm_errors_total", au_llm_name="failing_llm"
            )
            == 1
        )

    def test_metrics_are_recorded_once(
        self, tracer_provider, span_exporter, meter_provider, metric_reader
    ):
        bridge = _instrument_bridge(tracer_provider, meter_provider)
        try:
            StubLLM().call(prompt="hello")
        finally:
            bridge.uninstrument()

        collected = _collect_metrics(metric_reader)
        for name in _METRIC_NAMES[LLM_LAYER]:
            if name == "llm_errors_total":
                continue  # only an error produces this one
            assert name in collected, f"{name} was never recorded"
        assert (
            _counter(metric_reader, "llm_calls_total", au_llm_name="stub_llm")
            == 1
        )
        assert (
            _histogram_sum(
                metric_reader, "llm_total_tokens", au_llm_name="stub_llm"
            )
            == 8
        )
        assert (
            _histogram_sum(
                metric_reader, "llm_prompt_tokens", au_llm_name="stub_llm"
            )
            == 3
        )
        assert (
            _histogram_sum(
                metric_reader, "llm_completion_tokens", au_llm_name="stub_llm"
            )
            == 5
        )

    @pytest.mark.asyncio
    async def test_async_llm_call_produces_one_span(
        self, tracer_provider, span_exporter
    ):
        bridge = _instrument_bridge(tracer_provider)
        try:
            result = await AsyncLLM().call(prompt="hello")
        finally:
            bridge.uninstrument()

        assert result.text == "async:hello"
        span = _single_span(span_exporter)
        attributes = _assert_au_llm_span(span, "async_llm", content=False)
        assert attributes["au.llm.usage.total_tokens"] == 6
        _assert_gen_ai_llm_span(span)


# ---------------------------------------------------------------------------
# The tool layer
# ---------------------------------------------------------------------------


class TestToolLayer:
    def test_native_only_tool_span_has_no_gen_ai(
        self, tracer_provider, span_exporter
    ):
        native = _instrument_native(TOOL_LAYER, tracer_provider)
        try:
            result = StubTool().run(query="hello")
        finally:
            native.uninstrument()

        assert result == "tool-output:hello"
        span = _single_span(span_exporter)
        attributes = _assert_au_tool_span(span, "stub_tool")
        assert json.loads(attributes["au.tool.output"]) == "tool-output:hello"
        assert not _has_gen_ai_attributes(span)

    def test_tool_span_carries_both_namespaces(
        self, tracer_provider, span_exporter
    ):
        bridge = _instrument_bridge(tracer_provider)
        try:
            StubTool().run(query="hello")
        finally:
            bridge.uninstrument()

        span = _single_span(span_exporter)
        _assert_au_tool_span(span, "stub_tool", content=False)
        _assert_gen_ai_tool_span(span, "stub_tool")

    def test_a_tool_with_no_usage_gets_no_gen_ai_usage_attributes(
        self, tracer_provider, span_exporter
    ):
        bridge = _instrument_bridge(tracer_provider)
        try:
            StubTool().run(query="hello")
        finally:
            bridge.uninstrument()

        attributes = _attributes(_single_span(span_exporter))
        assert attributes["au.tool.usage.total_tokens"] == 0
        assert "gen_ai.usage.total_tokens" not in attributes
        assert "gen_ai.usage.input_tokens" not in attributes

    def test_tool_error_span_keeps_the_native_error_attributes(
        self, tracer_provider, span_exporter, meter_provider, metric_reader
    ):
        bridge = _instrument_bridge(tracer_provider, meter_provider)
        try:
            with pytest.raises(RuntimeError, match="tool exploded"):
                FailingTool().run(query="hello")
        finally:
            bridge.uninstrument()

        span = _single_span(span_exporter)
        assert span.status.status_code == StatusCode.ERROR
        attributes = _assert_au_tool_span(
            span, "failing_tool", status="error", content=False
        )
        assert attributes["au.tool.error.type"] == "RuntimeError"
        assert attributes["error.type"] == "RuntimeError"
        _assert_gen_ai_tool_span(span, "failing_tool")
        assert (
            _counter(
                metric_reader, "tool_errors_total", au_tool_name="failing_tool"
            )
            == 1
        )

    def test_metrics_are_recorded_once(
        self, tracer_provider, span_exporter, meter_provider, metric_reader
    ):
        bridge = _instrument_bridge(tracer_provider, meter_provider)
        try:
            StubTool().run(query="hello")
        finally:
            bridge.uninstrument()

        collected = _collect_metrics(metric_reader)
        for name in _METRIC_NAMES[TOOL_LAYER]:
            if name == "tool_errors_total":
                continue  # only an error produces this one
            assert name in collected, f"{name} was never recorded"
        assert (
            _counter(
                metric_reader, "tool_calls_total", au_tool_name="stub_tool"
            )
            == 1
        )
        assert (
            _only_point(
                metric_reader, "tool_call_duration", au_tool_name="stub_tool"
            ).count
            == 1
        )

    @pytest.mark.asyncio
    async def test_async_run_produces_one_span(
        self, tracer_provider, span_exporter
    ):
        bridge = _instrument_bridge(tracer_provider)
        try:
            result = await StubTool().async_run(query="hello async")
        finally:
            bridge.uninstrument()

        assert result == "tool-output:hello async"
        span = _single_span(span_exporter)
        _assert_au_tool_span(span, "stub_tool", content=False)
        _assert_gen_ai_tool_span(span, "stub_tool")

    def test_a_tool_call_inside_an_agent_nests_under_it(
        self, tracer_provider, span_exporter
    ):
        bridge = _instrument_bridge(tracer_provider)
        try:
            build_agent("tool_agent", ToolAgent).run(input="hello")
        finally:
            bridge.uninstrument()

        assert sorted(span.name for span in _spans(span_exporter)) == [
            "au.agent.tool_agent",
            "au.tool.stub_tool",
        ]
        agent_span = _single_layer_span(span_exporter, AGENT_LAYER)
        tool_span = _single_layer_span(span_exporter, TOOL_LAYER)
        assert tool_span.parent.span_id == agent_span.context.span_id
        assert _attributes(tool_span)["au.trace.caller_name"] == "tool_agent"
        # Both layers keep both namespaces even when they are nested.
        _assert_gen_ai_agent_span(agent_span, "tool_agent")
        _assert_gen_ai_tool_span(tool_span, "stub_tool")


# ---------------------------------------------------------------------------
# Token usage: the real field names, and the aggregation into the parent
# ---------------------------------------------------------------------------


class TestTokenUsage:
    def test_the_real_fields_are_text_in_and_text_out(self):
        usage = TokenUsage(text_in=3, text_out=5)
        assert usage.prompt_tokens == 3
        assert usage.completion_tokens == 5
        assert usage.total_tokens == 8

    def test_the_derived_property_names_are_silently_ignored(self):
        # ``TokenUsage`` is a pydantic-v1 model whose real fields are the
        # text/image/audio/cached/reasoning counters; ``prompt_tokens`` and
        # friends are read-only properties. Passing them sets nothing, which is
        # what made an earlier probe report "the aggregation is always zero".
        usage = TokenUsage(prompt=3, completion=5, total=8)
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0
        assert usage.dict()["text_in"] == 0

    def test_a_real_llm_child_aggregates_onto_the_agent_span(
        self, tracer_provider, span_exporter
    ):
        bridge = _instrument_bridge(tracer_provider)
        try:
            build_agent("token_agent", LLMAgent).run(input="hello")
        finally:
            bridge.uninstrument()

        assert sorted(span.name for span in _spans(span_exporter)) == [
            "au.agent.token_agent",
            "au.llm.stub_llm",
        ]
        agent_attributes = _attributes(
            _single_layer_span(span_exporter, AGENT_LAYER)
        )
        assert agent_attributes["au.agent.usage.prompt_tokens"] == 3
        assert agent_attributes["au.agent.usage.completion_tokens"] == 5
        assert agent_attributes["au.agent.usage.total_tokens"] == 8
        assert agent_attributes["gen_ai.usage.input_tokens"] == 3
        assert agent_attributes["gen_ai.usage.output_tokens"] == 5
        assert agent_attributes["gen_ai.usage.total_tokens"] == 8
        detail = json.loads(agent_attributes["au.agent.usage.detail_tokens"])
        assert detail["prompt_tokens"] == 3
        assert detail["completion_tokens"] == 5
        assert detail["total_tokens"] == 8

        llm_attributes = _attributes(
            _single_layer_span(span_exporter, LLM_LAYER)
        )
        assert llm_attributes["au.llm.usage.total_tokens"] == 8
        assert llm_attributes["gen_ai.usage.total_tokens"] == 8

    def test_a_real_llm_child_aggregates_onto_the_agent_metrics(
        self, tracer_provider, meter_provider, metric_reader
    ):
        bridge = _instrument_bridge(tracer_provider, meter_provider)
        try:
            build_agent("token_agent", LLMAgent).run(input="hello")
        finally:
            bridge.uninstrument()

        labels = {"au_agent_name": "token_agent"}
        assert _counter(metric_reader, "agent_calls_total", **labels) == 1
        assert (
            _histogram_sum(metric_reader, "agent_total_tokens", **labels) == 8
        )
        assert (
            _histogram_sum(metric_reader, "agent_prompt_tokens", **labels) == 3
        )
        assert (
            _histogram_sum(metric_reader, "agent_completion_tokens", **labels)
            == 5
        )
        assert (
            _counter(metric_reader, "llm_calls_total", au_llm_name="stub_llm")
            == 1
        )
        assert (
            _histogram_sum(
                metric_reader, "llm_total_tokens", au_llm_name="stub_llm"
            )
            == 8
        )

    def test_streaming_usage_is_double_counted_into_the_parent(
        self, tracer_provider, span_exporter, meter_provider, metric_reader
    ):
        """Pin an agentUniverse 0.0.19.1 defect, so it stays visible.

        ``llm_instrumentor.py`` adds the streamed usage to the LLM span's token
        entry in ``process_sync_stream`` (and its async twin) and again in
        ``_finalize_streaming_result``; ``LLMSpanManager.cleanup()`` then hands
        the parent twice that usage. The LLM span keeps the real numbers -- only
        the parent aggregate doubles. If agentUniverse ever fixes this, the
        aggregate asserted here becomes 4/6/10 and this test says so.
        """
        bridge = _instrument_bridge(tracer_provider, meter_provider)
        try:
            build_agent("stream_agent", StreamingLLMAgent).run(input="hello")
        finally:
            bridge.uninstrument()

        llm_attributes = _attributes(
            _single_layer_span(span_exporter, LLM_LAYER)
        )
        assert llm_attributes["au.llm.usage.prompt_tokens"] == 4
        assert llm_attributes["au.llm.usage.completion_tokens"] == 6
        assert llm_attributes["au.llm.usage.total_tokens"] == 10

        agent_attributes = _attributes(
            _single_layer_span(span_exporter, AGENT_LAYER)
        )
        assert agent_attributes["au.agent.usage.prompt_tokens"] == 8
        assert agent_attributes["au.agent.usage.completion_tokens"] == 12
        assert agent_attributes["au.agent.usage.total_tokens"] == 20
        # The mirrored gen_ai usage matches the native aggregate exactly.
        assert agent_attributes["gen_ai.usage.total_tokens"] == 20
        assert (
            _histogram_sum(
                metric_reader,
                "agent_total_tokens",
                au_agent_name="stream_agent",
            )
            == 20
        )


# ---------------------------------------------------------------------------
# Privacy: content capture is off by default
# ---------------------------------------------------------------------------


class TestContentPrivacy:
    def test_owned_bridge_with_capture_off_filters_all_three_layers(
        self, tracer_provider, span_exporter, monkeypatch
    ):
        monkeypatch.delenv(_CONTENT_ENV, raising=False)
        bridge = _instrument_bridge(tracer_provider)
        try:
            build_agent("rich_agent", RichAgent).run(input="hello")
        finally:
            bridge.uninstrument()

        _assert_one_span_per_layer(span_exporter)
        for layer in _ALL_LAYERS:
            span = _single_layer_span(span_exporter, layer)
            _assert_no_content_on_the_layer(layer, span)
            for key in _GEN_AI_CONTENT_KEYS:
                assert key not in _attributes(span), key
        # Everything that is not user content survives.
        agent_attributes = _attributes(
            _single_layer_span(span_exporter, AGENT_LAYER)
        )
        assert agent_attributes["au.agent.name"] == "rich_agent"
        assert agent_attributes["au.agent.usage.total_tokens"] == 8
        assert agent_attributes["gen_ai.usage.total_tokens"] == 8
        tool_attributes = _attributes(
            _single_layer_span(span_exporter, TOOL_LAYER)
        )
        assert tool_attributes["au.tool.name"] == "stub_tool"
        assert tool_attributes["au.tool.pair_id"]

    def test_owned_bridge_with_capture_on_writes_both_forms(
        self, tracer_provider, span_exporter, monkeypatch
    ):
        monkeypatch.setenv(_CONTENT_ENV, "SPAN_ONLY")
        bridge = _instrument_bridge(tracer_provider)
        try:
            build_agent("rich_agent", RichAgent).run(input="hello")
        finally:
            bridge.uninstrument()

        agent_attributes = _attributes(
            _single_layer_span(span_exporter, AGENT_LAYER)
        )
        assert json.loads(agent_attributes["au.agent.input"]) == {
            "kwargs": {"input": "hello"}
        }
        assert (
            json.loads(agent_attributes["gen_ai.input.messages"])
            == _USER_MESSAGE
        )
        llm_attributes = _attributes(
            _single_layer_span(span_exporter, LLM_LAYER)
        )
        assert json.loads(llm_attributes["au.llm.input"])["prompt"] == "hello"
        assert (
            json.loads(llm_attributes["gen_ai.input.messages"])
            == _USER_MESSAGE
        )
        assert llm_attributes["au.llm.output"] == "llm:hello"
        tool_attributes = _attributes(
            _single_layer_span(span_exporter, TOOL_LAYER)
        )
        assert json.loads(tool_attributes["au.tool.input"]) == {
            "kwargs": {"query": "hello"}
        }
        assert (
            tool_attributes["gen_ai.tool.call.arguments"]
            == tool_attributes["au.tool.input"]
        )
        assert (
            tool_attributes["gen_ai.tool.call.result"]
            == tool_attributes["au.tool.output"]
        )

    def test_event_only_capture_writes_no_span_content(
        self, tracer_provider, span_exporter, monkeypatch
    ):
        monkeypatch.setenv(_CONTENT_ENV, "EVENT_ONLY")
        bridge = _instrument_bridge(tracer_provider)
        try:
            build_agent("rich_agent", RichAgent).run(input="hello")
        finally:
            bridge.uninstrument()

        for layer in _ALL_LAYERS:
            span = _single_layer_span(span_exporter, layer)
            _assert_no_content_on_the_layer(layer, span)
            for key in _GEN_AI_CONTENT_KEYS:
                assert key not in _attributes(span), key

    def test_pre_activated_native_keeps_its_content_contract(
        self, tracer_provider, span_exporter, monkeypatch
    ):
        # An application that enabled the native instrumentors itself decides
        # what they may export, so the bridge must not filter them.
        monkeypatch.delenv(_CONTENT_ENV, raising=False)
        natives = _instrument_all_natives(tracer_provider)
        bridge = _instrument_bridge(tracer_provider)
        try:
            assert not any(layer.owned for layer in bridge._layers)
            build_agent("rich_agent", RichAgent).run(input="hello")
        finally:
            bridge.uninstrument()
            for native in natives:
                native.uninstrument()

        for layer in _ALL_LAYERS:
            span = _single_layer_span(span_exporter, layer)
            for key in _CONTENT_KEYS[layer]:
                assert key in _attributes(span), key
            # ... while the GenAI content still follows the capture switch.
            for key in _GEN_AI_CONTENT_KEYS:
                assert key not in _attributes(span), key

    def test_extra_parameters_do_not_leak_into_the_messages(
        self, tracer_provider, span_exporter, monkeypatch
    ):
        monkeypatch.setenv(_CONTENT_ENV, "SPAN_ONLY")
        bridge = _instrument_bridge(tracer_provider)
        try:
            build_agent().run(input="hello", callbacks=["cb"])
        finally:
            bridge.uninstrument()

        attributes = _attributes(_single_span(span_exporter))
        assert json.loads(attributes["gen_ai.input.messages"]) == _USER_MESSAGE
        assert "cb" not in attributes["gen_ai.input.messages"]


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_double_instrument_still_yields_one_span_per_layer(
        self, tracer_provider, span_exporter
    ):
        bridge = _instrument_bridge(tracer_provider)
        try:
            # The second call is a no-op, not a second patch.
            _instrument_bridge(tracer_provider)
            build_agent("rich_agent", RichAgent).run(input="hello")
        finally:
            bridge.uninstrument()
            bridge.uninstrument()

        _assert_one_span_per_layer(span_exporter)

    def test_uninstrument_restores_every_native_setter(self, tracer_provider):
        originals = {
            layer: {
                name: getattr(_SETTER_CLASSES[layer], name)
                for name in _BRIDGED_METHODS[layer]
            }
            for layer in _ALL_LAYERS
        }

        bridge = _instrument_bridge(tracer_provider)
        for layer in _ALL_LAYERS:
            for name, original in originals[layer].items():
                assert getattr(_SETTER_CLASSES[layer], name) is not original

        bridge.uninstrument()

        for layer in _ALL_LAYERS:
            for name, original in originals[layer].items():
                assert getattr(_SETTER_CLASSES[layer], name) is original

    def test_uninstrument_restores_the_wrapper_globals(self, tracer_provider):
        saved = {
            name: getattr(trace_module, name)
            for name in NATIVE_WRAPPER_GLOBALS
        }

        bridge = _instrument_bridge(tracer_provider)
        for native_cls in NATIVE_LAYER_GLOBALS:
            assert active_native_instrumentor(native_cls) is not None

        bridge.uninstrument()

        for name, original in saved.items():
            assert getattr(trace_module, name) is original, name
        for native_cls in NATIVE_LAYER_GLOBALS:
            assert active_native_instrumentor(native_cls) is None

    def test_uninstrument_only_tears_down_the_layers_it_owns(
        self, tracer_provider, span_exporter
    ):
        native_agent = _instrument_native(AGENT_LAYER, tracer_provider)
        bridge = _instrument_bridge(tracer_provider)
        try:
            bridge.uninstrument()
            assert (
                active_native_instrumentor(AgentInstrumentor) is native_agent
            )
            assert active_native_instrumentor(LLMInstrumentor) is None
            assert active_native_instrumentor(ToolInstrumentor) is None
            build_agent().run(input="hello")
        finally:
            if native_agent.__dict__.get("_is_instrumented_by_opentelemetry"):
                native_agent.uninstrument()

        # The application's own agent instrumentor is still producing spans.
        span = _single_span(span_exporter)
        assert span.name == "au.agent.test_agent"
        assert not _has_gen_ai_attributes(span)

    def test_reinstrument_after_uninstrument_bridges_again(
        self, tracer_provider, span_exporter
    ):
        first = _instrument_bridge(tracer_provider)
        first.uninstrument()

        second = _instrument_bridge(tracer_provider)
        try:
            build_agent("rich_agent", RichAgent).run(input="hello")
        finally:
            second.uninstrument()

        _assert_one_span_per_layer(span_exporter)
        for layer in _ALL_LAYERS:
            assert _has_gen_ai_attributes(
                _single_layer_span(span_exporter, layer)
            )

    def test_uninstrument_stops_span_production(
        self, tracer_provider, span_exporter, make_agent
    ):
        bridge = _instrument_bridge(tracer_provider)
        try:
            make_agent().run(input="hello")
            assert _spans(span_exporter)
        finally:
            bridge.uninstrument()

        span_exporter.clear()

        make_agent().run(input="hello")
        assert _spans(span_exporter) == []


# ---------------------------------------------------------------------------
# Session scope: the bridge activates the agent/LLM/tool layers and nothing
# else. ``au.trace.session.id`` belongs to agentUniverse's TelemetryManager.
# ---------------------------------------------------------------------------


class TestSessionScope:
    def test_bridge_touches_no_global_provider_or_propagator(
        self, tracer_provider
    ):
        provider_before = trace.get_tracer_provider()
        propagator_before = propagate.get_global_textmap()

        bridge = _instrument_bridge(tracer_provider)
        try:
            assert trace.get_tracer_provider() is provider_before
            assert propagate.get_global_textmap() is propagator_before
        finally:
            bridge.uninstrument()

        assert trace.get_tracer_provider() is provider_before
        assert propagate.get_global_textmap() is propagator_before

    def test_a_bridge_only_run_carries_no_session_attribute(
        self, tracer_provider, span_exporter, make_agent
    ):
        bridge = _instrument_bridge(tracer_provider)
        try:
            make_agent().run(input="hello")
        finally:
            bridge.uninstrument()

        assert SESSION_ATTR not in _attributes(_single_span(span_exporter))

    def test_telemetry_manager_session_path_in_an_isolated_process(self):
        """The documented session path, run for real.

        ``TelemetryManager.init_from_config`` installs the global provider, the
        propagator and the ``SessionSpanProcessor``, and it can only run once
        per process -- so it runs in a child process and reports what it saw.
        """
        payload = _run_session_child()

        assert payload["initialized"] is True
        assert payload["propagator"] == "CompositePropagator"
        assert payload["propagator_fields"] == ["AU-SessionId", "auSessionId"]
        assert payload["span_names"] == [
            "au.agent.session_agent",
            "au.llm.stub_llm",
            "au.tool.stub_tool",
        ]
        # One span per layer, and the session reaches every one of them.
        assert payload["session_attributes"] == {
            "au.agent.session_agent": _SESSION_ID,
            "au.llm.stub_llm": _SESSION_ID,
            "au.tool.stub_tool": _SESSION_ID,
        }
        # The LoongSuite bridge is active on that path too.
        assert payload["gen_ai_span_kinds"] == {
            "au.agent.session_agent": "AGENT",
            "au.llm.stub_llm": "LLM",
            "au.tool.stub_tool": "TOOL",
        }
        assert payload["agent_total_tokens"] == 8
        # inject/extract round trip, on the propagator's real contract.
        assert payload["carrier"] == {
            "AU-SessionId": _SESSION_ID,
            "auSessionId": _SESSION_ID,
        }
        assert payload["extracted_session_id"] == _CARRIER_SESSION
        # ``AUSessionPropagator.extract`` stores the header value it picked as
        # the baggage value for that header key -- a one-element list.
        assert payload["extracted_baggage_in_returned_context"] == [
            _CARRIER_SESSION
        ]


def test_framework_is_the_real_distribution():
    """Guard the whole suite: this must be the real agentUniverse."""

    from importlib.metadata import version

    import agentuniverse

    assert build_agent("probe").agent_model.info["name"] == "probe"
    assert version("agentUniverse").startswith("0.0.19")
    assert agentuniverse.__file__ is not None
