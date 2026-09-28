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

Everything here runs against a real ``agentUniverse`` install. Three
instrumentor paths are covered, and each one must produce exactly one span:

* native ``AgentInstrumentor`` only -- the ``au.*`` span, no ``gen_ai.*``;
* LoongSuite only -- that same ``au.*`` span plus the ``gen_ai.*`` attributes;
* both enabled -- still one span, carrying both namespaces.

Asserting the single-span property on every path is what catches a regression
back to wrapping ``Agent.run``/``async_run``: a duplicated span shows up here
rather than as a silent extra span in production.

Attribute names are spelled out as literals on purpose. They are the wire
contract, so the tests should fail if the instrumentation renames one.
"""

from __future__ import annotations

import json
import queue
from typing import Any

import pytest
from agentuniverse.base.annotation import trace as trace_module
from agentuniverse.base.tracing.otel.instrumentation.agent.agent_instrumentor import (  # noqa: E501
    AgentInstrumentor,
    AgentSpanAttributesSetter,
)

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

from .conftest import FailingAgent, StreamingAgent, build_agent

AU_AGENT_SPAN_PREFIX = "au.agent."
GEN_AI_PREFIX = "gen_ai."
_CONTENT_ENV = "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"
_USER_MESSAGE = [
    {"role": "user", "parts": [{"content": "hello", "type": "text"}]}
]

# The metric names the native instrumentor is expected to keep emitting. These
# are a wire contract too: a rename here means dashboards go blank.
_METRIC_NAMES = (
    "agent_calls_total",
    "agent_call_duration",
    "agent_total_tokens",
    "agent_prompt_tokens",
    "agent_completion_tokens",
    "agent_cached_tokens",
    "agent_reasoning_tokens",
    "agent_first_token_duration",
)


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


def _assert_au_span(
    span: ReadableSpan,
    agent_name: str,
    status: str = "success",
    content: bool = True,
) -> dict[str, Any]:
    """Assert the native ``au.*`` span shape the bridge must preserve.

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
    # Token usage is native, and must survive the bridge on every path.
    assert "au.agent.usage.total_tokens" in attributes
    assert "au.agent.usage.prompt_tokens" in attributes
    assert "au.agent.usage.completion_tokens" in attributes
    assert "au.agent.usage.detail_tokens" in attributes
    if content:
        assert "au.agent.input" in attributes
    else:
        assert "au.agent.input" not in attributes
    if status == "success":
        assert ("au.agent.output" in attributes) is content
    return attributes


def _assert_gen_ai_span(span: ReadableSpan, agent_name: str) -> dict[str, Any]:
    """Assert the LoongSuite ``gen_ai.*`` attributes landed on the span."""
    attributes = _attributes(span)
    assert attributes["gen_ai.span.kind"] == "AGENT"
    assert attributes["gen_ai.operation.name"] == "invoke_agent"
    assert attributes["gen_ai.framework"] == "agentuniverse"
    assert attributes["gen_ai.agent.name"] == agent_name
    return attributes


def _has_gen_ai_attributes(span: ReadableSpan) -> bool:
    return any(key.startswith(GEN_AI_PREFIX) for key in _attributes(span))


def _native_is_active() -> bool:
    wrapper = getattr(trace_module, "_agent_wrapper_sync", None)
    return isinstance(getattr(wrapper, "__self__", None), AgentInstrumentor)


def _instrument_native(
    tracer_provider, meter_provider=None
) -> AgentInstrumentor:
    instrumentor = AgentInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
        skip_dep_check=True,
    )
    return instrumentor


def _instrument_bridge(
    tracer_provider, meter_provider=None
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


def _only_point(points: list[Any]) -> Any:
    assert len(points) == 1, f"expected one data point, got {len(points)}"
    return points[0]


# ---------------------------------------------------------------------------
# Path 1: the native instrumentor on its own
# ---------------------------------------------------------------------------


class TestNativeOnly:
    @pytest.fixture
    def native(self, tracer_provider):
        instrumentor = _instrument_native(tracer_provider)
        yield instrumentor
        instrumentor.uninstrument()

    def test_single_au_span_without_any_gen_ai_attributes(
        self, native, span_exporter, make_agent
    ):
        result = make_agent().run(input="hello")

        assert result.get_data("output") == "echo:hello"
        span = _single_span(span_exporter)
        attributes = _assert_au_span(span, "test_agent")
        assert attributes["au.agent.output"] == '{"output": "echo:hello"}'
        assert not _has_gen_ai_attributes(span)


# ---------------------------------------------------------------------------
# Path 2: the LoongSuite bridge on its own
# ---------------------------------------------------------------------------


class TestLoongSuiteOnly:
    def test_single_span_carries_both_namespaces(
        self, tracer_provider, span_exporter, make_agent
    ):
        instrumentor = _instrument_bridge(tracer_provider)
        try:
            result = make_agent().run(input="hello")
        finally:
            instrumentor.uninstrument()

        assert result.get_data("output") == "echo:hello"
        span = _single_span(span_exporter)
        # No capture switch is set here, so this is the default posture: the
        # bridge owns the native instrumentor and filters its content carrier.
        _assert_au_span(span, "test_agent", content=False)
        _assert_gen_ai_span(span, "test_agent")

    def test_bridge_creates_the_native_instrumentor_when_none_is_active(
        self, tracer_provider, span_exporter, make_agent
    ):
        assert not _native_is_active()

        instrumentor = _instrument_bridge(tracer_provider)
        assert instrumentor._native_owned is True
        assert _native_is_active()

        make_agent().run(input="hello")
        _assert_gen_ai_span(_single_span(span_exporter), "test_agent")

        instrumentor.uninstrument()
        assert instrumentor._native_owned is False
        assert not _native_is_active()

    def test_bridge_adds_no_span_of_its_own(
        self, tracer_provider, span_exporter, make_agent
    ):
        """Span creation is the native instrumentor's job, and only its job."""
        instrumentor = _instrument_bridge(tracer_provider)
        try:
            # Take the native instrumentor back out from under the bridge --
            # through the live instance, never a fresh AgentInstrumentor(),
            # which would reset the saved wrapper originals. If the bridge
            # were still creating spans, one would show up here.
            instrumentor._native.uninstrument()

            make_agent().run(input="hello")

            assert _spans(span_exporter) == []
        finally:
            instrumentor.uninstrument()


# ---------------------------------------------------------------------------
# Path 3: native and LoongSuite enabled together
# ---------------------------------------------------------------------------


class TestBothEnabled:
    def test_one_span_with_both_namespaces(
        self, tracer_provider, span_exporter, make_agent
    ):
        native = _instrument_native(tracer_provider)
        bridge = _instrument_bridge(tracer_provider)
        try:
            make_agent().run(input="hello")
        finally:
            bridge.uninstrument()
            native.uninstrument()

        span = _single_span(span_exporter)
        _assert_au_span(span, "test_agent")
        _assert_gen_ai_span(span, "test_agent")

    def test_bridge_reuses_an_active_native_instrumentor(
        self, tracer_provider, span_exporter, make_agent
    ):
        native = _instrument_native(tracer_provider)
        bridge = _instrument_bridge(tracer_provider)
        try:
            assert bridge._native_owned is False
            assert bridge._native is native

            make_agent().run(input="hello")
            span = _single_span(span_exporter)
            _assert_au_span(span, "test_agent")
            _assert_gen_ai_span(span, "test_agent")
        finally:
            bridge.uninstrument()

        # The bridge must not tear down an instrumentor it did not create, and
        # must leave the native setters exactly as it found them.
        assert _native_is_active()

        span_exporter.clear()

        make_agent().run(input="hello")
        span = _single_span(span_exporter)
        _assert_au_span(span, "test_agent")
        assert not _has_gen_ai_attributes(span)

        native.uninstrument()


# ---------------------------------------------------------------------------
# Content privacy: who may put the prompt and the result on the span
# ---------------------------------------------------------------------------


class TestContentPrivacy:
    """``au.agent.input`` / ``au.agent.output`` are the native content carrier.

    Suppressing ``gen_ai.input.messages`` alone is not a privacy guarantee: the
    native setter writes the raw prompt and the agent result unconditionally, so
    a bridge-owned native instrumentor must be filtered as well. A native
    instrumentor the application activated itself is never filtered -- its
    behaviour stays governed by the application's own configuration.
    """

    _SECRET = "secret-prompt-8f21c"

    def _run_bridge_owned(
        self, tracer_provider, make_agent, monkeypatch, mode, text
    ):
        if mode is None:
            monkeypatch.delenv(_CONTENT_ENV, raising=False)
        else:
            monkeypatch.setenv(_CONTENT_ENV, mode)
        instrumentor = _instrument_bridge(tracer_provider)
        try:
            make_agent("privacy_agent").run(input=text)
            # Read the flags before uninstrument() resets them.
            return instrumentor._native_owned, instrumentor._privacy_filter
        finally:
            instrumentor.uninstrument()

    def test_owned_bridge_with_capture_off_writes_no_user_content(
        self, tracer_provider, span_exporter, make_agent, monkeypatch
    ):
        owned, filtered = self._run_bridge_owned(
            tracer_provider, make_agent, monkeypatch, None, self._SECRET
        )
        assert owned is True, (
            "the bridge must own the native instrumentor here"
        )
        assert filtered is True

        attributes = _assert_au_span(
            _single_span(span_exporter), "privacy_agent", content=False
        )
        _assert_gen_ai_span(_single_span(span_exporter), "privacy_agent")
        assert "gen_ai.input.messages" not in attributes
        # The strongest form of the claim: the prompt is nowhere on the span,
        # not merely missing from the two attribute names we know about.
        assert self._SECRET not in json.dumps(attributes)

    def test_owned_bridge_with_capture_on_writes_native_and_gen_ai_content(
        self, tracer_provider, span_exporter, make_agent, monkeypatch
    ):
        owned, filtered = self._run_bridge_owned(
            tracer_provider, make_agent, monkeypatch, "SPAN_ONLY", self._SECRET
        )
        assert owned is True
        assert filtered is False, (
            "capture-on must leave the native contract alone"
        )

        span = _single_span(span_exporter)
        attributes = _assert_au_span(span, "privacy_agent")
        _assert_gen_ai_span(span, "privacy_agent")
        # The filter must not be over-broad when the user opted in.
        assert self._SECRET in attributes["au.agent.input"]
        assert json.loads(attributes["gen_ai.input.messages"]) == [
            {
                "role": "user",
                "parts": [{"content": self._SECRET, "type": "text"}],
            }
        ]

    def test_pre_activated_native_keeps_its_own_content_contract(
        self, tracer_provider, span_exporter, make_agent, monkeypatch
    ):
        """The bridge never overrides a native instrumentor it did not create."""
        monkeypatch.delenv(_CONTENT_ENV, raising=False)

        native = _instrument_native(tracer_provider)
        bridge = _instrument_bridge(tracer_provider)
        try:
            assert bridge._native_owned is False
            assert bridge._privacy_filter is False

            make_agent("privacy_agent").run(input=self._SECRET)
        finally:
            bridge.uninstrument()
            native.uninstrument()

        span = _single_span(span_exporter)
        attributes = _assert_au_span(span, "privacy_agent")
        _assert_gen_ai_span(span, "privacy_agent")
        # Native wrote its content, as its own configuration dictates.
        assert self._SECRET in attributes["au.agent.input"]
        assert "gen_ai.input.messages" not in attributes


# ---------------------------------------------------------------------------
# The native metrics, streaming timing and token usage must survive intact
# ---------------------------------------------------------------------------


class TestMetrics:
    """Metrics are half the contract: the bridge must not disturb them.

    The native metric instruments are bound to whatever ``meter_provider`` the
    native instrumentor was given at ``instrument()`` time, so reading the
    metrics from that same provider also proves the bridge did not re-instrument
    the native instrumentor behind our back -- a second pass would rebind the
    instruments to the global provider and this reader would see nothing.
    """

    def _assert_native_metrics(
        self, reader: InMemoryMetricReader, agent_name: str
    ) -> dict[str, list[Any]]:
        metrics = _collect_metrics(reader)
        for name in _METRIC_NAMES:
            assert name in metrics, f"metric {name!r} was not recorded"
            _only_point(metrics[name])

        calls = _only_point(metrics["agent_calls_total"])
        assert calls.value == 1, "exactly one agent call must be recorded"
        assert calls.attributes["au_agent_name"] == agent_name
        assert _only_point(metrics["agent_call_duration"]).count == 1
        return metrics

    def test_both_enabled_records_each_metric_once(
        self,
        tracer_provider,
        meter_provider,
        metric_reader,
        span_exporter,
        make_agent,
    ):
        native = _instrument_native(tracer_provider, meter_provider)
        bridge = _instrument_bridge(tracer_provider, meter_provider)
        try:
            make_agent().run(input="hello")
        finally:
            bridge.uninstrument()
            native.uninstrument()

        _single_span(span_exporter)
        self._assert_native_metrics(metric_reader, "test_agent")

    def test_bridge_owned_records_each_metric_once(
        self,
        tracer_provider,
        meter_provider,
        metric_reader,
        span_exporter,
        make_agent,
    ):
        instrumentor = _instrument_bridge(tracer_provider, meter_provider)
        try:
            make_agent().run(input="hello")
        finally:
            instrumentor.uninstrument()

        _single_span(span_exporter)
        self._assert_native_metrics(metric_reader, "test_agent")

    def test_streaming_records_the_first_token(
        self,
        tracer_provider,
        meter_provider,
        metric_reader,
        span_exporter,
        make_agent,
    ):
        agent = make_agent("stream_agent", StreamingAgent)
        instrumentor = _instrument_bridge(tracer_provider, meter_provider)
        try:
            result = agent.run(input="hello", output_stream=queue.Queue())
        finally:
            instrumentor.uninstrument()

        assert result.get_data("output") == "streamed"
        span = _single_span(span_exporter)
        attributes = _assert_au_span(span, "stream_agent", content=False)
        assert attributes["au.agent.streaming"] is True
        assert "au.agent.first_token.duration" in attributes
        _assert_gen_ai_span(span, "stream_agent")

        # On the streaming path the native wrapper records the first token from
        # the queue callback, so this data point can only come from there.
        first_token = _only_point(
            _collect_metrics(metric_reader)["agent_first_token_duration"]
        )
        assert first_token.count == 1
        assert first_token.attributes["au_agent_streaming"] is True

    def test_token_usage_attributes_stay_on_the_span(
        self, tracer_provider, span_exporter, make_agent
    ):
        instrumentor = _instrument_bridge(tracer_provider)
        try:
            make_agent().run(input="hello")
        finally:
            instrumentor.uninstrument()

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


# ---------------------------------------------------------------------------
# The bridge must never wrap Agent.run / Agent.async_run itself
# ---------------------------------------------------------------------------


def test_bridge_does_not_wrap_agent_run(tracer_provider):
    from agentuniverse.agent.agent import Agent

    run_before = Agent.run
    async_run_before = Agent.async_run

    instrumentor = _instrument_bridge(tracer_provider)
    try:
        assert Agent.run is run_before
        assert Agent.async_run is async_run_before
    finally:
        instrumentor.uninstrument()

    assert Agent.run is run_before
    assert Agent.async_run is async_run_before


def test_session_propagation_is_left_to_the_application(tracer_provider):
    """The bridge activates the Agent instrumentor and nothing else.

    Session id propagation (``au.trace.session.id`` via ``AUSessionPropagator``)
    and the ``SessionSpanProcessor`` are installed by agentUniverse's
    ``TelemetryManager.init_from_config()``. The bridge neither registers them
    nor touches the global propagator, so session propagation requires
    TelemetryManager and is deliberately not covered by these unit tests.
    """
    from opentelemetry import propagate

    before = propagate.get_global_textmap()

    instrumentor = _instrument_bridge(tracer_provider)
    try:
        assert propagate.get_global_textmap() is before
    finally:
        instrumentor.uninstrument()

    assert propagate.get_global_textmap() is before


# ---------------------------------------------------------------------------
# Content capture
# ---------------------------------------------------------------------------


class TestContentCapture:
    def _run_once(self, tracer_provider, make_agent, monkeypatch, mode):
        if mode is None:
            monkeypatch.delenv(_CONTENT_ENV, raising=False)
        else:
            monkeypatch.setenv(_CONTENT_ENV, mode)
        instrumentor = _instrument_bridge(tracer_provider)
        try:
            return make_agent().run(input="hello")
        finally:
            instrumentor.uninstrument()

    def test_off_when_the_env_var_is_unset(
        self, tracer_provider, span_exporter, make_agent, monkeypatch
    ):
        self._run_once(tracer_provider, make_agent, monkeypatch, None)

        attributes = _assert_gen_ai_span(
            _single_span(span_exporter), "test_agent"
        )
        assert "gen_ai.input.messages" not in attributes

    @pytest.mark.parametrize("mode", ["SPAN_ONLY", "SPAN_AND_EVENT"])
    def test_on_writes_the_user_message(
        self, tracer_provider, span_exporter, make_agent, monkeypatch, mode
    ):
        self._run_once(tracer_provider, make_agent, monkeypatch, mode)

        span = _single_span(span_exporter)
        _assert_au_span(span, "test_agent")
        attributes = _assert_gen_ai_span(span, "test_agent")
        assert json.loads(attributes["gen_ai.input.messages"]) == _USER_MESSAGE

    def test_event_only_mode_does_not_write_the_span_attribute(
        self, tracer_provider, span_exporter, make_agent, monkeypatch
    ):
        self._run_once(tracer_provider, make_agent, monkeypatch, "EVENT_ONLY")

        attributes = _assert_gen_ai_span(
            _single_span(span_exporter), "test_agent"
        )
        assert "gen_ai.input.messages" not in attributes

    def test_only_the_input_is_captured_when_extra_parameters_are_passed(
        self, tracer_provider, span_exporter, make_agent, monkeypatch
    ):
        monkeypatch.setenv(_CONTENT_ENV, "SPAN_ONLY")
        instrumentor = _instrument_bridge(tracer_provider)
        try:
            make_agent().run(input="hello", callbacks=["cb"])
        finally:
            instrumentor.uninstrument()

        attributes = _attributes(_single_span(span_exporter))
        assert json.loads(attributes["gen_ai.input.messages"]) == _USER_MESSAGE


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_exception_keeps_the_native_error_span_and_the_gen_ai_attributes(
    tracer_provider, span_exporter, make_agent
):
    agent = make_agent("boom_agent", FailingAgent)
    instrumentor = _instrument_bridge(tracer_provider)
    try:
        with pytest.raises(RuntimeError, match="agent exploded"):
            agent.run(input="hello")
    finally:
        instrumentor.uninstrument()

    span = _single_span(span_exporter)
    assert span.status.status_code == StatusCode.ERROR
    attributes = _assert_au_span(
        span, "boom_agent", status="error", content=False
    )
    assert attributes["au.agent.error.type"] == "RuntimeError"
    assert "agent exploded" in attributes["au.agent.error.message"]
    _assert_gen_ai_span(span, "boom_agent")


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_double_instrument_still_yields_one_span(
        self, tracer_provider, span_exporter, make_agent
    ):
        instrumentor = _instrument_bridge(tracer_provider)
        try:
            # The second call is a no-op, not a second patch.
            _instrument_bridge(tracer_provider)

            make_agent().run(input="hello")
        finally:
            instrumentor.uninstrument()
            instrumentor.uninstrument()

        span = _single_span(span_exporter)
        _assert_au_span(span, "test_agent", content=False)
        _assert_gen_ai_span(span, "test_agent")

    def test_uninstrument_restores_the_native_setters(self, tracer_provider):
        originals = {
            name: getattr(AgentSpanAttributesSetter, name)
            for name in (
                "set_input_attributes",
                "set_success_attributes",
                "set_error_attributes",
            )
        }

        instrumentor = _instrument_bridge(tracer_provider)
        for name, original in originals.items():
            assert getattr(AgentSpanAttributesSetter, name) is not original

        instrumentor.uninstrument()

        for name, original in originals.items():
            assert getattr(AgentSpanAttributesSetter, name) is original

    def test_reinstrument_after_uninstrument_bridges_again(
        self, tracer_provider, span_exporter, make_agent
    ):
        first = _instrument_bridge(tracer_provider)
        first.uninstrument()

        second = _instrument_bridge(tracer_provider)
        try:
            make_agent().run(input="hello")
        finally:
            second.uninstrument()

        span = _single_span(span_exporter)
        _assert_au_span(span, "test_agent", content=False)
        _assert_gen_ai_span(span, "test_agent")

    def test_uninstrument_stops_span_production(
        self, tracer_provider, span_exporter, make_agent
    ):
        instrumentor = _instrument_bridge(tracer_provider)
        try:
            make_agent().run(input="hello")
            assert _spans(span_exporter)
        finally:
            instrumentor.uninstrument()

        span_exporter.clear()

        make_agent().run(input="hello")
        assert _spans(span_exporter) == []


# ---------------------------------------------------------------------------
# async_run goes through the native async wrapper and must stay single-span
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_run_produces_one_span_with_both_namespaces(
    tracer_provider, span_exporter, make_agent
):
    agent = make_agent("async_agent")
    instrumentor = _instrument_bridge(tracer_provider)
    try:
        result = await agent.async_run(input="hello async")
    finally:
        instrumentor.uninstrument()

    assert result.get_data("output") == "echo:hello async"
    span = _single_span(span_exporter)
    _assert_au_span(span, "async_agent", content=False)
    _assert_gen_ai_span(span, "async_agent")


def test_framework_is_the_real_distribution():
    """Guard the whole suite: this must be the real agentUniverse."""

    from importlib.metadata import version

    import agentuniverse

    assert build_agent("probe").agent_model.info["name"] == "probe"
    assert version("agentUniverse").startswith("0.0.19")
    assert agentuniverse.__file__ is not None
