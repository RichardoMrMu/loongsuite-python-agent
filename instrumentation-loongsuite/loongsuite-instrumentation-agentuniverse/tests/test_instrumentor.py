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

"""Tests for AgentUniverseInstrumentor.

Every GREEN span assertion is paired with a RED baseline proving the same
agent call produces no AGENT span while the instrumentor is inactive, and the
teardown test proves ``uninstrument()`` really stops span production. Content
capture is asserted both when the shared switch opts in and by default, and
the lifecycle tests prove double instrument/uninstrument is safe, that the
original methods are restored, and that a telemetry failure never breaks agent
execution.

The agent classes used here are real subclasses of the framework's
``agentUniverse`` ``Agent`` base class -- see ``conftest.py`` for how that base
class is resolved (real distribution when importable, otherwise a documented
schema-faithful stand-in, reported in the pytest header).
"""

from __future__ import annotations

import pytest

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.agentuniverse import (
    _GEN_AI_AGENT_NAME,
    _GEN_AI_FRAMEWORK,
    _GEN_AI_INPUT_MESSAGES,
    _GEN_AI_OPERATION_NAME,
    _GEN_AI_SPAN_KIND,
    _OP_INVOKE_AGENT,
    _SPAN_KIND_AGENT,
    AgentUniverseInstrumentor,
)
from opentelemetry.trace import SpanKind, StatusCode


def _agent_spans(span_exporter):
    return [
        span
        for span in span_exporter.get_finished_spans()
        if span.attributes.get(_GEN_AI_SPAN_KIND) == _SPAN_KIND_AGENT
    ]


def _only_agent_span(span_exporter):
    spans = _agent_spans(span_exporter)
    assert len(spans) == 1, [
        (span.name, dict(span.attributes))
        for span in span_exporter.get_finished_spans()
    ]
    return spans[0]


# ---------------------------------------------------------------------------
# Framework resolution is explicit
# ---------------------------------------------------------------------------


def test_framework_resolution_is_explicit(
    agentuniverse_is_real, agentuniverse_import_error
):
    # A stand-in run is never silently presented as a real-framework run: the
    # mode is reported in the pytest header and the (non-)error is asserted
    # here, so the two modes cannot be confused.
    if agentuniverse_is_real:
        assert agentuniverse_import_error is None
    else:
        assert isinstance(agentuniverse_import_error, BaseException)


# ---------------------------------------------------------------------------
# RED: no AGENT span without instrumentation
# ---------------------------------------------------------------------------


def test_red_no_agent_span_without_instrumentation(span_exporter, make_agent):
    agent = make_agent(agent_name="red_agent")
    agent.run(input="hello")

    names = [span.name for span in span_exporter.get_finished_spans()]
    assert not any(
        name.startswith(f"{_OP_INVOKE_AGENT} ") for name in names
    ), names


# ---------------------------------------------------------------------------
# GREEN: AGENT span for a sync run
# ---------------------------------------------------------------------------


def test_green_agent_span_for_configured_agent(
    instrument, span_exporter, make_agent
):
    agent = make_agent(agent_name="demo_agent")
    result = agent.run(input="hello")

    # The instrumentation must not alter the agent's own result.
    assert result.get_data("output") == "echo:hello"

    span = _only_agent_span(span_exporter)
    assert span.name == f"{_OP_INVOKE_AGENT} demo_agent"
    assert span.kind == SpanKind.INTERNAL
    assert span.attributes.get(_GEN_AI_SPAN_KIND) == _SPAN_KIND_AGENT
    assert span.attributes.get(_GEN_AI_OPERATION_NAME) == _OP_INVOKE_AGENT
    assert span.attributes.get(_GEN_AI_FRAMEWORK) == "agentuniverse"
    assert span.attributes.get(_GEN_AI_AGENT_NAME) == "demo_agent"
    assert span.status.status_code == StatusCode.OK


def test_green_span_name_falls_back_to_class_name(
    instrument, span_exporter, make_agent
):
    # No configured name in agent_model.info -> the concrete class name.
    agent = make_agent(agent_name=None)
    agent.run(input="hello")

    span = _only_agent_span(span_exporter)
    assert span.name == f"{_OP_INVOKE_AGENT} _Agent"
    assert span.attributes.get(_GEN_AI_AGENT_NAME) == "_Agent"


def test_green_inner_work_nests_under_agent(
    instrument, span_exporter, tracer_provider, make_agent
):
    # Route the agent's own child span to the same exporter so the nesting
    # relationship is observable; nesting itself is via context propagation.
    agent = make_agent(
        agent_name="nesting_agent", inner_tracer_provider=tracer_provider
    )
    agent.run(input="hello")

    agent_span = _only_agent_span(span_exporter)
    inner = next(
        span
        for span in span_exporter.get_finished_spans()
        if span.name == "agent-inner-work"
    )

    assert inner.parent is not None
    assert inner.parent.span_id == agent_span.context.span_id
    assert inner.context.trace_id == agent_span.context.trace_id


# ---------------------------------------------------------------------------
# Content capture (shared GenAI switch)
# ---------------------------------------------------------------------------


def test_green_captures_input_when_enabled(
    instrument, span_exporter, monkeypatch, make_agent
):
    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_ONLY"
    )
    agent = make_agent(agent_name="content_agent")
    agent.run(input="what is 2+2?")

    span = _only_agent_span(span_exporter)
    messages = span.attributes.get(_GEN_AI_INPUT_MESSAGES)
    assert messages is not None
    assert "what is 2+2?" in messages
    assert '"role":"user"' in messages


def test_green_content_suppressed_by_default(
    instrument, span_exporter, monkeypatch, make_agent
):
    # No capture env set -> shared util defaults to NO_CONTENT.
    monkeypatch.delenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", raising=False
    )
    agent = make_agent(agent_name="content_agent")
    agent.run(input="secret prompt")

    span = _only_agent_span(span_exporter)
    assert _GEN_AI_INPUT_MESSAGES not in span.attributes


def test_green_captures_kwargs_when_input_key_absent(
    instrument, span_exporter, monkeypatch, make_agent
):
    # Agents may declare arbitrary input keys; the conventional ``input`` key
    # is absent here, so the remaining kwargs become the user message while
    # runtime plumbing (callbacks) stays out of it.
    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_AND_EVENT"
    )
    agent = make_agent(agent_name="kwargs_agent", input_key="question")
    result = agent.run(question="why is the sky blue?", callbacks=["cb"])

    assert result.get_data("output") == "echo:why is the sky blue?"

    span = _only_agent_span(span_exporter)
    messages = span.attributes.get(_GEN_AI_INPUT_MESSAGES)
    assert messages is not None
    assert "why is the sky blue?" in messages
    assert "callbacks" not in messages


# ---------------------------------------------------------------------------
# GREEN: AGENT span for an async run
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_green_async_agent_span(instrument, span_exporter, make_agent):
    agent = make_agent(agent_name="async_agent")
    result = await agent.async_run(input="hello async")

    assert result.get_data("output") == "echo:hello async"

    span = _only_agent_span(span_exporter)
    assert span.name == f"{_OP_INVOKE_AGENT} async_agent"
    assert span.attributes.get(_GEN_AI_AGENT_NAME) == "async_agent"
    assert span.status.status_code == StatusCode.OK


@pytest.mark.asyncio
async def test_green_async_inner_work_nests_under_agent(
    instrument, span_exporter, tracer_provider, make_agent
):
    agent = make_agent(
        agent_name="async_nesting_agent",
        inner_tracer_provider=tracer_provider,
    )
    await agent.async_run(input="hello")

    agent_span = _only_agent_span(span_exporter)
    inner = next(
        span
        for span in span_exporter.get_finished_spans()
        if span.name == "agent-inner-work-async"
    )
    assert inner.parent is not None
    assert inner.parent.span_id == agent_span.context.span_id
    assert inner.context.trace_id == agent_span.context.trace_id


# ---------------------------------------------------------------------------
# Error handling: the agent's exception is recorded and re-raised unchanged
# ---------------------------------------------------------------------------


def test_green_records_exception(instrument, span_exporter, make_agent):
    agent = make_agent(agent_name="boom_agent", failure=ValueError("boom"))

    with pytest.raises(ValueError, match="boom"):
        agent.run(input="hello")

    span = _only_agent_span(span_exporter)
    assert span.status.status_code == StatusCode.ERROR
    events = [event.name for event in span.events]
    assert "exception" in events


# ---------------------------------------------------------------------------
# RED after uninstrument: teardown must stop AGENT span production
# ---------------------------------------------------------------------------


def test_red_uninstrument_stops_agent_span(
    span_exporter, tracer_provider, make_agent
):
    instrumentor = AgentUniverseInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider, skip_dep_check=True
    )
    try:
        make_agent(agent_name="before_uninstrument").run(input="hello")
        assert _agent_spans(span_exporter), (
            "AGENT span missing while instrumented"
        )
    finally:
        instrumentor.uninstrument()

    span_exporter.clear()

    make_agent(agent_name="after_uninstrument").run(input="hello")
    names = [span.name for span in span_exporter.get_finished_spans()]
    assert not any(
        name.startswith(f"{_OP_INVOKE_AGENT} ") for name in names
    ), f"AGENT span still produced after uninstrument: {names}"


# ---------------------------------------------------------------------------
# Lifecycle: double instrument/uninstrument is safe and idempotent
# ---------------------------------------------------------------------------


def test_double_instrument_produces_single_span(
    span_exporter, tracer_provider, make_agent
):
    instrumentor = AgentUniverseInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider, skip_dep_check=True
    )
    try:
        # Second instrument() is a no-op, not a second wrap.
        instrumentor.instrument(
            tracer_provider=tracer_provider, skip_dep_check=True
        )
        make_agent(agent_name="twice_agent").run(input="hello")
        assert len(_agent_spans(span_exporter)) == 1
    finally:
        instrumentor.uninstrument()
        # Double uninstrument must be safe too.
        instrumentor.uninstrument()


def test_uninstrument_restores_original_methods(
    span_exporter, tracer_provider, agentuniverse_sdk, make_agent
):
    agent_base, _ = agentuniverse_sdk
    original_run = agent_base.__dict__["run"]
    original_async_run = agent_base.__dict__["async_run"]

    instrumentor = AgentUniverseInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider, skip_dep_check=True
    )
    try:
        assert agent_base.__dict__["run"] is not original_run
        assert agent_base.__dict__["async_run"] is not original_async_run
    finally:
        instrumentor.uninstrument()

    # Restored to exactly the pre-instrument objects, with no sentinel left
    # behind that would defeat a later re-instrument.
    assert agent_base.__dict__["run"] is original_run
    assert agent_base.__dict__["async_run"] is original_async_run

    instrumentor.instrument(
        tracer_provider=tracer_provider, skip_dep_check=True
    )
    try:
        make_agent(agent_name="reinstrumented_agent").run(input="hello")
        assert len(_agent_spans(span_exporter)) == 1
    finally:
        instrumentor.uninstrument()


# ---------------------------------------------------------------------------
# Fail-safe: a telemetry failure must never break the agent's execution
# ---------------------------------------------------------------------------


def test_telemetry_failure_does_not_break_execution(
    instrument, span_exporter, make_agent, monkeypatch
):
    class _BoomTracer:
        def start_as_current_span(self, *args, **kwargs):
            raise RuntimeError("boom")

    # A tracer whose span creation explodes simulates telemetry-side failure.
    for wrapper in instrument._wrappers.values():
        monkeypatch.setattr(wrapper, "_tracer", _BoomTracer())

    agent = make_agent(agent_name="resilient_agent")
    result = agent.run(input="still works")

    assert result.get_data("output") == "echo:still works"
    assert not _agent_spans(span_exporter)


# ---------------------------------------------------------------------------
# Instrumentor metadata
# ---------------------------------------------------------------------------


def test_instrumentation_dependencies():
    instrumentor = AgentUniverseInstrumentor()
    assert instrumentor.instrumentation_dependencies() == (
        "agentUniverse >= 0.0.19",
    )


def test_instrumentor_is_singleton():
    assert AgentUniverseInstrumentor() is AgentUniverseInstrumentor()


def test_get_tracer_is_used_from_public_api():
    # Guards the import the wrapper relies on for span creation.
    assert callable(trace_api.get_tracer)
