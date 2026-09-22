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

"""Tests for A2AInstrumentor.

Every GREEN span assertion is paired with a RED baseline proving the same
executor call produces no AGENT span when the instrumentor is inactive.
The nesting assertions prove downstream spans share the AGENT span's trace,
and the "late subclass" test proves the ``__init_subclass__`` hook catches
executors defined after ``instrument()``.
"""

from __future__ import annotations

import pytest

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.a2a import (
    _GEN_AI_FRAMEWORK,
    _GEN_AI_OPERATION_NAME,
    _GEN_AI_SPAN_KIND,
    _SPAN_KIND_AGENT,
    A2AInstrumentor,
)


# ---------------------------------------------------------------------------
# Executor stubs (real a2a AgentExecutor subclasses)
# ---------------------------------------------------------------------------


def _make_executor_cls(inner_tracer_provider=None):
    """Build a fresh AgentExecutor subclass each call.

    A fresh class avoids cross-test contamination of the sentinel marker on
    a shared ``execute`` function object.

    ``inner_tracer_provider`` lets a test route the executor's own child span
    to the same exporter as the AGENT span, so the nesting relationship can be
    asserted. In production the executor's downstream spans come from whatever
    global/instrumentation provider is configured; nesting is guaranteed by
    context propagation from the AGENT span's ``start_as_current_span``.
    """
    from a2a.server.agent_execution import AgentExecutor

    class _Exec(AgentExecutor):
        def __init__(self):
            self.inner_tracer = trace_api.get_tracer(
                "test.inner", tracer_provider=inner_tracer_provider
            )
            self.captured_input = None

        async def execute(self, context, event_queue):
            # Simulate agent work that opens a nested child span.
            with self.inner_tracer.start_as_current_span("agent-inner-work"):
                pass

        async def cancel(self, context, event_queue):
            return None

    return _Exec


class _FakeContext:
    """Minimal RequestContext-like object."""

    def __init__(self, user_input="hello", context_id="ctx-1", task_id="task-1"):
        self._user_input = user_input
        self.context_id = context_id
        self.task_id = task_id

    def get_user_input(self):
        return self._user_input


async def _run(executor, context=None):
    await executor.execute(context or _FakeContext(), object())


# ---------------------------------------------------------------------------
# RED: no AGENT span without instrumentation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_red_no_agent_span_without_instrumentation(
    span_exporter, tracer_provider
):
    ExecCls = _make_executor_cls()
    await _run(ExecCls())
    names = [s.name for s in span_exporter.get_finished_spans()]
    assert "invoke_agent a2a-agent" not in names, names


# ---------------------------------------------------------------------------
# GREEN: AGENT span for an executor defined BEFORE instrument
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_green_agent_span_existing_subclass(instrument, span_exporter):
    ExecCls = _make_executor_cls()
    await _run(ExecCls())

    spans = span_exporter.get_finished_spans()
    agent_spans = [
        s
        for s in spans
        if s.attributes.get(_GEN_AI_SPAN_KIND) == _SPAN_KIND_AGENT
    ]
    assert agent_spans, [s.name for s in spans]
    span = agent_spans[0]
    assert span.attributes.get(_GEN_AI_FRAMEWORK) == "a2a"
    assert span.attributes.get(_GEN_AI_OPERATION_NAME) == "invoke_agent"
    assert span.attributes.get("a2a.context_id") == "ctx-1"
    assert span.attributes.get("a2a.task_id") == "task-1"


@pytest.mark.asyncio
async def test_green_inner_work_nests_under_agent(instrument, span_exporter, tracer_provider):
    # Route the executor's own child span to the same exporter so the nesting
    # relationship is observable; nesting itself is via context propagation.
    ExecCls = _make_executor_cls(inner_tracer_provider=tracer_provider)
    await _run(ExecCls())

    spans = span_exporter.get_finished_spans()
    assert len(spans) >= 2, [s.name for s in spans]

    agent = next(
        s
        for s in spans
        if s.attributes.get(_GEN_AI_SPAN_KIND) == _SPAN_KIND_AGENT
    )
    inner = next(s for s in spans if s.name == "agent-inner-work")

    assert inner.parent is not None
    assert inner.parent.span_id == agent.context.span_id
    assert inner.context.trace_id == agent.context.trace_id


@pytest.mark.asyncio
async def test_green_captures_user_input(instrument, span_exporter):
    ExecCls = _make_executor_cls()
    await _run(ExecCls(), context=_FakeContext(user_input="what is 2+2?"))

    spans = span_exporter.get_finished_spans()
    agent = next(
        s
        for s in spans
        if s.attributes.get(_GEN_AI_SPAN_KIND) == _SPAN_KIND_AGENT
    )
    msgs = agent.attributes.get("gen_ai.input.messages")
    assert msgs is not None
    assert "what is 2+2?" in msgs


# ---------------------------------------------------------------------------
# GREEN: AGENT span for a subclass defined AFTER instrument (init_subclass hook)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_green_agent_span_late_subclass(instrument, span_exporter):
    # Defined AFTER the instrument fixture ran -> must still be wrapped.
    ExecCls = _make_executor_cls()
    await _run(ExecCls())

    names = [s.name for s in span_exporter.get_finished_spans()]
    assert "invoke_agent a2a-agent" in names, names


# ---------------------------------------------------------------------------
# RED after uninstrument: teardown must stop AGENT span production
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_red_uninstrument_stops_agent_span(
    span_exporter, tracer_provider
):
    instrumentor = A2AInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider, skip_dep_check=True
    )
    ExecCls = _make_executor_cls()
    await _run(ExecCls())
    assert any(
        s.attributes.get(_GEN_AI_SPAN_KIND) == _SPAN_KIND_AGENT
        for s in span_exporter.get_finished_spans()
    )

    instrumentor.uninstrument()
    span_exporter.clear()

    ExecCls2 = _make_executor_cls()
    await _run(ExecCls2())
    names = [s.name for s in span_exporter.get_finished_spans()]
    assert "invoke_agent a2a-agent" not in names, (
        f"AGENT span still produced after uninstrument: {names}"
    )


# ---------------------------------------------------------------------------
# Error handling: exception in execute is recorded and re-raised
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_green_records_exception(instrument, span_exporter):
    from a2a.server.agent_execution import AgentExecutor

    class _BoomExec(AgentExecutor):
        async def execute(self, context, event_queue):
            raise ValueError("boom")

        async def cancel(self, context, event_queue):
            return None

    with pytest.raises(ValueError, match="boom"):
        await _run(_BoomExec())

    from opentelemetry.trace import StatusCode

    agent = next(
        s
        for s in span_exporter.get_finished_spans()
        if s.attributes.get(_GEN_AI_SPAN_KIND) == _SPAN_KIND_AGENT
    )
    assert agent.status.status_code == StatusCode.ERROR


# ---------------------------------------------------------------------------
# Lifecycle: double uninstrument is safe
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_double_uninstrument_safe(tracer_provider):
    instrumentor = A2AInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider, skip_dep_check=True
    )
    instrumentor.uninstrument()
    instrumentor.uninstrument()
