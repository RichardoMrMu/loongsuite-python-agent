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

"""Fixtures for the agentUniverse instrumentation tests.

These tests run against a real ``agentUniverse`` install, with no stand-in for
the framework: the native Agent, LLM and Tool instrumentors are what create the
spans under test, and the LoongSuite instrumentor must add ``gen_ai.*``
attributes to those same spans instead of creating a second one.

The agents, LLM and tool below are real framework objects -- a real ``Agent``
subclass, a real ``@trace_llm`` method and a real ``Tool`` subclass -- so the
native wrappers, setters, metrics and token aggregation all run for real.
"""

import os
import sys
import time
from pathlib import Path

# The shared GenAI util only exposes its capture switch in experimental mode,
# and the instrumentation reads that switch at instrument() time, so opt in
# before any test runs.
os.environ.setdefault(
    "OTEL_SEMCONV_STABILITY_OPT_IN", "gen_ai_latest_experimental"
)

# Import the package under test from its source tree, as the other
# instrumentation test suites do.
_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from collections.abc import Callable, Iterator  # noqa: E402
from typing import Any  # noqa: E402

import pytest  # noqa: E402
from agentuniverse.agent.action.tool.tool import Tool  # noqa: E402
from agentuniverse.agent.agent import Agent  # noqa: E402
from agentuniverse.agent.agent_model import AgentModel  # noqa: E402
from agentuniverse.base.annotation import trace as trace_module  # noqa: E402
from agentuniverse.base.annotation.trace import trace_llm  # noqa: E402
from agentuniverse.base.config.application_configer.app_configer import (  # noqa: E402
    AppConfiger,
)
from agentuniverse.base.config.application_configer.application_config_manager import (  # noqa: E402
    ApplicationConfigManager,
)
from agentuniverse.base.tracing.otel.instrumentation.agent.agent_instrumentor import (  # noqa: E402
    AgentInstrumentor,
)
from agentuniverse.base.tracing.otel.instrumentation.llm.llm_instrumentor import (  # noqa: E402
    LLMInstrumentor,
)
from agentuniverse.base.tracing.otel.instrumentation.tool.tool_instrumentor import (  # noqa: E402
    ToolInstrumentor,
)
from agentuniverse.llm.llm_output import LLMOutput, TokenUsage  # noqa: E402

from opentelemetry.sdk.metrics import MeterProvider  # noqa: E402
from opentelemetry.sdk.metrics.export import (  # noqa: E402
    InMemoryMetricReader,
)
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)

# The native instrumentor builds a ``ConversationMemoryModule`` at call time,
# which reads the application config manager on construction. Seed it before
# any agent runs, or the native wrapper fails on ``app_configer is None``.
ApplicationConfigManager().app_configer = AppConfiger()

# Each native instrumentor lives by swapping a pair of module-level globals in
# ``agentuniverse.base.annotation.trace``; the pair is how the instrumentor that
# owns a layer is recognised.
NATIVE_LAYER_GLOBALS: dict[type, tuple[str, str]] = {
    AgentInstrumentor: ("_agent_wrapper_sync", "_agent_wrapper_async"),
    LLMInstrumentor: ("_llm_wrapper_sync", "_llm_wrapper_async"),
    ToolInstrumentor: ("_tool_wrapper_sync", "_tool_wrapper_async"),
}
NATIVE_WRAPPER_GLOBALS: tuple[str, ...] = tuple(
    name for pair in NATIVE_LAYER_GLOBALS.values() for name in pair
)

# The three layers the bridge covers, as short keys the tests can loop over.
AGENT_LAYER = "agent"
LLM_LAYER = "llm"
TOOL_LAYER = "tool"


# ---------------------------------------------------------------------------
# Real agents
# ---------------------------------------------------------------------------


class StubAgent(Agent):
    """A real agentUniverse Agent whose body needs no planner or LLM."""

    def input_keys(self) -> list[str]:
        return ["input"]

    def output_keys(self) -> list[str]:
        return ["output"]

    def parse_input(self, input_object: Any, agent_input: dict) -> dict:
        return agent_input

    def parse_result(self, agent_result: dict) -> dict:
        return agent_result

    def execute(self, input_object: Any, agent_input: dict) -> dict:
        return {"output": f"echo:{input_object.get_data('input')}"}

    async def async_execute(
        self, input_object: Any, agent_input: dict
    ) -> dict:
        return {"output": f"echo:{input_object.get_data('input')}"}


class FailingAgent(StubAgent):
    """``StubAgent`` whose body raises, to exercise the error path."""

    def execute(self, input_object: Any, agent_input: dict) -> dict:
        raise RuntimeError("agent exploded")


class StreamingAgent(StubAgent):
    """``StubAgent`` that pushes one token onto ``output_stream``.

    The native wrapper swaps the caller's queue for one that records the first
    put, so putting an item here is what exercises the first-token hook -- on
    this path the native wrapper does not fall back to the end-of-call timing.
    """

    def execute(self, input_object: Any, agent_input: dict) -> dict:
        stream = input_object.get_data("output_stream")
        if stream is not None:
            # ``time.time()`` on Windows is only accurate to about 15 ms, so
            # put the token late enough that the first-token duration is
            # unambiguously positive.
            time.sleep(0.05)
            stream.put("first-token")
        return {"output": "streamed"}


class RichAgent(StubAgent):
    """An agent that really calls an LLM and a tool, as a real one would."""

    def execute(self, input_object: Any, agent_input: dict) -> dict:
        prompt = input_object.get_data("input")
        llm_result = StubLLM().call(prompt=prompt)
        tool_result = StubTool().run(query=prompt)
        return {"output": f"{llm_result.text}|{tool_result}"}


class AsyncRichAgent(StubAgent):
    """``RichAgent`` over the async wrappers of all three layers."""

    async def async_execute(
        self, input_object: Any, agent_input: dict
    ) -> dict:
        prompt = input_object.get_data("input")
        llm_result = await AsyncLLM().call(prompt=prompt)
        tool_result = await StubTool().async_run(query=prompt)
        return {"output": f"{llm_result.text}|{tool_result}"}


class LLMAgent(StubAgent):
    """An agent whose only child call is a single LLM call."""

    def execute(self, input_object: Any, agent_input: dict) -> dict:
        return {
            "output": StubLLM()
            .call(prompt=input_object.get_data("input"))
            .text
        }


class StreamingLLMAgent(StubAgent):
    """An agent whose only child call is a streaming LLM call."""

    def execute(self, input_object: Any, agent_input: dict) -> dict:
        stream = StreamingLLM().call(prompt=input_object.get_data("input"))
        return {"output": "".join(chunk.text for chunk in stream)}


class ToolAgent(StubAgent):
    """An agent whose only child call is a single tool call."""

    def execute(self, input_object: Any, agent_input: dict) -> dict:
        return {"output": StubTool().run(query=input_object.get_data("input"))}


# ---------------------------------------------------------------------------
# Real LLMs
# ---------------------------------------------------------------------------


class StubLLM:
    """A minimal faithful LLM: a real ``@trace_llm`` method is all the native
    ``LLMInstrumentor`` needs.

    ``_get_llm_info`` reads ``self.name``, ``self.channel_name`` and
    ``self.channel_model_config``, and ``_llm_plugins`` walks an (empty by
    default) plugin set from the application config, so no LLM base class is
    required to exercise the real native LLM wrapper.
    """

    name = "stub_llm"
    channel_name = "test_channel"
    channel_model_config: dict = {"temperature": 0.7}

    @trace_llm
    def call(self, prompt: str, **kwargs) -> LLMOutput:
        return LLMOutput(
            text=f"llm:{prompt}",
            usage=TokenUsage(text_in=3, text_out=5),
            finish_reason="stop",
        )


class AsyncLLM(StubLLM):
    """The async twin, taken by the native async LLM wrapper."""

    name = "async_llm"

    @trace_llm
    async def call(self, prompt: str, **kwargs) -> LLMOutput:
        return LLMOutput(
            text=f"async:{prompt}",
            usage=TokenUsage(text_in=2, text_out=4),
            finish_reason="stop",
        )


class StreamingLLM(StubLLM):
    """A ``@trace_llm`` generator, which the native wrapper treats as a stream.

    The sleep makes the first-token timing unambiguously positive, and the
    usage rides on the last chunk, which is where the native
    ``StreamingResultProcessor`` looks for it.
    """

    name = "stream_llm"

    @trace_llm
    def call(self, prompt: str, **kwargs):
        time.sleep(0.005)
        yield LLMOutput(text="part-1")
        yield LLMOutput(text="part-2", usage=TokenUsage(text_in=4, text_out=6))


class FailingLLM(StubLLM):
    """An LLM call that raises, to exercise the native LLM error path."""

    name = "failing_llm"

    @trace_llm
    def call(self, prompt: str, **kwargs) -> LLMOutput:
        raise RuntimeError("llm exploded")


class MessagesLLM(StubLLM):
    """An LLM called with a chat ``messages`` list instead of a prompt."""

    name = "messages_llm"

    @trace_llm
    def call(self, messages: list, **kwargs) -> LLMOutput:
        return LLMOutput(
            text="answered",
            usage=TokenUsage(text_in=1, text_out=2),
            finish_reason="stop",
        )


# ---------------------------------------------------------------------------
# Real tools
# ---------------------------------------------------------------------------


class StubTool(Tool):
    """A real agentUniverse ``Tool``; ``run``/``async_run`` are native seams."""

    name: str = "stub_tool"
    input_keys: list = ["query"]

    def execute(self, query: str) -> str:
        return f"tool-output:{query}"


class FailingTool(StubTool):
    """A tool whose body raises, to exercise the native tool error path."""

    name: str = "failing_tool"

    def execute(self, query: str) -> str:
        raise RuntimeError("tool exploded")


def build_agent(
    name: str = "test_agent", agent_cls: type = StubAgent
) -> Agent:
    """Build an agent the native instrumentor will name ``name``."""
    agent = agent_cls()
    agent.agent_model = AgentModel(info={"name": name})
    return agent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def span_exporter() -> Iterator[InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    yield exporter
    exporter.clear()


@pytest.fixture
def tracer_provider(
    span_exporter: InMemorySpanExporter,
) -> Iterator[TracerProvider]:
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    yield provider
    provider.shutdown()


@pytest.fixture
def metric_reader() -> Iterator[InMemoryMetricReader]:
    yield InMemoryMetricReader()


@pytest.fixture
def meter_provider(
    metric_reader: InMemoryMetricReader,
) -> Iterator[MeterProvider]:
    """A meter provider the native instrumentors can record their metrics into.

    The native instrumentors read the provider from ``instrument(...)``, so the
    metrics tests hand them this one instead of the global no-op provider.
    """
    provider = MeterProvider(metric_readers=[metric_reader])
    yield provider
    provider.shutdown()


@pytest.fixture
def make_agent() -> Callable[..., Agent]:
    return build_agent


def active_native_instrumentor(native_cls: type = AgentInstrumentor) -> Any:
    """The live native instrumentor for one layer, without constructing one.

    The native classes are ``BaseInstrumentor`` singletons whose ``__init__``
    resets the wrapper originals they saved, so calling ``AgentInstrumentor()``
    -- or its LLM/Tool twin -- while one is already instrumented would make a
    later ``uninstrument()`` blank the trace-module globals and break every
    call. Reach the live instance through the globals instead.
    """
    wrapper = getattr(trace_module, NATIVE_LAYER_GLOBALS[native_cls][0], None)
    owner = getattr(wrapper, "__self__", None)
    return owner if isinstance(owner, native_cls) else None


@pytest.fixture(autouse=True)
def isolate_instrumentation() -> Iterator[None]:
    """Keep the templated trace globals and the native singletons pristine.

    All four instrumentors are ``BaseInstrumentor`` singletons and each native
    one lives by swapping module-level globals in
    ``agentuniverse.base.annotation.trace``. A test that leaves any of them
    behind would silently change the next test's result, so force every layer
    back to its pre-test state.
    """
    from opentelemetry.instrumentation.agentuniverse import (
        AgentUniverseInstrumentor,
    )

    bridge = AgentUniverseInstrumentor()
    saved = {
        name: getattr(trace_module, name) for name in NATIVE_WRAPPER_GLOBALS
    }

    yield

    # Re-read the active native instrumentors at teardown: a test may have
    # enabled one that was not active when it started.
    instrumentors = [bridge] + [
        active_native_instrumentor(native_cls)
        for native_cls in NATIVE_LAYER_GLOBALS
    ]
    for instrumentor in instrumentors:
        if instrumentor is not None and instrumentor.__dict__.get(
            "_is_instrumented_by_opentelemetry"
        ):
            instrumentor.uninstrument()
    for name, value in saved.items():
        setattr(trace_module, name, value)
