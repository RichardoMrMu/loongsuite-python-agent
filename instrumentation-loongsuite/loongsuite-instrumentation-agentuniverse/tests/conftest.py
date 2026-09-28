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
the framework: the native ``AgentInstrumentor`` is what creates the span under
test, and the LoongSuite instrumentor must add ``gen_ai.*`` attributes to that
same span instead of creating a second one.
"""

import os
import sys
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
from agentuniverse.agent.agent import Agent  # noqa: E402
from agentuniverse.agent.agent_model import AgentModel  # noqa: E402
from agentuniverse.base.annotation import trace as trace_module  # noqa: E402
from agentuniverse.base.config.application_configer.app_configer import (  # noqa: E402
    AppConfiger,
)
from agentuniverse.base.config.application_configer.application_config_manager import (  # noqa: E402
    ApplicationConfigManager,
)
from agentuniverse.base.tracing.otel.instrumentation.agent.agent_instrumentor import (  # noqa: E402
    AgentInstrumentor,
)

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
            stream.put("first-token")
        return {"output": "streamed"}


def build_agent(
    name: str = "test_agent", agent_cls: type = StubAgent
) -> Agent:
    """Build an agent the native instrumentor will name ``name``."""
    agent = agent_cls()
    agent.agent_model = AgentModel(info={"name": name})
    return agent


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
    """A meter provider the native instrumentor can record its metrics into.

    The native instrumentor reads the provider from ``instrument(...)``, so the
    metrics tests hand it this one instead of the global no-op provider.
    """
    provider = MeterProvider(metric_readers=[metric_reader])
    yield provider
    provider.shutdown()


@pytest.fixture
def make_agent() -> Callable[..., Agent]:
    return build_agent


def active_native_instrumentor() -> AgentInstrumentor | None:
    """The native instrumentor currently installed, without constructing one.

    ``AgentInstrumentor.__init__`` resets the saved wrapper originals, so
    calling ``AgentInstrumentor()`` while one is already instrumented would
    make a later ``uninstrument()`` blank the trace-module globals and break
    every agent call. Reach the live instance through the globals instead.
    """
    wrapper = getattr(trace_module, "_agent_wrapper_sync", None)
    owner = getattr(wrapper, "__self__", None)
    return owner if isinstance(owner, AgentInstrumentor) else None


@pytest.fixture(autouse=True)
def isolate_instrumentation() -> Iterator[None]:
    """Keep the templated trace globals and the native singleton pristine.

    Both instrumentors are ``BaseInstrumentor`` singletons and the native one
    lives by swapping module-level globals in ``agentuniverse.base.annotation.
    trace``. A test that leaves either behind would silently change the next
    test's result, so force both back to their pre-test state.
    """
    from opentelemetry.instrumentation.agentuniverse import (
        AgentUniverseInstrumentor,
    )

    bridge = AgentUniverseInstrumentor()
    native = active_native_instrumentor()
    saved_sync = trace_module._agent_wrapper_sync
    saved_async = trace_module._agent_wrapper_async

    yield

    for instrumentor in (bridge, native):
        if instrumentor is not None and instrumentor.__dict__.get(
            "_is_instrumented_by_opentelemetry"
        ):
            instrumentor.uninstrument()
    trace_module._agent_wrapper_sync = saved_sync
    trace_module._agent_wrapper_async = saved_async
