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

"""Test configuration for agentUniverse instrumentation tests.

The tests exercise the instrumentation against the base class it actually
wraps -- ``agentuniverse.agent.agent.Agent`` -- and against real in-process
agent subclasses, asserting on the OTel spans exported to an
``InMemorySpanExporter``. No network access and no YAML application config are
required.

Framework resolution
--------------------
``agentUniverse`` requires Python <= 3.13 and pins dependencies
(``cffi<2``, ``numpy<2``, ``grpcio==1.63.0``, ``langchain==0.1.20`` and
friends) that publish no wheels for Python 3.14, so on a 3.14 interpreter the
real distribution cannot be installed at all. This conftest therefore resolves
``agentuniverse.agent.agent.Agent`` the same way the instrumentation does and,
only when that import fails, installs a **schema-faithful stand-in** of the
agentUniverse modules the instrumented call path touches.

The stand-in reproduces the upstream ``agentUniverse==0.0.19.1`` class shape
verbatim: identical abstract methods (``input_keys`` / ``output_keys`` /
``parse_input`` / ``parse_result``), the concrete ``run`` / ``async_run``
bodies, and the ``agent_model.info['name']`` name lookup. Which mode is active
is printed in the pytest header, so a stand-in run can never be mistaken for a
run against the real framework (see also
``test_framework_resolution_is_explicit``).
"""

from __future__ import annotations

import json
import os
import sys
import types
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import pytest

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def pytest_configure(config: pytest.Config):
    os.environ["OTEL_SEMCONV_STABILITY_OPT_IN"] = "gen_ai_latest_experimental"


# ---------------------------------------------------------------------------
# agentUniverse resolution: the real framework when importable, otherwise a
# documented schema-faithful stand-in.
# ---------------------------------------------------------------------------

REAL_AGENTUNIVERSE = True
AGENTUNIVERSE_IMPORT_ERROR: Optional[BaseException] = None


def _install_agentuniverse_standin() -> Tuple[type, type]:
    """Install a schema-faithful stand-in of the agentUniverse modules.

    Only used when the real distribution cannot be imported. Every class below
    mirrors ``agentUniverse==0.0.19.1`` for the surface the instrumentation
    touches; the ``run`` / ``async_run`` bodies and the abstract-method set are
    copied from ``agentuniverse/agent/agent.py``, and ``pre_parse_input`` keeps
    the upstream ``self.agent_model.info.get('name', '')`` lookup.
    """

    class InputObject:
        """Mirrors agentuniverse.agent.input_object.InputObject."""

        def __init__(self, params: dict):
            self._params = params
            for key, value in params.items():
                self.__dict__[key] = value

        def to_dict(self):
            return self._params

        def to_json_str(self):
            return json.dumps(self._params)

        def add_data(self, key, value):
            self._params[key] = value
            self.__dict__[key] = value

        def get_data(self, key, default=None):
            return self._params.get(key, default)

    class OutputObject:
        """Mirrors agentuniverse.agent.output_object.OutputObject."""

        def __init__(self, params: dict):
            self._params = params
            for key, value in params.items():
                self.__dict__[key] = value

        def to_dict(self):
            return self._params

        def to_json_str(self):
            return json.dumps(self._params, ensure_ascii=False)

        def get_data(self, key, default=None):
            return self._params.get(key, default)

    class AgentModel:
        """Mirrors agentuniverse.agent.agent_model.AgentModel attributes."""

        def __init__(
            self,
            info: Optional[dict] = None,
            profile: Optional[dict] = None,
            plan: Optional[dict] = None,
            memory: Optional[dict] = None,
            action: Optional[dict] = None,
        ):
            self.info = info if info is not None else {}
            self.profile = profile if profile is not None else {}
            self.plan = plan if plan is not None else {}
            self.memory = memory if memory is not None else {}
            self.action = action if action is not None else {}

    class Agent(ABC):
        """Mirrors agentuniverse.agent.agent.Agent.

        ``run`` and ``async_run`` are concrete on the base class exactly as
        upstream; only the four ``parse_*`` / ``*_keys`` methods are abstract.
        """

        agent_model: Optional[AgentModel] = None

        def __init__(self):
            pass

        @abstractmethod
        def input_keys(self) -> list:
            """Return the input keys of the Agent."""

        @abstractmethod
        def output_keys(self) -> list:
            """Return the output keys of the Agent."""

        @abstractmethod
        def parse_input(self, input_object, agent_input) -> dict:
            """Agent parameter parsing."""

        @abstractmethod
        def parse_result(self, agent_result) -> dict:
            """Agent result parser."""

        # -- run / async_run bodies copied from upstream ---------------------

        def run(self, **kwargs):
            """Agent instance running entry."""
            self.input_check(kwargs)
            input_object = InputObject(kwargs)

            self.update_trace_context(input_object)

            agent_input = self.pre_parse_input(input_object)

            planner_result = self.execute(input_object, agent_input)

            agent_result = self.parse_result(planner_result)

            self.output_check(agent_result)
            return OutputObject(agent_result)

        async def async_run(self, **kwargs):
            """Agent instance async running entry."""
            self.input_check(kwargs)
            input_object = InputObject(kwargs)
            self.update_trace_context(input_object)

            agent_input = self.pre_parse_input(input_object)

            agent_result = await self.async_execute(input_object, agent_input)

            agent_result = self.parse_result(agent_result)

            self.output_check(agent_result)
            return OutputObject(agent_result)

        def execute(self, input_object, agent_input) -> dict:
            """Upstream delegates to the configured planner; tests override."""
            raise NotImplementedError(
                "the stand-in has no planner; override execute()"
            )

        async def async_execute(self, input_object, agent_input) -> dict:
            """Upstream base implementation is a no-op."""
            return None

        def pre_parse_input(self, input_object) -> dict:
            """Agent execution parameter pre-parsing (upstream body)."""
            agent_input = dict()
            agent_input["chat_history"] = (
                input_object.get_data("chat_history") or ""
            )
            agent_input["background"] = (
                input_object.get_data("background") or ""
            )
            agent_input["image_urls"] = (
                input_object.get_data("image_urls") or []
            )
            agent_input["audio_url"] = input_object.get_data("audio_url") or ""
            agent_input["date"] = datetime.now().strftime("%Y-%m-%d")
            agent_input["session_id"] = (
                input_object.get_data("session_id") or ""
            )
            agent_input["agent_id"] = self.agent_model.info.get("name", "")
            self.parse_input(input_object, agent_input)
            return agent_input

        def update_trace_context(self, input_object):
            """Upstream tracks its own trace context; not needed here."""

        def input_check(self, kwargs: dict):
            """Agent parameter check."""
            for key in self.input_keys():
                if key not in kwargs.keys():
                    raise Exception(f"Input must have key: {key}.")

        def output_check(self, kwargs: dict):
            """Agent result check."""
            if not isinstance(kwargs, dict):
                raise Exception("Output type must be dict.")
            for key in self.output_keys():
                if key not in kwargs.keys():
                    raise Exception(f"Output must have key: {key}.")

    agentuniverse = types.ModuleType("agentuniverse")
    agentuniverse.__path__ = []
    agent_package = types.ModuleType("agentuniverse.agent")
    agent_package.__path__ = []

    agent_module = types.ModuleType("agentuniverse.agent.agent")
    agent_module.Agent = Agent
    model_module = types.ModuleType("agentuniverse.agent.agent_model")
    model_module.AgentModel = AgentModel
    input_module = types.ModuleType("agentuniverse.agent.input_object")
    input_module.InputObject = InputObject
    output_module = types.ModuleType("agentuniverse.agent.output_object")
    output_module.OutputObject = OutputObject

    agent_package.agent = agent_module
    agent_package.agent_model = model_module
    agent_package.input_object = input_module
    agent_package.output_object = output_module

    for name, module in (
        ("agentuniverse", agentuniverse),
        ("agentuniverse.agent", agent_package),
        ("agentuniverse.agent.agent", agent_module),
        ("agentuniverse.agent.agent_model", model_module),
        ("agentuniverse.agent.input_object", input_module),
        ("agentuniverse.agent.output_object", output_module),
    ):
        sys.modules[name] = module

    return Agent, AgentModel


try:
    from agentuniverse.agent.agent import Agent as _AGENT_BASE  # noqa: E402
    from agentuniverse.agent.agent_model import (  # noqa: E402
        AgentModel as _AGENT_MODEL,
    )
except Exception as _agentuniverse_import_error:  # pragma: no cover
    REAL_AGENTUNIVERSE = False
    AGENTUNIVERSE_IMPORT_ERROR = _agentuniverse_import_error
    _AGENT_BASE, _AGENT_MODEL = _install_agentuniverse_standin()


def pytest_report_header(config: pytest.Config) -> str:
    if REAL_AGENTUNIVERSE:
        return "agentuniverse sdk: real agentUniverse distribution"
    return (
        "agentuniverse sdk: schema stand-in "
        f"({type(AGENTUNIVERSE_IMPORT_ERROR).__name__}: "
        f"{AGENTUNIVERSE_IMPORT_ERROR})"
    )


from opentelemetry import trace as trace_api  # noqa: E402
from opentelemetry.instrumentation.agentuniverse import (  # noqa: E402
    AgentUniverseInstrumentor,
)
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)


@pytest.fixture(scope="function", name="span_exporter")
def fixture_span_exporter():
    exporter = InMemorySpanExporter()
    yield exporter
    exporter.clear()


@pytest.fixture(scope="function", name="tracer_provider")
def fixture_tracer_provider(span_exporter):
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    return provider


@pytest.fixture(scope="function")
def instrument(tracer_provider):
    instrumentor = AgentUniverseInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        skip_dep_check=True,
    )
    yield instrumentor
    instrumentor.uninstrument()


@pytest.fixture(scope="session", name="agentuniverse_is_real")
def fixture_agentuniverse_is_real():
    return REAL_AGENTUNIVERSE


@pytest.fixture(scope="session", name="agentuniverse_import_error")
def fixture_agentuniverse_import_error():
    return AGENTUNIVERSE_IMPORT_ERROR


@pytest.fixture(scope="session", name="agentuniverse_sdk")
def fixture_agentuniverse_sdk():
    """The resolved ``(Agent base class, AgentModel)`` pair."""
    return _AGENT_BASE, _AGENT_MODEL


@pytest.fixture(scope="function", name="make_agent")
def fixture_make_agent(agentuniverse_sdk):
    """Build a concrete agent instance over the resolved Agent base class.

    A fresh subclass per call keeps the sentinel marker on a shared ``run``
    function object from leaking between tests, and mirrors how a real
    application subclasses ``Agent``: only the abstract hooks (plus ``execute``,
    which upstream routes to a planner) are supplied.
    """
    agent_base, agent_model_cls = agentuniverse_sdk

    def _make_agent(
        agent_name=None,
        input_key="input",
        inner_tracer_provider=None,
        failure=None,
    ):
        inner_tracer = trace_api.get_tracer(
            "test.inner", tracer_provider=inner_tracer_provider
        )

        class _Agent(agent_base):
            def input_keys(self) -> list:
                return [input_key]

            def output_keys(self) -> list:
                return ["output"]

            def parse_input(self, input_object, agent_input) -> dict:
                return agent_input

            def parse_result(self, agent_result) -> dict:
                return agent_result

            def execute(self, input_object, agent_input) -> dict:
                # Simulate agent work that opens a nested child span.
                with inner_tracer.start_as_current_span("agent-inner-work"):
                    if failure is not None:
                        raise failure
                return {"output": f"echo:{input_object.get_data(input_key)}"}

            async def async_execute(self, input_object, agent_input) -> dict:
                with inner_tracer.start_as_current_span(
                    "agent-inner-work-async"
                ):
                    pass
                return {"output": f"echo:{input_object.get_data(input_key)}"}

        agent = _Agent()
        agent.agent_model = agent_model_cls(
            info={"name": agent_name} if agent_name else {}
        )
        return agent

    return _make_agent
