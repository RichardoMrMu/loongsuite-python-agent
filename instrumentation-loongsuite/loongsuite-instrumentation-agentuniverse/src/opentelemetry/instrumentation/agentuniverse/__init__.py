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
OpenTelemetry agentUniverse Instrumentation

Produces an ARMS gen-ai **AGENT** span around each execution of an
`agentUniverse <https://github.com/alipay/agentUniverse>`_ agent. The span
brackets ``agentuniverse.agent.agent.Agent.run`` (and its async twin
``Agent.async_run``), so every piece of work the agent performs -- planning,
tool calls, knowledge retrieval, memory access, LLM requests and any
downstream instrumentation -- nests underneath a single ``invoke_agent`` span
sharing one trace id.

Instrumentation seam
--------------------
Unlike frameworks whose execution entry point is an abstract method each agent
implements, agentUniverse defines ``run`` / ``async_run`` **concretely on the
base ``Agent`` class**: subclasses supply the pieces (``input_keys``,
``output_keys``, ``parse_input``, ``parse_result``) and the base class owns the
execution skeleton. Wrapping the two base-class methods therefore covers every
agent, including agents defined after ``instrument()``, with no
``__init_subclass__`` hook and no subclass walk.

Both methods are wrapped with ``wrapt`` and marked with a sentinel so
double-wrapping is impossible; ``uninstrument`` restores the originals.

Span
----
``kind=INTERNAL`` (agent execution is in-process work, not an inbound server
request), named ``invoke_agent {agent_name}``, where ``agent_name`` comes from
``instance.agent_model.info['name']`` and falls back to the concrete class name
when the agent has not been initialized from YAML yet.

Content capture
---------------
User input is recorded on ``gen_ai.input.messages`` only when the shared GenAI
util's content-capture switch enables span content -- i.e. when
``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`` is ``SPAN_ONLY`` or
``SPAN_AND_EVENT``. An absent or invalid value defaults to ``NO_CONTENT`` (no
message content), consistent with every other loongsuite instrumentation.

Fail-safe
---------
Every telemetry operation is guarded: an error raised while starting a span,
recording attributes or recording an exception is swallowed so instrumentation
can never interrupt or alter agent execution.
"""

import json
import logging
from typing import Any, Collection, Optional

from wrapt import wrap_function_wrapper

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.agentuniverse.package import _instruments
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.trace import SpanKind, Status, StatusCode
from opentelemetry.util.genai.extended_semconv.gen_ai_extended_attributes import (
    GEN_AI_SPAN_KIND,
    GenAiSpanKindValues,
)
from opentelemetry.util.genai.types import ContentCapturingMode
from opentelemetry.util.genai.utils import get_content_capturing_mode

logger = logging.getLogger(__name__)

# -- Framework identifier -----------------------------------------------------
_FRAMEWORK = "agentuniverse"

# -- GenAI semantic-convention attribute keys (sourced from the shared util) --
_GEN_AI_SPAN_KIND = GEN_AI_SPAN_KIND
_GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
_GEN_AI_FRAMEWORK = "gen_ai.framework"
_GEN_AI_AGENT_NAME = "gen_ai.agent.name"
_GEN_AI_INPUT_MESSAGES = "gen_ai.input.messages"

_SPAN_KIND_AGENT = GenAiSpanKindValues.AGENT.value
_OP_INVOKE_AGENT = "invoke_agent"

# -- Sentinel and wrapped methods ---------------------------------------------
_AGENTUNIVERSE_MARKER = "_otel_agentuniverse_wrapped"
_TARGET_METHODS = ("run", "async_run")

# Content-capture modes under which message text may be written onto spans.
_CONTENT_ON_SPAN_MODES = frozenset(
    {ContentCapturingMode.SPAN_ONLY, ContentCapturingMode.SPAN_AND_EVENT}
)

# Keys that carry runtime plumbing rather than user input; they are left out of
# the captured input when the conventional ``input`` key is absent.
_NON_INPUT_KEYS = frozenset({"callbacks"})


def _capture_content() -> bool:
    """True when message content should be written onto spans.

    Delegated to the shared util so an absent/invalid
    ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`` defaults to
    ``NO_CONTENT`` (no capture), matching the rest of loongsuite.
    """
    try:
        return get_content_capturing_mode() in _CONTENT_ON_SPAN_MODES
    except Exception:  # pragma: no cover - defensive: never break the app
        return False


def _text_message_json(role: str, content: Any) -> str:
    message = {
        "role": role,
        "parts": [{"type": "text", "content": str(content)}],
    }
    try:
        return json.dumps([message], ensure_ascii=False, separators=(",", ":"))
    except Exception:  # pragma: no cover - defensive
        return str([message])


def _extract_input(kwargs: Any) -> Optional[str]:
    """Best-effort extraction of the user input from the run() kwargs.

    ``agent.run(input=...)`` is the conventional call shape, so the ``input``
    key wins when present. Otherwise the remaining kwargs are serialized as
    JSON; agent execution is driven by these kwargs, which is the closest thing
    to a user message at this boundary.
    """
    if not kwargs:
        return None
    try:
        value = kwargs.get("input")
        if value is not None:
            return str(value)
        remaining = {
            key: val
            for key, val in kwargs.items()
            if key not in _NON_INPUT_KEYS
        }
        if not remaining:
            return None
        return json.dumps(
            remaining,
            ensure_ascii=False,
            default=str,
            separators=(",", ":"),
        )
    except Exception:  # pragma: no cover - defensive
        return None


def _agent_name(instance: Any) -> str:
    """Resolve the agent display name, falling back to the class name."""
    if instance is None:
        return _FRAMEWORK
    try:
        info = getattr(getattr(instance, "agent_model", None), "info", None)
        if isinstance(info, dict):
            name = info.get("name")
            if name:
                return str(name)
    except Exception:  # pragma: no cover - defensive: never break the app
        pass
    return type(instance).__name__


def _safe_set_attributes(span: Any, agent_name: str, kwargs: Any) -> None:
    """Populate span attributes; telemetry failures must never break execution."""
    try:
        span.set_attribute(_GEN_AI_SPAN_KIND, _SPAN_KIND_AGENT)
        span.set_attribute(_GEN_AI_OPERATION_NAME, _OP_INVOKE_AGENT)
        span.set_attribute(_GEN_AI_FRAMEWORK, _FRAMEWORK)
        span.set_attribute(_GEN_AI_AGENT_NAME, agent_name)

        if _capture_content():
            text = _extract_input(kwargs)
            if text:
                span.set_attribute(
                    _GEN_AI_INPUT_MESSAGES,
                    _text_message_json("user", text),
                )
    except Exception:  # pragma: no cover - defensive: never break the app
        logger.debug(
            "agentUniverse instrumentation failed to set span attributes",
            exc_info=True,
        )


def _safe_record_exception(span: Any, exc: BaseException) -> None:
    try:
        span.record_exception(exc)
        span.set_status(Status(StatusCode.ERROR))
    except Exception:  # pragma: no cover - defensive
        pass


def _safe_set_ok(span: Any) -> None:
    try:
        span.set_status(Status(StatusCode.OK))
    except Exception:  # pragma: no cover - defensive
        pass


class _RunWrapper:
    """Wrap ``Agent.run`` to produce the AGENT span for a sync execution."""

    def __init__(self, tracer):
        self._tracer = tracer

    def __call__(self, wrapped, instance, args, kwargs):
        try:
            agent_name = _agent_name(instance)
            span_cm = self._tracer.start_as_current_span(
                f"{_OP_INVOKE_AGENT} {agent_name}",
                kind=SpanKind.INTERNAL,
            )
        except Exception:  # pragma: no cover - defensive: never break the app
            logger.debug(
                "agentUniverse instrumentation could not start a span",
                exc_info=True,
            )
            return wrapped(*args, **kwargs)

        with span_cm as span:
            _safe_set_attributes(span, agent_name, kwargs)

            try:
                result = wrapped(*args, **kwargs)
            except Exception as exc:
                _safe_record_exception(span, exc)
                raise

            _safe_set_ok(span)
            return result


class _AsyncRunWrapper:
    """Wrap ``Agent.async_run`` to produce the AGENT span for an async run."""

    def __init__(self, tracer):
        self._tracer = tracer

    async def __call__(self, wrapped, instance, args, kwargs):
        try:
            agent_name = _agent_name(instance)
            span_cm = self._tracer.start_as_current_span(
                f"{_OP_INVOKE_AGENT} {agent_name}",
                kind=SpanKind.INTERNAL,
            )
        except Exception:  # pragma: no cover - defensive: never break the app
            logger.debug(
                "agentUniverse instrumentation could not start a span",
                exc_info=True,
            )
            return await wrapped(*args, **kwargs)

        with span_cm as span:
            _safe_set_attributes(span, agent_name, kwargs)

            try:
                result = await wrapped(*args, **kwargs)
            except Exception as exc:
                _safe_record_exception(span, exc)
                raise

            _safe_set_ok(span)
            return result


# ===========================================================================
# Wrap / unwrap helpers
# ===========================================================================


def _wrap_method(cls, name: str, wrapper) -> bool:
    """Wrap ``cls.<name>`` exactly once (idempotent via sentinel)."""
    own = cls.__dict__.get(name)
    if own is None:
        return False
    if getattr(own, _AGENTUNIVERSE_MARKER, False):
        return False
    try:
        wrap_function_wrapper(cls, name, wrapper)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Could not wrap %s.%s: %s", cls.__name__, name, exc)
        return False
    new = cls.__dict__.get(name)
    if new is not None:
        try:
            setattr(new, _AGENTUNIVERSE_MARKER, True)
        except Exception:  # pragma: no cover - defensive
            pass
    return True


def _unwrap_method(cls, name: str) -> None:
    own = cls.__dict__.get(name)
    if own is None or not getattr(own, _AGENTUNIVERSE_MARKER, False):
        return
    # Drop the marker first so it does not survive onto the restored original
    # and make a later re-instrument look like it is already wrapped.
    try:
        delattr(own, _AGENTUNIVERSE_MARKER)
    except (AttributeError, TypeError):
        pass
    try:
        unwrap(cls, name)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Could not unwrap %s.%s: %s", cls.__name__, name, exc)


# ===========================================================================
# Instrumentor
# ===========================================================================


class AgentUniverseInstrumentor(BaseInstrumentor):
    """Instrumentor for the agentUniverse agent framework."""

    def __init__(self):
        super().__init__()
        # BaseInstrumentor.__new__ returns a singleton, so __init__ may run
        # again on a later AgentUniverseInstrumentor() call. Only seed the
        # bookkeeping the first time, or a stray construct-after-instrument
        # would clear the active wrapper state.
        if not hasattr(self, "_agentuniverse_initialized"):
            self._agentuniverse_initialized = True
            self._agent = None
            self._wrapped_methods = []
            self._wrappers = {}

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        from agentuniverse.agent.agent import Agent

        tracer_provider = kwargs.get("tracer_provider")
        tracer = trace_api.get_tracer(
            __name__, "", tracer_provider=tracer_provider
        )

        self._agent = Agent
        self._wrapped_methods = []
        self._wrappers = {}
        wrappers = (
            ("run", _RunWrapper(tracer)),
            ("async_run", _AsyncRunWrapper(tracer)),
        )
        for method_name, wrapper in wrappers:
            self._wrappers[method_name] = wrapper
            if _wrap_method(Agent, method_name, wrapper):
                self._wrapped_methods.append(method_name)

    def _uninstrument(self, **kwargs: Any) -> None:
        agent = self._agent
        if agent is not None:
            for method_name in _TARGET_METHODS:
                _unwrap_method(agent, method_name)

        self._agent = None
        self._wrapped_methods = []
        self._wrappers = {}
