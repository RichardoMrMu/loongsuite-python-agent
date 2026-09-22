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
OpenTelemetry A2A (Agent2Agent) Instrumentation

Produces an ARMS gen-ai **AGENT** span around each server-side agent turn of
the official A2A Python SDK (``a2a-sdk``). The span brackets the user's
``AgentExecutor.execute`` invocation, so all work the agent does — including
the SDK's own transport / request-handler spans and any downstream LLM /
tool instrumentation — nests underneath a single ``invoke_agent`` span with a
shared trace id.

Relationship to a2a-sdk's built-in tracing
------------------------------------------
``a2a-sdk`` already ships an OpenTelemetry tracing layer
(``a2a.utils.telemetry``) that decorates its transports and request handlers
with generic spans under the instrumenting module ``a2a-python-sdk``. Those
spans describe the *protocol plumbing*; none of them carry gen-ai semantic
conventions and none of them wraps the user's ``execute`` implementation
(``AgentExecutor.execute`` is an abstract method the application overrides).

This package is therefore **complementary, not duplicative**: it adds the
gen-ai AGENT boundary that the SDK does not, mirroring how the sibling
``loongsuite`` packages layer ARMS gen-ai spans over frameworks that already
emit some telemetry (e.g. litellm).

Instrumentation seam
--------------------
``AgentExecutor`` is an ABC whose ``execute`` coroutine is overridden by
every concrete agent. To trace all of them we:

1. Walk the existing ``AgentExecutor`` subclass tree at ``instrument`` time
   and wrap each subclass's own ``execute`` (via ``wrapt``).
2. Install an ``__init_subclass__`` hook on ``AgentExecutor`` so that agent
   classes defined *after* instrumentation are wrapped as they are created.

Both paths mark the wrapped function with a sentinel so double-wrapping is
impossible, and ``uninstrument`` unwraps every marked ``execute`` and
restores the original ``__init_subclass__``.

Content capture
---------------
The user's input message is recorded as ``gen_ai.input.messages`` when
available. Set ``OTEL_INSTRUMENTATION_A2A_CAPTURE_CONTENT=false`` to suppress
message content while keeping the structural AGENT span.
"""

import json
import logging
import os
from typing import Any, Collection, Optional

from wrapt import wrap_function_wrapper

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.a2a.package import _instruments
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.trace import SpanKind, Status, StatusCode

logger = logging.getLogger(__name__)

# ── Framework identifier ─────────────────────────────────────────────────────
_FRAMEWORK = "a2a"
_AGENT_NAME = "a2a-agent"

# ── GenAI semantic-convention attribute keys (ARMS gen-ai semconv) ───────────
_GEN_AI_SPAN_KIND = "gen_ai.span.kind"
_GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
_GEN_AI_FRAMEWORK = "gen_ai.framework"
_GEN_AI_INPUT_MESSAGES = "gen_ai.input.messages"

_SPAN_KIND_AGENT = "AGENT"
_OP_INVOKE_AGENT = "invoke_agent"

# ── Sentinel + content toggle ────────────────────────────────────────────────
_A2A_MARKER = "_otel_a2a_wrapped"
_CAPTURE_CONTENT_ENV = "OTEL_INSTRUMENTATION_A2A_CAPTURE_CONTENT"


def _capture_content() -> bool:
    return os.getenv(_CAPTURE_CONTENT_ENV, "true").lower() != "false"


def _text_message_json(role: str, content: Any) -> str:
    message = {
        "role": role,
        "parts": [{"type": "text", "content": str(content)}],
    }
    try:
        return json.dumps([message], ensure_ascii=False, separators=(",", ":"))
    except Exception:
        return str([message])


def _extract_user_input(context: Any) -> Optional[str]:
    """Best-effort extraction of the user's text input from a RequestContext."""
    if context is None:
        return None
    getter = getattr(context, "get_user_input", None)
    if callable(getter):
        try:
            text = getter()
            if text:
                return str(text)
        except Exception:
            pass
    return None


class _ExecuteWrapper:
    """Wrap ``AgentExecutor.execute`` to produce the AGENT span."""

    def __init__(self, tracer):
        self._tracer = tracer

    async def __call__(self, wrapped, instance, args, kwargs):
        context = args[0] if args else kwargs.get("context")
        agent_name = (
            type(instance).__name__ if instance is not None else _AGENT_NAME
        )

        with self._tracer.start_as_current_span(
            f"{_OP_INVOKE_AGENT} {_AGENT_NAME}",
            kind=SpanKind.SERVER,
        ) as span:
            span.set_attribute(_GEN_AI_SPAN_KIND, _SPAN_KIND_AGENT)
            span.set_attribute(_GEN_AI_OPERATION_NAME, _OP_INVOKE_AGENT)
            span.set_attribute(_GEN_AI_FRAMEWORK, _FRAMEWORK)
            span.set_attribute("gen_ai.agent.name", agent_name)

            context_id = getattr(context, "context_id", None)
            if context_id:
                span.set_attribute("a2a.context_id", str(context_id))
            task_id = getattr(context, "task_id", None)
            if task_id:
                span.set_attribute("a2a.task_id", str(task_id))

            if _capture_content():
                user_input = _extract_user_input(context)
                if user_input:
                    span.set_attribute(
                        _GEN_AI_INPUT_MESSAGES,
                        _text_message_json("user", user_input),
                    )

            try:
                result = await wrapped(*args, **kwargs)
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR))
                raise

            span.set_status(Status(StatusCode.OK))
            return result


# ═══════════════════════════════════════════════════════════════════════════
# Wrap / unwrap helpers
# ═══════════════════════════════════════════════════════════════════════════


def _wrap_execute(cls, wrapper) -> None:
    """Wrap ``cls.execute`` exactly once (idempotent via sentinel)."""
    own = cls.__dict__.get("execute")
    if own is None:
        return  # abstract / not overridden on this class
    if getattr(own, _A2A_MARKER, False):
        return
    try:
        wrap_function_wrapper(cls, "execute", wrapper)
    except Exception as e:  # pragma: no cover - defensive
        logger.debug("Could not wrap %s.execute: %s", cls.__name__, e)
        return
    new = cls.__dict__.get("execute")
    if new is not None:
        try:
            setattr(new, _A2A_MARKER, True)
        except Exception:  # pragma: no cover - defensive
            pass


def _unwrap_execute(cls) -> None:
    own = cls.__dict__.get("execute")
    if own is None or not getattr(own, _A2A_MARKER, False):
        return
    try:
        delattr(own, _A2A_MARKER)
    except (AttributeError, TypeError):
        pass
    try:
        unwrap(cls, "execute")
    except Exception as e:  # pragma: no cover - defensive
        logger.debug("Could not unwrap %s.execute: %s", cls.__name__, e)


def _iter_subclasses(base):
    seen = set()
    stack = list(base.__subclasses__())
    while stack:
        cls = stack.pop()
        if id(cls) in seen:
            continue
        seen.add(id(cls))
        yield cls
        stack.extend(cls.__subclasses__())


# ═══════════════════════════════════════════════════════════════════════════
# Instrumentor
# ═══════════════════════════════════════════════════════════════════════════


class A2AInstrumentor(BaseInstrumentor):
    """Instrumentor for the official A2A Python SDK (``a2a-sdk``)."""

    def __init__(self):
        super().__init__()
        self._wrapper = None
        self._base = None
        self._saved_init_subclass = None
        self._wrapped_classes = []

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        from a2a.server.agent_execution import AgentExecutor

        tracer_provider = kwargs.get("tracer_provider")
        tracer = trace_api.get_tracer(
            __name__, "", tracer_provider=tracer_provider
        )
        wrapper = _ExecuteWrapper(tracer)
        self._wrapper = wrapper
        self._base = AgentExecutor

        # 1) Wrap all existing subclasses.
        for cls in _iter_subclasses(AgentExecutor):
            _wrap_execute(cls, wrapper)
            self._wrapped_classes.append(cls)

        # 2) Hook future subclasses via __init_subclass__.
        saved = AgentExecutor.__dict__.get("__init_subclass__")
        self._saved_init_subclass = saved

        def _new_init_subclass(cls, **kw):
            # Preserve any original __init_subclass__ behaviour first.
            try:
                if saved is not None:
                    saved.__func__(cls, **kw)
            except Exception:  # pragma: no cover - defensive
                pass
            _wrap_execute(cls, wrapper)

        AgentExecutor.__init_subclass__ = classmethod(_new_init_subclass)

    def _uninstrument(self, **kwargs: Any) -> None:
        base = self._base
        # Unwrap everything we touched, plus any subclass that carries the
        # sentinel (covers classes created via the __init_subclass__ hook).
        if base is not None:
            classes = set(self._wrapped_classes)
            classes.update(_iter_subclasses(base))
            for cls in classes:
                _unwrap_execute(cls)

            # Restore __init_subclass__.
            if self._saved_init_subclass is not None:
                base.__init_subclass__ = self._saved_init_subclass
            else:
                # Remove our override so it falls back to object's default.
                try:
                    del base.__dict__["__init_subclass__"]
                except (KeyError, TypeError):
                    try:
                        base.__init_subclass__ = classmethod(
                            lambda cls, **kw: None
                        )
                    except Exception:  # pragma: no cover - defensive
                        pass

        self._wrapper = None
        self._base = None
        self._saved_init_subclass = None
        self._wrapped_classes = []
