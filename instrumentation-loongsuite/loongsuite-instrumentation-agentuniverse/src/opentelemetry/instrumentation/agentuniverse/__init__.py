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

"""OpenTelemetry agentUniverse instrumentation: a bridge onto the framework's own.

agentUniverse ships an OpenTelemetry ``AgentInstrumentor`` (0.0.18+) that already
creates one ``au.agent.{source}`` INTERNAL span per ``Agent.run`` /
``Agent.async_run`` call, owning the ``au.*`` attributes, metrics, streaming
first-token timing, memory recording and error handling. This package therefore
creates no span and wraps neither method: that would add a duplicate span around
every agent call whenever the native instrumentor is enabled too, which is the
normal setup. It reuses an active native instrumentor -- or creates one when none
is active -- and patches the native ``AgentSpanAttributesSetter`` statics so the
LoongSuite GenAI conventions (``gen_ai.*``) land on the *same* span, right after
the native ``au.*`` attributes.

Content capture
---------------
``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`` is read once, at
``instrument()`` time, through the shared GenAI util. Only ``SPAN_ONLY`` and
``SPAN_AND_EVENT`` write ``gen_ai.input.messages``.

The native setter also writes the raw prompt to ``au.agent.input`` and the agent
result to ``au.agent.output`` unconditionally, so ``gen_ai.input.messages`` on
its own cannot express "do not capture content". When this bridge is the one
that activated the native instrumentor (``_native_owned``) and capture is off, it
hands the content-bearing setters a redacting span view that drops those two
attributes while every other ``au.*`` attribute is still recorded. A native
instrumentor the application activated itself is never filtered: its
``au.agent.input`` / ``au.agent.output`` behaviour stays governed by the
application's own configuration.

Every attribute write is fail-safe.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import asdict
from typing import Any, Collection, Optional

from opentelemetry.instrumentation.agentuniverse.package import _instruments
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.util.genai.extended_semconv.gen_ai_extended_attributes import (
    GEN_AI_SPAN_KIND,
    GenAiSpanKindValues,
)
from opentelemetry.util.genai.types import (
    ContentCapturingMode,
    InputMessage,
    Text,
)
from opentelemetry.util.genai.utils import (
    gen_ai_json_dumps,
    get_content_capturing_mode,
)

logger = logging.getLogger(__name__)

# GenAI conventions this bridge adds to the native agent span.
_GEN_AI_IDENTITY = {
    "gen_ai.operation.name": "invoke_agent",
    "gen_ai.framework": "agentuniverse",
}
_GEN_AI_AGENT_NAME = "gen_ai.agent.name"
_GEN_AI_INPUT_MESSAGES = "gen_ai.input.messages"
_CONTENT_ON_SPAN_MODES = frozenset(
    {ContentCapturingMode.SPAN_ONLY, ContentCapturingMode.SPAN_AND_EVENT}
)
_NON_INPUT_KEYS = frozenset({"callbacks"})  # runtime plumbing, not user input

# Native instrumentation surface this bridge addresses.
_NATIVE_AGENT_NAME_ATTR = "au.agent.name"
_TRACE_WRAPPER_GLOBALS = ("_agent_wrapper_sync", "_agent_wrapper_async")
_INPUT_SETTER = "set_input_attributes"
_BRIDGED_METHODS = (
    _INPUT_SETTER,
    "set_success_attributes",
    "set_error_attributes",
)
_BRIDGE_MARKER = "_loongsuite_genai_bridge"

# The native attributes that carry user content, and the setters that write
# them. Only these are suppressible; every other ``au.*`` attribute is kept.
_PRIVATE_ATTRS = frozenset({"au.agent.input", "au.agent.output"})
_CONTENT_SETTERS = frozenset({_INPUT_SETTER, "set_success_attributes"})


class _RedactingSpan:
    """A span view that drops the content-bearing attributes.

    The native setters write through ``span.set_attribute``, so passing them this
    proxy while content capture is off keeps the raw prompt (``au.agent.input``)
    and the agent result (``au.agent.output``) off the span, while every other
    ``au.*`` attribute is still recorded.
    """

    __slots__ = ("_span",)

    def __init__(self, span: Any) -> None:
        self._span = span

    def set_attribute(self, key: str, value: Any) -> None:
        if key not in _PRIVATE_ATTRS:
            self._span.set_attribute(key, value)

    def __getattr__(self, name: str) -> Any:
        # Anything else the setter reaches for -- set_status, is_recording, ...
        return getattr(self._span, name)


def _safe_set(span: Any, key: str, value: Any) -> None:
    try:
        span.set_attribute(key, value)
    except Exception:  # defensive: instrumentation must never break the app
        logger.debug("could not set %s on the agent span", key, exc_info=True)


def _find_active_native(trace_module: Any, native_cls: type) -> Optional[Any]:
    """The native AgentInstrumentor already installed, if any."""
    for name in _TRACE_WRAPPER_GLOBALS:
        owner = getattr(getattr(trace_module, name, None), "__self__", None)
        if isinstance(owner, native_cls):
            return owner
    return None


def _native_agent_name(span: Any) -> Optional[str]:
    """The agent name the native input setter already recorded on the span."""
    try:
        name = (getattr(span, "attributes", None) or {}).get(
            _NATIVE_AGENT_NAME_ATTR
        )
    except Exception:  # defensive: instrumentation must never break the app
        return None
    return str(name) if name else None


def _set_common_attributes(span: Any, source_name: Any = None) -> None:
    """Write the LoongSuite GenAI identity attributes for an agent span."""
    _safe_set(span, GEN_AI_SPAN_KIND, GenAiSpanKindValues.AGENT.value)
    for key, value in _GEN_AI_IDENTITY.items():
        _safe_set(span, key, value)
    name = source_name or _native_agent_name(span)
    if name:
        _safe_set(span, _GEN_AI_AGENT_NAME, str(name))


def _extract_user_input(input_params: Any) -> Optional[str]:
    """Recover the user input from the native input params.

    The native ``_get_input`` binds ``run(self, **kwargs)``, so the payload is
    normally nested under ``kwargs``. ``input`` wins when present; otherwise
    the remaining parameters are serialized, minus runtime plumbing.
    """
    if not isinstance(input_params, Mapping):
        return None
    params = input_params.get("kwargs")
    if not isinstance(params, Mapping):
        params = input_params
    if params.get("input") is not None:
        return str(params["input"])
    remaining = {k: v for k, v in params.items() if k not in _NON_INPUT_KEYS}
    try:
        return gen_ai_json_dumps(remaining) if remaining else None
    except Exception:  # defensive: instrumentation must never break the app
        logger.debug("could not serialize the agent input", exc_info=True)
        return None


def _set_input_messages(span: Any, input_params: Any) -> None:
    """Record the user input as ``gen_ai.input.messages`` (capture opted in)."""
    try:
        text = _extract_user_input(input_params)
        if text:
            message = InputMessage(role="user", parts=[Text(content=text)])
            _safe_set(
                span,
                _GEN_AI_INPUT_MESSAGES,
                gen_ai_json_dumps([asdict(message)]),
            )
    except Exception:  # defensive: instrumentation must never break the app
        logger.debug("could not capture the agent input", exc_info=True)


def _make_bridged_setter(
    method_name: str, original: Any, capture: bool, redact: bool
) -> Any:
    """Wrap a native setter so it also writes ``gen_ai.*`` on the same span."""
    # Only the input setter receives the agent name and the input payload.
    captures_input = capture and method_name == _INPUT_SETTER
    # Only the content-bearing setters can leak the prompt or the result.
    redacts_content = redact and method_name in _CONTENT_SETTERS

    def bridged(span, *args):
        # The native setter writes through ``target``; the LoongSuite attributes
        # below always go to the real span so they can never be filtered out.
        target = _RedactingSpan(span) if redacts_content else span
        original(target, *args)
        _set_common_attributes(span, args[0] if captures_input else None)
        if captures_input and len(args) > 1:
            _set_input_messages(span, args[1])

    bridged.__name__ = method_name
    setattr(bridged, _BRIDGE_MARKER, True)
    return bridged


def _install_bridge(
    setter_cls: type, capture: bool, redact: bool
) -> dict[str, Any]:
    """Replace the native setter statics; return the originals to restore.

    Returns an empty mapping when the setters are already bridged, so a later
    ``uninstrument()`` never tears down a bridge this call did not install.
    """
    originals: dict[str, Any] = {}
    for name in _BRIDGED_METHODS:
        descriptor = setter_cls.__dict__.get(name)
        if descriptor is None:
            continue
        original = getattr(setter_cls, name)
        if getattr(original, _BRIDGE_MARKER, False):
            return {}
        originals[name] = descriptor
        setattr(
            setter_cls,
            name,
            staticmethod(
                _make_bridged_setter(name, original, capture, redact)
            ),
        )
    return originals


def _remove_bridge(setter_cls: type, originals: dict[str, Any]) -> None:
    for name, descriptor in originals.items():
        try:
            setattr(setter_cls, name, descriptor)
        except (
            Exception
        ):  # defensive: instrumentation must never break the app
            logger.debug("could not restore %s", name, exc_info=True)


class AgentUniverseInstrumentor(BaseInstrumentor):
    """Compatibility bridge over agentUniverse's native instrumentation.

    ``Agent.run`` / ``Agent.async_run`` are never wrapped: the native
    ``AgentInstrumentor`` owns the span and this instrumentor only augments it.
    """

    def __init__(self):
        super().__init__()
        # BaseInstrumentor.__new__ returns a singleton, so __init__ runs again on
        # every AgentUniverseInstrumentor() call. Seed the bookkeeping once only,
        # or a stray construct-after-instrument would erase the state that
        # uninstrument() needs.
        if not hasattr(self, "_agentuniverse_bridge_ready"):
            self._agentuniverse_bridge_ready = True
            self._native = None
            self._native_owned = False
            self._setter_cls = None
            self._originals: dict[str, Any] = {}
            self._content_mode = ContentCapturingMode.NO_CONTENT
            self._privacy_filter = False

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        try:
            from agentuniverse.base.annotation import trace as trace_module
            from agentuniverse.base.tracing.otel.instrumentation.agent.agent_instrumentor import (  # noqa: E501
                AgentInstrumentor,
                AgentSpanAttributesSetter,
            )
        except Exception:
            logger.warning(
                "agentUniverse native AgentInstrumentor is unavailable; the "
                "LoongSuite compatibility bridge stays inactive",
                exc_info=True,
            )
            return

        try:
            self._content_mode = get_content_capturing_mode()
        except Exception:  # defensive: never break the caller's instrument()
            logger.debug(
                "could not resolve the GenAI capture mode", exc_info=True
            )
            self._content_mode = ContentCapturingMode.NO_CONTENT
        capture = self._content_mode in _CONTENT_ON_SPAN_MODES
        self._setter_cls = AgentSpanAttributesSetter

        # Span creation is delegated: reuse the native instrumentor when the
        # application already enabled it, otherwise own the one we create.
        self._native = _find_active_native(trace_module, AgentInstrumentor)
        self._native_owned = self._native is None
        if self._native_owned:
            self._native = AgentInstrumentor()
            self._native.instrument(
                tracer_provider=kwargs.get("tracer_provider"),
                meter_provider=kwargs.get("meter_provider"),
                skip_dep_check=True,
            )

        # Only a native instrumentor this bridge activated may be filtered: one
        # the application activated itself keeps its own au.agent.input/output
        # contract, whatever the application configured it to do.
        self._privacy_filter = self._native_owned and not capture
        self._originals = _install_bridge(
            AgentSpanAttributesSetter, capture, self._privacy_filter
        )

    def _uninstrument(self, **kwargs: Any) -> None:
        if self._originals and self._setter_cls is not None:
            _remove_bridge(self._setter_cls, self._originals)
        self._originals = {}
        self._setter_cls = None

        # Only tear down the native instrumentor if this bridge created it.
        if self._native_owned and self._native is not None:
            try:
                self._native.uninstrument()
            except (
                Exception
            ):  # defensive: instrumentation must never break the app
                logger.debug(
                    "could not uninstrument the native bridge", exc_info=True
                )
        self._native = None
        self._native_owned = False
        self._content_mode = ContentCapturingMode.NO_CONTENT
        self._privacy_filter = False
