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

"""OpenTelemetry agentUniverse instrumentation: a bridge onto the
framework's own.

agentUniverse ships its own OpenTelemetry instrumentors (0.0.18+): one for
``Agent.run`` / ``Agent.async_run``, one for the ``@trace_llm`` decorator and
one for ``@trace_tool``. Each of them creates exactly one INTERNAL span per
invocation -- ``au.agent.{source}``, ``au.llm.{source}``, ``au.tool.{source}``
-- and owns its ``au.*`` attributes, its metrics, streaming first-token timing,
conversation memory recording, token usage aggregation and error handling.

This package therefore creates no span and wraps no framework method: wrapping
them would add a duplicate span around every agent, LLM and tool call whenever
the native instrumentors are enabled, which is the normal setup. Instead, for
each of the three layers, it reuses the native instrumentor that is already
active -- or creates and owns one when that layer is not instrumented -- and
patches that layer's native ``*SpanAttributesSetter`` statics so the LoongSuite
GenAI conventions (``gen_ai.*``) land on the *same* span, right after the
native ``au.*`` attributes.

Ownership and privacy
---------------------
Each layer is bridged independently, so any mixture of pre-activated and
bridge-owned layers works (``owned`` means this bridge created that layer's
native instrumentor):

* bridge-owned layer, content capture off -- the native content carriers
  (``au.agent.input`` / ``au.agent.output``, ``au.llm.input`` /
  ``au.llm.output``, ``au.tool.input`` / ``au.tool.output``) are dropped, and
  no ``gen_ai.*`` content is written either. Structure, status, duration,
  token usage, error and metrics attributes are all kept.
* bridge-owned layer, content capture on -- the native content carriers and the
  ``gen_ai.*`` content are both written, on the same span.
* pre-activated layer (``owned`` false) -- the native ``au.*`` content
  contract belongs to the application and is never filtered. ``gen_ai.*``
  content still follows the capture switch.

``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`` is read once, at
``instrument()`` time, through the shared GenAI util.

Session propagation is out of scope: ``au.trace.session.id`` and
``AUSessionPropagator`` are installed by agentUniverse's own
``TelemetryManager``, not by this bridge.

Every attribute write is fail-safe.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Callable, Collection, Optional

from opentelemetry.instrumentation.agentuniverse.package import _instruments
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.semconv.attributes import error_attributes
from opentelemetry.util.genai.extended_semconv.gen_ai_extended_attributes import (  # noqa: E501
    GEN_AI_RESPONSE_TIME_TO_FIRST_TOKEN,
    GEN_AI_SPAN_KIND,
    GEN_AI_USAGE_TOTAL_TOKENS,
    GenAiSpanKindValues,
)
from opentelemetry.util.genai.types import (
    ContentCapturingMode,
    InputMessage,
    OutputMessage,
    Text,
)
from opentelemetry.util.genai.utils import (
    gen_ai_json_dumps,
    get_content_capturing_mode,
)

logger = logging.getLogger(__name__)

_TRACE_MODULE = "agentuniverse.base.annotation.trace"
_INSTRUMENTATION_ROOT = "agentuniverse.base.tracing.otel.instrumentation"

# GenAI conventions this bridge adds, on top of the native ``au.*`` ones.
_GEN_AI_FRAMEWORK_KEY = "gen_ai.framework"
_GEN_AI_FRAMEWORK_NAME = "agentuniverse"
_CONTENT_ON_SPAN_MODES = frozenset(
    {ContentCapturingMode.SPAN_ONLY, ContentCapturingMode.SPAN_AND_EVENT}
)
_NON_INPUT_KEYS = frozenset({"callbacks"})  # runtime plumbing, not user input
# Not present in every vendored semconv release; the shared util reads them
# defensively the same way.
_GEN_AI_TOOL_CALL_ARGUMENTS = getattr(
    GenAI, "GEN_AI_TOOL_CALL_ARGUMENTS", "gen_ai.tool.call.arguments"
)
_GEN_AI_TOOL_CALL_RESULT = getattr(
    GenAI, "GEN_AI_TOOL_CALL_RESULT", "gen_ai.tool.call.result"
)

# Native setter methods this bridge replaces, and the ones that carry content.
_INPUT_SETTER = "set_input_attributes"
_SUCCESS_SETTER = "set_success_attributes"
_ERROR_SETTER = "set_error_attributes"
_FIRST_TOKEN_SETTER = "set_first_token_attributes"
_BRIDGED_METHODS = (_INPUT_SETTER, _SUCCESS_SETTER, _ERROR_SETTER)
_CONTENT_SETTERS = frozenset({_INPUT_SETTER, _SUCCESS_SETTER})
_BRIDGE_MARKER = "_loongsuite_genai_bridge"

# Native ``au.*`` token counters, mirrored onto the GenAI usage attributes.
_USAGE_MIRROR = (
    ("total_tokens", GEN_AI_USAGE_TOTAL_TOKENS),
    ("prompt_tokens", GenAI.GEN_AI_USAGE_INPUT_TOKENS),
    ("completion_tokens", GenAI.GEN_AI_USAGE_OUTPUT_TOKENS),
)


class _RedactingSpan:
    """A span view that drops one layer's content-bearing attributes.

    The native setters write through ``span.set_attribute``, so passing them
    this proxy while content capture is off keeps the raw input and result off
    the span, while every other ``au.*`` attribute is still recorded. Only the
    native call sees the proxy; this bridge's own ``gen_ai.*`` writes always go
    to the real span.
    """

    __slots__ = ("_span", "_private")

    def __init__(self, span: Any, private: frozenset) -> None:
        self._span = span
        self._private = private

    def set_attribute(self, key: str, value: Any) -> None:
        if key not in self._private:
            self._span.set_attribute(key, value)

    def __getattr__(self, name: str) -> Any:
        # Anything else the setter reaches for -- set_status, is_recording...
        return getattr(self._span, name)


def _safe_set(span: Any, key: str, value: Any) -> None:
    try:
        span.set_attribute(key, value)
    except Exception:  # defensive: instrumentation must never break the app
        logger.debug("could not set %s", key, exc_info=True)


def _attr(span: Any, key: str) -> Any:
    """Read back an attribute the native setter just wrote."""
    try:
        return (getattr(span, "attributes", None) or {}).get(key)
    except Exception:  # defensive: instrumentation must never break the app
        return None


def _find_active_native(
    trace_module: Any, native_cls: type, wrapper_globals: Sequence[str]
) -> Optional[Any]:
    """The native instrumentor already installed for one layer, if any."""
    for name in wrapper_globals:
        owner = getattr(getattr(trace_module, name, None), "__self__", None)
        if isinstance(owner, native_cls):
            return owner
    return None


def _unwrap_params(input_params: Any) -> Mapping[str, Any]:
    """The bound-argument mapping, preferring a lone nested ``kwargs`` payload.

    The native ``_get_input`` binds through ``inspect.signature().bind()``, so
    ``run(self, **kwargs)`` collects every argument under a single ``kwargs``
    key, while ``call(self, prompt, **kwargs)`` leaves an empty ``kwargs`` as a
    sibling of ``prompt``. Only the first shape is a payload to unwrap.
    """
    if not isinstance(input_params, Mapping):
        return {}
    if len(input_params) == 1:
        params = input_params.get("kwargs")
        if isinstance(params, Mapping):
            return params
    return input_params


def _set_identity(span: Any, span_kind: str, operation: str) -> None:
    """Write the GenAI identity attributes shared by every layer."""
    _safe_set(span, GEN_AI_SPAN_KIND, span_kind)
    _safe_set(span, GenAI.GEN_AI_OPERATION_NAME, operation)
    _safe_set(span, _GEN_AI_FRAMEWORK_KEY, _GEN_AI_FRAMEWORK_NAME)


def _mirror_usage(span: Any, native_prefix: str) -> None:
    """Mirror the native token counters onto ``gen_ai.usage.*``.

    Reading the counters the native setter just wrote keeps the two
    namespaces consistent by construction, and non-zero values only, matching
    the shared util's response-attribute helper.
    """
    for suffix, gen_ai_key in _USAGE_MIRROR:
        value = _attr(span, f"{native_prefix}.{suffix}")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if value > 0:
                _safe_set(span, gen_ai_key, value)


def _set_error_type(span: Any, error: Any) -> None:
    """Record ``error.type`` the way the shared util does."""
    if isinstance(error, BaseException):
        _safe_set(span, error_attributes.ERROR_TYPE, type(error).__qualname__)


def _as_input_message(entry: Any) -> Optional[InputMessage]:
    """Map one native chat message onto a GenAI ``InputMessage``."""
    if isinstance(entry, Mapping):
        role, content = entry.get("role"), entry.get("content")
    else:
        role = getattr(entry, "role", None)
        content = getattr(entry, "content", None)
    if content is None:
        return None
    return InputMessage(
        role=str(role or "user"), parts=[Text(content=str(content))]
    )


def _write_messages(span: Any, key: str, messages: Sequence[Any]) -> None:
    if messages:
        _safe_set(span, key, gen_ai_json_dumps([asdict(m) for m in messages]))


def _user_message(content: Any) -> list[InputMessage]:
    if content is None:
        return []
    return [InputMessage(role="user", parts=[Text(content=str(content))])]


def _set_json_attr(span: Any, native_key: str, gen_ai_key: str) -> None:
    """Copy a native JSON content carrier onto its GenAI counterpart."""
    value = _attr(span, native_key)
    if value is not None:
        _safe_set(span, gen_ai_key, value)


def _apply_agent(capture: bool) -> Callable[..., None]:
    """GenAI attributes for one agent span, after the native ``au.*`` ones."""

    def apply(span: Any, method_name: str, args: tuple) -> None:
        _set_identity(span, GenAiSpanKindValues.AGENT.value, "invoke_agent")
        if method_name == _INPUT_SETTER:
            name = args[0] if args else None
            if not name:
                name = _attr(span, "au.agent.name")
            if name:
                _safe_set(span, GenAI.GEN_AI_AGENT_NAME, str(name))
            if capture and len(args) > 1:
                params = _unwrap_params(args[1])
                if params.get("input") is not None:
                    _write_messages(
                        span,
                        GenAI.GEN_AI_INPUT_MESSAGES,
                        _user_message(params["input"]),
                    )
                else:
                    remaining = {
                        key: value
                        for key, value in params.items()
                        if key not in _NON_INPUT_KEYS
                    }
                    _write_messages(
                        span,
                        GenAI.GEN_AI_INPUT_MESSAGES,
                        _user_message(
                            gen_ai_json_dumps(remaining) if remaining else None
                        ),
                    )
        elif method_name == _ERROR_SETTER:
            _set_error_type(span, args[0] if args else None)
        _mirror_usage(span, "au.agent.usage")

    return apply


def _apply_llm(capture: bool) -> Callable[..., None]:
    """GenAI attributes for one LLM span, after the native ``au.*`` ones."""

    def apply(span: Any, method_name: str, args: tuple) -> None:
        _set_identity(span, GenAiSpanKindValues.LLM.value, "chat")
        if method_name == _INPUT_SETTER:
            # (span, source_name, channel_name, input_params, llm_params,
            #  caller_info)
            params = args[3] if len(args) > 3 else None
            if isinstance(params, Mapping):
                temperature = params.get("temperature")
                # The native params carry -1 when the temperature is unknown.
                if (
                    isinstance(temperature, (int, float))
                    and not isinstance(temperature, bool)
                    and temperature >= 0
                ):
                    _safe_set(
                        span,
                        GenAI.GEN_AI_REQUEST_TEMPERATURE,
                        float(temperature),
                    )
            if capture and len(args) > 2:
                _write_messages(
                    span,
                    GenAI.GEN_AI_INPUT_MESSAGES,
                    _llm_input_messages(args[2]),
                )
        elif method_name == _SUCCESS_SETTER:
            if capture and len(args) > 1:
                _write_messages(
                    span,
                    GenAI.GEN_AI_OUTPUT_MESSAGES,
                    _llm_output_messages(args[1]),
                )
        elif method_name == _ERROR_SETTER:
            _set_error_type(span, args[0] if args else None)
        elif method_name == _FIRST_TOKEN_SETTER:
            duration = args[0] if args else None
            if isinstance(duration, (int, float)) and duration > 0:
                _safe_set(
                    span,
                    GEN_AI_RESPONSE_TIME_TO_FIRST_TOKEN,
                    int(duration * 1_000_000_000),
                )
        _mirror_usage(span, "au.llm.usage")

    return apply


def _llm_input_messages(input_params: Any) -> list[InputMessage]:
    """Map the native LLM call arguments onto GenAI input messages."""
    params = _unwrap_params(input_params)
    if not params:
        return _user_message(input_params)
    messages = params.get("messages")
    if isinstance(messages, (list, tuple)):
        built = [
            message
            for message in (_as_input_message(item) for item in messages)
            if message is not None
        ]
        if built:
            return built
    for key in ("prompt", "input", "query"):
        if params.get(key) is not None:
            return _user_message(params[key])
    remaining = {
        key: value
        for key, value in params.items()
        if key not in _NON_INPUT_KEYS and key != "kwargs"
    }
    return _user_message(gen_ai_json_dumps(remaining) if remaining else None)


def _llm_output_messages(result: Any) -> list[OutputMessage]:
    """Map the native ``LLMOutput`` onto a GenAI output message."""
    text = getattr(result, "text", None)
    if text is None:
        return []
    finish_reason = getattr(result, "finish_reason", None) or "stop"
    return [
        OutputMessage(
            role="assistant",
            parts=[Text(content=str(text))],
            finish_reason=str(finish_reason),
        )
    ]


def _apply_tool(capture: bool) -> Callable[..., None]:
    """GenAI attributes for one tool span, after the native ``au.*`` ones."""

    def apply(span: Any, method_name: str, args: tuple) -> None:
        _set_identity(span, GenAiSpanKindValues.TOOL.value, "execute_tool")
        if method_name == _INPUT_SETTER:
            # (span, source_name, input_params, caller_info, pair_id)
            name = args[0] if args else None
            if name:
                _safe_set(span, GenAI.GEN_AI_TOOL_NAME, str(name))
            _safe_set(span, GenAI.GEN_AI_TOOL_TYPE, "function")
            # set_input_attributes(span, source_name, input_params,
            #                      caller_info, pair_id)
            pair_id = args[3] if len(args) > 3 else None
            if pair_id:
                _safe_set(span, GenAI.GEN_AI_TOOL_CALL_ID, str(pair_id))
            if capture:
                _set_json_attr(
                    span, "au.tool.input", _GEN_AI_TOOL_CALL_ARGUMENTS
                )
        elif method_name == _SUCCESS_SETTER:
            if capture:
                _set_json_attr(
                    span, "au.tool.output", _GEN_AI_TOOL_CALL_RESULT
                )
        elif method_name == _ERROR_SETTER:
            _set_error_type(span, args[0] if args else None)
        _mirror_usage(span, "au.tool.usage")

    return apply


@dataclass(frozen=True)
class _LayerSpec:
    """One native instrumentor layer this bridge augments."""

    key: str
    instrumentor_path: str
    setter_path: str
    wrapper_globals: tuple
    private_attrs: frozenset
    applier: Callable[[bool], Callable[..., None]]
    bridged_methods: tuple = _BRIDGED_METHODS


_LAYER_SPECS = (
    _LayerSpec(
        key="agent",
        instrumentor_path=(
            f"{_INSTRUMENTATION_ROOT}.agent.agent_instrumentor."
            "AgentInstrumentor"
        ),
        setter_path=(
            f"{_INSTRUMENTATION_ROOT}.agent.agent_instrumentor."
            "AgentSpanAttributesSetter"
        ),
        wrapper_globals=("_agent_wrapper_sync", "_agent_wrapper_async"),
        private_attrs=frozenset({"au.agent.input", "au.agent.output"}),
        applier=_apply_agent,
    ),
    _LayerSpec(
        key="llm",
        instrumentor_path=(
            f"{_INSTRUMENTATION_ROOT}.llm.llm_instrumentor.LLMInstrumentor"
        ),
        setter_path=(
            f"{_INSTRUMENTATION_ROOT}.llm.llm_instrumentor."
            "LLMSpanAttributesSetter"
        ),
        wrapper_globals=("_llm_wrapper_sync", "_llm_wrapper_async"),
        private_attrs=frozenset({"au.llm.input", "au.llm.output"}),
        applier=_apply_llm,
        bridged_methods=_BRIDGED_METHODS + (_FIRST_TOKEN_SETTER,),
    ),
    _LayerSpec(
        key="tool",
        instrumentor_path=(
            f"{_INSTRUMENTATION_ROOT}.tool.tool_instrumentor.ToolInstrumentor"
        ),
        setter_path=(
            f"{_INSTRUMENTATION_ROOT}.tool.tool_instrumentor."
            "ToolSpanAttributesSetter"
        ),
        wrapper_globals=("_tool_wrapper_sync", "_tool_wrapper_async"),
        private_attrs=frozenset({"au.tool.input", "au.tool.output"}),
        applier=_apply_tool,
    ),
)


def _make_bridged_setter(
    method_name: str,
    original: Any,
    *,
    redact: bool,
    private: frozenset,
    apply: Callable[..., None],
) -> Any:
    """Wrap a native setter so it also writes ``gen_ai.*`` on the same span."""
    redacts_content = redact and method_name in _CONTENT_SETTERS

    def bridged(span, *args):
        # The native setter writes through ``target``; the LoongSuite
        # attributes below always go to the real span so they can never be
        # filtered out.
        target = _RedactingSpan(span, private) if redacts_content else span
        original(target, *args)
        try:
            apply(span, method_name, args)
        except Exception:  # defensive: never break the app
            logger.debug("could not add the gen_ai attributes", exc_info=True)

    bridged.__name__ = method_name
    setattr(bridged, _BRIDGE_MARKER, True)
    return bridged


def _install_layer_bridge(
    setter_cls: type,
    spec: _LayerSpec,
    *,
    capture: bool,
    redact: bool,
) -> dict:
    """Replace one layer's native setter statics; return the originals.

    Returns an empty mapping when the setters are already bridged, so a later
    ``uninstrument()`` never tears down a bridge this call did not install.
    """
    originals: dict = {}
    apply = spec.applier(capture)
    for name in spec.bridged_methods:
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
                _make_bridged_setter(
                    name,
                    original,
                    redact=redact,
                    private=spec.private_attrs,
                    apply=apply,
                )
            ),
        )
    return originals


def _remove_bridge(setter_cls: type, originals: dict) -> None:
    for name, descriptor in originals.items():
        try:
            setattr(setter_cls, name, descriptor)
        except Exception:  # defensive: never break the app
            logger.debug("could not restore %s", name, exc_info=True)


def _import_member(path: str) -> Any:
    module_name, _, member = path.rpartition(".")
    return getattr(importlib.import_module(module_name), member)


class _LayerBridge:
    """The bridge state for one native instrumentor layer."""

    def __init__(self, spec: _LayerSpec) -> None:
        self._spec = spec
        self._native: Optional[Any] = None
        self._owned = False
        self._setter_cls: Optional[type] = None
        self._originals: dict = {}

    @property
    def key(self) -> str:
        return self._spec.key

    @property
    def owned(self) -> bool:
        """Whether this bridge created the layer's native instrumentor."""
        return self._owned

    @property
    def native(self) -> Optional[Any]:
        return self._native

    def install(
        self,
        *,
        capture: bool,
        tracer_provider: Any = None,
        meter_provider: Any = None,
    ) -> None:
        try:
            trace_module = importlib.import_module(_TRACE_MODULE)
            native_cls = _import_member(self._spec.instrumentor_path)
            setter_cls = _import_member(self._spec.setter_path)
        except Exception:
            logger.warning(
                "agentUniverse native %s instrumentation is unavailable; the "
                "LoongSuite compatibility bridge stays inactive for that "
                "layer",
                self._spec.key,
                exc_info=True,
            )
            return

        self._setter_cls = setter_cls

        # Span creation is delegated: reuse the native instrumentor when the
        # application already enabled it, otherwise own the one we create.
        # The native classes reuse a ``BaseInstrumentor`` singleton whose
        # ``__init__`` resets the saved wrapper originals, so a second
        # construction would break a later ``uninstrument()``.
        self._native = _find_active_native(
            trace_module, native_cls, self._spec.wrapper_globals
        )
        self._owned = self._native is None
        if self._owned:
            self._native = native_cls()
            self._native.instrument(
                tracer_provider=tracer_provider,
                meter_provider=meter_provider,
                skip_dep_check=True,
            )

        # Only a native instrumentor this bridge activated may be filtered:
        # one the application activated itself keeps its own content
        # contract, whatever the application configured it to do.
        self._originals = _install_layer_bridge(
            setter_cls,
            self._spec,
            capture=capture,
            redact=self._owned and not capture,
        )

    def uninstall(self) -> None:
        if self._originals and self._setter_cls is not None:
            _remove_bridge(self._setter_cls, self._originals)
        self._originals = {}
        self._setter_cls = None

        # Only tear down the native instrumentor if this bridge created it.
        if self._owned and self._native is not None:
            try:
                self._native.uninstrument()
            except Exception:  # defensive: never break the app
                logger.debug(
                    "could not uninstrument the native %s layer",
                    self._spec.key,
                    exc_info=True,
                )
        self._native = None
        self._owned = False


class AgentUniverseInstrumentor(BaseInstrumentor):
    """Compatibility bridge over agentUniverse's native instrumentation.

    ``Agent.run`` / ``Agent.async_run``, ``@trace_llm`` and ``@trace_tool`` are
    never wrapped: the native instrumentors own the spans and this instrumentor
    only augments them with the LoongSuite GenAI conventions.
    """

    def __init__(self):
        super().__init__()
        # BaseInstrumentor.__new__ returns a singleton, so __init__ runs again
        # on every AgentUniverseInstrumentor() call. Seed the bookkeeping once
        # only, or a stray construct-after-instrument would erase the state
        # that uninstrument() needs.
        if not hasattr(self, "_agentuniverse_bridge_ready"):
            self._agentuniverse_bridge_ready = True
            self._layers: list = []
            self._content_mode = ContentCapturingMode.NO_CONTENT

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        try:
            self._content_mode = get_content_capturing_mode()
        except Exception:  # defensive: never break the caller's instrument()
            logger.debug(
                "could not resolve the GenAI capture mode", exc_info=True
            )
            self._content_mode = ContentCapturingMode.NO_CONTENT
        capture = self._content_mode in _CONTENT_ON_SPAN_MODES

        self._layers = []
        for spec in _LAYER_SPECS:
            layer = _LayerBridge(spec)
            layer.install(
                capture=capture,
                tracer_provider=kwargs.get("tracer_provider"),
                meter_provider=kwargs.get("meter_provider"),
            )
            self._layers.append(layer)

    def _uninstrument(self, **kwargs: Any) -> None:
        # Unwind in reverse order, so a partial failure cannot leave a layer
        # patched without its native instrumentor being tracked.
        for layer in reversed(self._layers):
            layer.uninstall()
        self._layers = []
        self._content_mode = ContentCapturingMode.NO_CONTENT
