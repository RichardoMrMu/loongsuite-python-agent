# LoongSuite agentUniverse Instrumentation

OpenTelemetry instrumentation for the
[agentUniverse](https://github.com/alipay/agentUniverse) multi-agent framework
(`agentUniverse`).

agentUniverse ships its own OpenTelemetry instrumentors (0.0.18+): one for
`Agent.run` / `Agent.async_run`, one for the `@trace_llm` decorator and one for
`@trace_tool`. Each already produces exactly one INTERNAL span per invocation --
`au.agent.{source}`, `au.llm.{source}`, `au.tool.{source}` -- together with the
`au.*` metrics, streaming first-token timing, conversation-memory recording,
token usage aggregation and error handling.

This package does **not** create spans and does **not** wrap `Agent.run`,
`Agent.async_run`, `@trace_llm` or `@trace_tool`. It is a minimal compatibility
bridge across all three layers: for each one it delegates span creation to the
native instrumentor and adds the ARMS/LoongSuite gen-ai semantic conventions
(`gen_ai.*`) to that very same span, so a whole agent call tree -- agent, LLM
requests, tool calls -- carries one span per layer, each with both namespaces.

Enabling this package next to the native instrumentors, which is the normal
setup, still yields exactly one span per layer, and one data point per native
metric.

## Requirements

The instrumentation itself is pure Python and runs on Python 3.10+. It needs
the `agentUniverse` distribution to import, and agentUniverse 0.0.19 pins
`numpy<2`, `grpcio==1.63.0` and `pyarrow<17`, none of which publish cp313
wheels (numpy 1.x cannot build on 3.13 either). agentUniverse therefore
installs on Python 3.10-3.12 today, and the LoongSuite test matrix mirrors
that range.

`requires-python` is bounded accordingly (`>=3.10,<3.13`), so the declared
range, the classifiers and the tox environments all say the same thing. On
3.13 the `instruments` extra cannot be resolved, and this bridge is inert
without agentUniverse.

`agentUniverse >= 0.0.19` is declared as a dependency, which implies the
native instrumentors this bridge delegates to (`0.0.18+`).

## Installation

```bash
pip install loongsuite-instrumentation-agentuniverse
```

## Usage

```python
from opentelemetry.instrumentation.agentuniverse import (
    AgentUniverseInstrumentor,
)

AgentUniverseInstrumentor().instrument()
```

One call bridges all three layers. Instrumentation covers every agent, LLM
method and tool, including classes defined after `instrument()` is called: each
native instrumentor replaces module-level wrapper globals in
`agentuniverse.base.annotation.trace` that the `@trace_agent`, `@trace_llm` and
`@trace_tool` decorators read at call time.

The application can also enable the native instrumentors first, in any
combination, and instrument the bridge afterwards -- or let agentUniverse's own
`TelemetryManager.init_from_config()` enable all three native instrumentors plus
this bridge by class path:

```python
from agentuniverse.base.tracing.otel.telemetry_manager import TelemetryManager

TelemetryManager().init_from_config({
    "service_name": "my-agent-app",
    "instrumentations": [
        "agentuniverse.base.tracing.otel.instrumentation.llm.llm_instrumentor.LLMInstrumentor",
        "agentuniverse.base.tracing.otel.instrumentation.tool.tool_instrumentor.ToolInstrumentor",
        "agentuniverse.base.tracing.otel.instrumentation.agent.agent_instrumentor.AgentInstrumentor",
        "opentelemetry.instrumentation.agentuniverse:AgentUniverseInstrumentor",
    ],
})
```

## Compatibility with native agentUniverse instrumentation

Every layer is bridged independently, by the same two steps.

1. **Span creation is delegated.** For a layer the bridge looks for a native
   instrumentor already installed, recognised as a bound method of that
   instrumentor class sitting in the layer's wrapper globals
   (`_agent_wrapper_sync` / `_agent_wrapper_async`, `_llm_wrapper_sync` /
   `_llm_wrapper_async`, `_tool_wrapper_sync` / `_tool_wrapper_async` in
   `agentuniverse.base.annotation.trace`). If one is active the bridge reuses
   that live instance untouched; only when none is active does it create and
   instrument one, and it then remembers that it owns it. Ownership is tracked
   per layer, so any mixture of pre-activated and bridge-owned layers works.
2. **`gen_ai.*` attributes are added on the native span.** For each layer the
   bridge patches that layer's native `*SpanAttributesSetter` statics, so that
   immediately after the native `au.*` attributes are written the LoongSuite
   conventions are written onto the same span. Each wrapper calls the original
   setter first and fails safe.

| Layer | Native instrumentor (`...otel.instrumentation.`) | Setter seam patched | Native span |
| --- | --- | --- | --- |
| Agent | `agent.agent_instrumentor.AgentInstrumentor` | `AgentSpanAttributesSetter.set_input_attributes`, `set_success_attributes`, `set_error_attributes` | `au.agent.{name}`, `SpanKind.INTERNAL` |
| LLM | `llm.llm_instrumentor.LLMInstrumentor` | `LLMSpanAttributesSetter.set_input_attributes`, `set_success_attributes`, `set_error_attributes`, `set_first_token_attributes` | `au.llm.{name}`, `SpanKind.INTERNAL` |
| Tool | `tool.tool_instrumentor.ToolInstrumentor` | `ToolSpanAttributesSetter.set_input_attributes`, `set_success_attributes`, `set_error_attributes` | `au.tool.{name}`, `SpanKind.INTERNAL` |

Native code keeps full ownership of span lifetime and status, metrics,
streaming first-token timing, token usage aggregation, conversation memory
recording and error handling, on all three layers. `uninstrument()` restores
the original setters of every layer and uninstruments only the native
instrumentors this bridge created; a pre-activated layer is left running, and
exactly as the application configured it.

Both `run` and `async_run`, and `@trace_llm` / `@trace_tool` on sync and async
methods, are covered, because the native sync and async wrappers both go
through the same setters. Instrument and uninstrument are sentinelled, so
repeating either is a no-op and re-instrumenting afterwards works.

### What the bridge adds

| Layer | gen_ai attributes added (after the native ones) | Native attributes kept |
| --- | --- | --- |
| Agent | `gen_ai.span.kind=AGENT`, `gen_ai.operation.name=invoke_agent`, `gen_ai.framework=agentuniverse`, `gen_ai.agent.name`, `gen_ai.input.messages` (capture on), `gen_ai.usage.total_tokens/input_tokens/output_tokens` | `au.span.kind`, `au.agent.name`, `au.agent.input`, `au.agent.output`, `au.agent.status`, `au.agent.duration`, `au.agent.pair_id`, `au.agent.streaming`, `au.agent.first_token.duration`, `au.agent.error.type`, `au.agent.error.message`, `au.trace.caller_name`, `au.trace.caller_type`, `au.agent.usage.*` |
| LLM | `gen_ai.span.kind=LLM`, `gen_ai.operation.name=chat`, `gen_ai.framework=agentuniverse`, `gen_ai.request.temperature` (only when the caller set one), `gen_ai.input.messages` / `gen_ai.output.messages` (capture on), `gen_ai.response.time_to_first_token`, `gen_ai.usage.*` | `au.span.kind`, `au.llm.name`, `au.llm.channel_name`, `au.llm.input`, `au.llm.output`, `au.llm.llm_params`, `au.llm.streaming`, `au.llm.duration`, `au.llm.status`, `au.llm.first_token.duration`, `au.llm.error.type`, `au.llm.error.message`, `au.trace.caller_name`, `au.trace.caller_type`, `au.llm.usage.*` |
| Tool | `gen_ai.span.kind=TOOL`, `gen_ai.operation.name=execute_tool`, `gen_ai.framework=agentuniverse`, `gen_ai.tool.name`, `gen_ai.tool.type=function`, `gen_ai.tool.call.id`, `gen_ai.tool.call.arguments` / `gen_ai.tool.call.result` (capture on), `gen_ai.usage.*` | `au.span.kind`, `au.tool.name`, `au.tool.input`, `au.tool.output`, `au.tool.duration`, `au.tool.status`, `au.tool.pair_id`, `au.tool.error.type`, `au.tool.error.message`, `au.trace.caller_name`, `au.trace.caller_type`, `au.tool.usage.*` |

`gen_ai.agent.name` comes from the source name the native input setter already
recorded as `au.agent.name`, and `gen_ai.tool.call.id` from the native
`au.tool.pair_id`, so the two namespaces can never disagree about identity.
`gen_ai.usage.*` is mirrored from the `au.*.usage.*` counters the native setter
just wrote, non-zero values only -- which is also why a layer that used no
tokens carries no `gen_ai.usage.*` attributes at all.

The native instrumentors write no `gen_ai.*` attribute themselves, so nothing
here duplicates or overwrites a value the framework already set.

| Layer | Native metrics, emitted unchanged |
| --- | --- |
| Agent | `agent_calls_total`, `agent_errors_total`, `agent_call_duration`, `agent_first_token_duration`, `agent_total_tokens`, `agent_prompt_tokens`, `agent_completion_tokens`, `agent_cached_tokens`, `agent_reasoning_tokens` |
| LLM | `llm_calls_total`, `llm_errors_total`, `llm_call_duration`, `llm_first_token_duration`, `llm_total_tokens`, `llm_prompt_tokens`, `llm_completion_tokens`, `llm_cached_tokens`, `llm_reasoning_tokens` |
| Tool | `tool_calls_total`, `tool_errors_total`, `tool_call_duration`, `tool_total_tokens`, `tool_prompt_tokens`, `tool_completion_tokens`, `tool_cached_tokens`, `tool_reasoning_tokens` |

## Content capture

Content capture is governed by the shared GenAI switch used by every loongsuite
instrumentation. Its mode is read once, at `instrument()` time. An absent or
invalid value defaults to `NO_CONTENT` (no message content), so sensitive
prompts are never exported without opt-in:

```bash
# Record message content on spans (default is NO_CONTENT):
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_ONLY
```

The native span carries user content in two places on each layer, and one
switch covers both:

| | capture off (default) | capture on (`SPAN_ONLY` / `SPAN_AND_EVENT`) |
| --- | --- | --- |
| `gen_ai.input.messages`, `gen_ai.output.messages`, `gen_ai.tool.call.arguments`, `gen_ai.tool.call.result` | not written | written |
| `au.agent.input` / `au.agent.output`, `au.llm.input` / `au.llm.output`, `au.tool.input` / `au.tool.output` (bridge-owned layer) | not written | written by the native setter |
| the same attributes on a layer the application activated itself | governed by the application's config | governed by the application's config |

Suppressing the `gen_ai.*` content alone would not be a privacy guarantee: the
native setters write the raw prompt and the raw result to their `au.*` content
carriers unconditionally, and would still export them. So for a layer this
bridge activated, the content-bearing setters are handed a redacting view of
the span that drops exactly that layer's two content attributes -- while the
identity, caller, timing, status, pairing, token usage and error attributes are
all still recorded, and every metric is untouched. Content capture is
all-or-nothing per layer: either the input and the output are both on the span,
or neither is.

Only `SPAN_ONLY` and `SPAN_AND_EVENT` count as "on" -- `EVENT_ONLY` does not,
because the bridge never emits events. `agent.run(input=...)`, the LLM
`prompt` / `messages` arguments and the tool's arguments are mapped onto the
conventional GenAI messages; when no recognized input key is present the
remaining call arguments are serialized as JSON (runtime plumbing such as
`callbacks` is left out).

**A native instrumentor the application activated itself is never filtered.**
When the native instrumentor is pre-activated by the application (e.g. via
TelemetryManager), its `au.agent.input/output` behavior is governed by the
application's config; the LoongSuite bridge does not override it. The same holds
for the LLM and tool layers. The bridge only ever changes the behaviour of an
instrumentor it created, which is why the two ownership modes are called out
separately above. On such a layer the `gen_ai.*` content still follows the
capture switch.

## Session propagation

Session ID propagation (`au.trace.session.id`, `AUSessionPropagator`) requires
agentUniverse's `TelemetryManager.init_from_config()`. The LoongSuite bridge
activates only the Agent, LLM and Tool instrumentors; it does not register the
`SessionSpanProcessor` or the propagator, and it deliberately never touches
global propagator or tracer provider state. Applications that need session
propagation should use TelemetryManager, or register the processor and the
propagator themselves.

That path is verified for real: a test runs
`TelemetryManager.init_from_config()` with an in-memory exporter in a separate
process, sets the session id through the framework, and asserts that the
agent, LLM and tool spans all carry `au.trace.session.id`, that the propagator
injects both `AU-SessionId` and `auSessionId` into a carrier, and that
extracting a carrier restores the session. Running it out of process keeps the
one-shot global provider and propagator out of the pytest process, which is
also what the bridge's own tests assert -- instrumenting through this package
leaves `trace.get_tracer_provider()` and `propagate.get_global_textmap()`
untouched.

## Token usage

`agentuniverse.llm.llm_output.TokenUsage` is a pydantic-v1 model whose real
fields are `text_in`, `image_in`, `audio_in`, `cached_in`, `text_out`,
`image_out`, `audio_out`, `cached_out` and `reasoning_out`; `prompt_tokens`,
`completion_tokens`, `cached_tokens`, `reasoning_tokens` and `total_tokens`
are read-only derived properties. Constructing
`TokenUsage(prompt=3, completion=5, total=8)` therefore sets nothing at all
(pydantic drops the unknown fields, leaving every counter zero) --
`TokenUsage(text_in=3, text_out=5)` is the non-zero form. The child-LLM
aggregation into the parent agent span and metrics is verified with a real
`@trace_llm` method returning that usage, so the agent span ends up with
`au.agent.usage.total_tokens=8`, `prompt_tokens=3` and `completion_tokens=5`,
and `gen_ai.usage.*` mirrors exactly those numbers.

### Known upstream caveats

These are agentUniverse 0.0.19.1 behaviours the bridge does not change, pinned
by tests so a framework upgrade shows up as a failing test rather than a silent
change:

* **Streaming usage is double counted into the parent.** On the streaming LLM
  path the native instrumentor adds the streamed usage to the LLM span's token
  entry once in `process_sync_stream` / `process_async_stream` and again in
  `_finalize_streaming_result`, so `LLMSpanManager.cleanup()` hands the parent
  twice the usage. The LLM span keeps the real numbers (4 / 6 / 10 for the test
  stream); the parent agent span and the `agent_*_tokens` metrics see 8 / 12 /
  20.
* **A native instrumentor's `__init__` resets its saved state.** The native
  classes are `BaseInstrumentor` singletons, but `__init__` runs on every
  construction and clears the wrapper originals and metric recorder the
  instrumentor is already serving calls with. Constructing one while it is
  active therefore breaks a later `uninstrument()` (it blanks the trace-module
  globals) and can leave a running wrapper with no metrics recorder:

  ```python
  instrumentor = AgentInstrumentor()
  instrumentor.instrument()           # saved original = _default_agent_wrapper_sync

  AgentInstrumentor()                 # same object, but __init__ runs again:
                                      # saved original is now None

  AgentInstrumentor().uninstrument()  # sets the trace-module global to None
  agent.run(input="hi")               # TypeError: 'NoneType' object is not callable
  ```

  The bridge is immune by construction: it reaches the live instance through
  the trace-module globals and only constructs a native instrumentor when that
  layer has none, always immediately before instrumenting it. When you
  uninstrument a native instrumentor yourself, hold on to the handle you
  instrumented with rather than constructing a fresh one.

## Fail-safe telemetry

Instrumentation never changes agent behaviour: a failure while setting an
attribute or capturing content is swallowed and the agent's own exception is
re-raised unchanged by the native wrapper.

## Tests

The suite runs against a real `agentUniverse` install with no stand-in, and
covers every ownership combination: native instrumentors only, LoongSuite only,
both enabled, and partial ownership with one layer pre-activated and the others
bridge-owned. In each setup the call tree is asserted to be exactly one span
per layer -- `au.agent.rich_agent` + `au.llm.stub_llm` + `au.tool.stub_tool` --
with the native `au.*` attributes and, when the bridge is engaged, the
`gen_ai.*` ones on those same spans.

Beyond span shape, the suite asserts the native metrics through an
`InMemoryMetricReader` (each metric recorded exactly once, `*_calls_total` == 1,
the `*_tokens` histograms carrying the real non-zero totals), sync and async
paths on all three layers, streaming first-token timing for the agent and the
LLM (positive duration on the span and in the `*_first_token_duration`
histogram), error status and error metrics, the content-capture matrix in both
ownership modes, the instrument/uninstrument lifecycle including setter and
wrapper-global restoration, and the session path described above.

Mutation checks confirm the tests are load-bearing: neutering the agent, LLM or
tool bridge, disabling the privacy filter, disabling the `gen_ai` usage mirror,
or removing the session id from the session probe each turns the corresponding
tests red (16, 16, 12, 14, 4 and 1 failures respectively), and the suite is
green again once restored.

```bash
python -m pytest tests -v
```
