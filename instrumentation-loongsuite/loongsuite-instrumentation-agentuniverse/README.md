# LoongSuite agentUniverse Instrumentation

OpenTelemetry instrumentation for the
[agentUniverse](https://github.com/alipay/agentUniverse) multi-agent framework
(`agentUniverse`).

agentUniverse ships its own OpenTelemetry `AgentInstrumentor` (0.0.18+), which
already produces one `au.agent.{source}` span per agent execution -- together
with the `au.*` metrics, streaming first-token timing, conversation-memory
recording and error handling. This package does **not** create spans and does
**not** wrap `Agent.run` / `Agent.async_run`. It is a minimal compatibility
bridge that delegates span creation to the native instrumentor and adds the
ARMS/LoongSuite gen-ai **AGENT** semantic conventions (`gen_ai.*`) to the very
same span, so downstream work -- planning, tool calls, knowledge retrieval,
memory access, LLM requests and any LLM/tool instrumentation -- nests
underneath a single span with a shared trace id.

Enabling this package next to the native instrumentor, which is the normal
setup, still yields exactly one span per agent call.

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
native `AgentInstrumentor` this bridge delegates to (`0.0.18+`).

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

Instrumentation covers every agent, including agents whose classes are defined
after `instrument()` is called: the native instrumentor replaces the
module-level wrapper globals read by the `@trace_agent` decorator that the base
`Agent` class already applies to `run` and `async_run`.

## Compatibility with native agentUniverse instrumentation

`AgentUniverseInstrumentor().instrument()` does two things:

1. **Delegates span creation.** It looks for an `AgentInstrumentor` already
   installed -- recognised as a bound method of `AgentInstrumentor` sitting in
   `agentuniverse.base.annotation.trace._agent_wrapper_sync` /
   `_agent_wrapper_async`. If one is active it reuses that live instance
   untouched; only when none is active does the bridge create and instrument
   one, and it then remembers that it owns it.
2. **Adds `gen_ai.*` attributes on the native span.** It patches the three
   static setters on `agentuniverse.base.tracing.otel.instrumentation.agent.
   agent_instrumentor.AgentSpanAttributesSetter` -- `set_input_attributes`,
   `set_success_attributes`, `set_error_attributes` -- so that, immediately
   after the native `au.*` attributes are written, the LoongSuite conventions
   are written onto the same span. Each wrapper calls the original first and
   fails safe.

Both `run` and `async_run` are covered because the native async and sync
wrappers both go through those setters. Native code keeps full ownership of
span lifetime, span status, metrics, streaming first-token timing and memory
recording; `uninstrument()` restores the original setters and, only if this
bridge created the native instrumentor, uninstruments it. Both calls are
sentinelled, so repeating them is a no-op and re-instrumenting afterwards works.

### Span

| Field         | Value                                                       |
| ------------- | ----------------------------------------------------------- |
| Name          | `au.agent.{agent_name}` (set by the native instrumentor)     |
| Kind          | `INTERNAL` (in-process agent work, not an inbound request)   |
| Native attrs  | `au.span.kind=agent`, `au.agent.name`, `au.agent.input`, `au.agent.output`, `au.agent.status`, `au.agent.duration`, `au.agent.pair_id`, `au.agent.streaming`, `au.agent.first_token.duration`, `au.agent.error.type`, `au.agent.error.message`, `au.trace.caller_name`, `au.trace.caller_type`, `au.agent.usage.*` |
| LoongSuite attrs | `gen_ai.span.kind=AGENT`, `gen_ai.operation.name=invoke_agent`, `gen_ai.framework=agentuniverse`, `gen_ai.agent.name={agent_name}` |
| Metrics       | The native `agent_calls_total`, `agent_errors_total`, `agent_call_duration`, `agent_first_token_duration` and `agent_*_tokens` metrics are emitted unchanged. |

`gen_ai.agent.name` comes from the source name the native input setter already
recorded as `au.agent.name`, so the two namespaces can never disagree.

`au.agent.input` and `au.agent.output` are the native content carrier: the
native setters write them unconditionally. They are withheld whenever this
bridge is the one that activated the native instrumentor and content capture is
off -- see [Content capture](#content-capture). Every other `au.*` attribute,
and all of the native metrics, are unaffected.

### Known upstream caveat

`AgentInstrumentor` is a `BaseInstrumentor` singleton, but its `__init__` runs
on every `AgentInstrumentor()` call and resets the saved wrapper originals:

```python
instrumentor = AgentInstrumentor()
instrumentor.instrument()          # saved original = _default_agent_wrapper_sync

AgentInstrumentor()                # same object, but __init__ runs again:
                                   # saved original is now None

AgentInstrumentor().uninstrument()  # sets the trace-module global to None
agent.run(input="hi")              # TypeError: 'NoneType' object is not callable
```

The bridge is immune by construction: it reaches the live instance through the
trace-module globals and only constructs `AgentInstrumentor()` when none is
active, always immediately before instrumenting it. When you uninstrument the
native instrumentor yourself, hold on to the handle you instrumented with
rather than constructing a fresh one.

## Content capture

Content capture is governed by the shared GenAI switch used by every loongsuite
instrumentation. Its mode is read once, at `instrument()` time. An absent or
invalid value defaults to `NO_CONTENT` (no message content), so sensitive
prompts are never exported without opt-in:

```bash
# Record message content on spans (default is NO_CONTENT):
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_ONLY
```

Two things are covered by that one switch, because the native span carries user
content in two places:

| | capture off (default) | capture on (`SPAN_ONLY` / `SPAN_AND_EVENT`) |
| --- | --- | --- |
| `gen_ai.input.messages` | not written | written |
| `au.agent.input`, `au.agent.output` (bridge-owned native) | not written | written by the native setter |
| `au.agent.input`, `au.agent.output` (native activated by the application) | governed by the application's config | governed by the application's config |

Suppressing `gen_ai.input.messages` alone would not be a privacy guarantee: the
native setter writes the raw prompt to `au.agent.input` and the result to
`au.agent.output` unconditionally, and would still export them. So when this
bridge activated the native instrumentor, it hands the two content-bearing
setters a redacting view of the span that drops exactly those two attributes
while the agent identity, caller, timing, status, pairing and token usage are
all still recorded. Content capture is all-or-nothing: either the prompt and
the result are both on the span, or neither is.

Only `SPAN_ONLY` and `SPAN_AND_EVENT` count as "on" -- `EVENT_ONLY` does not,
because the bridge never emits events. The conventional `agent.run(input=...)`
value is captured as the user message; when `input` is absent the remaining call
kwargs are serialized as JSON (runtime plumbing such as `callbacks` is left out).

**A native instrumentor the application activated itself is never filtered.**
When the native instrumentor is pre-activated by the application (e.g. via
TelemetryManager), its `au.agent.input/output` behavior is governed by the
application's config; the LoongSuite bridge does not override it. The bridge
only ever changes the behaviour of an instrumentor it created, which is why the
two ownership modes are called out separately above.

## Session propagation

Session ID propagation (`au.trace.session.id`, `AUSessionPropagator`) requires
agentUniverse's `TelemetryManager.init_from_config()`. The LoongSuite bridge
activates only the Agent instrumentor; applications that need session
propagation should use TelemetryManager or register the processor/propagator
separately. The bridge deliberately never touches global propagator or tracer
provider state, and its unit tests assert that the global propagator is
unchanged across `instrument()` / `uninstrument()`.

## Fail-safe telemetry

Instrumentation never changes agent behaviour: a failure while setting an
attribute or capturing content is swallowed and the agent's own exception is
re-raised unchanged by the native wrapper.

## Tests

The suite runs against a real `agentUniverse` install with no stand-in, and
covers three instrumentor paths -- native only, LoongSuite only, and both
enabled -- asserting exactly one span per agent call in every case. It also
covers content capture and the privacy policy in both ownership modes, the
error path, `async_run`, and the instrument/uninstrument lifecycle.

The native metrics are asserted directly through an `InMemoryMetricReader`:
each of `agent_calls_total`, `agent_call_duration`, the `agent_*_tokens`
histograms and `agent_first_token_duration` is recorded exactly once, and
`agent_calls_total` reads 1 on the both-enabled path, which is what proves the
native instrumentor was not instrumented a second time. Streaming first-token
timing is exercised with a real `queue.Queue` `output_stream`, and the
`au.agent.usage.*` attributes are asserted to keep their native shape.

```bash
python -m pytest tests -v
```
