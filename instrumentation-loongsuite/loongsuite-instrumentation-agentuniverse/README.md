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

User input is recorded on `gen_ai.input.messages` **only when** content capture
is explicitly enabled through the shared GenAI switch used by every loongsuite
instrumentation. The mode is read once, at `instrument()` time. An absent or
invalid value defaults to `NO_CONTENT` (no message content), so sensitive
prompts are never exported without opt-in:

```bash
# Record message content on spans (default is NO_CONTENT):
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_ONLY
```

Only `SPAN_ONLY` and `SPAN_AND_EVENT` write the attribute -- `EVENT_ONLY` does
not, because the bridge never emits events. The conventional
`agent.run(input=...)` value is captured as the user message; when `input` is
absent the remaining call kwargs are serialized as JSON (runtime plumbing such
as `callbacks` is left out).

## Fail-safe telemetry

Instrumentation never changes agent behaviour: a failure while setting an
attribute or capturing content is swallowed and the agent's own exception is
re-raised unchanged by the native wrapper.

## Tests

The suite runs against a real `agentUniverse` install with no stand-in, and
covers three instrumentor paths -- native only, LoongSuite only, and both
enabled -- asserting exactly one span per agent call in every case, plus
content capture, the error path, `async_run` and the instrument/uninstrument
lifecycle:

```bash
python -m pytest tests -v
```
