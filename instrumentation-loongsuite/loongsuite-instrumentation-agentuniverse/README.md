# LoongSuite agentUniverse Instrumentation

OpenTelemetry instrumentation for the
[agentUniverse](https://github.com/alipay/agentUniverse) multi-agent framework
(`agentUniverse`).

This package adds an ARMS gen-ai **AGENT** span around each agent execution,
bracketing `agentuniverse.agent.agent.Agent.run` (and `Agent.async_run`) so that
all downstream work -- planning, tool calls, knowledge retrieval, memory
access, LLM requests and any LLM/tool instrumentation -- nests underneath a
single `invoke_agent` span with a shared trace id.

## Requirements

The instrumentation itself is pure Python and runs on Python 3.10+. It needs
the `agentUniverse` distribution to import, and agentUniverse 0.0.19 pins
`numpy<2`, `grpcio==1.63.0` and `pyarrow<17`, none of which publish cp313
wheels (numpy 1.x cannot build on 3.13 either). agentUniverse therefore
installs on Python 3.10-3.12 today, and the LoongSuite test matrix mirrors
that range.

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
after `instrument()` is called: `run` and `async_run` are concrete methods on
the base `Agent` class, so wrapping the base class is enough.

## Instrumentation seam

agentUniverse does not ask each agent to implement its own execution entry
point. Subclasses provide the pieces (`input_keys`, `output_keys`,
`parse_input`, `parse_result`) and the base class owns the execution skeleton:

```
Agent.run(**kwargs)        # sync entry point  -> invoke_agent AGENT span
Agent.async_run(**kwargs)  # async entry point -> invoke_agent AGENT span
```

Both methods are wrapped with `wrapt` and marked with a sentinel, so
double-wrapping is impossible and `uninstrument()` restores the originals.

## Span

| Field      | Value                                                     |
| ---------- | --------------------------------------------------------- |
| Name       | `invoke_agent {agent_name}`                                |
| Kind       | `INTERNAL` (in-process agent work, not an inbound request) |
| Attributes | `gen_ai.span.kind=AGENT`, `gen_ai.operation.name=invoke_agent`, `gen_ai.framework=agentuniverse`, `gen_ai.agent.name={agent_name}` |

`agent_name` is read from `instance.agent_model.info['name']` and falls back to
the concrete agent class name when the agent has not been initialized from YAML
yet.

## Content capture

User input is recorded on `gen_ai.input.messages` **only when** content capture
is explicitly enabled through the shared GenAI switch used by every loongsuite
instrumentation. An absent or invalid value defaults to `NO_CONTENT` (no
message content), so sensitive prompts are never exported without opt-in:

```bash
# Record message content on spans (default is NO_CONTENT):
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_ONLY
```

The conventional `agent.run(input=...)` value is captured as the user message;
when `input` is absent the remaining call kwargs are serialized as JSON
(runtime plumbing such as `callbacks` is left out).

## Fail-safe telemetry

Instrumentation never changes agent behaviour: a failure while starting a span,
setting attributes or recording an exception is swallowed, and the agent's own
exception is always re-raised unchanged.
