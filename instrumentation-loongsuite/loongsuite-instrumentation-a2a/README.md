# LoongSuite A2A Instrumentation

OpenTelemetry instrumentation for the official
[A2A (Agent2Agent) Python SDK](https://github.com/a2aproject/a2a-python)
(`a2a-sdk`).

This package adds an ARMS gen-ai **AGENT** span around each server-side agent
turn, bracketing the user's `AgentExecutor.execute` implementation so that all
downstream work (the SDK's own transport/request-handler spans, plus any LLM
or tool instrumentation) nests underneath a single `invoke_agent` span with a
shared trace id.

It is **complementary** to the tracing already built into `a2a-sdk`
(`a2a.utils.telemetry`): the SDK traces protocol plumbing under the
`a2a-python-sdk` instrumenting module with generic span names and no gen-ai
semantic conventions, and it does not wrap the user's `execute` method. This
package supplies exactly that missing gen-ai agent boundary.

## Installation

```bash
pip install loongsuite-instrumentation-a2a
```

## Usage

```python
from opentelemetry.instrumentation.a2a import A2AInstrumentor

A2AInstrumentor().instrument()
```

Instrumentation covers both `AgentExecutor` subclasses that already exist when
`instrument()` is called and any defined afterwards (via an
`__init_subclass__` hook installed on `AgentExecutor`).

## Content capture

The user's input message is captured on `gen_ai.input.messages` by default. To
suppress it while keeping the structural AGENT span:

```bash
export OTEL_INSTRUMENTATION_A2A_CAPTURE_CONTENT=false
```
