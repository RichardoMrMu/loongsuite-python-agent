# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- Initial release of `loongsuite-instrumentation-agentuniverse`: instrumentation
  for the [agentUniverse](https://github.com/alipay/agentUniverse) multi-agent
  framework (`agentUniverse >= 0.0.19`).
- Emits the ARMS gen-ai semantic conventions on all three layers of an agent
  call tree, on the span the framework's own instrumentor created:
  - AGENT: `gen_ai.span.kind=AGENT`, `gen_ai.operation.name=invoke_agent`,
    `gen_ai.framework=agentuniverse`, `gen_ai.agent.name`, `gen_ai.usage.*`;
  - LLM: `gen_ai.span.kind=LLM`, `gen_ai.operation.name=chat`,
    `gen_ai.framework=agentuniverse`, `gen_ai.request.temperature` (when the
    caller set one), `gen_ai.response.time_to_first_token`, `gen_ai.usage.*`;
  - TOOL: `gen_ai.span.kind=TOOL`, `gen_ai.operation.name=execute_tool`,
    `gen_ai.framework=agentuniverse`, `gen_ai.tool.name`,
    `gen_ai.tool.type=function`, `gen_ai.tool.call.id`, `gen_ai.usage.*`.
- Content capture is governed by the shared GenAI util
  (`opentelemetry-util-genai`): the span kinds come from `GenAiSpanKindValues`
  and message capture from the standard
  `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` switch (default
  `NO_CONTENT`), so user prompts are never exported without explicit opt-in.
- Content capture covers both places every native span can carry user content.
  `gen_ai.input.messages` / `gen_ai.output.messages` /
  `gen_ai.tool.call.arguments` / `gen_ai.tool.call.result` are not written
  without opt-in, and neither are the native carriers (`au.agent.input` /
  `au.agent.output`, `au.llm.input` / `au.llm.output`, `au.tool.input` /
  `au.tool.output`), which the native setters would otherwise write
  unconditionally. With capture off on a bridge-owned native instrumentor, the
  content attributes of that layer are withheld while its identity, caller,
  timing, status, pairing, `au.*.usage.*` and `error.*` attributes are still
  recorded and every native metric is still emitted.
- Telemetry is fail-safe: any error while setting span attributes or capturing
  content is swallowed so instrumentation can never interrupt agent execution;
  a user exception is re-raised unchanged.

### Changed

- **The instrumentor no longer wraps `Agent.run` / `Agent.async_run`, and it
  does not wrap `@trace_llm` or `@trace_tool` either.** It is a minimal
  compatibility bridge over agentUniverse's own instrumentors
  (`agentuniverse.base.tracing.otel.instrumentation.{agent,llm,tool}`, shipped
  since 0.0.18), which create the `au.agent.{source}`, `au.llm.{source}` and
  `au.tool.{source}` `INTERNAL` spans and emit the `au.*` attributes and
  metrics. The previous direct-wrap implementation produced a duplicate agent
  span whenever the native instrumentor was enabled as well, and bridging only
  the agent layer would have left the LLM and tool spans without the LoongSuite
  conventions.
  - Span creation delegates per layer: an already-active native instrumentor is
    reused as-is, and one is created and instrumented only when that layer has
    none. Ownership is tracked per layer, so pre-activated and bridge-owned
    layers can be mixed freely, in any assembly order.
  - The LoongSuite `gen_ai.*` attributes are added to those same spans by
    patching each layer's `*SpanAttributesSetter` statics so each calls the
    native original first -- `set_input_attributes`, `set_success_attributes`
    and `set_error_attributes` on all three layers, plus
    `set_first_token_attributes` on the LLM layer for
    `gen_ai.response.time_to_first_token`.
  - Span name, span kind, span status, metrics, streaming first-token timing,
    token usage aggregation, conversation-memory recording and error handling
    all remain native and unchanged.
  - `uninstrument()` restores the original setters of every layer and
    uninstruments a native instrumentor only when this bridge created it;
    instrument/uninstrument are sentinel-guarded and idempotent. The nested
    agent/LLM/tool call tree stays exactly one span per layer.
  - **Ownership decides the privacy policy, per layer.** A native instrumentor
    this bridge created is filtered: with capture off, that layer's `au.*`
    content carriers are withheld. A native instrumentor the application
    activated itself (e.g. via `TelemetryManager`) is never modified -- its
    content behavior stays governed by the application's config, and the bridge
    only adds its own `gen_ai.*` attributes to the span.
  - Session propagation is explicitly out of scope and documented as such.
    `au.trace.session.id` and `AUSessionPropagator` come from agentUniverse's
    `TelemetryManager.init_from_config()`; the bridge never touches global
    propagator or tracer provider state, and its tests assert that.
- Tests now run against a real `agentUniverse` 0.0.19.1 install with no
  stand-in, initializing the `ApplicationConfigManager` the native
  `ConversationMemoryModule` requires. They cover the native-only,
  LoongSuite-only, both-enabled and partially-owned paths across all three
  layers, asserting exactly one span per layer in every case -- a typical call
  tree is exactly one agent, one LLM and one tool span.
  - The native metrics are asserted through an `InMemoryMetricReader`: every
    metric is recorded exactly once, `*_calls_total` reads 1, and the
    `*_tokens` histograms carry the real non-zero totals, which is what proves
    no native instrumentor was instrumented twice.
  - Sync and async paths are covered on all three layers; streaming
    first-token timing is exercised with a real `queue.Queue` `output_stream`
    (agent) and a real streaming `@trace_llm` generator (LLM), asserting a
    positive duration both on the span and in the histogram.
  - Non-zero token usage is verified end to end: a real `@trace_llm` child
    returning `TokenUsage(text_in=3, text_out=5)` aggregates into
    `au.agent.usage.total_tokens=8` / `prompt_tokens=3` / `completion_tokens=5`
    on the parent span and into the parent metrics, with `gen_ai.usage.*`
    mirroring the same numbers.
  - The session path is verified for real, in a separate process:
    `TelemetryManager.init_from_config()` registers `SessionSpanProcessor` and
    `AUSessionPropagator`, and the agent, LLM and tool spans all carry
    `au.trace.session.id`, while the propagator injects and extracts both
    `AU-SessionId` and `auSessionId`.
  - Mutation checks confirm the tests are load-bearing: neutering the agent, LLM
    or tool bridge, disabling the privacy filter, disabling the `gen_ai` usage
    mirror, or removing the session id from the session probe each turn the
    corresponding tests red.
- Documented two agentUniverse 0.0.19.1 behaviours the bridge does not change,
  each pinned by a test: streamed token usage is counted twice into the parent
  agent span and metrics (the LLM span keeps the real numbers), and a native
  instrumentor's `__init__` resets its saved state, so it must not be
  constructed while it is already active.
- `wrapt` is no longer a declared dependency. The bridge never used it -- it
  patches the native static setters with `setattr` -- and the only remaining
  consumer, `opentelemetry-instrumentation`, already depends on it.
- `requires-python` is bounded to `>=3.10,<3.13` and the Python 3.13 classifier
  is dropped, matching the tox matrix: agentUniverse's pins (`numpy<2`,
  `grpcio==1.63.0`, `pyarrow<17`) have no cp313 wheels, so the `instruments`
  extra cannot be installed there.
