# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- Initial release of `loongsuite-instrumentation-agentuniverse`: instrumentation
  for the [agentUniverse](https://github.com/alipay/agentUniverse) multi-agent
  framework (`agentUniverse >= 0.0.19`).
- Emits the ARMS gen-ai `AGENT` semantic conventions
  (`gen_ai.span.kind=AGENT`, `gen_ai.operation.name=invoke_agent`,
  `gen_ai.framework=agentuniverse`, `gen_ai.agent.name`) on the agent span.
- Content capture is governed by the shared GenAI util
  (`opentelemetry-util-genai`): the span kind comes from `GenAiSpanKindValues`
  and message capture from the standard
  `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` switch (default
  `NO_CONTENT`), so user prompts are never exported without explicit opt-in.
- Telemetry is fail-safe: any error while setting span attributes or capturing
  content is swallowed so instrumentation can never interrupt agent execution;
  an agent's own exception is re-raised unchanged.

### Changed

- **The instrumentor no longer wraps `Agent.run` / `Agent.async_run`.** It is
  now a minimal compatibility bridge over agentUniverse's own
  `AgentInstrumentor` (`agentuniverse.base.tracing.otel.instrumentation.agent
  .agent_instrumentor.AgentInstrumentor`, shipped since 0.0.18), which creates
  the `au.agent.{source}` `INTERNAL` span and emits the `au.*` attributes and
  metrics. The previous direct-wrap implementation produced a duplicate span
  whenever the native instrumentor was enabled as well.
  - Span creation delegates to the native instrumentor: an already-active
    native `AgentInstrumentor` is reused as-is, and one is created and
    instrumented only when none is active.
  - The LoongSuite `gen_ai.*` attributes are added to that same span by
    patching `AgentSpanAttributesSetter.set_input_attributes`,
    `set_success_attributes` and `set_error_attributes` so each calls the
    native original first. Span name, span kind, span status, metrics,
    streaming first-token timing and conversation-memory recording all remain
    native and unchanged.
  - `uninstrument()` restores the original setters and uninstruments the native
    instrumentor only when this bridge created it; instrument/uninstrument are
    sentinel-guarded and idempotent.
  - The AGENT span is therefore named `au.agent.{agent_name}` (native naming)
    rather than `invoke_agent {agent_name}`, and its status is set by the
    native error path.
- Tests now run against a real `agentUniverse` 0.0.19.1 install with no
  stand-in, initializing the `ApplicationConfigManager` the native
  `ConversationMemoryModule` requires. They cover the native-only,
  LoongSuite-only and both-enabled paths plus content capture, the error path,
  `async_run` and the instrument/uninstrument lifecycle, asserting exactly one
  span per agent call in every case.
