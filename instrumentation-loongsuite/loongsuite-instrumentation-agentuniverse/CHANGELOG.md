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
- Content capture covers both places the native span carries user content.
  `gen_ai.input.messages` is not written without opt-in, and neither are the
  native `au.agent.input` / `au.agent.output`, which the native setter would
  otherwise write unconditionally. With capture off on a bridge-owned native
  instrumentor, both are withheld while `au.span.kind`, `au.agent.name`,
  `au.trace.caller_name`, `au.trace.caller_type`, `au.agent.pair_id`,
  `au.agent.duration`, `au.agent.status`, `au.agent.usage.*` and `error.*` are
  still recorded.
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
  - **Ownership decides the privacy policy.** A native instrumentor this bridge
    created is filtered: with capture off, `au.agent.input` / `au.agent.output`
    are withheld. A native instrumentor the application activated itself (e.g.
    via `TelemetryManager`) is never modified -- its `au.agent.input/output`
    behavior stays governed by the application's config, and the bridge only
    adds its own `gen_ai.*` attributes to the span.
  - Session propagation is explicitly out of scope and documented as such.
    `au.trace.session.id` and `AUSessionPropagator` come from agentUniverse's
    `TelemetryManager.init_from_config()`; the bridge activates only the Agent
    instrumentor and never touches global propagator or tracer provider state.
- Tests now run against a real `agentUniverse` 0.0.19.1 install with no
  stand-in, initializing the `ApplicationConfigManager` the native
  `ConversationMemoryModule` requires. They cover the native-only,
  LoongSuite-only and both-enabled paths plus content capture and the privacy
  policy in both ownership modes, the error path, `async_run` and the
  instrument/uninstrument lifecycle, asserting exactly one span per agent call
  in every case.
  - The native metrics are asserted through an `InMemoryMetricReader`: every
    native metric is recorded exactly once, and `agent_calls_total` reads 1 on
    the both-enabled path, which is what proves the native instrumentor was not
    instrumented twice.
  - Streaming first-token timing is exercised with a real `queue.Queue`
    `output_stream`, and the `au.agent.usage.*` attributes are asserted to keep
    their native shape.
- `wrapt` is no longer a declared dependency. The bridge never used it -- it
  patches the native static setters with `setattr` -- and the only remaining
  consumer, `opentelemetry-instrumentation`, already depends on it.
- `requires-python` is bounded to `>=3.10,<3.13` and the Python 3.13 classifier
  is dropped, matching the tox matrix: agentUniverse's pins (`numpy<2`,
  `grpcio==1.63.0`, `pyarrow<17`) have no cp313 wheels, so the `instruments`
  extra cannot be installed there.
