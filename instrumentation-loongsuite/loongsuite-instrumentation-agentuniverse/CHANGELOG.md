# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- Initial release of `loongsuite-instrumentation-agentuniverse`: automatic
  instrumentation for the [agentUniverse](https://github.com/alipay/agentUniverse)
  multi-agent framework (`agentUniverse >= 0.0.19`). Produces an ARMS gen-ai
  `AGENT` span around each agent execution by wrapping the concrete
  `Agent.run` / `Agent.async_run` methods on the base `Agent` class, so every
  agent -- including those defined after `instrument()` -- is covered.
- Span kind and content-capture are sourced from the shared GenAI util
  (`opentelemetry-util-genai`): the span kind comes from `GenAiSpanKindValues`
  and content capture is governed by the standard
  `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` switch (default
  `NO_CONTENT`), so user prompts are never exported without explicit opt-in.
- The AGENT span name uses the agent's configured name
  (`invoke_agent {agent_name}`) and falls back to the concrete class name when
  the agent has no `agent_model.info['name']` yet.
- Telemetry is fail-safe: any error while starting a span or recording span
  attributes/exceptions is swallowed so instrumentation can never interrupt
  agent execution; an agent's own exception is re-raised unchanged.
