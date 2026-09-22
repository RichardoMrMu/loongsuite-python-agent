# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- Initial release of `loongsuite-instrumentation-a2a`: automatic
  instrumentation for the official A2A (Agent2Agent) Python SDK (`a2a-sdk`).
  Produces an ARMS gen-ai `AGENT` span around each server-side
  `AgentExecutor.execute` invocation (covering both existing and
  future-defined executor subclasses), complementing — rather than
  duplicating — the SDK's built-in transport/request-handler tracing.
  ([#28](https://github.com/alibaba/loongsuite-python/issues/28))
