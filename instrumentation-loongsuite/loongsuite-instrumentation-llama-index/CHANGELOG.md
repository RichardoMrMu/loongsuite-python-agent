# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- Initial release of `loongsuite-instrumentation-llama-index`: automatic
  instrumentation for LlamaIndex (`llama-index-core`) via its native
  instrumentation dispatcher, projecting LlamaIndex spans/events onto
  OpenTelemetry spans that follow the ARMS gen-ai semantic conventions
  (LLM / EMBEDDING / RETRIEVER / RERANKER / TASK / CHAIN / AGENT), with
  parent/child trace relationships preserved from `parent_span_id`.
  ([#18](https://github.com/alibaba/loongsuite-python/issues/18))
