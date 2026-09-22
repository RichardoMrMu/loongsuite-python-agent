# Copyright The OpenTelemetry Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for LlamaIndexInstrumentor.

Every span-producing assertion has a paired RED check: without the
instrumentor active (before ``instrument`` / after ``uninstrument``), the
same LlamaIndex call produces **zero** OTel spans. This guards against the
test passing for reasons unrelated to the instrumentation.
"""

from __future__ import annotations

import pytest

from opentelemetry.instrumentation.llama_index import (
    _GEN_AI_FRAMEWORK,
    _GEN_AI_OPERATION_NAME,
    _GEN_AI_SPAN_KIND,
    _SPAN_KIND_EMBEDDING,
    _SPAN_KIND_LLM,
    LlamaIndexInstrumentor,
    _classify,
    _span_id_prefix,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_llm():
    from llama_index.core.llms import MockLLM

    return MockLLM(max_tokens=8)


def _chat_once(llm):
    from llama_index.core.llms import ChatMessage

    return llm.chat([ChatMessage(role="user", content="hi there")])


# ---------------------------------------------------------------------------
# Pure classification unit tests (no dispatcher needed)
# ---------------------------------------------------------------------------


def test_span_id_prefix_strips_uuid():
    assert (
        _span_id_prefix("MockLLM.chat-8c7b6315-7a0e-401b-9185-59474d2632c0")
        == "MockLLM.chat"
    )
    assert _span_id_prefix("") == ""


@pytest.mark.parametrize(
    "prefix,expected_kind",
    [
        ("MockLLM.chat", _SPAN_KIND_LLM),
        ("OpenAI.complete", _SPAN_KIND_LLM),
        ("MockEmbedding.get_text_embedding", _SPAN_KIND_EMBEDDING),
        ("VectorIndexRetriever.retrieve", "RETRIEVER"),
        ("LLMRerank.postprocess_nodes", "RERANKER"),
        ("CompactAndRefine.synthesize", "TASK"),
        ("RetrieverQueryEngine.query", "CHAIN"),
        ("ReActAgent.run", "AGENT"),
    ],
)
def test_classify(prefix, expected_kind):
    kind, _op = _classify(prefix)
    assert kind == expected_kind


# ---------------------------------------------------------------------------
# RED: no spans when not instrumented
# ---------------------------------------------------------------------------


def test_red_no_spans_without_instrumentation(span_exporter, tracer_provider):
    """Baseline: driving an LLM chat with NO instrumentor active must not
    produce any OTel spans on our exporter."""
    llm = _mock_llm()
    _chat_once(llm)
    assert span_exporter.get_finished_spans() == ()


# ---------------------------------------------------------------------------
# GREEN: spans appear and nest correctly when instrumented
# ---------------------------------------------------------------------------


def test_green_chat_produces_llm_span(instrument, span_exporter):
    llm = _mock_llm()
    _chat_once(llm)

    spans = span_exporter.get_finished_spans()
    assert len(spans) >= 1

    chat_spans = [
        s
        for s in spans
        if s.attributes.get(_GEN_AI_SPAN_KIND) == _SPAN_KIND_LLM
    ]
    assert chat_spans, f"no LLM span among {[s.name for s in spans]}"
    for s in chat_spans:
        assert s.attributes.get(_GEN_AI_FRAMEWORK) == "llama_index"
        assert s.attributes.get(_GEN_AI_OPERATION_NAME) == "chat"


def test_green_chat_complete_share_trace_and_nest(instrument, span_exporter):
    """MockLLM.chat internally calls MockLLM.complete. The two spans must
    share one trace_id and the complete span must be a child of the chat
    span — proving parent_span_id is faithfully mapped."""
    llm = _mock_llm()
    _chat_once(llm)

    spans = span_exporter.get_finished_spans()
    assert len(spans) >= 2, [s.name for s in spans]

    trace_ids = {s.context.trace_id for s in spans}
    assert len(trace_ids) == 1, f"spans split across traces: {trace_ids}"

    by_span_id = {s.context.span_id: s for s in spans}
    chat = next(s for s in spans if s.name.endswith(".chat"))
    complete = next(s for s in spans if s.name.endswith(".complete"))

    # complete's parent chain must reach the chat span within the same trace.
    assert complete.parent is not None
    assert complete.parent.span_id in by_span_id
    assert complete.context.trace_id == chat.context.trace_id


def test_green_embedding_span(instrument, span_exporter):
    from llama_index.core.embeddings import MockEmbedding

    emb = MockEmbedding(embed_dim=4)
    emb.get_text_embedding("hello world")

    spans = span_exporter.get_finished_spans()
    emb_spans = [
        s
        for s in spans
        if s.attributes.get(_GEN_AI_SPAN_KIND) == _SPAN_KIND_EMBEDDING
    ]
    assert emb_spans, f"no EMBEDDING span among {[s.name for s in spans]}"


# ---------------------------------------------------------------------------
# RED after uninstrument: teardown must stop span production
# ---------------------------------------------------------------------------


def test_red_uninstrument_stops_spans(span_exporter, tracer_provider):
    instrumentor = LlamaIndexInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider, skip_dep_check=True
    )
    llm = _mock_llm()
    _chat_once(llm)
    assert len(span_exporter.get_finished_spans()) >= 1

    instrumentor.uninstrument()
    span_exporter.clear()

    _chat_once(_mock_llm())
    assert span_exporter.get_finished_spans() == (), (
        "spans still produced after uninstrument"
    )


# ---------------------------------------------------------------------------
# Lifecycle: double instrument / uninstrument is safe
# ---------------------------------------------------------------------------


def test_instrument_is_idempotent_on_uninstrument(
    span_exporter, tracer_provider
):
    instrumentor = LlamaIndexInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider, skip_dep_check=True
    )
    instrumentor.uninstrument()
    # second uninstrument must not raise
    instrumentor.uninstrument()
