"""Tests for the retriever and retrieve-then-generate chain (src/rag_chain.py).

Gemini is always faked, so no API key or quota is used.

Run from the repo root:
    pytest -v tests/test_rag_chain.py               # all (the slow one builds a small Chroma store)
    pytest -v tests/test_rag_chain.py -m "not slow"
"""
import json
import sys
from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import rag_chain  # noqa: E402
from config import GeminiConfig  # noqa: E402


def _doc(paper_id, page, end_page=None, text="text"):
    return Document(
        page_content=text,
        metadata={
            "paper_id": paper_id,
            "title": f"Paper {paper_id}",
            "page_number": page,
            "end_page": end_page or page,
            "id": f"{paper_id}_0000",
            "distance": 0.2,
        },
    )


DOCS = [_doc("2005.11401", 3, text="RAG-Sequence uses one document."), _doc("2106.09685", 4, 5), _doc("2205.14135", 1)]


class FixedRetriever(BaseRetriever):
    docs: list
    seen: list = []

    def _get_relevant_documents(self, query, *, run_manager):
        self.seen.append(query)
        return self.docs


def fake_llm(reply, prompts=None):
    """A chat 'model' that records the prompt it was sent and returns `reply`."""
    def call(prompt_value):
        if prompts is not None:
            prompts.append(prompt_value.to_string())
        return AIMessage(content=reply)
    return RunnableLambda(call)


# ---------------------------------------------------------------- citations
@pytest.mark.parametrize("text, expected", [
    ("Claim [1]. Other claim [3].", [1, 3]),
    ("Both [2, 1] say so [1][2].", [2, 1]),     # order of first use, no duplicates
    ("Cites [7] and [0] which don't exist [2].", [2]),  # invented numbers dropped
    ("The provided sources do not contain this information.", []),
])
def test_parse_citations(text, expected):
    citations = rag_chain.parse_citations(text, DOCS)
    assert [c["source"] for c in citations] == expected
    for c in citations:
        m = DOCS[c["source"] - 1].metadata
        assert (c["paper_id"], c["page_number"], c["end_page"]) == (m["paper_id"], m["page_number"], m["end_page"])


# ---------------------------------------------------------------- chain wiring
def test_chain_sends_question_and_numbered_sources_to_llm():
    prompts = []
    retriever = FixedRetriever(docs=DOCS, seen=[])
    chain = rag_chain.build_chain(retriever, fake_llm("RAG-Sequence uses one document [1].", prompts))
    out = chain.invoke({"question": "What is RAG-Sequence?"})

    assert retriever.seen == ["What is RAG-Sequence?"]
    assert out["docs"] == DOCS
    assert out["answer"] == "RAG-Sequence uses one document [1]."
    sent = prompts[0]
    assert "What is RAG-Sequence?" in sent
    assert "[1] Paper 2005.11401 (2005.11401, p.3)" in sent
    assert "[2] Paper 2106.09685 (2106.09685, pp.4-5)" in sent
    assert "Answer the question using ONLY the sources provided" in sent  # Niv's system prompt


def test_gemini_call_retried_on_rate_limit_but_retrieval_is_not():
    class RateLimited(Exception):
        code = 429

    calls = {"llm": 0}

    def flaky(prompt_value):
        calls["llm"] += 1
        if calls["llm"] == 1:
            raise RateLimited("429 Resource exhausted")
        return AIMessage(content="Answer [1].")

    retriever = FixedRetriever(docs=DOCS, seen=[])
    fast = GeminiConfig(api_key="test", wait_min=0, wait_max=0.01, wait_multiplier=0.001)
    out = rag_chain.build_chain(retriever, RunnableLambda(flaky), retry_config=fast).invoke({"question": "q"})

    assert out["answer"] == "Answer [1]."
    assert calls["llm"] == 2
    assert len(retriever.seen) == 1


def test_answer_without_key_needs_injected_llm(monkeypatch):
    """answer() only builds Gemini (and needs the key) when no llm is passed."""
    monkeypatch.setattr(rag_chain, "ChromaRetriever", lambda **kw: FixedRetriever(docs=DOCS[:kw["k"]], seen=[]))
    result = rag_chain.answer("q", k=2, llm=fake_llm("From LoRA [2]."))

    assert result["answer"] == "From LoRA [2]."
    assert result["citations"] == [{
        "source": 2, "paper_id": "2106.09685", "title": "Paper 2106.09685", "page_number": 4, "end_page": 5,
    }]
    assert [s["rank"] for s in result["sources"]] == [1, 2]


# ---------------------------------------------------------------- real Chroma retriever
@pytest.mark.slow
def test_chroma_retriever_and_answer_on_small_store(tmp_path):
    import vector_store

    chunks = json.loads((ROOT / "data" / "chunks.json").read_text(encoding="utf-8"))
    keep = {"2106.09685", "2205.14135"}  # LoRA + FlashAttention, first 15 chunks each
    subset = [c for p in keep for c in [c for c in chunks if c["metadata"]["paper_id"] == p][:15]]
    path = tmp_path / "chunks.json"
    path.write_text(json.dumps(subset), encoding="utf-8")
    collection = vector_store.build_collection(chunks_path=path, db_path=tmp_path / "db")

    docs = rag_chain.ChromaRetriever(k=3, collection=collection).invoke("low-rank adaptation of weight matrices")
    assert len(docs) == 3
    assert docs[0].metadata["paper_id"] == "2106.09685"
    assert [d.metadata["distance"] for d in docs] == sorted(d.metadata["distance"] for d in docs)

    only = rag_chain.ChromaRetriever(k=4, where={"paper_id": "2205.14135"}, collection=collection).invoke("weights")
    assert len(only) == 4 and {d.metadata["paper_id"] for d in only} == {"2205.14135"}

    result = rag_chain.answer("What is LoRA?", k=2, llm=fake_llm("LoRA freezes weights [1]."), collection=collection)
    assert len(result["sources"]) == 2
    assert result["citations"][0]["paper_id"] == "2106.09685"
