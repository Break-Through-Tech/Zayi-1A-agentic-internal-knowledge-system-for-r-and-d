"""Retrieve-then-generate RAG chain over the Chroma store.

Shared interface:
    from rag_chain import answer
    result = answer("What is the difference between RAG-Sequence and RAG-Token?", k=5)
    result["answer"]     # Gemini's answer, with [n] markers pointing at the sources
    result["citations"]  # the sources the answer cites: paper_id, title, page_number, end_page
    result["sources"]    # every retrieved chunk, in rank order (for debugging / evaluation)

How the pieces connect:
    question -> ChromaRetriever (top-k chunks, vector_store.py)
             -> format_sources + PROMPT (generation.py, Niv)
             -> Gemini with exponential backoff (gemini_client.py, Godwin)
             -> answer text -> citations parsed from its [n] markers

Usage (from the repo root, needs GOOGLE_API_KEY and a built store):
    python src/rag_chain.py "What does LoRA stand for?" -k 5
"""
import argparse
import re
from typing import Any, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough

import vector_store
from generation import PROMPT, format_sources, source_label

DEFAULT_K = 5

# Matches [1], [2, 3] and [1][2]; each bracket may hold a comma-separated list
_CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


class ChromaRetriever(BaseRetriever):
    """LangChain retriever over our Chroma collection.

    Uses vector_store.query(), so questions are embedded with the same model and
    query prefix as the stored chunks. Each Document keeps the chunk's metadata
    plus its Chroma `id` and cosine `distance`.
    """

    k: int = DEFAULT_K
    where: Optional[dict] = None  # Chroma metadata filter, e.g. {"paper_id": "2005.11401"}
    collection: Any = None  # defaults to the store at vector_store.DB_PATH

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        if self.collection is None:
            self.collection = vector_store.get_collection()
        results = vector_store.query(query, k=self.k, where=self.where, collection=self.collection)
        return [
            Document(
                page_content=r["text"],
                metadata={**r["metadata"], "id": r["id"], "distance": r["distance"]},
            )
            for r in results
        ]


def parse_citations(answer_text, docs):
    """Map the [n] markers in the answer to the retrieved docs, in order of first use.

    Numbers outside 1..len(docs) are invented citations and are dropped.
    """
    cited = []
    for match in _CITATION_RE.finditer(answer_text):
        for n in match.group(1).split(","):
            n = int(n)
            if 1 <= n <= len(docs) and n not in cited:
                cited.append(n)
    citations = []
    for n in cited:
        m = docs[n - 1].metadata
        citations.append({
            "source": n,
            "paper_id": m["paper_id"],
            "title": m["title"],
            "page_number": m["page_number"],
            "end_page": m["end_page"],
        })
    return citations


def build_chain(retriever: BaseRetriever, llm: Runnable, retry_config=None) -> Runnable:
    """{"question"} -> {"question", "docs", "answer"}.

    If `retry_config` (a GeminiConfig) is given, only the Gemini call is retried
    with backoff; retrieval is local and never retried.
    """
    generate = PROMPT | llm | StrOutputParser()
    if retry_config is not None:
        from gemini_client import invoke_with_retry

        generate_step = RunnableLambda(lambda inputs: invoke_with_retry(generate, inputs, retry_config))
    else:
        generate_step = generate

    to_prompt_inputs = RunnableLambda(
        lambda x: {"question": x["question"], "context": format_sources(x["docs"])}
    )
    return (
        RunnablePassthrough.assign(docs=RunnableLambda(lambda x: x["question"]) | retriever)
        .assign(answer=to_prompt_inputs | generate_step)
    )


_default_llm = None
_default_config = None


def _gemini():
    """Gemini model + retry config, built once (needs GOOGLE_API_KEY)."""
    global _default_llm, _default_config
    if _default_llm is None:
        from config import load_config
        from gemini_client import get_llm

        _default_config = load_config()
        _default_llm = get_llm(_default_config)
    return _default_llm, _default_config


def answer(question, k=DEFAULT_K, *, where=None, llm=None, retry_config=None, collection=None):
    """Answer a question from the papers, with citations (paper ID and page).

    `llm` defaults to Gemini with backoff; pass a fake chat model in tests.
    `where` restricts retrieval with a Chroma metadata filter.
    """
    if llm is None:
        llm, retry_config = _gemini()
    retriever = ChromaRetriever(k=k, where=where, collection=collection)
    out = build_chain(retriever, llm, retry_config).invoke({"question": question})
    docs = out["docs"]
    return {
        "question": question,
        "answer": out["answer"],
        "citations": parse_citations(out["answer"], docs),
        "sources": [
            {
                "rank": i,
                "id": d.metadata["id"],
                "paper_id": d.metadata["paper_id"],
                "title": d.metadata["title"],
                "page_number": d.metadata["page_number"],
                "end_page": d.metadata["end_page"],
                "distance": d.metadata["distance"],
                "text": d.page_content,
            }
            for i, d in enumerate(docs, 1)
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question")
    parser.add_argument("-k", type=int, default=DEFAULT_K)
    args = parser.parse_args()

    result = answer(args.question, k=args.k)
    print(result["answer"], "\n")
    print("Citations:")
    for c in result["citations"]:
        print(f"  [{c['source']}] {source_label(c)}")
    if not result["citations"]:
        print("  (none)")


if __name__ == "__main__":
    main()
