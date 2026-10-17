"""Prompt template and source formatting for the generation step.

Copied verbatim from Niv's notebooks/generate_response.ipynb (niv/generation-prompt
branch) so the RAG chain can import it. Niv owns the wording: edit it here and the
chain picks it up.
"""
from langchain_core.prompts import ChatPromptTemplate

SYSTEM = """You are a research assistant for an R&D team. Answer the question using ONLY the sources provided.
- Use only information found in the sources. If they don't contain the answer, say: "The provided sources do not contain this information."
- After every claim, cite the source it came from by its bracket number, e.g. [1].
- Do not use outside knowledge, and never invent a citation."""
HUMAN = "Question: {question}\n\nSources:\n{context}"

PROMPT = ChatPromptTemplate.from_messages([("system", SYSTEM), ("human", HUMAN)])


def source_label(m):
    pages = f"p.{m['page_number']}" if m['page_number'] == m['end_page'] \
            else f"pp.{m['page_number']}-{m['end_page']}"
    return f"{m['title']} ({m['paper_id']}, {pages})"


def format_sources(docs):
    blocks = []
    for i, d in enumerate(docs, 1):
        blocks.append(f"[{i}] {source_label(d.metadata)}\n{d.page_content}")
    return "\n\n".join(blocks)
