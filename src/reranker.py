"""Cross-encoder reranker (stretch goal, suggested by our CA).

bge-small embeds the question and each chunk separately, then compares the two
vectors -- fast, but it never "reads" them together. A cross-encoder reads the
question and a chunk *together* and scores how well the chunk answers it. That
is more accurate but too slow to run on all chunks, so we:
    1. retrieve `fetch_k` (e.g. 20) candidates with bge-small (Chroma),
    2. rerank only those with the cross-encoder,
    3. keep the top k.

Model chosen in eval/tune_semantic_chunking.py (see eval/semantic_chunking_report.md).
Runs locally through sentence-transformers -- no API key, no Gemini quota.
"""
from langchain_core.documents import Document

DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L6-v2"

_models = {}


def get_cross_encoder(model_name=DEFAULT_RERANKER):
    if model_name not in _models:
        from sentence_transformers import CrossEncoder

        _models[model_name] = CrossEncoder(model_name)
    return _models[model_name]


class CrossEncoderReranker:
    """rerank(question, docs, top_n) -> the top_n docs, best first, with a `rerank_score`."""

    def __init__(self, model_name=DEFAULT_RERANKER, model=None):
        self.model_name = model_name
        self._model = model  # inject a fake in tests

    @property
    def model(self):
        if self._model is None:
            self._model = get_cross_encoder(self.model_name)
        return self._model

    def rerank(self, question: str, docs: list[Document], top_n: int) -> list[Document]:
        if not docs:
            return []
        scores = self.model.predict([(question, d.page_content) for d in docs])
        order = sorted(range(len(docs)), key=lambda i: -float(scores[i]))[:top_n]
        return [
            Document(page_content=docs[i].page_content,
                     metadata={**docs[i].metadata, "rerank_score": float(scores[i])})
            for i in order
        ]
