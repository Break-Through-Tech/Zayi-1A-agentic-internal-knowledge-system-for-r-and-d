"""Tests for the semantic chunker (src/semantic_chunking.py).

Embeddings are faked (one topic vector per sentence), so these run in a second
and never load the model.

Run from the repo root:
    pytest -v tests/test_semantic_chunking.py
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import semantic_chunking as sc  # noqa: E402


def test_sentence_spans_keep_exact_offsets():
    text = "First one. Second?  Third!\nFourth without stop"
    spans = sc.sentence_spans(text)
    assert [text[s:e] for s, e in spans] == ["First one.", "Second?", "Third!", "Fourth without stop"]


def test_window_texts_buffer():
    text = "A. B. C. D."
    spans = sc.sentence_spans(text)
    assert sc.window_texts(text, spans, 0) == ["A.", "B.", "C.", "D."]
    assert sc.window_texts(text, spans, 1) == ["A. B.", "A. B. C.", "B. C. D.", "C. D."]


def test_thresholds():
    d = np.array([0.1, 0.1, 0.1, 0.9])
    assert sc.threshold(d, "percentile", 50) == pytest.approx(0.1)
    assert sc.threshold(d, "standard_deviation", 1) == pytest.approx(d.mean() + d.std())
    assert sc.threshold(np.zeros(0), "percentile", 90) == np.inf
    with pytest.raises(ValueError):
        sc.threshold(d, "gradient", 95)


SPANS = [(i * 10, i * 10 + 9) for i in range(6)]  # six 9-character sentences
DIST = np.array([0.1, 0.1, 0.8, 0.1, 0.1])        # topic changes after sentence 2


def settings(**kw):
    base = dict(breakpoint_threshold_type="percentile", breakpoint_threshold_amount=70,
                buffer_size=0, min_chunk_size=0, max_chunk_size=None)
    return sc.SemanticSettings(**{**base, **kw})


def test_cut_where_topic_changes():
    assert sc.group_sentences(SPANS, DIST, settings()) == [(0, 3), (3, 6)]


def test_min_chunk_size_skips_early_cut():
    # first chunk would be 29 chars; with min 40 the cut is skipped
    assert sc.group_sentences(SPANS, DIST, settings(min_chunk_size=40)) == [(0, 6)]


def test_max_chunk_size_splits_at_biggest_distance():
    dist = np.array([0.1, 0.3, 0.2, 0.1, 0.1])
    groups = sc.group_sentences(SPANS, dist, settings(breakpoint_threshold_amount=100, max_chunk_size=30))
    assert groups == [(0, 2), (2, 3), (3, 6)]
    assert all(SPANS[hi - 1][1] - SPANS[lo][0] <= 30 for lo, hi in groups)


def test_single_long_sentence_falls_back_to_character_splitter():
    text = "word " * 100  # 500 chars, no sentence end
    spans = sc.sentence_spans(text)
    out = sc.chunk_spans(text, spans, np.zeros(0), settings(max_chunk_size=120))
    assert len(out) > 1 and all(e - s <= 120 for s, e in out)
    assert all(text[s:e].strip() for s, e in out)


def _paper(pid, pages):
    return {"id": pid, "title": f"Paper {pid}", "category": "Test", "pdf_url": "", "total_pages": len(pages),
            "pages": [{"page_number": i + 1, "text": t} for i, t in enumerate(pages)]}


def test_paper_chunks_follow_topics_and_pages(monkeypatch):
    # Fake embedder (buffer 0, so each window is one sentence): cats vs engines
    def fake_embed(texts, batch_size=64):
        return np.array([[1.0, 0.0] if "Cats" in t else [0.0, 1.0] for t in texts])

    monkeypatch.setattr(sc, "embed_passages", fake_embed)
    page1 = "Cats purr loudly. Cats sleep a lot. Cats chase mice."
    page2 = "Engines burn fuel. Engines need oil. Engines get hot."
    papers = [_paper("1234.5678", [page1, page2])]
    chunks = sc.semantic_chunk_papers(papers, settings(breakpoint_threshold_amount=50))

    assert [c["page_content"] for c in chunks] == [page1, page2]
    assert [(c["metadata"]["page_number"], c["metadata"]["end_page"]) for c in chunks] == [(1, 1), (2, 2)]
    full = page1 + "\n" + page2
    for c in chunks:
        s = c["metadata"]["start_index"]
        assert full[s:s + len(c["page_content"])] == c["page_content"]
        assert c["metadata"]["paper_id"] == "1234.5678" and c["metadata"]["chunker"] == "semantic"
