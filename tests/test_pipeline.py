"""End-to-end tests: raw curated JSON -> clean -> chunk -> embed + store -> query.

Run from the repo root:
    pytest -v tests/                 # everything (the E2E test loads the embedding model)
    pytest -v tests/ -m "not slow"   # data checks only, no model needed (~10 s)
"""
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import chunking  # noqa: E402
import cleaning  # noqa: E402

RAW = json.loads((ROOT / "data" / "curated_papers_text.json").read_text(encoding="utf-8"))
COMMITTED_CLEAN = json.loads((ROOT / "data" / "cleaned_papers_text.json").read_text(encoding="utf-8"))
COMMITTED_CHUNKS = json.loads((ROOT / "data" / "chunks.json").read_text(encoding="utf-8"))
QA = json.loads((ROOT / "eval" / "qa_pairs.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def cleaned():
    return cleaning.clean_papers(RAW)


@pytest.fixture(scope="module")
def chunks(cleaned):
    return chunking.chunk_papers(cleaned)


def _paper(papers, pid):
    return next(p for p in papers if p["id"] == pid)


# ---------------------------------------------------------------- cleaning
def test_all_papers_cleaned(cleaned):
    assert len(cleaned) == 10
    for p in cleaned:
        assert len(p["pages"]) == p["total_pages"]
        assert p["clean_full_text"]


def test_no_control_or_non_ascii_chars(cleaned):
    for p in cleaned:
        for pg in p["pages"]:
            bad = [c for c in pg["text"] if (ord(c) < 32 and c not in "\n\t") or ord(c) > 127]
            assert not bad, f"{p['id']} p.{pg['page_number']}: {bad[:5]}"


def test_react_cipher_decoded(cleaned):
    """Regression test for the Figure 1 bug flagged on Godwin's PR."""
    text = "\n".join(pg["text"] for pg in _paper(cleaned, "2210.03629")["pages"])
    assert "Front Row Motorsports" in text
    for garbage in ["BnkjpNks", "7RXFK", "$OI:RUOG", "ZDWFK"]:
        assert garbage not in text
    assert "discontinued media center software ..." in text  # ellipsis, not a stray 'a'


def test_running_headers_and_page_numbers_removed(cleaned):
    for p in cleaned:
        for pg in p["pages"]:
            first = pg["text"].split("\n", 1)[0].strip()
            last = pg["text"].rsplit("\n", 1)[-1].strip()
            assert first not in {"Preprint.", "Published as a conference paper at ICLR 2023"}
            assert not re.fullmatch(r"\d{1,4}", last), f"{p['id']} p.{pg['page_number']} ends with page number"


def test_committed_cleaned_file_is_up_to_date(cleaned):
    """data/cleaned_papers_text.json should equal a fresh run of src/cleaning.py.
    If this fails: python src/cleaning.py && python src/chunking.py"""
    assert cleaned == COMMITTED_CLEAN


# ---------------------------------------------------------------- chunking
def test_committed_chunks_are_up_to_date(chunks):
    """data/chunks.json should equal a fresh run of src/chunking.py."""
    assert chunks == COMMITTED_CHUNKS


def test_chunk_metadata(chunks):
    assert {c["metadata"]["paper_id"] for c in chunks} == {p["id"] for p in RAW}
    for c in chunks:
        m = c["metadata"]
        assert 0 < len(c["page_content"]) <= chunking.CHUNK_SIZE
        assert 1 <= m["page_number"] <= m["end_page"] <= m["total_pages"]
        for key in ["paper_id", "title", "category", "pdf_url", "start_index"]:
            assert key in m


def test_chunks_cover_every_page(cleaned, chunks):
    for p in cleaned:
        covered = set()
        for c in chunks:
            if c["metadata"]["paper_id"] == p["id"]:
                covered.update(range(c["metadata"]["page_number"], c["metadata"]["end_page"] + 1))
        nonempty = {pg["page_number"] for pg in p["pages"] if pg["text"].strip()}
        assert nonempty <= covered, f"{p['id']} pages never chunked: {sorted(nonempty - covered)}"


def test_drop_references_keeps_appendices(cleaned):
    kept = chunking.chunk_papers(cleaned, drop_references=True)
    assert len(kept) < len(chunking.chunk_papers(cleaned))
    react = " ".join(c["page_content"] for c in kept if c["metadata"]["paper_id"] == "2210.03629")
    assert "ALFWorld" in react and "Front Row Motorsports" in react  # appendix + Figure 1 survive


# ---------------------------------------------------------------- Q&A file
def test_qa_evidence_exists_in_papers(cleaned):
    """Every Q&A pair must be answerable from the cleaned text."""
    norm = lambda s: " ".join(s.lower().split())  # noqa: E731
    for qa in QA:
        text = norm("\n".join(pg["text"] for pg in _paper(cleaned, qa["paper_id"])["pages"]))
        assert any(norm(e) in text for e in qa["evidence"]), qa["id"]


# ---------------------------------------------------------------- end to end
@pytest.mark.slow
def test_end_to_end_clean_to_query(tmp_path, cleaned, chunks):
    """raw -> clean -> chunk -> embed -> Chroma -> query, in a temp folder."""
    import vector_store

    chunks_path = tmp_path / "chunks.json"
    chunks_path.write_text(json.dumps(chunks), encoding="utf-8")
    collection = vector_store.build_collection(chunks_path=chunks_path, db_path=tmp_path / "db")
    assert collection.count() == len(chunks)

    hits = 0
    for qa in QA:
        results = vector_store.query(qa["question"], k=5, collection=collection)
        assert len(results) == 5
        assert all(0.0 <= r["distance"] <= 2.0 for r in results)
        hits += any(r["metadata"]["paper_id"] == qa["paper_id"] for r in results)
    # the expected paper should show up in the top 5 for (nearly) every question
    assert hits / len(QA) >= 0.9, f"paper hit@5 = {hits}/{len(QA)}"

    # metadata filter works (used later for per-paper questions)
    only = vector_store.query("attention", k=3, where={"paper_id": "2205.14135"}, collection=collection)
    assert {r["metadata"]["paper_id"] for r in only} == {"2205.14135"}
