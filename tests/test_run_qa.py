"""Tests for the Q&A runner (eval/run_qa.py).

answer() is replaced with a fake, so these need no API key, no quota, no Chroma store and
no embedding model. They run in under a second.

Run from the repo root:
    pytest -v tests/test_run_qa.py
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

import run_qa  # noqa: E402
from config import MissingAPIKeyError  # noqa: E402

QA = [
    {"paper_id": "2106.09685", "question": "What does LoRA freeze?",
     "key_keywords": ["pre-trained weights", "Frozen", "rank"]},
    {"paper_id": "2205.14135", "question": "What does FlashAttention reduce?",
     "key_keywords": ["memory", "HBM"]},
]


def result(answer, source_papers, cited_papers):
    """Build something shaped like rag_chain.answer()'s return value."""
    return {
        "question": "q",
        "answer": answer,
        "sources": [{"rank": i, "paper_id": p} for i, p in enumerate(source_papers, 1)],
        "citations": [{"source": i, "paper_id": p} for i, p in enumerate(cited_papers, 1)],
    }


def fake_answer(replies):
    """A stand-in for answer(): returns the reply for each question and records every call."""
    calls = []

    def answer(question, k=5, **kwargs):
        calls.append(question)
        reply = replies[question]
        if isinstance(reply, Exception):
            raise reply
        return reply

    answer.calls = calls
    return answer


def run(tmp_path, answer_fn, **kwargs):
    return run_qa.run([dict(q) for q in QA], tmp_path / "out.jsonl", answer_fn=answer_fn,
                      log=lambda *a: None, **kwargs)


GOOD = {
    QA[0]["question"]: result("LoRA keeps the pre-trained weights frozen [1].",
                              ["2005.11401", "2106.09685"], ["2106.09685"]),
    QA[1]["question"]: result("The provided sources do not contain this information.",
                              ["2004.04906"], []),
}


# ---------------------------------------------------------------- loading
def test_load_qa_adds_ids_only_when_missing(tmp_path):
    path = tmp_path / "qa.json"
    path.write_text(json.dumps([{"question": "a", "paper_id": "x"},
                                {"id": "q07", "question": "b", "paper_id": "y"}]))
    assert [qa["id"] for qa in run_qa.load_qa(path)] == ["gt01", "q07"]


def test_ground_truth_file_loads():
    pairs = run_qa.load_qa(run_qa.DEFAULT_QA)
    assert len(pairs) == 40
    assert len({qa["id"] for qa in pairs}) == 40
    assert all({"question", "paper_id", "key_keywords"} <= qa.keys() for qa in pairs)


# ---------------------------------------------------------------- scoring
def test_keyword_coverage_ignores_case():
    assert run_qa.keyword_coverage("The FROZEN pre-trained weights", ["frozen", "rank"]) == 0.5
    assert run_qa.keyword_coverage("anything", []) is None


def test_score_hit_rank_and_citation():
    qa = {**QA[0], "id": "gt01"}
    row = run_qa.score(qa, GOOD[QA[0]["question"]])
    assert row["paper_hit"] is True and row["paper_rank"] == 2
    assert row["cited_paper"] is True and row["has_citation"] is True
    assert row["said_no_info"] is False
    assert row["keyword_coverage"] == pytest.approx(2 / 3)  # "pre-trained weights", "frozen"


def test_score_miss_and_no_info():
    qa = {**QA[1], "id": "gt02"}
    row = run_qa.score(qa, GOOD[QA[1]["question"]])
    assert row["paper_hit"] is False and row["paper_rank"] is None
    assert row["cited_paper"] is False and row["has_citation"] is False
    assert row["said_no_info"] is True


def test_cited_wrong_paper_is_not_cited_paper():
    qa = {**QA[0], "id": "gt01"}
    row = run_qa.score(qa, result("Something [1].", ["2106.09685", "2004.04906"], ["2004.04906"]))
    assert row["paper_hit"] is True
    assert row["has_citation"] is True and row["cited_paper"] is False


# ---------------------------------------------------------------- running
def test_run_answers_every_question_and_saves_each_row(tmp_path):
    answer = fake_answer(GOOD)
    rows = run(tmp_path, answer)
    assert [r["id"] for r in rows] == ["gt01", "gt02"]
    saved = [json.loads(line) for line in (tmp_path / "out.jsonl").read_text().splitlines()]
    assert [r["id"] for r in saved] == ["gt01", "gt02"]
    s = run_qa.summarize(rows)
    assert s["paper_hit"] == 0.5 and s["cited_paper"] == 0.5 and s["errors"] == 0


def test_rerun_skips_done_questions_so_no_quota_is_spent_twice(tmp_path):
    run(tmp_path, fake_answer(GOOD))
    again = fake_answer(GOOD)
    rows = run(tmp_path, again)
    assert again.calls == []
    assert len(rows) == 2


def test_limit_runs_only_that_many_new_questions(tmp_path):
    answer = fake_answer(GOOD)
    rows = run(tmp_path, answer, limit=1)
    assert answer.calls == [QA[0]["question"]] and len(rows) == 1
    rest = fake_answer(GOOD)
    run(tmp_path, rest, limit=1)
    assert rest.calls == [QA[1]["question"]]  # picks up where the first run stopped


def test_one_failing_question_does_not_stop_the_run_and_is_retried(tmp_path):
    replies = {**GOOD, QA[0]["question"]: RuntimeError("429 Resource exhausted")}
    rows = run(tmp_path, fake_answer(replies))
    assert rows[0]["error"].startswith("RuntimeError") and rows[1]["error"] is None
    assert run_qa.summarize(rows)["errors"] == 1

    retry = fake_answer(GOOD)
    rows = run(tmp_path, retry)
    assert retry.calls == [QA[0]["question"]]  # only the failed one runs again
    assert [r["error"] for r in rows] == [None, None]


def test_missing_api_key_stops_immediately(tmp_path):
    replies = {q["question"]: MissingAPIKeyError("GOOGLE_API_KEY not found") for q in QA}
    answer = fake_answer(replies)
    with pytest.raises(MissingAPIKeyError):
        run(tmp_path, answer)
    assert len(answer.calls) == 1


def test_csv_has_one_row_per_question(tmp_path):
    rows = run(tmp_path, fake_answer(GOOD))
    run_qa.write_csv(rows, tmp_path / "out.csv")
    lines = (tmp_path / "out.csv").read_text().splitlines()
    assert lines[0].startswith("id,paper_id,question")
    assert len(lines) == 3


def test_fake_llm_works_with_the_real_chain():
    """--fake mode plugs a fake Gemini into Tanish's chain; check it parses as a citation."""
    import rag_chain

    docs = [rag_chain.Document(page_content="t", metadata={
        "paper_id": "2106.09685", "title": "LoRA", "page_number": 1, "end_page": 1,
        "id": "x", "distance": 0.1})]

    class Fixed(rag_chain.BaseRetriever):
        def _get_relevant_documents(self, query, *, run_manager):
            return docs

    out = rag_chain.build_chain(Fixed(), run_qa.fake_llm()).invoke({"question": "q"})
    assert [c["paper_id"] for c in rag_chain.parse_citations(out["answer"], docs)] == ["2106.09685"]