"""Tests for eval/answer_checks.py. Fake answer() results only: no API key, no quota,
no Chroma store and no embedding model.

Run from the repo root:
    pytest -v tests/test_answer_checks.py
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))

import answer_checks as ac  # noqa: E402

LORA_SRC = ("LoRA freezes the pre-trained model weights and injects trainable rank decomposition "
            "matrices into each layer. With rank 4 it trains 4.7M parameters instead of 175B.")
OTHER_SRC = "Dense passage retrieval encodes questions and passages with two BERT encoders."

QA = {"paper_id": "2106.09685", "question": "What does LoRA freeze?",
      "key_keywords": ["pre-trained", "rank", "freezes"]}


def result(answer, sources=(LORA_SRC, OTHER_SRC), cited=()):
    return {
        "answer": answer,
        "sources": [{"rank": i, "paper_id": "p", "text": t} for i, t in enumerate(sources, 1)],
        "citations": [{"source": n} for n in cited],
    }


# ---------------------------------------------------------------- text helpers
def test_cited_numbers_handles_lists_and_adjacent_markers():
    assert ac.cited_numbers("A [1]. B [2, 3][4]. C [1].") == [1, 2, 3, 4, 1]


def test_split_sentences_handles_bullets_and_keeps_markers():
    text = "LoRA freezes weights [1]. It trains fewer parameters [1].\n- Bullet point [2]"
    assert ac.split_sentences(text) == [
        "LoRA freezes weights [1].", "It trains fewer parameters [1].", "Bullet point [2]"]


def test_numbers_skip_lone_digits_but_keep_percentages_and_decimals():
    assert ac._numbers("Uses 2 layers, 4.7M params, 175 billion, 12% gain [1]") == {"4.7", "175", "12%"}


# ---------------------------------------------------------------- citations
def test_cited_chunk_support_counts_keywords_in_cited_chunks_only():
    good = ac.check_citations(QA, result("LoRA freezes the pre-trained weights [1].", cited=[1]))
    assert good["cited_chunk_support"] == 1.0
    wrong_chunk = ac.check_citations(QA, result("LoRA freezes the pre-trained weights [2].", cited=[2]))
    assert wrong_chunk["cited_chunk_support"] == 0.0


def test_no_citations_means_zero_support():
    out = ac.check_citations(QA, result("LoRA freezes the pre-trained weights."))
    assert out["cited_chunk_support"] == 0.0
    assert out["uncited_sentence_share"] == 1.0


def test_invented_citation_numbers_are_counted():
    out = ac.check_citations(QA, result("LoRA freezes weights [1]. It is fast [7][9].", cited=[1]))
    assert out["invented_citations"] == 2


def test_no_keywords_gives_none_not_a_crash():
    out = ac.check_citations({"paper_id": "x"}, result("Something [1].", cited=[1]))
    assert out["cited_chunk_support"] is None


def test_uncited_sentence_share_ignores_the_no_info_sentence():
    out = ac.check_citations(QA, result("LoRA freezes weights [1]. It is cheap. "
                                        "The provided sources do not contain this information."))
    assert out["uncited_sentence_share"] == 0.5


# ---------------------------------------------------------------- hallucination
def test_supported_answer_is_not_flagged():
    out = ac.check_grounding(result("LoRA freezes the pre-trained weights and adds rank "
                                    "decomposition matrices [1]."))
    assert out["grounded_share"] == 1.0
    assert out["hallucination_flag"] is False and out["unsupported_sentences"] == []


def test_made_up_sentence_is_unsupported_and_flagged():
    out = ac.check_grounding(result(
        "LoRA freezes the pre-trained weights [1]. "
        "Training happened on thousands of quantum processors in Antarctica [1]."))
    assert out["grounded_share"] == 0.5
    assert out["unsupported_sentences"] == ["Training happened on thousands of quantum "
                                            "processors in Antarctica [1]."]
    assert out["hallucination_flag"] is True


def test_number_not_in_any_source_is_flagged_even_if_words_match():
    out = ac.check_grounding(result("LoRA freezes the pre-trained model weights and trains "
                                    "9.9M parameters [1]."))
    assert out["unsupported_numbers"] == ["9.9"]
    assert out["hallucination_flag"] is True


def test_number_found_in_sources_is_fine():
    out = ac.check_grounding(result("With rank 4, LoRA trains 4.7M parameters instead of "
                                    "175B [1]."))
    assert out["unsupported_numbers"] == []


def test_sentence_is_checked_against_the_source_it_cites():
    # words are in source 1, but the sentence cites source 2 -> unsupported
    out = ac.check_grounding(result("LoRA freezes the pre-trained weights [2]."))
    assert out["unsupported_sentences"] != []


def test_pure_abstention_has_nothing_to_check():
    out = ac.check_grounding(result("The provided sources do not contain this information."))
    assert out["grounded_share"] is None and out["hallucination_flag"] is False


def test_custom_support_function_can_replace_the_lexical_one():
    out = ac.check_grounding(result("Totally unrelated words here [1]."), support_fn=lambda s, t: 1.0)
    assert out["grounded_share"] == 1.0


# ---------------------------------------------------------------- "I don't know"
@pytest.mark.parametrize("text,n_cited,expected", [
    ("The provided sources do not contain this information.", 0, True),
    ("The provided sources do not contain this information.", 1, True),   # exact phrase wins
    ("I cannot find this in the sources.", 0, True),
    ("There is no relevant information here.", 0, True),
    ("The paper does not mention dropout, but it freezes weights [1].", 1, False),
    ("LoRA freezes the weights.", 0, False),
])
def test_is_no_info(text, n_cited, expected):
    assert ac.is_no_info(text, n_cited) is expected


UNANSWERABLE = {"answerable": False, "key_keywords": [], "paper_id": None}


def test_unanswerable_question_abstained_correctly():
    out = ac.check_abstention(UNANSWERABLE, result("The provided sources do not contain this information."))
    assert out["abstained_correctly"] is True and out["over_abstained"] is None


def test_unanswerable_question_answered_anyway_is_a_failure():
    out = ac.check_abstention(UNANSWERABLE, result("Canberra is the capital [1].", cited=[1]))
    assert out["abstained_correctly"] is False


def test_unanswerable_abstention_that_still_cites_is_not_clean():
    out = ac.check_abstention(UNANSWERABLE, result(
        "The provided sources do not contain this information [1].", cited=[1]))
    assert out["said_no_info"] is True and out["abstained_correctly"] is False


def test_answerable_question_over_abstained():
    out = ac.check_abstention(QA, result("The provided sources do not contain this information."))
    assert out["over_abstained"] is True and out["abstained_correctly"] is None


def test_answered_without_evidence_when_sources_miss_every_keyword():
    out = ac.check_abstention(QA, result("It freezes things [2].", sources=(OTHER_SRC,), cited=[1]))
    assert out["answered_without_evidence"] is True
    ok = ac.check_abstention(QA, result("LoRA freezes weights [1].", cited=[1]))
    assert ok["answered_without_evidence"] is False


# ---------------------------------------------------------------- all together
def test_score_answer_quality_returns_all_columns():
    row = ac.score_answer_quality(QA, result("LoRA freezes the pre-trained weights [1].", cited=[1]))
    assert set(row) == {
        "cited_chunk_support", "invented_citations", "uncited_sentence_share",
        "grounded_share", "unsupported_sentences", "unsupported_numbers", "hallucination_flag",
        "said_no_info", "abstained_correctly", "over_abstained", "answered_without_evidence"}


def test_summarize_quality_splits_answerable_and_unanswerable():
    good = {**ac.score_answer_quality(QA, result("LoRA freezes the pre-trained weights [1].", cited=[1]))}
    abst = {**ac.score_answer_quality(UNANSWERABLE, result("The provided sources do not contain this information."))}
    fab = {**ac.score_answer_quality(UNANSWERABLE, result("Canberra [1].", cited=[1]))}
    s = ac.summarize_quality([good, abst, fab, {"error": "429"}])
    assert s["scored"] == 3
    assert s["abstained_correctly_rate"] == 0.5
    assert s["over_abstained_rate"] == 0.0


def test_summarize_quality_with_nothing_scored():
    assert ac.summarize_quality([]) == {"scored": 0}


# ---------------------------------------------------------------- the question file
def test_unanswerable_question_file_is_well_formed():
    qs = json.loads((ROOT / "eval" / "unanswerable_qa.json").read_text(encoding="utf-8"))
    assert len(qs) == 10 and len({q["id"] for q in qs}) == 10
    assert all(q["answerable"] is False and q["paper_id"] is None for q in qs)
    assert all(q["question"].endswith("?") for q in qs)


# ---------------------------------------------------------------- run_qa.py integration
import run_qa  # noqa: E402

RUN_QA = {"id": "gt01", "paper_id": "2106.09685", "question": "What does LoRA freeze?",
          "key_keywords": ["pre-trained", "rank", "freezes"]}
RUN_UNANS = {"id": "un01", "paper_id": None, "answerable": False,
             "question": "What is a good recipe for sourdough bread?", "key_keywords": []}


def full_result(answer, cited=(1,), paper="2106.09685"):
    return {
        "question": "q", "answer": answer,
        "sources": [{"rank": 1, "id": "c1", "paper_id": paper, "title": "LoRA", "page_number": 3,
                     "end_page": 3, "distance": 0.1, "text": LORA_SRC},
                    {"rank": 2, "id": "c2", "paper_id": "2004.04906", "title": "DPR", "page_number": 5,
                     "end_page": 5, "distance": 0.2, "text": OTHER_SRC}],
        "citations": [{"source": n, "paper_id": paper} for n in cited],
    }


def test_score_keeps_ivans_columns_and_adds_the_quality_ones():
    row = run_qa.score(RUN_QA, full_result("LoRA freezes the pre-trained weights [1]."))
    assert row["paper_hit"] is True and row["cited_paper"] is True and row["has_citation"] is True
    assert row["keyword_coverage"] == pytest.approx(2 / 3)  # "rank" is not in the answer text
    assert row["cited_chunk_support"] == 1.0 and row["hallucination_flag"] is False
    assert {"grounded_share", "invented_citations", "over_abstained"} <= row.keys()


def test_score_saves_sources_and_citations_for_later_rescoring():
    row = run_qa.score(RUN_QA, full_result("LoRA freezes the pre-trained weights [1]."))
    assert row["sources"][0]["text"] == LORA_SRC
    assert row["sources"][0]["page_number"] == 3 and row["citations"][0]["source"] == 1
    json.dumps(row)  # must be serialisable for the .jsonl


def test_unanswerable_question_scores_without_an_expected_paper():
    row = run_qa.score(RUN_UNANS, full_result("The provided sources do not contain this information.", cited=()))
    assert row["paper_hit"] is None and row["cited_paper"] is None
    assert row["said_no_info"] is True and row["abstained_correctly"] is True


def test_unanswerable_question_answered_anyway_is_reported():
    row = run_qa.score(RUN_UNANS, full_result("Mix flour, water and starter [1]."))
    assert row["abstained_correctly"] is False
    assert run_qa._status(row).startswith("ANSWERED anyway")


def test_summary_keeps_paper_metrics_to_answerable_questions():
    good = run_qa.score(RUN_QA, full_result("LoRA freezes the pre-trained weights [1]."))
    abst = run_qa.score(RUN_UNANS, full_result("The provided sources do not contain this information.", cited=()))
    s = run_qa.summarize([good, abst])
    assert s["paper_hit"] == 1.0 and s["cited_paper"] == 1.0   # the unanswerable row doesn't drag them down
    assert s["abstained_correctly_rate"] == 1.0 and s["over_abstained_rate"] == 0.0


def test_csv_handles_lists_and_leaves_bulky_fields_in_the_jsonl(tmp_path):
    row = run_qa.score(RUN_QA, full_result("LoRA freezes weights [1]. Quantum processors in Antarctica train it 999M times [1]."))
    run_qa.write_csv([row], tmp_path / "out.csv")
    header, line = (tmp_path / "out.csv").read_text().splitlines()[:2]
    assert "hallucination_flag" in header and "sources" not in header
    assert "999M" in line  # unsupported number and sentence are readable in the CSV


def test_run_saves_sources_and_rescore_needs_no_gemini(tmp_path):
    out = tmp_path / "out.jsonl"
    calls = []

    def answer(question, k=5, **kwargs):
        calls.append(question)
        return full_result("LoRA freezes the pre-trained weights [1].")

    run_qa.run([dict(RUN_QA)], out, answer_fn=answer, log=lambda *a: None)
    saved = json.loads(out.read_text().splitlines()[0])
    assert saved["sources"][0]["text"] == LORA_SRC

    rows = run_qa.rescore([dict(RUN_QA)], out, log=lambda *a: None)
    assert len(rows) == 1 and rows[0]["cited_chunk_support"] == 1.0
    assert calls == [RUN_QA["question"]]  # rescoring did not call answer() again


def test_rescore_skips_rows_saved_before_sources_were_kept(tmp_path):
    out = tmp_path / "old.jsonl"
    out.write_text(json.dumps({"id": "gt01", "answer": "x", "error": None}) + "\n")
    assert run_qa.rescore([dict(RUN_QA)], out, log=lambda *a: None) == []


def test_unanswerable_file_loads_through_run_qa():
    pairs = run_qa.load_qa(run_qa.UNANSWERABLE_QA)
    assert [q["id"] for q in pairs][:2] == ["un01", "un02"] and len(pairs) == 10
