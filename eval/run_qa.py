"""Run Q&A pairs through the full RAG pipeline (answer()) and score the results.

For every question we call rag_chain.answer(question, k) and record:
  paper_hit         - the expected paper is among the retrieved sources (top k)
  paper_rank        - rank of the first source from the expected paper (blank if none)
  cited_paper       - the answer cites at least one source from the expected paper
  has_citation      - the answer cites anything at all
  said_no_info      - the model answered "The provided sources do not contain this information."
  keyword_coverage  - share of the question's key_keywords found in the answer (case-insensitive)

Results are appended to a .jsonl file one question at a time, so if the run stops halfway
(rate limit, crash, Ctrl+C) you can run the same command again and it skips the questions
already done. That way no Gemini quota is spent twice.

Usage (from the repo root, with GOOGLE_API_KEY in .env and a built Chroma store):
    python eval/run_qa.py --limit 3          # try 3 questions first
    python eval/run_qa.py                    # all 40 questions in data/ground_truth_qa.json
    python eval/run_qa.py --fake             # no Gemini calls: real retrieval, fake answers
    python eval/run_qa.py --qa eval/qa_pairs.json --out eval/results/qa_run_ours.jsonl

Writes eval/results/qa_run.jsonl (full answers) and eval/results/qa_run.csv (one row per question).
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_QA = ROOT / "data" / "ground_truth_qa.json"
DEFAULT_OUT = ROOT / "eval" / "results" / "qa_run.jsonl"
NO_INFO = "do not contain this information"

CSV_FIELDS = [
    "id", "paper_id", "question", "paper_hit", "paper_rank", "cited_paper", "has_citation",
    "said_no_info", "keyword_coverage", "cited_papers", "error", "answer",
]


def load_qa(path):
    """Load Q&A pairs and give each one an id (Jose's file has none, ours uses q01...)."""
    return add_ids(json.loads(Path(path).read_text(encoding="utf-8")))


def add_ids(pairs):
    for i, qa in enumerate(pairs, 1):
        qa.setdefault("id", f"gt{i:02d}")
    return pairs


def keyword_coverage(answer_text, keywords):
    """Share of keywords that appear in the answer, ignoring case. None if there are no keywords."""
    if not keywords:
        return None
    text = answer_text.lower()
    return sum(1 for kw in keywords if kw.lower() in text) / len(keywords)


def score(qa, result):
    """Turn one answer() result into a flat row of checks for this question."""
    expected = qa["paper_id"]
    source_papers = [s["paper_id"] for s in result["sources"]]
    cited_papers = [c["paper_id"] for c in result["citations"]]
    answer_text = result["answer"]
    return {
        "id": qa["id"],
        "paper_id": expected,
        "question": qa["question"],
        "paper_hit": expected in source_papers,
        "paper_rank": source_papers.index(expected) + 1 if expected in source_papers else None,
        "cited_paper": expected in cited_papers,
        "has_citation": bool(cited_papers),
        "said_no_info": NO_INFO in answer_text.lower(),
        "keyword_coverage": keyword_coverage(answer_text, qa.get("key_keywords")),
        "cited_papers": cited_papers,
        "error": None,
        "answer": answer_text,
    }


def error_row(qa, exc):
    """Row for a question whose answer() call failed, so the run can keep going."""
    return {
        "id": qa["id"], "paper_id": qa["paper_id"], "question": qa["question"],
        "paper_hit": None, "paper_rank": None, "cited_paper": None, "has_citation": None,
        "said_no_info": None, "keyword_coverage": None, "cited_papers": [],
        "error": f"{type(exc).__name__}: {exc}", "answer": "",
    }


def load_done(out_path):
    """Rows already saved by an earlier run, keyed by id. Failed questions are retried."""
    done = {}
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if not row.get("error"):
                    done[row["id"]] = row
    return done


def summarize(rows):
    """Averages over the questions that ran without errors."""
    ok = [r for r in rows if not r["error"]]
    n = len(ok)

    def share(key):
        return sum(1 for r in ok if r[key]) / n if n else 0.0

    covs = [r["keyword_coverage"] for r in ok if r["keyword_coverage"] is not None]
    return {
        "questions": len(rows),
        "errors": len(rows) - n,
        "paper_hit": share("paper_hit"),
        "cited_paper": share("cited_paper"),
        "has_citation": share("has_citation"),
        "said_no_info": share("said_no_info"),
        "keyword_coverage": sum(covs) / len(covs) if covs else None,
    }


def write_csv(rows, csv_path):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({**r, "cited_papers": " ".join(r["cited_papers"])})


def fake_llm():
    """A stand-in for Gemini that always cites source [1]. Tests retrieval with zero API calls."""
    from langchain_core.messages import AIMessage
    from langchain_core.runnables import RunnableLambda

    return RunnableLambda(lambda prompt: AIMessage(content="Fake answer for a dry run [1]."))


def run(qa_pairs, out_path, k=5, limit=None, sleep=0.0, answer_fn=None, llm=None, log=print):
    """Answer every question not already in out_path, append each result, and return all rows."""
    if answer_fn is None:
        from rag_chain import answer as answer_fn

    from config import MissingAPIKeyError

    qa_pairs = add_ids(qa_pairs)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out_path)
    todo = [qa for qa in qa_pairs if qa["id"] not in done]
    if limit is not None:
        todo = todo[:limit]
    log(f"{len(qa_pairs)} questions, {len(done)} already done, running {len(todo)} now")

    new_rows = []
    with open(out_path, "a", encoding="utf-8") as f:
        for i, qa in enumerate(todo, 1):
            try:
                kwargs = {"llm": llm} if llm is not None else {}
                row = score(qa, answer_fn(qa["question"], k=k, **kwargs))
            except MissingAPIKeyError:
                raise  # no point trying the other questions without a key
            except Exception as exc:  # one bad question shouldn't stop the whole run
                row = error_row(qa, exc)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()  # saved now, so a crash on the next question loses nothing
            new_rows.append(row)
            status = "ERROR " + row["error"] if row["error"] else (
                f"paper {'hit' if row['paper_hit'] else 'MISS'}, "
                f"cited {'right' if row['cited_paper'] else 'wrong/none'}")
            log(f"[{i}/{len(todo)}] {qa['id']} {status}")
            if sleep and i < len(todo):
                time.sleep(sleep)  # stay under Gemini's requests-per-minute limit

    # Keep the order of the Q&A file; failed rows from this run replace nothing already saved.
    by_id = {**done, **{r["id"]: r for r in new_rows}}
    return [by_id[qa["id"]] for qa in qa_pairs if qa["id"] in by_id]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qa", default=str(DEFAULT_QA), help="Q&A JSON file (default: Jose's 40 questions)")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="results .jsonl (a .csv is written next to it)")
    ap.add_argument("-k", type=int, default=5, help="chunks retrieved per question")
    ap.add_argument("--limit", type=int, help="only run this many new questions")
    ap.add_argument("--sleep", type=float, default=4.0, help="seconds between Gemini calls")
    ap.add_argument("--fake", action="store_true", help="no Gemini: real retrieval, fake answers")
    args = ap.parse_args()

    out_path = Path(args.out)
    if args.fake and out_path == DEFAULT_OUT:
        out_path = out_path.with_name("qa_run_fake.jsonl")  # never mix fake and real answers

    rows = run(load_qa(args.qa), out_path, k=args.k, limit=args.limit,
               sleep=0 if args.fake else args.sleep, llm=fake_llm() if args.fake else None)
    csv_path = out_path.with_suffix(".csv")
    write_csv(rows, csv_path)

    s = summarize(rows)
    print()
    print(f"questions={s['questions']}  errors={s['errors']}  paper_hit@{args.k}={s['paper_hit']:.2f}  "
          f"cited_paper={s['cited_paper']:.2f}  has_citation={s['has_citation']:.2f}  "
          f"said_no_info={s['said_no_info']:.2f}  keyword_coverage="
          + ("-" if s["keyword_coverage"] is None else f"{s['keyword_coverage']:.2f}"))
    print(f"Results: {csv_path}")


if __name__ == "__main__":
    main()