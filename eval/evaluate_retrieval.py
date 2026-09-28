"""Retrieval evaluation with Q&A pairs (Milestone 1: "Test retrieval and pipeline").

For every question in eval/qa_pairs.json we check whether the right chunk comes
back. A retrieved chunk counts as CORRECT when it is from the expected paper AND
contains one of the question's evidence phrases (case/whitespace-insensitive).
Matching the paper alone is also reported, since it is a looser signal.

Metrics (k = 5 by default):
  hit@1 / hit@3 / hit@k  - share of questions with a correct chunk in the top 1/3/k
  MRR                    - mean of 1/rank of the first correct chunk (0 if none in top k)
  paper hit@k            - share of questions with ANY chunk from the right paper in top k
  bibliography in top k  - share of questions whose top k contains a References-section chunk

Three configurations are compared, all with Tanish's model and query prefix:
  A  current      data/chunks.json as committed, queried through the Chroma
                  store (src/vector_store.py) -- the real pipeline
  B  cipher-fix   raw -> src/cleaning.py (ReAct fix) -> src/chunking.py (same settings)
  C  no-bib       B, plus --drop-references (bibliography chunks removed)
B and C are searched with exact cosine similarity (same embeddings Chroma uses);
A is also re-scored that way as a sanity check that Chroma returns the same results.

Usage (from the repo root, after `python src/vector_store.py build`):
    python eval/evaluate_retrieval.py
    python eval/evaluate_retrieval.py -k 10 --only A

Writes eval/results/retrieval_report.md and eval/results/per_question.csv.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import chunking  # noqa: E402
import cleaning  # noqa: E402
import vector_store  # noqa: E402

QA_PATH = ROOT / "eval" / "qa_pairs.json"
RESULTS_DIR = ROOT / "eval" / "results"


def _norm(s):
    return " ".join(s.lower().split())


def is_correct(chunk_text, chunk_meta, qa):
    if chunk_meta["paper_id"] != qa["paper_id"]:
        return False
    text = _norm(chunk_text)
    return any(_norm(e) in text for e in qa["evidence"])


def bibliography_flags(chunks, cleaned_papers):
    """True for chunks whose midpoint lies inside their paper's References section."""
    spans = {}
    for p in cleaned_papers:
        spans[p["id"]] = chunking.reference_span(
            chunking.SEP.join(pg["text"].strip() for pg in p["pages"])
        )
    flags = []
    for c in chunks:
        span = spans.get(c["metadata"]["paper_id"])
        mid = c["metadata"]["start_index"] + len(c["page_content"]) // 2
        flags.append(bool(span and span[0] <= mid < span[1]))
    return flags


class Embedder:
    """Wraps Tanish's model settings; caches passage embeddings by text so the
    three configurations (which share most chunks) are only embedded once."""

    def __init__(self):
        self.model = vector_store.get_model()
        self._cache = {}

    def passages(self, texts):
        todo = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if todo:
            embs = self.model.encode(todo, batch_size=64, normalize_embeddings=True,
                                     show_progress_bar=len(todo) > 64)
            self._cache.update(zip(todo, np.asarray(embs)))
        return np.stack([self._cache[t] for t in texts])

    def queries(self, questions):
        return np.asarray(self.model.encode(
            [vector_store.QUERY_PREFIX + q for q in questions], normalize_embeddings=True
        ))


def exact_search(q_embs, p_embs, k):
    sims = q_embs @ p_embs.T
    return np.argsort(-sims, axis=1)[:, :k]


def score_config(name, qa_pairs, ranked, chunks, bib_flags, k):
    rows = []
    for qa, idxs in zip(qa_pairs, ranked):
        first = next((r for r, i in enumerate(idxs, 1)
                      if is_correct(chunks[i]["page_content"], chunks[i]["metadata"], qa)), None)
        paper_first = next((r for r, i in enumerate(idxs, 1)
                            if chunks[i]["metadata"]["paper_id"] == qa["paper_id"]), None)
        top = chunks[idxs[0]]["metadata"]
        rows.append({
            "config": name,
            "id": qa["id"],
            "question": qa["question"],
            "expected_paper": qa["paper_id"],
            "rank_correct": first or "",
            "rank_paper": paper_first or "",
            "bib_in_topk": sum(bib_flags[i] for i in idxs),
            "top1": f'{top["paper_id"]} p.{top["page_number"]}-{top["end_page"]}',
        })
    n = len(rows)
    ranks = [r["rank_correct"] or None for r in rows]
    summary = {
        "config": name,
        "chunks": len(chunks),
        "hit@1": sum(1 for r in ranks if r and r <= 1) / n,
        "hit@3": sum(1 for r in ranks if r and r <= 3) / n,
        f"hit@{k}": sum(1 for r in ranks if r) / n,
        "MRR": sum(1 / r for r in ranks if r) / n,
        f"paper hit@{k}": sum(1 for r in rows if r["rank_paper"]) / n,
        f"bibliography in top{k}": sum(1 for r in rows if r["bib_in_topk"]) / n,
    }
    return summary, rows


def chroma_ranked(qa_pairs, chunks, k):
    """Config A through the real Chroma store (Tanish's query function)."""
    ids = vector_store.chunk_ids(chunks)
    pos = {cid: i for i, cid in enumerate(ids)}
    collection = vector_store.get_collection()
    if collection.count() != len(chunks):
        raise SystemExit(
            f"Chroma has {collection.count()} chunks but data/chunks.json has {len(chunks)}. "
            "Rebuild with: python src/vector_store.py build"
        )
    return [[pos[r["id"]] for r in vector_store.query(qa["question"], k=k, collection=collection)]
            for qa in qa_pairs]


def write_report(summaries, rows, k, notes):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "per_question.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    cols = list(summaries[0].keys())
    lines = [
        "# Retrieval evaluation",
        "",
        f"Model: `{vector_store.MODEL_NAME}` (query prefix on), top-k = {k}, "
        f"{len({r['id'] for r in rows})} Q&A pairs from `eval/qa_pairs.json`.",
        "A retrieved chunk is **correct** if it is from the expected paper and contains an evidence phrase.",
        "",
        "| " + " | ".join(cols) + " |",
        "|" + "---|" * len(cols),
    ]
    for s in summaries:
        lines.append("| " + " | ".join(
            f"{v:.2f}" if isinstance(v, float) else str(v) for v in s.values()) + " |")
    lines += ["", *notes, "", "## Misses (no correct chunk in top k)", ""]
    for cfg in [s["config"] for s in summaries]:
        missed = [r for r in rows if r["config"] == cfg and not r["rank_correct"]]
        lines.append(f"**{cfg}**: {len(missed)} missed")
        for r in missed:
            lines.append(f"- {r['id']} ({r['expected_paper']}): {r['question']} "
                         f"-> top1 {r['top1']}, right paper at rank {r['rank_paper'] or '-'}")
        lines.append("")
    (RESULTS_DIR / "retrieval_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--only", choices=["A", "B", "C"], nargs="+", default=["A", "B", "C"])
    args = ap.parse_args()
    k = args.k

    qa_pairs = json.loads(QA_PATH.read_text(encoding="utf-8"))
    questions = [qa["question"] for qa in qa_pairs]
    emb = Embedder()
    q_embs = emb.queries(questions)
    summaries, rows, notes = [], [], []

    committed_clean = json.loads((ROOT / "data" / "cleaned_papers_text.json").read_text(encoding="utf-8"))
    if "A" in args.only:
        chunks_a = vector_store.load_chunks()
        bib_a = bibliography_flags(chunks_a, committed_clean)
        ranked = chroma_ranked(qa_pairs, chunks_a, k)
        exact = exact_search(q_embs, emb.passages([c["page_content"] for c in chunks_a]), k)
        same_top1 = sum(r[0] == e[0] for r, e in zip(ranked, exact))
        notes.append(f"- Sanity check: Chroma and exact cosine search agree on the top-1 chunk "
                     f"for {same_top1}/{len(qa_pairs)} questions.")
        s, r = score_config("A current", qa_pairs, ranked, chunks_a, bib_a, k)
        summaries.append(s); rows += r

    if {"B", "C"} & set(args.only):
        raw = json.loads((ROOT / "data" / "curated_papers_text.json").read_text(encoding="utf-8"))
        fixed_clean = cleaning.clean_papers(raw)
        for name, drop in [("B cipher-fix", False), ("C no-bib", True)]:
            if name[0] not in args.only:
                continue
            chunks = chunking.chunk_papers(fixed_clean, drop_references=drop)
            ranked = exact_search(q_embs, emb.passages([c["page_content"] for c in chunks]), k)
            s, r = score_config(name, qa_pairs, ranked, chunks, bibliography_flags(chunks, fixed_clean), k)
            summaries.append(s); rows += r

    write_report(summaries, rows, k, notes)
    for s in summaries:
        print("  ".join(f"{key}={v:.2f}" if isinstance(v, float) else f"{key}={v}" for key, v in s.items()))
    print(f"\nReport: {RESULTS_DIR / 'retrieval_report.md'}")


if __name__ == "__main__":
    main()
