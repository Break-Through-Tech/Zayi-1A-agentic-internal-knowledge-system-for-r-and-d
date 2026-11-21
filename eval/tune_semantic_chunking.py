"""Tune the semantic chunker and compare it with the current chunks (November task).

Same scoring as eval/evaluate_retrieval.py (Ivan): a retrieved chunk is correct if
it is from the expected paper and contains one of the question's evidence phrases
(eval/qa_pairs.json, 30 questions). Every config is embedded with our model
(bge-small + query prefix) and searched with exact cosine similarity, which gives
the same ranking as Chroma.

Fairness: bigger chunks contain more text, so they "contain the evidence" more
often. To compare chunkers of different sizes, we also report
`hit@budget`: is a correct chunk retrieved before the retrieved text adds up to
5,000 characters (= about 5 of the current 1,000-character chunks, i.e. the same
amount of text sent to Gemini)?

Stages:
  1. breakpoint threshold x buffer size     (min_chunk_size=0, max_chunk_size=2000)
  2. min / max chunk size                   (best threshold + buffer from stage 1)
  3. stretch: cross-encoder reranker on the baseline and the best semantic config
     (retrieve 20 with bge-small, rerank, keep the top 5)
Also reported: paper-level hit@1/@5 on Jose's 40 ground-truth questions
(data/ground_truth_qa.json has no evidence phrases, so only the paper can be checked).

Usage (from the repo root; ~45 min on a laptop CPU, window embeddings are cached):
    python eval/tune_semantic_chunking.py
    python eval/tune_semantic_chunking.py --no-rerank
Writes eval/semantic_chunking_report.md (tracked) and eval/results/semantic_sweep.csv.
"""
import argparse
import csv
import json
import pickle
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

import chunking  # noqa: E402
import semantic_chunking as sc  # noqa: E402
import vector_store  # noqa: E402
from evaluate_retrieval import Embedder, is_correct  # noqa: E402

RESULTS_DIR = ROOT / "eval" / "results"
CACHE_DIR = RESULTS_DIR / "semantic_cache"
REPORT_PATH = ROOT / "eval" / "semantic_chunking_report.md"
K = 5
BUDGET = 5000  # characters of context, about 5 baseline chunks
RANK_DEPTH = 50

BUFFERS = [0, 1, 2, 3]
THRESHOLDS = [("percentile", 80), ("percentile", 90), ("percentile", 95),
              ("standard_deviation", 1.5), ("standard_deviation", 3),  # 3 = SemanticChunker default
              ("interquartile", 1.5)]
MIN_SIZES = [0, 250, 500]
MAX_SIZES = [1000, 1500, 2000]
RERANKERS = ["cross-encoder/ms-marco-MiniLM-L6-v2", "BAAI/bge-reranker-base"]
FETCH_K = 20
CONCLUSION_MARKER = "<!-- conclusion: hand-written, kept on re-run -->"
CONCLUSION_END = "<!-- end conclusion -->"


def load_distances(papers, buffer_size):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"buffer{buffer_size}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    t = time.time()
    pd = sc.PaperDistances(papers, buffer_size)
    path.write_bytes(pickle.dumps(pd))
    print(f"  embedded sentence windows for buffer={buffer_size} in {time.time() - t:.0f}s")
    return pd


def n_truncated(chunks):
    tok = vector_store.get_model().tokenizer
    lens = tok([c["page_content"] for c in chunks], add_special_tokens=True, truncation=False)["input_ids"]
    return sum(len(x) > 512 for x in lens)


def rank_metrics(chunks, ranked, qa_pairs):
    """Evidence-phrase metrics for ranked chunk indices (one list per question)."""
    first, budget_hit = [], []
    for qa, idxs in zip(qa_pairs, ranked):
        r = next((r for r, i in enumerate(idxs[:K], 1)
                  if is_correct(chunks[i]["page_content"], chunks[i]["metadata"], qa)), None)
        first.append(r)
        used, hit = 0, False
        for i in idxs:
            used += len(chunks[i]["page_content"])
            if used > BUDGET and used != len(chunks[i]["page_content"]):  # always allow the first chunk
                break
            if is_correct(chunks[i]["page_content"], chunks[i]["metadata"], qa):
                hit = True
                break
        budget_hit.append(hit)
    n = len(qa_pairs)
    return {
        "hit@1": sum(1 for r in first if r == 1) / n,
        "hit@3": sum(1 for r in first if r and r <= 3) / n,
        "hit@5": sum(1 for r in first if r) / n,
        "MRR": sum(1 / r for r in first if r) / n,
        "hit@budget": sum(budget_hit) / n,
    }, first


def paper_metrics(chunks, ranked, gt):
    n = len(gt)
    return {
        "GT paper hit@1": sum(chunks[idxs[0]]["metadata"]["paper_id"] == g["paper_id"]
                              for g, idxs in zip(gt, ranked)) / n,
        "GT paper hit@5": sum(any(chunks[i]["metadata"]["paper_id"] == g["paper_id"] for i in idxs[:K])
                              for g, idxs in zip(gt, ranked)) / n,
    }


class Evaluator:
    def __init__(self, qa_pairs, gt):
        self.qa, self.gt = qa_pairs, gt
        self.emb = Embedder()  # caches chunk embeddings by text across configs
        self.q_embs = self.emb.queries([q["question"] for q in qa_pairs])
        self.g_embs = self.emb.queries([g["question"] for g in gt])

    def ranked(self, chunks, q_embs, depth=RANK_DEPTH):
        p = self.emb.passages([c["page_content"] for c in chunks])
        return np.argsort(-(q_embs @ p.T), axis=1)[:, :depth]

    def evaluate(self, name, chunks, settings=None):
        t = time.time()
        rq, rg = self.ranked(chunks, self.q_embs), self.ranked(chunks, self.g_embs)
        m, first = rank_metrics(chunks, rq, self.qa)
        sizes = [len(c["page_content"]) for c in chunks]
        row = {
            "config": name,
            **({k: v for k, v in asdict(settings).items()} if settings else {}),
            "chunks": len(chunks),
            "avg chars": round(float(np.mean(sizes))),
            "max chars": max(sizes),
            "truncated (>512 tok)": n_truncated(chunks),
            **m,
            **paper_metrics(chunks, rg, self.gt),
        }
        print(f"  {name:<45} chunks={row['chunks']:<5} avg={row['avg chars']:<5} "
              f"hit@1={m['hit@1']:.2f} hit@5={m['hit@5']:.2f} MRR={m['MRR']:.3f} "
              f"budget={m['hit@budget']:.2f}  ({time.time() - t:.0f}s)")
        return row, rq, rg, first


def score(row):
    """Selection rule: MRR, then hit@budget, then hit@5."""
    return (round(row["MRR"], 3), row["hit@budget"], row["hit@5"])


def name_of(s):
    t = {"percentile": "pct", "standard_deviation": "std", "interquartile": "iqr"}[s.breakpoint_threshold_type]
    return (f"semantic {t}={s.breakpoint_threshold_amount:g} buf={s.buffer_size} "
            f"min={s.min_chunk_size} max={s.max_chunk_size}")


def rerank_rows(ev, name, chunks, ranked_q, ranked_g, model_name):
    from sentence_transformers import CrossEncoder

    ce = CrossEncoder(model_name)
    t = time.time()

    def rerank(questions, ranked):
        out = []
        for q, idxs in zip(questions, ranked):
            cand = list(idxs[:FETCH_K])
            scores = ce.predict([(q, chunks[i]["page_content"]) for i in cand])
            out.append([cand[j] for j in np.argsort(-np.asarray(scores))] + list(idxs[FETCH_K:]))
        return out

    rq = rerank([q["question"] for q in ev.qa], ranked_q)
    rg = rerank([g["question"] for g in ev.gt], ranked_g)
    m, first = rank_metrics(chunks, rq, ev.qa)
    secs = (time.time() - t) / (len(ev.qa) + len(ev.gt))
    row = {"config": f"{name} + rerank {model_name.split('/')[-1]}", **m, **paper_metrics(chunks, rg, ev.gt),
           "rerank sec/query": round(secs, 2)}
    print(f"  {row['config']:<70} hit@1={m['hit@1']:.2f} hit@5={m['hit@5']:.2f} MRR={m['MRR']:.3f} "
          f"budget={m['hit@budget']:.2f}  ({secs:.2f}s/query)")
    return row, first


def table(rows, cols):
    def fmt(v):
        return f"{v:.2f}" if isinstance(v, float) else ("-" if v is None else str(v))
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(r.get(c)) for c in cols) + " |" for r in rows]
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-rerank", action="store_true")
    args = ap.parse_args()

    papers = json.loads(chunking.CLEAN_PATH.read_text(encoding="utf-8"))
    qa = json.loads((ROOT / "eval" / "qa_pairs.json").read_text(encoding="utf-8"))
    gt = json.loads((ROOT / "data" / "ground_truth_qa.json").read_text(encoding="utf-8"))
    ev = Evaluator(qa, gt)
    all_rows = []

    print("Baseline (current data/chunks.json pipeline: RecursiveCharacterTextSplitter 1000/125)")
    base_chunks = chunking.chunk_papers(papers)
    base, base_rq, base_rg, base_first = ev.evaluate("baseline recursive 1000/125", base_chunks)
    all_rows.append({**base, "stage": "baseline"})

    print("\nStage 1: breakpoint threshold x buffer size (min=0, max=2000)")
    stage1 = []
    dists = {}
    for b in BUFFERS:
        dists[b] = load_distances(papers, b)
        for kind, amount in THRESHOLDS:
            s = sc.SemanticSettings(kind, amount, b, 0, 2000)
            row = ev.evaluate(name_of(s), dists[b].chunks(s), s)[0]
            stage1.append({**row, "stage": "1"})
    all_rows += stage1
    best1 = max(stage1, key=score)
    s1 = sc.SemanticSettings(best1["breakpoint_threshold_type"], best1["breakpoint_threshold_amount"],
                             best1["buffer_size"], 0, 2000)

    print(f"\nStage 2: min / max chunk size around {name_of(s1)}")
    stage2 = []
    for mn in MIN_SIZES:
        for mx in MAX_SIZES:
            s = replace(s1, min_chunk_size=mn, max_chunk_size=mx)
            row = ev.evaluate(name_of(s), dists[s.buffer_size].chunks(s), s)[0]
            stage2.append({**row, "stage": "2"})
    all_rows += stage2
    best = max(stage2, key=score)
    best_s = sc.SemanticSettings(best["breakpoint_threshold_type"], best["breakpoint_threshold_amount"],
                                 best["buffer_size"], best["min_chunk_size"], best["max_chunk_size"])
    best_chunks = dists[best_s.buffer_size].chunks(best_s)
    _, best_rq, best_rg, best_first = ev.evaluate(name_of(best_s), best_chunks, best_s)

    rerank = []
    firsts = {"baseline": base_first, "semantic": best_first}
    if not args.no_rerank:
        print(f"\nStage 3 (stretch): rerank top {FETCH_K} -> top {K}")
        for model in RERANKERS:
            for label, chunks, rq, rg in [("baseline", base_chunks, base_rq, base_rg),
                                         ("best semantic", best_chunks, best_rq, best_rg)]:
                row, first = rerank_rows(ev, label, chunks, rq, rg, model)
                rerank.append(row)
                firsts[row["config"]] = first

    # ------------------------------------------------------------ outputs
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in all_rows + rerank for k in r))
    with open(RESULTS_DIR / "semantic_sweep.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(all_rows + rerank)

    metric_cols = ["hit@1", "hit@3", "hit@5", "MRR", "hit@budget", "GT paper hit@1", "GT paper hit@5"]
    size_cols = ["chunks", "avg chars", "max chars", "truncated (>512 tok)"]
    per_q = ["| question | baseline | best semantic |" + "".join(
        f" {c} |" for c in firsts if c not in ("baseline", "semantic")), "|---|---|---|" + "---|" * (len(firsts) - 2)]
    for i, q in enumerate(qa):
        per_q.append(f"| {q['id']} | " + " | ".join(str(f[i] or "-") for f in firsts.values()) + " |")

    lines = [
        "# Semantic chunking: tuning and comparison",
        "",
        f"Generated by `python eval/tune_semantic_chunking.py`. Embeddings: `{vector_store.MODEL_NAME}` "
        f"(query prefix on), exact cosine search, top-k = {K}.",
        f"Questions: {len(qa)} Q&A pairs with evidence phrases (`eval/qa_pairs.json`) and "
        f"{len(gt)} ground-truth questions (`data/ground_truth_qa.json`, paper-level only).",
        "",
        "- **hit@k / MRR**: a retrieved chunk is correct if it is from the right paper *and* contains an "
        "evidence phrase (same rule as `eval/evaluate_retrieval.py`).",
        f"- **hit@budget**: correct chunk found before the retrieved text passes {BUDGET:,} characters. "
        "This is the fair comparison when chunk sizes differ, because bigger chunks contain evidence phrases "
        "more often just by being bigger.",
        "- **truncated**: chunks longer than bge-small's 512-token limit (the end of the chunk is not embedded).",
        "",
        "## Baseline vs best semantic settings",
        "",
        *table([base, best], ["config"] + size_cols + metric_cols),
        "",
        "## Stage 1: breakpoint threshold x buffer size (min_chunk_size=0, max_chunk_size=2000)",
        "",
        *table(sorted(stage1, key=score, reverse=True), ["config"] + size_cols + metric_cols),
        "",
        "## Stage 2: min / max chunk size",
        "",
        *table(sorted(stage2, key=score, reverse=True), ["config"] + size_cols + metric_cols),
        "",
    ]
    if rerank:
        lines += [f"## Stage 3 (stretch): reranker (retrieve {FETCH_K} with bge-small, rerank, keep {K})", "",
                  *table(rerank, ["config"] + metric_cols + ["rerank sec/query"]), ""]
    lines += ["## Rank of the first correct chunk per question (- = not in top 5)", "", *per_q, ""]
    # Keep the hand-written conclusion (between the two markers) across re-runs
    old = REPORT_PATH.read_text(encoding="utf-8") if REPORT_PATH.exists() else ""
    if CONCLUSION_MARKER in old and CONCLUSION_END in old:
        conclusion = old[old.index(CONCLUSION_MARKER):old.index(CONCLUSION_END) + len(CONCLUSION_END)]
        lines.insert(lines.index("## Baseline vs best semantic settings"), conclusion + "\n")
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nBest semantic settings: {asdict(best_s)}")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
