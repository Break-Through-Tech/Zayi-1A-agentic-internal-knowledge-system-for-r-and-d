"""Semantic chunking: cut each paper where the topic changes, not every 1000 characters.

Same algorithm and parameter names as LangChain's SemanticChunker
(langchain_experimental.text_splitter), written here so it:
  * reuses our embedding model (bge-small, src/vector_store.py) -- no new dependency,
  * keeps character offsets, so page_number / end_page are mapped exactly like
    src/chunking.py does,
  * adds `max_chunk_size`, which SemanticChunker lacks. bge-small only reads the
    first 512 tokens (~2000 characters) of a chunk, so longer chunks lose text.

How it works, per paper:
  1. Split the text into sentences.
  2. For every sentence, embed a small window: the sentence plus `buffer_size`
     sentences on each side (smooths out short/noisy sentences).
  3. Distance between neighbouring windows = 1 - cosine similarity.
     A big distance means the topic changed.
  4. Cut wherever the distance is above a threshold:
       percentile          -> the `amount`-th percentile of the paper's distances
       standard_deviation  -> mean + amount * std
       interquartile       -> mean + amount * IQR
  5. `min_chunk_size`: a cut is skipped while the current chunk is still shorter
     than this (avoids tiny chunks such as a lone heading).
     `max_chunk_size`: an oversize chunk is split again at its biggest internal
     distance, until every piece fits (a single huge "sentence", e.g. a table,
     falls back to RecursiveCharacterTextSplitter).

Usage (from the repo root):
    python src/semantic_chunking.py                          # tuned defaults -> data/chunks_semantic.json
    python src/semantic_chunking.py --out data/chunks.json   # swap it into the pipeline
    python src/semantic_chunking.py --amount 90 --buffer-size 1 --min-chunk-size 200 --max-chunk-size 1500
Parameters were chosen with eval/tune_semantic_chunking.py. Result: semantic chunks
rank the right chunk first more often, but with the same amount of context they
find it no more often than the current chunks, so data/chunks.json is NOT
swapped yet -- see the conclusion in eval/semantic_chunking_report.md.
"""
import argparse
import bisect
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter

import chunking
import vector_store

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "data" / "chunks_semantic.json"

# Same sentence rule as SemanticChunker: split after . ? ! followed by whitespace
SENTENCE_RE = re.compile(r"(?<=[.?!])\s+")


@dataclass
class SemanticSettings:
    """Defaults = best balanced settings from eval/semantic_chunking_report.md:
    same hit@5 / hit@budget as the current chunks, higher hit@1 and MRR, and
    fewer over-length chunks than max_chunk_size=2000."""
    breakpoint_threshold_type: str = "percentile"  # percentile | standard_deviation | interquartile
    breakpoint_threshold_amount: float = 80
    buffer_size: int = 3
    min_chunk_size: int = 500
    max_chunk_size: Optional[int] = 1500


def sentence_spans(text):
    """(start, end) character spans of the sentences in `text`, whitespace excluded."""
    spans, pos = [], 0
    for m in SENTENCE_RE.finditer(text):
        if text[pos:m.start()].strip():
            spans.append((pos, m.start()))
        pos = m.end()
    if text[pos:].strip():
        spans.append((pos, len(text)))
    return spans


def window_texts(text, spans, buffer_size):
    """Each sentence joined with `buffer_size` neighbours on each side."""
    n = len(spans)
    return [
        " ".join(text[s:e] for s, e in spans[max(0, i - buffer_size):min(n, i + buffer_size + 1)])
        for i in range(n)
    ]


def embed_passages(texts, batch_size=64):
    return np.asarray(vector_store.get_model().encode(
        texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=len(texts) > 500
    ))


def neighbour_distances(embeddings):
    """distance[i] = 1 - cos(window i, window i+1); embeddings are already normalized."""
    if len(embeddings) < 2:
        return np.zeros(0)
    return 1.0 - np.sum(embeddings[:-1] * embeddings[1:], axis=1)


def threshold(distances, kind, amount):
    if len(distances) == 0:
        return np.inf
    if kind == "percentile":
        return float(np.percentile(distances, amount))
    if kind == "standard_deviation":
        return float(np.mean(distances) + amount * np.std(distances))
    if kind == "interquartile":
        q1, q3 = np.percentile(distances, [25, 75])
        return float(np.mean(distances) + amount * (q3 - q1))
    raise ValueError(f"unknown breakpoint_threshold_type: {kind}")


def _split_oversize(spans, distances, lo, hi, max_size, out):
    """Sentences lo..hi-1 form one group; split at the biggest distance until it fits."""
    if spans[hi - 1][1] - spans[lo][0] <= max_size or hi - lo == 1:
        out.append((lo, hi))
        return
    cut = lo + int(np.argmax(distances[lo:hi - 1]))  # cut after sentence `cut`
    _split_oversize(spans, distances, lo, cut + 1, max_size, out)
    _split_oversize(spans, distances, cut + 1, hi, max_size, out)


def group_sentences(spans, distances, settings):
    """Sentence index ranges [(lo, hi), ...] that make up each chunk."""
    t = threshold(distances, settings.breakpoint_threshold_type, settings.breakpoint_threshold_amount)
    groups, lo = [], 0
    for i, d in enumerate(distances):
        if d > t and spans[i][1] - spans[lo][0] >= settings.min_chunk_size:
            groups.append((lo, i + 1))
            lo = i + 1
    if spans:
        groups.append((lo, len(spans)))
    if settings.max_chunk_size:
        fitted = []
        for lo, hi in groups:
            _split_oversize(spans, distances, lo, hi, settings.max_chunk_size, fitted)
        groups = fitted
    return groups


def chunk_spans(text, spans, distances, settings):
    """Character spans (start, end) of the chunks of one paper."""
    out = []
    fallback = None
    for lo, hi in group_sentences(spans, distances, settings):
        start, end = spans[lo][0], spans[hi - 1][1]
        if settings.max_chunk_size and end - start > settings.max_chunk_size:
            # a single sentence longer than the limit (tables, equations, reference lists)
            fallback = fallback or RecursiveCharacterTextSplitter(
                chunk_size=settings.max_chunk_size, chunk_overlap=0, add_start_index=True
            )
            for piece in fallback.create_documents([text[start:end]]):
                s = start + piece.metadata["start_index"]
                out.append((s, s + len(piece.page_content)))
        else:
            out.append((start, end))
    return out


class PaperDistances:
    """Sentences + neighbour distances for every paper, for one buffer_size.

    Embedding is the slow part; thresholds and min/max sizes can then be swept
    for free with `chunks(settings)`.
    """

    def __init__(self, papers, buffer_size):
        self.buffer_size = buffer_size
        self.documents, self.page_tables, self.ref_spans = chunking.build_documents(papers)
        self.spans, windows, owner = [], [], []
        for pi, doc in enumerate(self.documents):
            sp = sentence_spans(doc.page_content)
            self.spans.append(sp)
            windows += window_texts(doc.page_content, sp, buffer_size)
            owner += [pi] * len(sp)
        embs = embed_passages(windows)
        owner = np.asarray(owner)
        self.distances = [neighbour_distances(embs[owner == pi]) for pi in range(len(self.documents))]

    def chunks(self, settings, drop_references=False):
        assert settings.buffer_size == self.buffer_size
        out = []
        for doc, sp, dist in zip(self.documents, self.spans, self.distances):
            text, meta = doc.page_content, doc.metadata
            starts, nums = self.page_tables[meta["paper_id"]]
            ref = self.ref_spans[meta["paper_id"]]
            for s, e in chunk_spans(text, sp, dist, settings):
                if drop_references and ref and ref[0] <= (s + e) // 2 < ref[1]:
                    continue
                out.append({
                    "page_content": text[s:e],
                    "metadata": {
                        **meta,
                        "start_index": s,
                        "page_number": nums[bisect.bisect_right(starts, s) - 1],
                        "end_page": nums[bisect.bisect_right(starts, e - 1) - 1],
                        "chunker": "semantic",
                    },
                })
        return out


def semantic_chunk_papers(papers, settings=None, drop_references=False):
    settings = settings or SemanticSettings()
    return PaperDistances(papers, settings.buffer_size).chunks(settings, drop_references)


def main():
    d = SemanticSettings()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threshold-type", default=d.breakpoint_threshold_type,
                    choices=["percentile", "standard_deviation", "interquartile"])
    ap.add_argument("--amount", type=float, default=d.breakpoint_threshold_amount)
    ap.add_argument("--buffer-size", type=int, default=d.buffer_size)
    ap.add_argument("--min-chunk-size", type=int, default=d.min_chunk_size)
    ap.add_argument("--max-chunk-size", type=int, default=d.max_chunk_size, help="0 = no limit")
    ap.add_argument("--drop-references", action="store_true")
    ap.add_argument("--out", default=str(OUT_PATH))
    args = ap.parse_args()

    settings = SemanticSettings(args.threshold_type, args.amount, args.buffer_size,
                                args.min_chunk_size, args.max_chunk_size or None)
    papers = json.loads(chunking.CLEAN_PATH.read_text(encoding="utf-8"))
    chunks = semantic_chunk_papers(papers, settings, drop_references=args.drop_references)
    Path(args.out).write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    sizes = [len(c["page_content"]) for c in chunks]
    print(f"Wrote {len(chunks)} chunks (avg {np.mean(sizes):.0f}, max {max(sizes)} chars) -> {args.out}")
    print("Settings:", asdict(settings))


if __name__ == "__main__":
    main()
