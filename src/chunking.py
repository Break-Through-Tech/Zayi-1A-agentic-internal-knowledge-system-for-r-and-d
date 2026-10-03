"""Chunk the cleaned papers: data/cleaned_papers_text.json -> data/chunks.json.

Same logic and settings as notebooks/chunking_documents.ipynb (Niv):
RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=125) over each
paper's per-page text joined with '\n', with page_number / end_page mapped
back from start_index.

Optional: `drop_references=True` removes bibliography chunks. The notebook
chunks the per-page text, which still contains each paper's References
section. Those chunks are dense with other papers' titles, so they can
outrank the real answer at query time. We drop only chunks whose midpoint
falls between the 'References' heading and the first appendix heading, so
appendices after the bibliography (ReAct trajectories, Self-RAG details,
prompt templates...) are kept -- unlike clean_full_text, which cuts
everything after 'References'.

Usage (from the repo root):
    python src/chunking.py                    # same output as the notebook
    python src/chunking.py --drop-references  # also filter bibliography chunks
"""
import argparse
import bisect
import json
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT = Path(__file__).resolve().parent.parent
CLEAN_PATH = ROOT / "data" / "cleaned_papers_text.json"
CHUNKS_PATH = ROOT / "data" / "chunks.json"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 125
SEP = "\n"

_REF_HEADING_RE = re.compile(r"^\s*(?:References|REFERENCES|Bibliography)\s*$", re.M)
# First appendix heading after the bibliography: 'APPENDIX', 'Appendices for ...',
# 'A Task Prompts', or small-caps 'A A DDITIONAL RESULTS'. Author initials like
# 'A. Sharma' do not match (no period allowed after the 'A').
_APPENDIX_RE = re.compile(
    r"^(?:APPENDIX|Appendix|Appendices)\b|^A\s+[A-Z](?:[a-z]+|[A-Z]+|\s[A-Z]{2,})\b", re.M
)


def reference_span(full_text):
    """(start, end) character span of the bibliography, or None.

    Starts at the LAST 'References' heading and ends at the first appendix
    heading after it (or the end of the paper if there is no appendix).
    """
    heads = list(_REF_HEADING_RE.finditer(full_text))
    if not heads:
        return None
    start = heads[-1].start()
    app = _APPENDIX_RE.search(full_text, heads[-1].end())
    return start, (app.start() if app else len(full_text))


def build_documents(papers):
    documents, page_tables, ref_spans = [], {}, {}
    for paper in papers:
        texts = [page["text"].strip() for page in paper["pages"]]
        nums = [page["page_number"] for page in paper["pages"]]
        starts, pos = [], 0
        for t in texts:
            starts.append(pos)
            pos += len(t) + len(SEP)
        page_tables[paper["id"]] = (starts, nums)
        ref_spans[paper["id"]] = reference_span(SEP.join(texts))
        documents.append(Document(
            page_content=SEP.join(texts),
            metadata={
                "paper_id": paper["id"],
                "title": paper["title"],
                "category": paper["category"],
                "pdf_url": paper["pdf_url"],
                "total_pages": paper["total_pages"],
            },
        ))
    return documents, page_tables, ref_spans


def chunk_papers(papers, drop_references=False):
    documents, page_tables, ref_spans = build_documents(papers)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True
    )
    chunks = splitter.split_documents(documents)
    out = []
    for c in chunks:
        starts, nums = page_tables[c.metadata["paper_id"]]
        si = c.metadata["start_index"]
        ei = si + len(c.page_content) - 1
        c.metadata["page_number"] = nums[bisect.bisect_right(starts, si) - 1]
        c.metadata["end_page"] = nums[bisect.bisect_right(starts, ei) - 1]
        span = ref_spans[c.metadata["paper_id"]]
        mid = si + len(c.page_content) // 2
        if drop_references and span and span[0] <= mid < span[1]:
            continue  # chunk is mostly bibliography
        out.append({"page_content": c.page_content, "metadata": c.metadata})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drop-references", action="store_true")
    ap.add_argument("--out", default=str(CHUNKS_PATH))
    args = ap.parse_args()
    with open(CLEAN_PATH, "r", encoding="utf-8") as f:
        papers = json.load(f)
    chunks = chunk_papers(papers, drop_references=args.drop_references)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(chunks)} chunks -> {args.out}")


if __name__ == "__main__":
    main()
