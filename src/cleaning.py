"""Clean the curated papers: data/curated_papers_text.json -> data/cleaned_papers_text.json.

This is the cleaning logic from notebooks/data_exploration.ipynb (Godwin) moved
into an importable module so the end-to-end pipeline test can run it, plus:

  * Fix for the ReAct (2210.03629) font-cipher decoder:
      - the shift search tie-broke on the FIRST shift reaching the best score, so a
        wrong shift that turned a whole line into one space-less "word" (e.g.
        'BnkjpNksIkpknolknpo' instead of 'Front Row Motorsports') could win.
        Ties now go to the decode with more real words (more tokens).
      - the shift is now chosen once per page (all cipher lines on a page use the
        same font), so tiny fragments can't pick their own wrong shift.
      - short cipher lines with no control characters (e.g. '7RXFK' -> 'Touch')
        are decoded when they sit next to a decoded cipher line.
      - the cipher font's ellipsis glyph 'ª' became a stray 'a' after NFKC; it now
        becomes '...'.
  * Niv's manual edits to cleaned_papers_text.json (removing per-page running
    headers such as 'Preprint.' / 'Published as a conference paper at ICLR 2023'
    and page-number lines at the top/bottom of each page) are now done in code,
    so re-running cleaning reproduces them instead of silently undoing them.

Usage (from the repo root):
    python src/cleaning.py
"""
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import ftfy

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "curated_papers_text.json"
CLEAN_PATH = ROOT / "data" / "cleaned_papers_text.json"

ARXIV_HEADER_RE = re.compile(
    r"arXiv:\d{4}\.\d{4,5}v\d+\s*\[[\w\.\-]+\]\s*\d{1,2}\s+\w+\s+\d{4}"
)
PAGE_NUMBER_RE = re.compile(r"\n\s*\d{1,4}\s*\n")
REFERENCES_HEADING_RE = re.compile(r"\n\s*References\s*\n", re.IGNORECASE)
HYPHEN_LINEBREAK_RE = re.compile(r"(\w)-\s*\n\s*(\w)")
TRAILING_PAGE_NUMBER_RE = re.compile(r"\n\s*\d{1,4}\s*$")
NON_ASCII_RE = re.compile(r"[^\x00-\x7F]+")
_CLEAN_TOKEN_RE = re.compile(r"^[A-Za-z0-9,.:;!?'\"()\[\]/\-]+$")

# Glyphs the ReAct cipher font uses that fall outside the shifted ASCII range
_CIPHER_GLYPHS = {"ª": "..."}  # 'ª' is the font's ellipsis


# --------------------------------------------------------------------------
# Font-cipher repair (ReAct appendix / Figure 1)
# --------------------------------------------------------------------------
def _shift_decode_char(ch, shift):
    o = ord(ch)
    if ch in "\n\t\r" or o >= 127:
        return ch
    return chr((o + shift) % 127)


def _shift_decode(text, shift):
    return "".join(_shift_decode_char(c, shift) for c in text)


def _token_purity(text):
    """(fraction of whitespace tokens that look like plain words, token count)."""
    tokens = text.split()
    if not tokens:
        return 0.0, 0
    clean = sum(1 for t in tokens if _CLEAN_TOKEN_RE.match(t))
    return clean / len(tokens), len(tokens)


def _has_ctrl(line):
    return any(ord(c) < 32 and c not in "\n\t\r" for c in line)


def _is_suspicious(line):
    """A line written in the cipher font: no real spaces, has control chars
    (the font's space glyph lands on \\x03)."""
    return len(line) > 8 and " " not in line and _has_ctrl(line)


def _best_shift(block):
    """Shift with the highest purity; ties go to the decode with MORE tokens.

    The old version kept the first shift reaching the top score, and a shift
    that maps the cipher's space glyph to a letter yields a single all-letter
    token (purity 1.0) -- which is how 'Front Row Motorsports' became
    'BnkjpNksIkpknolknpo'.
    """
    best = (-1.0, -1, 0)
    for shift in range(127):
        purity, n_tokens = _token_purity(_shift_decode(block, shift))
        if (purity, n_tokens) > best[:2]:
            best = (purity, n_tokens, shift)
    return best[2], best[0]


def _decode_line(line, shift):
    out = _shift_decode(line, shift)
    for glyph, repl in _CIPHER_GLYPHS.items():
        out = out.replace(glyph, repl)
    return out


def repair_font_encoding_artifacts(text, min_score=0.85):
    """Decode runs of cipher-font text (see module docstring).

    Detection is line by line: cipher lines contain raw control characters and
    no real spaces, which normal extracted text never does. All cipher lines on
    one page share a font, so the shift is chosen once per page. If the best
    decode doesn't clear `min_score`, the page is left untouched (so a stray
    control character inside an equation is not scrambled).
    """
    lines = text.split("\n")
    flagged = [i for i, l in enumerate(lines) if _is_suspicious(l)]
    if not flagged:
        return text

    shift, score = _best_shift("\n".join(lines[i] for i in flagged))
    if score < min_score:
        return text

    decode = set(flagged)
    # Short cipher fragments ('7RXFK', ', \x03ª]') have no control chars or are
    # too short to be flagged. Pick them up when they touch a flagged line and
    # clearly decode to cleaner, lower-case text than the original.
    changed = True
    while changed:
        changed = False
        for i, line in enumerate(lines):
            if i in decode or not line.strip() or " " in line:
                continue
            if (i - 1) not in decode and (i + 1) not in decode:
                continue
            dec = _decode_line(line, shift)
            dec_purity, _ = _token_purity(dec)
            orig_lower = sum(c.islower() for c in line)
            dec_lower = sum(c.islower() for c in dec)
            if _has_ctrl(line) or (dec_purity == 1.0 and dec_lower > orig_lower and orig_lower == 0):
                decode.add(i)
                changed = True

    return "\n".join(_decode_line(l, shift) if i in decode else l for i, l in enumerate(lines))


# --------------------------------------------------------------------------
# Unicode + page cleaning (unchanged from Godwin's notebook)
# --------------------------------------------------------------------------
def _strip_accents(text):
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def normalize_unicode_text(text):
    text = repair_font_encoding_artifacts(text)  # must run first -- needs raw control chars
    text = ftfy.fix_text(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace(" ", " ")
    text = "".join(ch for ch in text if ch.isprintable() or ch in "\n\t")
    text = text.replace("​", "").replace("﻿", "")
    text = _strip_accents(text)
    text = NON_ASCII_RE.sub(" ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def clean_page_text(text):
    text = normalize_unicode_text(text)
    text = ARXIV_HEADER_RE.sub(" ", text)
    text = PAGE_NUMBER_RE.sub("\n", text)
    # Drop a page number on the page's last line BEFORE re-joining hyphenated
    # words, otherwise 'lan-\n8' becomes 'lan8'.
    text = TRAILING_PAGE_NUMBER_RE.sub("", text)
    text = HYPHEN_LINEBREAK_RE.sub(r"\1\2", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


# --------------------------------------------------------------------------
# Page-edge cleanup (codifies Niv's manual edits)
# --------------------------------------------------------------------------
_PAGE_NUM_LINE_RE = re.compile(r"^\s*\d{1,4}\s*$")


def find_running_header(page_texts, min_fraction=0.5):
    """A first line repeated at the top of >= min_fraction of a paper's pages
    (e.g. 'Preprint.', 'Published as a conference paper at ICLR 2023')."""
    firsts = [t.split("\n", 1)[0].strip() for t in page_texts if t.strip()]
    if len(firsts) < 3:
        return None
    line, count = Counter(firsts).most_common(1)[0]
    if count / len(firsts) >= min_fraction and 0 < len(line) <= 100:
        return line
    return None


def strip_page_edges(text, running_header=None):
    lines = text.split("\n")
    if running_header and lines and lines[0].strip() == running_header:
        lines = lines[1:]
    if lines and _PAGE_NUM_LINE_RE.match(lines[-1]):
        lines = lines[:-1]
    if lines and _PAGE_NUM_LINE_RE.match(lines[0]):
        lines = lines[1:]
    return "\n".join(lines).strip()


# --------------------------------------------------------------------------
# Whole-paper helpers
# --------------------------------------------------------------------------
def strip_references_section(full_text):
    match = REFERENCES_HEADING_RE.search(full_text)
    if match:
        return full_text[:match.start()].strip()
    return full_text


def join_and_clean_pages(cleaned_pages):
    joined = "\n".join(page["text"] for page in cleaned_pages)
    joined = PAGE_NUMBER_RE.sub("\n", joined)
    joined = re.sub(r"\n{2,}", "\n", joined)
    return joined.strip()


def clean_papers(papers):
    cleaned_papers = []
    for p in papers:
        texts = [clean_page_text(page["text"]) for page in p["pages"]]
        header = find_running_header(texts)
        cleaned_pages = [
            {"page_number": page["page_number"], "text": strip_page_edges(t, header)}
            for page, t in zip(p["pages"], texts)
        ]
        full_clean_text = strip_references_section(join_and_clean_pages(cleaned_pages))
        cleaned_papers.append({
            "id": p["id"],
            "title": p["title"],
            "category": p["category"],
            "pdf_url": p["pdf_url"],
            "total_pages": p["total_pages"],
            "pages": cleaned_pages,
            "clean_full_text": full_clean_text,
        })
    return cleaned_papers


def main(raw_path=RAW_PATH, out_path=CLEAN_PATH):
    with open(raw_path, "r", encoding="utf-8") as f:
        papers = json.load(f)
    cleaned = clean_papers(papers)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)
    print(f"Cleaned {len(cleaned)} papers -> {out_path}")


if __name__ == "__main__":
    main()
