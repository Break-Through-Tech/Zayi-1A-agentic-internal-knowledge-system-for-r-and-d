"""Answer-quality checks for the RAG pipeline (citations, hallucination, "I don't know").

evaluate_retrieval.py only asks "did the right chunk come back?". These checks look at the
generated ANSWER. Each one is a plain function that takes the dict returned by
rag_chain.answer() plus one ground-truth entry, so the same code works on a live result
or on a row replayed from a saved .jsonl, and none of it calls Gemini.

    from answer_checks import score_answer_quality
    cols = score_answer_quality(qa, result)     # flat dict of new columns

Three groups of checks:

  1. Citations      cited_chunk_support, invented_citations, uncited_sentence_share
  2. Hallucination  grounded_share, unsupported_sentences, unsupported_numbers, hallucination_flag
  3. "I don't know" said_no_info, abstained_correctly, over_abstained, answered_without_evidence

The hallucination check is a cheap text-overlap approximation, not a judge: it flags
sentences whose words (and any numbers) are not found in the retrieved sources. It can
miss a wrong claim that reuses source words, and can flag a correct paraphrase. The
thresholds below are starting values, to be tuned on the real answers.

Inputs
  result  {"answer": str,
           "citations": [{"source": n, "paper_id": ...}, ...],
           "sources":   [{"rank", "paper_id", "text", ...}, ...]}   (rag_chain.answer())
  qa      {"key_keywords": [...], "paper_id": ..., "answerable": bool (default True)}
          Questions whose answer is NOT in the papers set "answerable": false
          (see eval/unanswerable_qa.json).
"""
import re

# [1], [2, 3] and [1][2] -- same pattern as rag_chain.py (copied so this module needs no
# langchain/chromadb to import)
CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")

# The exact sentence Niv's prompt tells the model to use (src/generation.py)
NO_INFO_PHRASE = "do not contain this information"
# Looser wording the model might use instead. Only counted when the answer cites nothing,
# because "the paper does not mention X, but ... [1]" is a normal answer, not an abstention.
LOOSE_NO_INFO = re.compile(
    r"(?:do(?:es)? not|don't|doesn't|cannot|can't|unable to)\s+"
    r"(?:contain|include|mention|provide|cover|find|answer|address)"
    r"|no (?:relevant |such )?information",
    re.IGNORECASE,
)

# Tunable defaults (calibrate on real answers once we have a key)
SENTENCE_SUPPORT_MIN = 0.5   # share of a sentence's content words found in its sources
GROUNDED_SHARE_MIN = 0.75    # below this, the answer is flagged

STOPWORDS = frozenset("""
a about above after again all also am an and any are as at be because been before being
below between both but by can could did do does doing down during each few for from further
had has have having he her here hers him his how i if in into is it its just may me might
more most my no nor not of off on once only or other our out over own same she should so
some such than that the their them then there these they this those through to too under
until up us very was we were what when where which while who whom why will with would you
your paper papers model models method approach
""".split())

_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*%?")


# ------------------------------------------------------------------ text helpers
def cited_numbers(answer_text):
    """Every [n] marker in the answer, in order, including duplicates and bad numbers."""
    out = []
    for m in CITATION_RE.finditer(answer_text):
        out.extend(int(n) for n in m.group(1).split(","))
    return out


def strip_citations(text):
    return CITATION_RE.sub("", text)


def split_sentences(text):
    """Split an answer into sentences/bullets. Citation markers stay attached."""
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\[(])|\n+", text.strip())
    cleaned = (re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", p).strip() for p in parts)
    return [p for p in cleaned if p]


def _norm_token(tok):
    # tiny plural stemmer so "weights" matches "weight"; leaves "class", "loss" alone
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def content_tokens(text):
    """Lower-cased, de-pluralised words that carry meaning (no stopwords, no [n] markers)."""
    toks = re.findall(r"[a-z0-9]+", strip_citations(text).lower())
    return {_norm_token(t) for t in toks if t not in STOPWORDS and (len(t) > 2 or t.isdigit())}


def _numbers(text):
    """Numbers worth checking: 2+ digits, decimals, or percentages. Lone digits are skipped
    because papers write 'two'/'three' in words."""
    found = set()
    for m in _NUMBER_RE.findall(strip_citations(text)):
        norm = m.replace(",", "")
        if len(norm.rstrip("%")) >= 2 or "." in norm or norm.endswith("%"):
            found.add(norm)
    return found


def _source_text(result, indices=None):
    srcs = result.get("sources", [])
    if indices is not None:
        srcs = [srcs[i - 1] for i in indices if 1 <= i <= len(srcs)]
    return "\n".join(s.get("text", "") for s in srcs)


def _valid_cited_indices(text, n_sources):
    seen = []
    for n in cited_numbers(text):
        if 1 <= n <= n_sources and n not in seen:
            seen.append(n)
    return seen


# ------------------------------------------------------------------ "I don't know"
def is_no_info(answer_text, n_valid_citations=0):
    """True if the answer says the sources don't contain the information."""
    if NO_INFO_PHRASE in answer_text.lower():
        return True
    return n_valid_citations == 0 and bool(LOOSE_NO_INFO.search(answer_text))


# ------------------------------------------------------------------ 1. citations
def check_citations(qa, result):
    answer = result["answer"]
    n_sources = len(result.get("sources", []))
    raw = cited_numbers(answer)
    valid = _valid_cited_indices(answer, n_sources)
    invented = sum(1 for n in raw if not 1 <= n <= n_sources)

    keywords = [k.lower() for k in qa.get("key_keywords") or []]
    if not keywords:
        support = None
    elif not valid:
        support = 0.0
    else:
        cited_text = _source_text(result, valid).lower()
        support = sum(1 for k in keywords if k in cited_text) / len(keywords)

    sentences = [s for s in split_sentences(answer) if not is_no_info(s)]
    uncited = (sum(1 for s in sentences if not CITATION_RE.search(s)) / len(sentences)
               if sentences else None)
    return {
        "cited_chunk_support": support,        # share of key_keywords found in the CITED chunks
        "invented_citations": invented,        # [n] markers pointing at a source that doesn't exist
        "uncited_sentence_share": uncited,     # share of answer sentences with no [n] at all
    }


# ------------------------------------------------------------------ 2. hallucination
def lexical_support(sentence, source_text):
    """Share of the sentence's content words that appear in the source text (0..1)."""
    words = content_tokens(sentence)
    if not words:
        return 1.0
    return len(words & content_tokens(source_text)) / len(words)


def check_grounding(result, support_fn=lexical_support,
                    sentence_min=SENTENCE_SUPPORT_MIN, share_min=GROUNDED_SHARE_MIN):
    """Is each sentence of the answer supported by the sources it cites (or, if it cites
    nothing, by any retrieved source)? `support_fn(sentence, source_text) -> 0..1` can be
    swapped for an embedding-based scorer."""
    answer = result["answer"]
    n_sources = len(result.get("sources", []))
    all_text = _source_text(result)
    all_numbers = {n.replace(",", "") for n in _NUMBER_RE.findall(all_text)}

    checked, unsupported, bad_numbers = 0, [], []
    for sent in split_sentences(answer):
        if is_no_info(sent):
            continue
        checked += 1
        idx = _valid_cited_indices(sent, n_sources)
        text = _source_text(result, idx) if idx else all_text
        if support_fn(sent, text) < sentence_min:
            unsupported.append(sent)
        for num in sorted(_numbers(sent)):
            if num not in all_numbers and num.rstrip("%") not in all_numbers:
                bad_numbers.append(num)

    share = (checked - len(unsupported)) / checked if checked else None
    return {
        "grounded_share": share,                       # share of sentences supported by the sources
        "unsupported_sentences": unsupported,          # the sentences that were not
        "unsupported_numbers": sorted(set(bad_numbers)),  # numbers in the answer found in NO source
        "hallucination_flag": bool(bad_numbers) or (share is not None and share < share_min),
    }


# ------------------------------------------------------------------ 3. "I don't know"
def check_abstention(qa, result):
    answer = result["answer"]
    n_sources = len(result.get("sources", []))
    valid = _valid_cited_indices(answer, n_sources)
    said = is_no_info(answer, len(valid))
    answerable = qa.get("answerable", True)

    keywords = [k.lower() for k in qa.get("key_keywords") or []]
    sources_lower = _source_text(result).lower()
    evidence = any(k in sources_lower for k in keywords) if keywords else None

    return {
        "said_no_info": said,
        # unanswerable question: did it say so instead of making something up?
        "abstained_correctly": (said and not valid) if not answerable else None,
        # answerable question: did it refuse although we can answer it?
        "over_abstained": said if answerable else None,
        # answered confidently although no key term from the ground truth is in the sources
        "answered_without_evidence": (not said and evidence is False) if answerable else None,
    }


# ------------------------------------------------------------------ everything together
def score_answer_quality(qa, result, support_fn=lexical_support):
    """All new columns for one question, ready to merge into a row (e.g. in run_qa.score())."""
    return {**check_citations(qa, result),
            **check_grounding(result, support_fn=support_fn),
            **check_abstention(qa, result)}


def summarize_quality(rows):
    """Averages over rows that have the quality columns (skips errored / unscored rows)."""
    ok = [r for r in rows if not r.get("error") and "hallucination_flag" in r]
    n = len(ok)
    if not n:
        return {"scored": 0}

    def share(key, subset=None):
        subset = ok if subset is None else subset
        return sum(1 for r in subset if r.get(key)) / len(subset) if subset else None

    def mean(key):
        vals = [r[key] for r in ok if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    unanswerable = [r for r in ok if r.get("abstained_correctly") is not None]
    answerable = [r for r in ok if r.get("over_abstained") is not None]
    return {
        "scored": n,
        "cited_chunk_support": mean("cited_chunk_support"),
        "invented_citation_rate": share("invented_citations"),
        "uncited_sentence_share": mean("uncited_sentence_share"),
        "grounded_share": mean("grounded_share"),
        "hallucination_flag_rate": share("hallucination_flag"),
        "abstained_correctly_rate": share("abstained_correctly", unanswerable),   # unanswerable Qs
        "over_abstained_rate": share("over_abstained", answerable),               # answerable Qs
        "answered_without_evidence_rate": share("answered_without_evidence", answerable),
    }
