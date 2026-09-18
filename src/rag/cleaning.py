"""Element cleaning.

Three corrections to the prototype live here.

**Furniture detection.** The prototype dropped elements whose Unstructured
category was ``Header``/``Footer``. That is not enough: on a real book the
``fast`` strategy labelled the running header a ``Title``, so 174 copies of
``rlhfbook.com`` sailed through and landed mid-paragraph, cutting sentences in
half. Category is a hint, not evidence. What actually identifies furniture is
*repetition across pages*, so that is what :func:`detect_furniture` measures.

**Fragment reassembly.** Display equations get shattered by PDF text
extraction into runs of tiny elements (``'θ'``, ``'∞ X'``, ``'PK−1'``). The
prototype kept each as a standalone element -- 169 of them -- and some became
``Title``\\ s. Runs of consecutive fragments are merged back together here.

**Dedup stays local.** The prototype's sliding-window dedup was already
correct in design and is preserved: a global dedup would delete a sentence
that an author legitimately repeated in a later chapter, whereas text
repeating a few elements apart is nearly always an extraction artifact. Only
the comparison is upgraded, from exact hash to normalised near-match.
"""

from __future__ import annotations

import hashlib
import re
from collections import deque

import ftfy
from langchain_core.documents import Document

from rag.config import Settings, get_settings
from rag.logging_utils import get_logger
from rag.state import MetadataKeys as MK

logger = get_logger(__name__)

# Categories that are noise under every strategy. Note that Header/Footer are
# *not* relied upon alone -- see detect_furniture.
ALWAYS_DROP_CATEGORIES: frozenset[str] = frozenset({"PageNumber", "PageBreak"})

# Categories whose text must never be merged or reflowed.
STRUCTURAL_CATEGORIES: frozenset[str] = frozenset({"Table"})

# A run of this many consecutive tiny elements on one page is treated as a
# shattered equation rather than as several real short headings.
_MIN_FRAGMENT_RUN = 2
_FRAGMENT_MAX_CHARS = 15

# Repeated text longer than this is likely real content (a recurring
# definition, say), not a running header.
_FURNITURE_MAX_CHARS = 200

_BULLET_CHARS = "•▪▫◦‣⁃∙·"
_INLINE_BULLET_RE = re.compile(rf"\s*[{_BULLET_CHARS}]\s+")
_LEADING_BULLET_RE = re.compile(rf"^\s*[{_BULLET_CHARS}]\s*")
_HORIZONTAL_WS_RE = re.compile(r"[ \t\x0b\f\r]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_DIGITS_RE = re.compile(r"\d+")


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------


def clean_text(text: str) -> str:
    """Normalise one element's text.

    ``ftfy`` runs first so that later passes operate on repaired characters
    rather than mojibake.
    """
    text = ftfy.fix_text(text)

    # Curly quotes and dashes -> ASCII, so that a query typed with a straight
    # apostrophe matches text that was typeset with a curly one.
    text = (
        text.replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
        .replace("–", "-")
        .replace("—", "-")
    )

    # PDF extraction frequently merges several list items into one element,
    # leaving bullets stranded mid-line ("...signal. • Some models..."). Turn
    # those back into line-led items instead of dropping the marker.
    text = _LEADING_BULLET_RE.sub("", text)
    text = _INLINE_BULLET_RE.sub("\n- ", text)

    # Collapse horizontal runs but keep newlines: paragraph structure is what
    # the chunker splits on later.
    text = _HORIZONTAL_WS_RE.sub(" ", text)
    text = _BLANK_LINES_RE.sub("\n\n", text)
    text = "\n".join(line.strip() for line in text.split("\n"))

    return text.strip()


def _normalise_for_matching(text: str) -> str:
    """Aggressive normalisation used only for comparison, never for storage.

    Digits are stripped so that ``rlhfbook.com 12`` and ``rlhfbook.com 13``
    compare equal -- page-numbered running headers are the single most common
    furniture pattern and would otherwise look unique on every page.
    """
    lowered = text.lower()
    without_digits = _DIGITS_RE.sub("", lowered)
    return _NON_ALNUM_RE.sub("", without_digits)


def _normalise_for_dedupe(text: str) -> str:
    """Normalisation for duplicate detection; digits kept.

    Unlike furniture matching, dedupe must NOT strip digits: "accuracy
    improved to 71%" and "accuracy improved to 92%" are different facts, and
    collapsing them would silently delete real content. Only case and
    punctuation are normalised.
    """
    return _NON_ALNUM_RE.sub("", text.lower())


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Furniture detection
# ---------------------------------------------------------------------------


def _furniture_key(text: str) -> str:
    """Comparison key for furniture detection.

    Short non-sentence texts (running headers with page numbers) are matched
    digit-blind, so ``rlhfbook.com 12`` and ``rlhfbook.com 13`` compare
    equal. Anything that reads as a sentence (terminal punctuation) keeps
    its digits: ``Chapter 1 discusses X.`` and ``Chapter 2 discusses X.``
    are different content, and numbered lines like ``Exercise 1`` appearing
    at each chapter start must not be mistaken for a running header.
    """
    if text.rstrip().endswith((".", "!", "?")):
        return _normalise_for_dedupe(text)
    return _normalise_for_matching(text)


def detect_furniture(documents: list[Document], page_ratio: float) -> set[str]:
    """Return normalised texts that repeat across enough pages to be furniture.

    Counts *distinct pages* rather than occurrences, so a phrase used three
    times on one page is not mistaken for a running header.
    """
    pages_by_text: dict[str, set[int]] = {}
    all_pages: set[int] = set()

    for doc in documents:
        page = doc.metadata.get(MK.PAGE)
        if page is None:
            continue
        all_pages.add(page)

        text = doc.page_content.strip()
        if not text or len(text) > _FURNITURE_MAX_CHARS:
            continue

        key = _furniture_key(text)
        if not key:
            continue
        pages_by_text.setdefault(key, set()).add(page)

    total_pages = len(all_pages)
    if total_pages < 3:
        # Too few pages for repetition to mean anything.
        return set()

    threshold = max(2, int(total_pages * page_ratio))
    furniture = {key for key, pages in pages_by_text.items() if len(pages) >= threshold}

    if furniture:
        logger.info(
            "  furniture: %d repeated pattern(s) over %d pages (threshold %d pages)",
            len(furniture),
            total_pages,
            threshold,
        )
    return furniture


def remove_furniture(
    documents: list[Document], furniture: set[str]
) -> tuple[list[Document], int]:
    kept: list[Document] = []
    removed = 0

    for doc in documents:
        if _furniture_key(doc.page_content) in furniture:
            removed += 1
            continue
        kept.append(doc)

    return kept, removed


# ---------------------------------------------------------------------------
# Fragment reassembly
# ---------------------------------------------------------------------------


def _is_structural(doc: Document) -> bool:
    return doc.metadata.get(MK.CATEGORY) in STRUCTURAL_CATEGORIES


def reassemble_fragments(documents: list[Document]) -> tuple[list[Document], int]:
    """Merge runs of consecutive tiny elements back into single elements.

    Only runs of two or more are merged. A lone short element between two long
    ones is usually a genuine short heading (``Abstract``), whereas five
    consecutive two-character elements are the debris of a display equation.
    Requiring a run is what keeps real headings intact.
    """
    out: list[Document] = []
    run: list[Document] = []
    merged = 0

    def flush() -> None:
        nonlocal merged
        if not run:
            return
        if len(run) < _MIN_FRAGMENT_RUN:
            out.extend(run)
            run.clear()
            return

        text = " ".join(d.page_content.strip() for d in run if d.page_content.strip())
        base = dict(run[0].metadata)
        # These fragments are not headings, whatever Unstructured decided.
        base[MK.CATEGORY] = "Formula"
        base[MK.PAGE_END] = run[-1].metadata.get(MK.PAGE, base.get(MK.PAGE))
        out.append(Document(page_content=text, metadata=base))
        merged += len(run)
        run.clear()

    current_page: int | None = None

    for doc in documents:
        page = doc.metadata.get(MK.PAGE)
        short = len(doc.page_content.strip()) < _FRAGMENT_MAX_CHARS

        if short and not _is_structural(doc):
            # A run must stay on one page; equations do not span pages.
            if run and page != current_page:
                flush()
            current_page = page
            run.append(doc)
            continue

        flush()
        out.append(doc)

    flush()
    return out, merged


# ---------------------------------------------------------------------------
# Local-window deduplication
# ---------------------------------------------------------------------------


def dedupe_local(
    documents: list[Document], window: int
) -> tuple[list[Document], int]:
    """Drop near-duplicates that occur within ``window`` elements of each other.

    Deliberately local. A global dedup would delete a sentence an author
    repeated on purpose in a later chapter; text repeating a few elements
    apart is an extraction artifact -- a duplicated header, or overlap at a
    batch boundary.
    """
    if window <= 0:
        return documents, 0

    recent: deque[str] = deque(maxlen=window)
    recent_set: set[str] = set()
    kept: list[Document] = []
    removed = 0

    for doc in documents:
        key = _hash(_normalise_for_dedupe(doc.page_content))

        if key in recent_set:
            removed += 1
            continue

        if len(recent) == recent.maxlen and recent:
            evicted = recent[0]
            # Only forget the evicted key if no other slot still holds it.
            if sum(1 for item in recent if item == evicted) == 1:
                recent_set.discard(evicted)

        recent.append(key)
        recent_set.add(key)
        kept.append(doc)

    return kept, removed


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def clean_documents(
    documents: list[Document], settings: Settings | None = None
) -> list[Document]:
    """Run the full cleaning pass over one document's elements."""
    settings = settings or get_settings()

    if not documents:
        return []

    start_count = len(documents)

    # 1. Drop categories that are noise under any strategy.
    stage = [
        d
        for d in documents
        if d.metadata.get(MK.CATEGORY) not in ALWAYS_DROP_CATEGORIES
    ]

    # 2. Normalise text, dropping anything that cleans away to nothing.
    normalised: list[Document] = []
    for doc in stage:
        text = clean_text(doc.page_content)
        if not text:
            continue
        normalised.append(Document(page_content=text, metadata=dict(doc.metadata)))

    # 3. Furniture, detected by cross-page repetition rather than category.
    furniture = detect_furniture(normalised, settings.furniture_page_ratio)
    deduped_furniture, furniture_removed = remove_furniture(normalised, furniture)

    # 4. Reassemble shattered equations.
    reassembled, fragments_merged = reassemble_fragments(deduped_furniture)

    # 5. Local near-duplicate removal.
    final, duplicates_removed = dedupe_local(reassembled, settings.dedupe_window)

    logger.info(
        "  cleaned %d -> %d elements (furniture -%d, fragments merged %d, dupes -%d)",
        start_count,
        len(final),
        furniture_removed,
        fragments_merged,
        duplicates_removed,
    )
    return final
