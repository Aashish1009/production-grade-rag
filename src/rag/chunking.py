"""Structure-aware, token-accurate chunking.

The prototype split the *whole document's* cleaned text with a
``RecursiveCharacterTextSplitter`` and threw away every element boundary the
parser had recovered. That discards the two most valuable signals Unstructured
gives us: where sections begin (so a chunk can carry a
``"Chapter 11 > Policy Gradients > GRPO"`` breadcrumb) and which elements are
tables (which must never be reflowed or split).

Chunk sizes are measured in *dense-model tokens*, not characters. A
character-based budget silently drifts whenever the model changes, and the
embedding model truncates at its context window -- so :func:`token_length`
uses the dense model's own tokenizer, loaded once and shared.

Elements are *packed* up to that budget rather than emitted one chunk each:
`chunk_size` is the context budget the answer stage spends, and a parser that
emits sentence-sized elements would otherwise shrink every passage to a
sentence. See :func:`chunk_documents`.

Chunk ids are content hashes (source + text), so re-ingesting an unchanged
document produces byte-identical ids and the Qdrant upsert is a no-op rather
than a duplicate. ``prev_id``/``next_id`` links are what the query stage uses
to widen a winning passage with the chunks around it.

Every chunk opens with its section breadcrumb, as text rather than metadata --
see :func:`_section_header` for why that placement is the difference between a
heading that helps retrieval and one that does not.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.config import Settings, get_settings
from rag.logging_utils import get_logger
from rag.state import MetadataKeys as MK

logger = get_logger(__name__)

# Split priorities: paragraphs first (chunks keep whole paragraphs together),
# then lines, then words. Newlines are separators, not just whitespace, so
# paragraph structure survives splitting.
_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# Headings deepen the breadcrumb path. Above this depth the hierarchy is
# usually extraction noise (a bolded word mistaken for a level-4 heading).
_MAX_SECTION_DEPTH = 3


def _section_header(section: str | None) -> str:
    """The breadcrumb line that opens a chunk, or ``""`` where there is none.

    Prepended into the chunk's ``page_content`` rather than left in metadata,
    because ``page_content`` is the *only* thing the retrieval stack ever sees:
    it is what the dense model embeds, what BM25 tokenises, what the
    cross-encoder scores against the question, and what the answer is written
    from. A breadcrumb that lives only in metadata is invisible to all four.

    That invisibility was costing real recall. A passage reading "the
    probability that i is preferred to j is the ratio of their strengths"
    contains none of the words "Bradley-Terry", yet it is the passage that
    answers a question about the Bradley-Terry equation -- the section heading
    it sits under says so, and nothing in the retrieval path could see it. With
    the heading in the text, the embedding for that chunk lands near the
    question, BM25 matches "bradley" and "terry" outright, and the reranker has
    the same evidence the reader does.
    """
    return f"{section}\n\n" if section else ""


# ---------------------------------------------------------------------------
# Token measurement
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _get_tokenizer(model_name: str):
    """Load the dense model's tokenizer once per process.

    Only the tokenizer files are downloaded (a few MB), never the model
    weights, so this stays cheap even though the embedding model itself is
    gigabytes.
    """
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name)


def token_length(text: str, model_name: str | None = None) -> int:
    """Measure ``text`` in dense-model tokens.

    Falls back to a character estimate (4 chars/token) if the tokenizer cannot
    be loaded offline, with a warning -- an estimate is still a better chunk
    budget than a character count. The warning is emitted once per process:
    this runs per element and per chunk, so repeating it would bury the rest
    of the log under thousands of identical lines.
    """
    global _TOKENIZER_WARNING_EMITTED
    model_name = model_name or get_settings().dense_model
    try:
        return len(_get_tokenizer(model_name).encode(text, add_special_tokens=False))
    except Exception as exc:
        if not _TOKENIZER_WARNING_EMITTED:
            _TOKENIZER_WARNING_EMITTED = True
            logger.warning(
                "tokenizer unavailable (%s: %s); estimating tokens as len/4 "
                "for the rest of this run",
                type(exc).__name__,
                exc,
            )
        return max(1, len(text) // 4)


_TOKENIZER_WARNING_EMITTED = False


def _content_hash(source: str, index: int, text: str) -> str:
    """Chunk id from (source, position, text).

    Position is part of the hash so a paragraph an author deliberately
    repeated beyond the dedupe window (which cleaning keeps, by design)
    still yields two distinct chunks instead of colliding on one Qdrant
    point id -- a collision would silently drop one copy and tangle the
    prev/next neighbour links of both.
    """
    digest = hashlib.sha256()
    digest.update(source.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(str(index).encode("utf-8"))
    digest.update(b"\x00")
    digest.update(text.encode("utf-8"))
    return digest.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Section breadcrumbs
# ---------------------------------------------------------------------------


def _section_path(documents: list[Document]) -> tuple[list[str | None], set[int]]:
    """Compute a ``"A > B > C"`` breadcrumb for each element.

    Walks the element stream keeping a stack of open headings; ``Title``
    elements at depth ``d`` replace everything above depth ``d`` and become
    the current innermost section. Non-heading elements inherit the current
    stack. Returns ``None`` where no heading has been seen yet.

    The second return value is the indices of the elements *consumed* as
    headings -- the ones whose text became part of a breadcrumb. They are
    reported rather than merely detected so the caller can leave them out of
    the chunk text: their words are already the line that opens the chunk they
    head, and packing them as content as well printed every heading twice.

    Unstructured reports heading depth 0-based where it reports it at all
    (DOCX, HTML, Markdown). ``category_depth = 0`` therefore means "top-level
    heading" and must reset the stack to depth 1 -- treating 0 as "unknown"
    would nest a new chapter under the previous one and produce a breadcrumb
    claiming the wrong parent.

    **A document that reports no depth anywhere is flat, and is treated as
    such.** Nesting its headings instead -- the ``len(stack) + 1`` fallback
    that has to guess when one element's depth is missing -- grows the stack
    until it exceeds ``_MAX_SECTION_DEPTH``, and every heading after that is
    discarded rather than recorded. The stack then never changes again, so
    the breadcrumb freezes on whatever the first few headings happened to be.
    On the 241-page RLHF PDF that produced 1812 chunks sharing one breadcrumb
    taken from page 5, including a chapter on direct alignment algorithms
    labelled "Instruction Fine-Tuning". The fallback is right for a single
    unlabelled heading in an otherwise depth-reporting document; it is wrong
    as a document-wide policy, so the policy is decided once, up front.

    Flat documents instead reset the stack on every heading: siblings replace
    one another, so each breadcrumb names the section its text is actually in.
    """
    flat = not any(
        isinstance(d.metadata.get(MK.CATEGORY_DEPTH), int)
        and d.metadata[MK.CATEGORY_DEPTH] >= 0
        for d in documents
    )

    paths: list[str | None] = []
    consumed: set[int] = set()
    stack: list[str] = []

    for i, doc in enumerate(documents):
        category = doc.metadata.get(MK.CATEGORY)
        if category == "Title":
            text = doc.page_content.strip()
            if not text:
                paths.append(" > ".join(stack) or None)
                continue
            raw_depth = doc.metadata.get(MK.CATEGORY_DEPTH)
            if flat:
                depth = 1
            elif isinstance(raw_depth, int) and raw_depth >= 0:
                depth = raw_depth + 1
            else:
                depth = len(stack) + 1
            if depth > _MAX_SECTION_DEPTH:
                # Too deep to be a real heading level; treat as content.
                paths.append(" > ".join(stack) or None)
                continue
            del stack[depth - 1 :]
            stack.append(text)
            consumed.add(i)
        paths.append(" > ".join(stack) or None)

    return paths, consumed


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def _splitter(settings: Settings, model_name: str) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=_SEPARATORS,
        keep_separator=False,
        strip_whitespace=False,
        length_function=lambda text: token_length(text, model_name),
    )


def _chunk_metadata(
    base: dict,
    section: str | None,
    pieces: list[Document],
    text: str,
    index: int,
    model_name: str,
) -> dict:
    """Merge a parent element's metadata with chunk-level bookkeeping."""
    meta = dict(base)
    meta.pop(MK.CATEGORY, None)  # element-level; meaningless at chunk level
    meta.pop(MK.CATEGORY_DEPTH, None)
    meta.pop(MK.ELEMENT_ID, None)
    meta.pop(MK.PARENT_ID, None)
    meta.pop(MK.PAGE, None)  # replaced by the page span below

    pages = [p.metadata.get(MK.PAGE) for p in pieces if p.metadata.get(MK.PAGE) is not None]
    if pages:
        meta[MK.PAGE_START] = min(pages)
        meta[MK.PAGE_END] = max(pages)

    if section:
        meta[MK.SECTION_PATH] = section
    meta[MK.TOKEN_COUNT] = token_length(text, model_name)
    meta[MK.CHUNK_INDEX] = index
    meta[MK.IS_TABLE] = False
    return meta


def chunk_documents(
    documents: list[Document], settings: Settings | None = None
) -> list[Document]:
    """Split cleaned elements into retrievable chunks of roughly ``chunk_size``.

    Consecutive elements are packed into a buffer and flushed as one chunk as
    soon as the next element would push it past ``chunk_size``. An element
    that alone exceeds ``chunk_size`` is split on its own rather than
    buffered, and a table passes through whole.

    **Packing is the point, not a detail.** ``chunk_size`` is the budget the
    query stage actually spends: ``top_k`` chunks of context are what the
    answer is written from. Emitting one chunk per element is therefore only
    correct when the parser emits paragraph-sized elements. The ``fast`` PDF
    strategy does not -- it emits sentence-sized ones -- and under it the real
    chunk size collapsed to a sentence: a 241-page book produced 1816 chunks
    at a *median of 57 tokens* against a configured 384, so five passages of
    context were five clauses. A question whose answer spanned the section
    got three unrelated fragments and a truncated sentence, and the model
    correctly reported that the context did not cover it.

    Merging never crosses a section boundary, so a chunk's breadcrumb is true
    of all the text it holds. It also never touches a table: a table split
    across chunks is two halves of a fact, and neither half retrieves well.
    """
    settings = settings or get_settings()
    if not documents:
        return []

    model_name = settings.dense_model
    splitter = _splitter(settings, model_name)
    paths, headings = _section_path(documents)
    source = str(documents[0].metadata.get(MK.SOURCE, ""))

    chunks: list[Document] = []

    def emit(pieces: list[Document], section: str | None, text: str, is_table: bool) -> None:
        """Append one chunk built from ``pieces``, the elements it covers.

        ``pieces`` is what the page span is computed from, so it must list
        every element the text was drawn from -- a merged chunk reports the
        pages it really spans, not just its first element's.

        The breadcrumb is prepended here, at the last moment before the chunk
        exists, so every path into this function -- packed, split, and table --
        gets it exactly once. Hashing the text *after* the prepend is what
        keeps a re-ingest idempotent: the chunk id is still a pure function of
        what is stored.
        """
        index = len(chunks)
        full = _section_header(section) + text
        chunk_meta = _chunk_metadata(
            pieces[0].metadata, section, pieces, full, index, model_name
        )
        if is_table:
            chunk_meta[MK.IS_TABLE] = True
        chunk_meta[MK.CONTENT_HASH] = _content_hash(source, index, full)
        chunk_meta[MK.CHUNK_ID] = chunk_meta[MK.CONTENT_HASH]
        chunks.append(Document(page_content=full, metadata=chunk_meta))

    buffer: list[Document] = []
    buffer_section: str | None = None
    buffer_tokens = 0

    def flush() -> None:
        nonlocal buffer, buffer_section, buffer_tokens
        if not buffer:
            return
        # Paragraphs stay separated inside the chunk: the blank line is what
        # keeps the passage readable to the model rather than one run-on line.
        emit(buffer, buffer_section, "\n\n".join(p.page_content for p in buffer), False)
        buffer = []
        buffer_section = None
        buffer_tokens = 0

    for position, (element, section) in enumerate(zip(documents, paths)):
        if position in headings:
            # A heading's words are already the breadcrumb that opens its
            # chunk. Buffering the element as content too printed every
            # heading twice -- "The Bradley-Terry model" as the chunk's first
            # line and again immediately below it -- and spent the tokens of a
            # duplicated line on every chunk in the document.
            continue

        meta = element.metadata

        if meta.get(MK.CATEGORY) == "Table":
            # Prefer the HTML rendering: it preserves cell structure that
            # flattened text loses, and the reranker sees the layout. Tables
            # are never split or merged -- a table split across chunks is two
            # halves of a fact and neither half retrieves well.
            flush()
            emit(
                [element],
                section,
                meta.get(MK.TEXT_AS_HTML) or element.page_content,
                True,
            )
            continue

        text = element.page_content
        tokens = token_length(text, model_name)
        # The breadcrumb is part of the stored chunk, so it is part of the
        # budget. Charged at the moment a buffer opens rather than per element,
        # because a chunk carries it once however many elements it packs.
        header_tokens = token_length(_section_header(section), model_name)

        if header_tokens + tokens > settings.chunk_size:
            # Too large to share a chunk with anything, so split it alone.
            # Each piece keeps the element's metadata as its base; the dicts
            # are copies -- sharing one dict across pieces would let any
            # later mutation leak between chunks.
            flush()
            for piece in splitter.split_text(text):
                emit([element], section, piece, False)
            continue

        if buffer and (
            section != buffer_section or buffer_tokens + tokens > settings.chunk_size
        ):
            flush()

        if not buffer:
            buffer_tokens = header_tokens
        buffer.append(element)
        buffer_section = section
        buffer_tokens += tokens

    flush()

    # Drop fragments: extraction debris that survived cleaning but is too
    # short to be a meaningful retrieval target on its own.
    keep = [
        c
        for c in chunks
        if c.metadata.get(MK.IS_TABLE)
        or c.metadata[MK.TOKEN_COUNT] >= settings.min_chunk_tokens
    ]
    dropped = len(chunks) - len(keep)
    if dropped:
        logger.info("  dropped %d fragment chunk(s) below %d tokens", dropped, settings.min_chunk_tokens)
    chunks = keep

    # Link neighbours last: indices are final only after fragment removal.
    for i, chunk in enumerate(chunks):
        meta = chunk.metadata
        if i > 0:
            meta[MK.PREV_ID] = chunks[i - 1].metadata[MK.CHUNK_ID]
        if i < len(chunks) - 1:
            meta[MK.NEXT_ID] = chunks[i + 1].metadata[MK.CHUNK_ID]

    logger.info(
        "  chunked %d elements -> %d chunks (max %d tokens)",
        len(documents),
        len(chunks),
        max((c.metadata[MK.TOKEN_COUNT] for c in chunks), default=0),
    )
    return chunks
