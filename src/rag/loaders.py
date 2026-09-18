"""Document loading.

Wraps LangChain's ``UnstructuredLoader`` and fixes the provenance bug that
made the prototype's output uncitable.

Large PDFs are split into page batches so that parsing never holds an entire
book in memory. That split is why provenance broke before: Unstructured
numbers pages relative to *the file it is handed*, so every batch restarted at
page 1 and the recorded filename was a temp file like
``batch_0180_0200.pdf``. A 241-page book came out claiming a maximum page
number of 20. Here each batch carries the page offset it started at, and
:func:`_restore_provenance` adds it back so ``page_number`` is absolute and
``source`` always names the original document.
"""

from __future__ import annotations

import hashlib
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path

from langchain_core.documents import Document
from langchain_unstructured import UnstructuredLoader
from pypdf import PdfReader, PdfWriter
from pypdf.errors import PdfReadError

from rag.config import PdfStrategy, Settings, get_settings
from rag.logging_utils import get_logger
from rag.state import MetadataKeys as MK, ProgressFn, ProgressStage

logger = get_logger(__name__)

# Formats Unstructured can partition. Anything else is skipped rather than
# handed to the parser, which would otherwise raise on archives and binaries
# that happen to share a directory with real documents.
SUPPORTED_SUFFIXES: frozenset[str] = frozenset(
    {
        ".pdf",
        ".docx", ".doc", ".odt", ".rtf",
        ".pptx", ".ppt",
        ".xlsx", ".xls", ".csv", ".tsv",
        ".html", ".htm", ".xml",
        ".md", ".markdown", ".txt", ".rst", ".org",
        ".epub",
        ".eml", ".msg",
        ".json",
        ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".heic",
    }
)

_HASH_CHUNK_BYTES = 1 << 20  # 1 MiB

# Directories never walked during discovery. A recursive scan of a project
# root would otherwise descend into the virtualenv and ingest every README
# and JSON file inside site-packages, and into the vector store itself.
EXCLUDED_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "env",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "node_modules",
        "site-packages",
        ".tox",
        ".idea",
        ".vscode",
        "qdrant_db",
        "data",
    }
)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _is_excluded(path: Path, root: Path) -> bool:
    """Whether any directory between ``root`` and ``path`` is excluded.

    Checked against the path *relative to the scan root*, so a file that
    happens to sit in a directory called ``data`` is still ingested when that
    directory was named explicitly on the command line.
    """
    return any(
        part in EXCLUDED_DIRS or part.startswith(".")
        for part in path.relative_to(root).parts[:-1]
    )


def discover_files(input_path: str | Path) -> list[Path]:
    """Return every supported document under ``input_path``.

    Recurses into subdirectories (the prototype used a non-recursive
    ``iterdir``) and filters by extension so unsupported files are skipped
    cleanly instead of failing mid-parse.
    """
    root = Path(input_path).expanduser().resolve()

    if root.is_file():
        if root.suffix.lower() not in SUPPORTED_SUFFIXES:
            logger.warning("Unsupported file type, skipping: %s", root.name)
            return []
        return [root]

    if not root.is_dir():
        raise FileNotFoundError(f"No such file or directory: {root}")

    files = sorted(
        p
        for p in root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in SUPPORTED_SUFFIXES
        and not p.name.startswith(".")
        and not _is_excluded(p, root)
    )
    logger.info("Discovered %d supported file(s) under %s", len(files), root)
    return files


def compute_doc_id(path: Path) -> str:
    """Stable content hash for a document.

    Content-addressed rather than path-addressed so that re-ingesting an
    unchanged file is a no-op, while an edited file produces a new id and its
    stale chunks can be deleted.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(_HASH_CHUNK_BYTES):
            digest.update(block)
    return digest.hexdigest()[:16]


# ---------------------------------------------------------------------------
# PDF batching
# ---------------------------------------------------------------------------


def _split_pdf(
    path: Path, out_dir: Path, batch_size: int
) -> list[tuple[Path, int, int]]:
    """Split a PDF into physical page batches.

    Returns ``(batch_path, start_page, end_page)`` triples, all 0-based and
    half-open-exclusive on ``end_page``. ``start_page`` is the index of the
    batch's first page in the original document and is what makes absolute
    page numbers recoverable later; ``end_page`` exists so progress logging
    reports real page ranges instead of running past the last page.
    """
    try:
        reader = PdfReader(str(path))
    except PdfReadError as exc:
        raise ValueError(f"Corrupt or unreadable PDF: {path.name} ({exc})") from exc

    if reader.is_encrypted:
        # An empty password covers the common "restricted permissions but not
        # actually password protected" case.
        try:
            if reader.decrypt("") == 0:
                raise ValueError(f"PDF is password protected: {path.name}")
        except (NotImplementedError, PdfReadError) as exc:
            raise ValueError(
                f"PDF uses an unsupported encryption scheme: {path.name} ({exc})"
            ) from exc

    total_pages = len(reader.pages)
    out_dir.mkdir(parents=True, exist_ok=True)
    batches: list[tuple[Path, int, int]] = []

    for start in range(0, total_pages, batch_size):
        end = min(start + batch_size, total_pages)

        writer = PdfWriter()
        for page_index in range(start, end):
            writer.add_page(reader.pages[page_index])

        batch_path = out_dir / f"batch_{start:05d}_{end:05d}.pdf"
        with batch_path.open("wb") as handle:
            writer.write(handle)

        batches.append((batch_path, start, end))

    logger.info("  split %d pages into %d batch(es)", total_pages, len(batches))
    return batches


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _partition_kwargs(path: Path, settings: Settings) -> dict:
    """Build Unstructured options appropriate to the file type."""
    kwargs: dict = {
        "mode": "elements",
        "partition_via_api": False,
    }

    suffix = path.suffix.lower()
    is_image = suffix in {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".heic"}

    if suffix == ".pdf" or is_image:
        kwargs["strategy"] = settings.pdf_strategy.value
        kwargs["languages"] = settings.ocr_languages
        # Table structure inference only runs under hi_res. Without it,
        # Unstructured emits no Table elements at all -- which is why the
        # prototype recovered zero tables from a book full of them.
        if settings.pdf_strategy is PdfStrategy.HI_RES:
            kwargs["infer_table_structure"] = True

    return kwargs


def _load_one(path: Path, settings: Settings) -> list[Document]:
    """Parse a single physical file into element-level Documents.

    ``hi_res`` needs optional native dependencies (layout models, poppler,
    tesseract). When those are missing the failure is an unhelpful import or
    binary error deep inside Unstructured, so it is caught and downgraded to
    ``fast`` with an explicit warning rather than aborting the whole run.
    """
    kwargs = _partition_kwargs(path, settings)

    try:
        return UnstructuredLoader(file_path=str(path), **kwargs).load()
    except Exception as exc:
        if kwargs.get("strategy") != PdfStrategy.HI_RES.value:
            raise
        logger.warning(
            "hi_res parsing failed for %s (%s: %s). Falling back to 'fast'. "
            "Tables will not be extracted -- install the hi_res extras "
            "(poppler, tesseract, unstructured-inference) to recover them.",
            path.name,
            type(exc).__name__,
            exc,
        )
        kwargs["strategy"] = PdfStrategy.FAST.value
        kwargs.pop("infer_table_structure", None)
        return UnstructuredLoader(file_path=str(path), **kwargs).load()


def _restore_provenance(
    documents: list[Document],
    *,
    source_path: Path,
    doc_id: str,
    page_offset: int | None,
) -> list[Document]:
    """Rewrite batch-local metadata into document-absolute metadata.

    This is the correction for the prototype's central defect. Without the
    offset, page numbers reset once per batch and every citation points at the
    wrong place.

    ``page_offset`` is the 0-based index of the batch's first page, or ``None``
    for a format that has no page numbers at all. The distinction matters:
    for a PDF, an element with no page number still has a known lower bound
    (its batch start), but for a Markdown file inventing page 1 would be a
    fabrication.
    """
    for doc in documents:
        meta = doc.metadata

        raw_page = meta.get(MK.PAGE)
        if isinstance(raw_page, int):
            meta[MK.PAGE] = raw_page + (page_offset or 0)
        elif page_offset is not None:
            # Element carried no page number; the batch start is the best
            # lower bound available rather than losing provenance entirely.
            meta[MK.PAGE] = page_offset + 1

        meta[MK.SOURCE] = str(source_path)
        meta[MK.SOURCE_NAME] = source_path.name
        meta[MK.DOC_ID] = doc_id
        # Unstructured records the file it actually parsed, which for a batch
        # is a temp file that will not exist after this run.
        meta.pop("filename", None)
        meta.pop("file_directory", None)

    return documents


def load_document(
    path: Path,
    settings: Settings | None = None,
    *,
    on_progress: ProgressFn | None = None,
) -> list[Document]:
    """Parse one document into element-level Documents with correct provenance.

    PDFs are batched through temporary files; everything else is parsed
    directly. Temporary files are always removed, including on failure.

    ``on_progress`` receives one event per page batch, so the browser can show
    a book advancing rather than a spinner that has been still for a minute.
    The page range is reported in *original-document* numbering -- the same
    numbers the citations use -- and ``total`` is the document's real page
    count, so the bar never resets between batches. Only the PDF path has
    pages to report; every other format produces no events and is parsed in
    one call.
    """
    settings = settings or get_settings()
    path = Path(path).expanduser().resolve()
    doc_id = compute_doc_id(path)

    if path.suffix.lower() != ".pdf":
        documents = _load_one(path, settings)
        return _restore_provenance(
            documents, source_path=path, doc_id=doc_id, page_offset=None
        )

    # Batch temp files live outside the corpus so a crashed run never leaves
    # partial PDFs where discover_files would pick them up next time.
    temp_root = Path(tempfile.mkdtemp(prefix=f"rag_split_{path.stem[:24]}_"))
    collected: list[Document] = []

    try:
        batches = _split_pdf(path, temp_root, settings.pdf_batch_size)
        # Each batch records its exclusive end page, so the last one's end is
        # the document's page count -- no second read of the PDF needed.
        total_pages = batches[-1][2] if batches else 0

        for index, (batch_path, start_page, end_page) in enumerate(batches, start=1):
            logger.info(
                "  [%d/%d] parsing pages %d-%d",
                index,
                len(batches),
                start_page + 1,
                end_page,
            )
            if on_progress is not None:
                on_progress(
                    {
                        "stage": ProgressStage.LOAD.value,
                        "done": end_page,
                        "total": total_pages,
                        "message": (
                            f"parsing pages {start_page + 1}-{end_page} of {total_pages}"
                        ),
                        "path": str(path),
                    }
                )
            batch_docs = _load_one(batch_path, settings)
            collected.extend(
                _restore_provenance(
                    batch_docs,
                    source_path=path,
                    doc_id=doc_id,
                    page_offset=start_page,
                )
            )
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)

    return collected


def iter_documents(
    input_path: str | Path, settings: Settings | None = None
) -> Iterator[tuple[Path, list[Document]]]:
    """Yield ``(path, elements)`` for each discovered document.

    One document failing never stops the run; the error is logged and the
    next file is attempted.
    """
    settings = settings or get_settings()

    for path in discover_files(input_path):
        try:
            yield path, load_document(path, settings)
        except Exception as exc:
            logger.error(
                "Failed to load %s: %s: %s", path.name, type(exc).__name__, exc
            )
            yield path, []
