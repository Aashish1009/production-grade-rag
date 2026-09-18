# Local Production RAG Pipeline

A production-grade, fully local Retrieval-Augmented Generation pipeline built
on **LangChain + LangGraph**, with an embedded **Qdrant** vector database,
hybrid retrieval, and a browser UI. No Docker, no paid APIs, no cloud
services. Everything except the optional answer-writing stage runs on your own
machine, on CPU, for free.

```
ingest:  discover -> load (Unstructured) -> clean -> chunk -> index (Qdrant)
query:   guard -> hybrid retrieve -> rerank -> generate (optional) -> guard
```

**Jump to:** [What it does](#what-it-does) · [Tech stack](#tech-stack) ·
[Requirements](#requirements) · [How to run](#how-to-run) ·
[Using the app](#using-the-app) · [HTTP API](#http-api) ·
[Configuration](#configuration) · [Architecture](#architecture) ·
[Where your data lives](#where-your-data-lives) ·
[Testing](#testing) · [Troubleshooting](#troubleshooting)

---

## What it does

- **Any document format** — PDF (including scanned, via OCR), DOCX, PPTX,
  XLSX, HTML, Markdown, EPUB, email, images, and everything else Unstructured
  handles. A 600-page book is a first-class case, not an afterthought.
- **Provenance-correct parsing** — large PDFs are split into page batches
  internally, but page numbers in citations always refer to the *original*
  document, never a temp batch file.
- **Aggressive cleaning** — running headers/footers detected by cross-page
  repetition (not by parser category), shattered equations reassembled,
  local-window near-duplicate removal that never deletes intentional
  repetition.
- **Hybrid retrieval** — dense (bge-small-en-v1.5) + sparse (BM25) searches
  fused server-side inside Qdrant with Reciprocal Rank Fusion, with a
  dense+BM25 ensemble fallback if the hybrid path fails.
- **Cross-encoder reranking** — `ms-marco-MiniLM-L-6-v2`, fast on CPU.
- **Metadata filtering** — narrow a query to one document, source file or
  section; filters are pushed into both the primary hybrid search and the
  ensemble fallback, so a filtered query cannot receive unfiltered hits.
- **Guardrails** — prompt-injection rejection on input; citation validity,
  grounding, and PII-redaction checks on output.
- **Optional generation** — LiteLLM gateway with automatic model fallbacks
  (Groq free tier by default, OpenAI if a key is present). With no keys, the
  pipeline returns ranked, cited context without an LLM answer.
- **Sliced, cancellable ingest** — a book is written to the index in slices,
  so progress is visible per page batch and per slice, an interrupted run
  keeps what it already wrote, and you can ask questions *while* it indexes.
- **A browser UI** — drag and drop, live progress, answers with clickable
  citations. This is the interface; there is no CLI.

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.12 (`.python-version`) | 3.12 is the floor for the LangChain 1.x line |
| Orchestration | LangGraph 1.2 + LangChain 1.4 | The ingest and query flows are genuine graphs — the ingest side fans out per file with `Send` |
| Vector store | Qdrant via `qdrant-client` **embedded** (`path=`) | No server to run, no Docker. See the single-process note below |
| Dense embeddings | `BAAI/bge-small-en-v1.5`, 384-dim | 33M params, ~130MB, fast on CPU, strong on retrieval benchmarks |
| Sparse embeddings | `Qdrant/bm25` via FastEmbed | Real BM25, computed server-side inside Qdrant |
| Fusion | Qdrant's own RRF (`Fusion.RRF`) | Hybrid search without a second round trip |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` | ~90MB, 10–30ms per pair on CPU |
| Parsing | Unstructured `[all-docs]` | One API across PDF/Office/HTML/EPUB/images |
| LLM gateway | LiteLLM + `langchain-litellm` | One interface for Groq and OpenAI, plus `with_fallbacks` |
| API | FastAPI + uvicorn | Serves the JSON API **and** the built UI from one process |
| Validation | Pydantic v2 + pydantic-settings | `Settings` is the single source of truth for every tunable |
| UI | React 19 + TypeScript 7 + Vite 8 | No UI kit; hand-written CSS with design tokens |
| Tests | pytest (Python) + `tsc --noEmit` (UI) | Model-free — no test downloads a model or opens the store |

Resolved versions of everything are pinned in `uv.lock` and
`frontend/package-lock.json`. The Python packages that matter most:
`fastapi 0.141`, `langchain-core 1.6.2`, `langgraph 1.2.11`,
`qdrant-client 1.19`, `sentence-transformers 6.0.1`, `torch 2.14`,
`unstructured 0.27.5`.

> **Why LangChain 1.x is a requirement, not a preference.** The code uses
> `langchain_core.cross_encoders`, which does not exist in the 0.3 line, and
> `EnsembleRetriever`, which moved out of the `langchain` umbrella into the
> separate `langchain-classic` package. A 0.3-era floor would declare
> compatibility with a combination that cannot resolve.

## Requirements

- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/)** — for the Python side
- **Node 20+** and npm — only to build the UI (dev machine and first build)
- **~2 GB free disk** — models plus the vector store for a book-scale corpus
- Optional, for better PDF table extraction: **poppler** and **tesseract** on
  `PATH`, plus `unstructured-inference`. Without them everything still
  ingests; tables just arrive as plain text.

No GPU is required. `RAG_DEVICE=auto` resolves to `cuda`, then `mps`, then
`cpu`.

## How to run

Run these from the repository root. The first four are **setup, done once**;
`uv run rag-server` is the only command you run day to day.

```powershell
# 1. Python dependencies (once)
uv sync --all-extras

# 2. UI dependencies (once)
cd frontend
npm install
cd ..

# 3. Build the UI (once -- repeat after changing anything in frontend/src)
cd frontend
npm run build
cd ..

# 4. Start the app

uv run rag-server
```

Then open **http://127.0.0.1:8000**.

That single command serves the API *and* the built UI on one origin, so there
is no CORS to configure and no second process to start. Leave it running in
its own terminal; stop it with Ctrl+C.

The very first ingest additionally downloads the models (bge-small-en-v1.5
~130MB, reranker ~90MB, BM25 vocab ~10MB) and the first query after that loads
them, so the first run of each is slow. Everything after is offline.

> **Frontend or backend?** Work only on Python: commands 1 and 4 are all you
> need — omit the build and `/` returns a note saying there is no UI. Work on
> the UI as well: keep `uv run rag-server` running and, in a second terminal,
> `cd frontend; npm run dev` for hot reload on http://localhost:5173 (it
> proxies `/api` to port 8000). Re-run `npm run build` when you are done, so
> `rag-server` serves the new bundle.

If port 8000 is taken, set `RAG_API_PORT` before starting (and point the Vite
proxy at the same port in `frontend/vite.config.ts` if you are using the dev
server).

### UI commands

```powershell
npm install          # once
npm run dev          # hot reload on :5173, proxying /api to the server
npm run typecheck    # tsc --noEmit; also the first half of `build`
npm run build        # writes frontend/dist, which rag-server serves at /
npm run preview      # serve the production build standalone (no API)
```

`StaticFiles` reads from disk per request, so a rebuild is served without a
restart — *except* the very first time: the mount is decided when the server
starts, so if `dist/` did not exist then, build it and restart `rag-server`.

## Using the app

The workspace is two panes: the library on the left, asking on the right.

1. **Drop a file** onto the dropzone (or click to browse). It uploads with a
   real transfer-progress bar, then becomes an *ingest job*.
2. **Watch it index.** The job card steps through Scan → Parse → Clean →
   Split → Index. Parsing reports pages (`parsing pages 241-260 of 600`),
   indexing reports chunks, and the stages with no countable unit get an
   indeterminate bar rather than a fake percentage. A 600-page book is a
   multi-minute job on CPU; the page is fine to leave open.
3. **Ask.** Answers cite their sources inline as `[1]`, `[2]`; hovering a
   marker highlights the matching passage below, with its file, page range
   and section path. Ctrl/⌘ + Enter submits.
4. **Scope a question** to one document by selecting it in the library. That
   is a filter on the query, not a mode — the scope is shown on the ask panel
   the whole time it is narrowing what you get back.

The **passages slider** sets how many chunks survive reranking (1 to
`RAG_FETCH_K`). Higher means more context for the LLM and more to read; lower
means a tighter, higher-confidence answer.

### Ask while it ingests

Ingest writes in slices, and the store lock is released between them, so a
question asked mid-ingest waits for at most one slice (~2–4s of embedding on
CPU) rather than for the whole book. Slice size is `RAG_INGEST_SLICE_SIZE`.

Two consequences worth knowing:

- **A cancelled or crashed ingest keeps what it wrote.** The chunks already
  indexed are searchable, and the document is listed as *partial* until a
  re-ingest completes it. Before this change an interrupted run discarded
  everything it had done.
- **An index built before this change is re-ingested once.** Documents are
  marked complete by a payload stamp written after the final slice, and
  points written by the old all-or-nothing code carry no stamp. The pipeline
  cannot tell "written by an older version" from "interrupted", so it takes
  the honest reading and treats them as incomplete. Upload the file again and
  it self-heals. Existing chunks are dropped per `doc_id` as the re-ingest
  reaches them, so nothing is duplicated.

### What a job status means

| Status | Meaning |
|---|---|
| `queued` | Accepted; waiting for the single ingest worker. Jobs run one at a time. |
| `running` | Parsing, cleaning, chunking or embedding. `cancel` stops it at the next checkpoint (within one page batch, seconds). |
| `done` | Finished and stamped complete. `chunks` is how many landed in the index. |
| `failed` | The per-file error is in `error`. The common one is "no ingestable text" — a scanned PDF with no text layer. |
| `cancelled` | Stopped on request. Whatever was written stays, unstamped, so it shows as *partial*. |

## HTTP API

Every route is under `/api`. The UI is the intended client, but the surface is
plain JSON and usable with `curl`.

| Route | Does |
|---|---|
| `GET /health` | Liveness. Touches no store and loads no model. |
| `POST /documents` | Multipart upload → a queued ingest job. Streams to disk; capped by `RAG_MAX_UPLOAD_MB`. |
| `GET /jobs` · `GET /jobs/{id}` | Job list / one job. Polled by the UI. |
| `POST /jobs/{id}/cancel` | Ask the job to stop at its next checkpoint. |
| `POST /ask` | `{question, top_k?, filters?}` → `{answer, citations, context, refused}`. |
| `GET /documents` | Indexed documents: id, name, chunk count, page span, complete flag. |
| `DELETE /documents/{id}` | Drop every chunk of one document, and its uploaded file. |
| `GET /stats` | Collection size, models, device, generation state. |
| `GET /config` | Every effective setting, secrets excluded. |

```powershell
$job = Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/documents `
  -Form @{ file = Get-Item "rlhf book.pdf" }
$job.id
Invoke-RestMethod http://127.0.0.1:8000/api/jobs/$($job.id)
```

Errors use status codes rather than exit codes: **400** for an unsupported
file type, a malformed filter or a rejected question; **413** for an oversize
upload; **404** for an unknown job or document; **500** otherwise. The body is
always `{"detail": "..."}` with a sentence meant to be shown to a person.

### Filtering a query

`filters` is a JSON object of equality matches:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/ask `
  -ContentType application/json -Body (@{
    question = "What is the reward model?"
    filters  = @{ source_name = "rlhf book.pdf" }
    top_k    = 5
  } | ConvertTo-Json)
```

Any filterable metadata key works (`doc_id`, `source_name`, `section_path`,
`chunk_index`, `page_start`, `is_table`, …). An unknown key is a 400 rather
than an empty result: `build_filter` turns any key into
`metadata.<key> == value`, and a key that does not exist matches zero points,
so a misspelt `docID` would otherwise look like "the corpus is empty".

`doc_id`, `source`, `source_name`, `section_path` and `chunk_index` are
declared as payload indexes, which would let a Qdrant *server* seek on them —
the embedded store ignores payload indexes entirely, so every filter scans
locally and returns correct results either way.

### From your own code

The graphs are ordinary callables:

```python
from rag.graphs import run_ingest, run_query

def show(event):
    print(event["stage"], event.get("message", ""))

# Raises rag.state.JobCancelled if `show` does.
state = run_ingest("rlhf book.pdf", progress=show)

result = run_query("What is GRPO?", filters={"source_name": "rlhf book.pdf"})
print(result["answer"] or result["context"])
for citation in result["citations"]:
    print(citation["marker"], citation["source"], citation["page_start"])
```

`run_query` raises `rag.guardrails.InputRejectedError` when the input
guardrail rejects a question; every other failure mode returns empty rather
than raising, so querying an empty index yields "no passages found" instead
of a traceback.

> **One process at a time.** The embedded Qdrant store holds an exclusive lock
> on `qdrant_db/`. A second process that opens it — a script, or a stray
> `python -c` — will fail or hang until the server exits. This is why there is
> no CLI: it would have had to be a second process, and it would have
> deadlocked against the server holding the store. Use the HTTP API instead.

## Configuration

Every tunable is an environment variable (prefix `RAG_`) or an entry in
`.env` — see [`.env.example`](.env.example) for the full annotated list, or
`GET /api/config` for what the running server actually resolved. Highlights:

| Variable | Default | Meaning |
|---|---|---|
| `RAG_PDF_STRATEGY` | `hi_res` | `fast` is ~5x quicker but extracts no tables; `ocr_only` for scans |
| `RAG_PDF_BATCH_SIZE` | `20` | pages per internal parse batch — the page granularity of ingest progress |
| `RAG_CHUNK_SIZE` | `384` | chunk size in dense-model tokens |
| `RAG_CHUNK_OVERLAP` | `48` | overlap between neighbouring chunks |
| `RAG_RETRIEVAL_MODE` | `hybrid` | `dense`, `sparse`, or `hybrid` |
| `RAG_FETCH_K` / `RAG_TOP_K` | `30` / `5` | candidates before/after rerank |
| `RAG_INGEST_SLICE_SIZE` | `64` | chunks per upsert; the query-latency knob |
| `RAG_MAX_UPLOAD_MB` | `512` | upload ceiling |
| `RAG_API_HOST` / `RAG_API_PORT` | `127.0.0.1` / `8000` | where the app listens |
| `RAG_ENABLE_NEIGHBOUR_EXPANSION` | `false` | pull prev/next chunks of each hit |
| `RAG_ENABLE_GENERATION` | `false` | turn on the LLM answer stage |
| `RAG_GROQ_MODEL` | `groq/openai/gpt-oss-120b` | primary generation model |
| `RAG_GROQ_FALLBACK_MODELS` | gpt-oss-20b, qwen3.6-27b | tried in order on failure |
| `RAG_LLM_MAX_TOKENS` | `1024` | cap on the generated answer |
| `RAG_LOG_LEVEL` | `INFO` | console logging; `DEBUG` names which guardrail fired |

> `RAG_API_HOST` binds loopback by default, and that default is load-bearing:
> the server has no authentication and can read *and delete* the index.
> Binding it to a public interface is an explicit decision.

### Enabling generation

Generation is the only stage that calls a hosted API. To enable it, set a
key in `.env` and flip the flag:

```dotenv
RAG_ENABLE_GENERATION=true
GROQ_API_KEY=gsk_...            # free tier; no OpenAI key needed
```

With an `OPENAI_API_KEY` present, OpenAI (gpt-4o-mini) takes priority. The
fallback chain (LiteLLM + `with_fallbacks`) moves to the next model on rate
limits or provider errors automatically, so a single dead model never breaks
a query — which cuts both ways: a decommissioned primary still answers, from
the next model down, so the only symptom is latency. Groq retires models on
a rolling schedule (it decommissioned both Llama ids this project used to
default to on 2026-08-16), so check the
[deprecation table](https://console.groq.com/docs/deprecations) before
trusting the ids above.

With generation off, the UI labels the result **context only** and shows the
retrieved passages instead of an answer. Nothing about retrieval changes.

## Architecture

```
frontend/                   # React + TypeScript UI (Vite), no UI kit
├── index.html
├── vite.config.ts          # /api proxied to :8000 in dev; dist/ in prod
└── src/
    ├── main.tsx            # mount
    ├── App.tsx             # two-pane workspace, upload + ask orchestration
    ├── api.ts              # typed client; XHR for real upload progress
    ├── types.ts            # wire types, mirroring api/schemas.py
    ├── formats.ts          # accepted suffixes (server stays the authority)
    ├── styles.css          # design tokens, dark + light, responsive
    ├── hooks/              # usePolling, useJobs, useDocuments, useStats
    └── components/         # Dropzone, JobCard, DocumentList, AskPanel,
                            #   AnswerCard, CitationList, StatsBar

src/rag/
├── config.py       # all tunables, env-overridable, validated at startup
├── logging_utils.py
├── state.py        # LangGraph states, canonical metadata keys, progress contract
├── loaders.py      # Unstructured loading, PDF batching, provenance repair
├── cleaning.py     # furniture/fragments/dedupe
├── chunking.py     # token-accurate, structure-aware, tables kept whole
├── embedding.py    # bge-small-en-v1.5 (disk-cached) + FastEmbed BM25
├── indexing.py     # embedded Qdrant, hybrid collection, sliced upserts
├── retrieval.py    # hybrid search, ensemble fallback, neighbour expansion
├── reranking.py    # cross-encoder reranking over retrieved candidates
├── generation.py   # LiteLLM gateway + model fallbacks
├── guardrails.py   # input injection checks; output citation/grounding/PII
├── graphs.py       # LangGraph ingest + query graphs, run_ingest/run_query
└── api/
    ├── app.py      # FastAPI routes, error contract, serves the built UI
    ├── jobs.py     # in-process ingest job registry + cancellation
    ├── schemas.py  # request/response validation
    └── __main__.py # `rag-server`

tests/              # model-free pytest suite
data/               # uploads + embedding cache (gitignored)
qdrant_db/          # the vector store (gitignored)
```

### What happens on an upload

1. `POST /api/documents` streams the body to `data/uploads/<random>/<name>`
   in 1 MB chunks — never into memory — and returns a job id with 202.
2. The job goes to a `ThreadPoolExecutor(max_workers=1)`. One worker, because
   the store admits one writer and ingest is CPU-bound anyway.
3. `run_ingest` walks the graph: discover → load (per page batch) → clean →
   chunk → index (per slice). Every step calls a progress sink.
4. The sink writes a snapshot into the job under a lock; the UI polls
   `GET /api/jobs` every 1.5s while anything is running.
5. After the last slice, one `set_payload` stamps the document complete.
   Only then is it reported as a finished document.

A question asked in the middle of all this runs on Starlette's threadpool
(the handlers are sync `def` on purpose) and takes the same store guard, so it
waits for one slice, not for the book.

### Design notes

**Why a server instead of a CLI.** The embedded Qdrant store takes an
exclusive lock on its directory, so exactly one process may own it. A browser
cannot import Python, so something has to sit between them, and that
something has to be the process holding the store. A CLI could only have been
a second process contending for the same lock.

**Why ingest handlers are sync `def`.** Starlette runs them in its
threadpool. A handler waiting on the store guard would otherwise block the
event loop — and with it the `GET /api/jobs` polls that exist to show the
ingest progressing. Only the upload handler is `async`, because it awaits the
request body.

**Why one lock, defined once.** The embedded store's file lock is
cross-*process*; it does nothing for two threads. `indexing.store_guard` is a
process-wide `RLock` that `retrieval.py` enters too, held per store operation
rather than per document, so any embedder is thread-safe and a query waits a
slice rather than a book.

**Why the completion stamp is a new payload key.** Two measured traps: a
dot-notation write (`{"metadata.ingest_status": ...}`) is *not* expanded
locally — it creates a literal top-level key and the filter then matches
nothing; and writing `{"metadata": {...}}` **replaces** the whole metadata
object, destroying `doc_id`, `chunk_id` and the page numbers with it. The
stamp is therefore a fresh root key (`{"ingest": {"status": "complete"}}`)
merged additively, which touches nothing else.

**Why content-hash chunk ids.** Chunk ids are hashes of (source, text), so
re-ingest produces identical ids and the Qdrant upsert replaces rather than
duplicates. Stale chunks of an edited document are removed by `doc_id`
before its new chunks are written.

**Why furniture is detected by repetition, not category.** On real books the
`fast` parse strategy labels running headers as `Title`, so category-based
filtering let 174 copies of a site URL through. Cross-page repetition is the
actual signal. Digit-insensitive matching is applied only to short texts so
"Chapter 1 …" / "Chapter 2 …" sentences are never mistaken for headers.

**Why local-window dedupe.** A global dedup deletes a sentence an author
legitimately repeated in a later chapter; text repeating a few elements
apart is an extraction artifact. The window is deliberately local.

**Why tables embed their HTML rendering.** Unstructured gives a detected
table both a plain-text `page_content` and a `text_as_html` reconstruction
with the row/column structure intact. The table is chunked whole — splitting
it across chunks destroys the header→cell relationships — and the text that
gets embedded and returned is the HTML, so a table is retrievable by the
terms printed in it rather than by whichever cells happened to survive
reading order.

**Why the lexical fallback splits on word boundaries.** Its tokeniser must
preserve the spaces between words. Routing it through the cleaning pass's
dedupe normaliser (which strips *all* non-alphanumerics) fused each chunk
into a single token, so every realistic query scored zero against every
document: a fallback that ran, reported no error, and silently retrieved
nothing.

**Why neighbour expansion is off by default.** Pulling prev/next chunks
widens context for the reranker but roughly doubles candidate volume. Enable
with `RAG_ENABLE_NEIGHBOUR_EXPANSION=true` when answer quality matters more
than latency.

## Where your data lives

Everything is local and gitignored. Nothing is uploaded anywhere except the
question and the retrieved passages, and only when generation is on.

| Path | Holds | Notes |
|---|---|---|
| `qdrant_db/` | the vector store | Locked by whichever process owns it |
| `data/uploads/<id>/` | your uploaded originals | Deleted when you delete the document in the UI |
| `data/embedding_cache/` | BM25 vocabulary, embedding cache | Regenerable; deleting it only costs time |
| HF cache (`~/.cache/huggingface`) | dense model + reranker | ~220MB total, downloaded once |

Models are loaded lazily: `GET /api/stats` reports the model *names* without
downloading anything, so a fresh checkout can be inspected before the first
ingest.

## Testing

```powershell
uv run pytest                    # Python: model-free, fast
cd frontend; npm run typecheck   # UI types
```

The Python suite is model-free. It covers the pure seams — provenance
arithmetic and discovery filtering (`test_loaders.py`), cleaning
(`test_cleaning.py`), chunking and table handling (`test_chunking.py`),
filter translation and BM25 scoring (`test_retrieval.py`), context rendering
and the prompt/citation-guardrail contract (`test_generation.py`), sliced
indexing and the completion marker (`test_indexing.py`), progress emission
and cancellation (`test_ingest_progress.py`), the HTTP surface
(`test_api.py`), and every startup validator (`test_config.py`). No test
downloads a model, opens the vector store, or calls an API, so the suite runs
on a clean checkout.

`test_api.py` drives the real FastAPI app through `TestClient` with the
pipeline seams monkeypatched, so routing, validation and the error contract
are exercised without a model. Anything that needs the actual pipeline is
exercised by hand through the UI.

## Troubleshooting

- **`uv run rag-server` says there is no built UI** — run
  `cd frontend; npm install; npm run build`. The API works either way; only
  `/` is missing.
- **The UI says "The API is not responding"** — the server is not running, or
  it is on another port. Check `RAG_API_PORT`.
- **Upload rejected as an unsupported file type** — the extension is not one
  Unstructured parses. The message lists what it accepts.
- **`hi_res parsing failed ... falling back to 'fast'`** — install the
  hi_res extras to recover table extraction: poppler and tesseract on PATH,
  plus `unstructured-inference`. Without them PDFs still ingest, but tables
  come through as plain text.
- **A job failed with "No ingestable text was found"** — the parse worked and
  nothing survived cleaning. Usually a scanned PDF whose pages are images
  with no text layer; set `RAG_PDF_STRATEGY=ocr_only`.
- **`Storage lock ... already accessed by another instance`** — something else
  holds `qdrant_db/`. Only one process may, by design: stop the server before
  running a script against the store directly.
- **A document shows as *partial*** — its ingest was cancelled or crashed
  before the final slice. Its chunks are searchable; re-ingest the file to
  finish it.
- **First ingest is slow** — model downloads plus first-time embedding of the
  corpus. Subsequent ingests hit the disk embedding cache and are much
  faster; unchanged documents are skipped.
- **Generation stays off / returns context only** — no API key found, or
  every fallback model failed. `GET /api/config` reports the resolved
  provider (secrets excluded).
- **An answer was withheld with a refusal message** — the output guardrails
  fired. Citation refusal means the model invented a source marker;
  grounding refusal means the answer shared too little vocabulary with the
  retrieved passages. Both are the pipeline declining to show an ungrounded
  answer rather than a crash; `RAG_LOG_LEVEL=DEBUG` logs which check fired.

## Known limits

- **One process, one writer.** The embedded store is a directory with an
  exclusive lock, not a server. If you need concurrent access from several
  machines, that is the point to switch `qdrant-client` to server mode.
- **CPU-only by default.** Ingestion of a 600-page book is minutes, not
  seconds. `RAG_DEVICE=cuda` and a larger `RAG_RERANK_BATCH_SIZE` are the
  levers.
- **No authentication.** Loopback only, single user, by design.
- **`RAG_TOP_K` and `RAG_FETCH_K` must stay ordered.** Reranking can only
  narrow the candidate set, so the UI caps its slider at `fetch_k` and the
  API answers 400 above it rather than silently returning fewer passages.
