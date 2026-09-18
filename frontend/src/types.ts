/**
 * Wire types, mirroring rag/api/schemas.py.
 *
 * Hand-written rather than generated: the surface is small, and a generated
 * client would be one more build step between a schema change and the type
 * error that should catch it.
 */

export type JobStatus = 'queued' | 'running' | 'done' | 'failed' | 'cancelled'

/** Stages a document passes through, mirroring rag.state.ProgressStage. */
export type Stage = 'discover' | 'load' | 'clean' | 'chunk' | 'index'

/** Stages a *question* passes through, mirroring rag.state.QueryStage.
 *  Deliberately a separate union: an ingest parses and embeds, a query
 *  retrieves and generates, and the two lifecycles share no step.
 *
 *  `search` and `fetch` are reached only on a web search. They come after the
 *  corpus stages rather than before them because that is where they happen in
 *  time: the corpus answers first, and the web is what a user asks for once it
 *  has come back empty-handed. */
export type QueryStage =
  | 'guard'
  | 'retrieve'
  | 'rerank'
  | 'generate'
  | 'verify'
  | 'search'
  | 'fetch'

/** One stage of an ingest that was entered, and how it went. */
export interface StageRecord {
  stage: Stage
  /** The last thing the stage said before it was left; null if it never spoke. */
  message: string | null
  /** Seconds spent in the stage, or null while it is still running. */
  seconds: number | null
}

export interface Job {
  id: string
  filename: string
  status: JobStatus
  stage: Stage | null
  done: number
  /** Units of the *current* stage: pages while parsing, chunks while indexing. */
  total: number | null
  /** Percent complete of the current stage, or null when it has no total. */
  percent: number | null
  message: string | null
  /** Every stage the job has entered so far, in order. Outlives `message`,
   *  which only ever holds the current stage's latest line. */
  stages: StageRecord[]
  chunks: number
  error: string | null
  stats: Record<string, unknown> | null
  cancel_requested: boolean
  created_at: number
  started_at: number | null
  finished_at: number | null
}

export interface Citation {
  marker: string
  source: string
  page_start: number | null
  page_end: number | null
  section: string | null
  rerank_score: number | null
}

export interface AskResponse {
  question: string
  /** null when generation is off, every model failed, or nothing was retrieved. */
  answer: string | null
  /** True when the output guardrails withheld an answer rather than producing one. */
  refused: boolean
  context: string
  citations: Citation[]
  generation_enabled: boolean
  elapsed_ms: number
  /** Whether a web search could run at all on this server: a backend is
   *  configured and generation is on. False is what explains the absence of the
   *  search offer rather than leaving the user to guess. */
  web_search_enabled: boolean
  /** Whether *this* answer came back empty-handed, which is what the search
   *  offer is shown on. Computed server-side because part of it compares
   *  against a sentence the server's own prompt produced. */
  search_suggested: boolean
}

/** One web page behind a web answer. */
export interface WebSource {
  marker: string
  title: string
  url: string
  /** The search engine's summary of the page; absent when it gave none. */
  snippet: string | null
}

/**
 * The result of one web search.
 *
 * `error` is an ordinary outcome rather than a thrown failure: a search that
 * finds nothing, a rate limit, a backend that is not configured. There is
 * already a corpus answer on screen and this is an addition to it, so the
 * section says what happened instead of the request failing.
 */
export interface WebAskResponse {
  question: string
  /** null when generation is off or every model failed -- the pages are still
   *  worth showing, which is what `sources` holds. */
  answer: string | null
  refused: boolean
  sources: WebSource[]
  provider: string | null
  error: string | null
  generation_enabled: boolean
  elapsed_ms: number
}

/** One stage of a query reporting in, as it happens. */
export interface StageEvent {
  stage: QueryStage
  message: string
  /**
   * Milliseconds from the question to this event, stamped server-side.
   *
   * Server-side on purpose: a client-side clock would measure the wire and the
   * render as well as the work, and the trace's whole claim is that it shows
   * what the pipeline did rather than what the browser saw.
   */
  elapsed_ms: number
}

/**
 * One line of an NDJSON stream: a stage, the result, or a failure.
 *
 * `result` carries the response byte for byte, so a client that ignores the
 * stages mounted before it sees no difference between the streaming and plain
 * endpoints. Generic over the response because both `POST /api/ask/stream` and
 * `POST /api/web/stream` speak it, and the two differ only in what they carry.
 */
export type StreamEvent<T> =
  | ({ type: 'stage' } & StageEvent)
  | ({ type: 'result' } & T)
  | { type: 'error'; detail: string; status: number }

export interface DocumentSummary {
  doc_id: string
  source_name: string
  source: string | null
  chunks: number
  page_start: number | null
  page_end: number | null
  /** False for a document whose ingest was interrupted before it finished. */
  complete: boolean
}

export interface Stats {
  collection: string
  exists: boolean
  points: number
  dense_model: string
  sparse_model: string
  reranker_model: string
  device: string
  retrieval_mode: string
  fetch_k: number
  top_k: number
  max_upload_mb: number
  generation_enabled: boolean
  llm_provider: string | null
  /** The web-search backend in use, or null if none is available. Reported so
   *  the UI can say why the search offer never appears rather than leaving the
   *  user to guess. */
  web_search_provider: string | null
}
