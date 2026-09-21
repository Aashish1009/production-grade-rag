import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import {
  askStream,
  cancelJob,
  deleteDocument,
  errorMessage,
  uploadDocument,
  webSearchStream,
} from './api'
import { AnswerCard } from './components/AnswerCard'
import { AskPanel } from './components/AskPanel'
import { CitationList } from './components/CitationList'
import { DocumentList } from './components/DocumentList'
import { Dropzone } from './components/Dropzone'
import { JobCard } from './components/JobCard'
import { ModelNotice } from './components/ModelNotice'
import { PipelineTrace } from './components/PipelineTrace'
import { StatsBar } from './components/StatsBar'
import { WebSearchCard } from './components/WebSearchCard'
import { useJobs, isActive } from './hooks/useJobs'
import { useDocuments } from './hooks/useDocuments'
import { useModelWarmup } from './hooks/useModelWarmup'
import { useStats } from './hooks/useStats'
import type { AskResponse, StageEvent, WebAskResponse } from './types'

/** One file being transferred, before the server has a job for it. */
interface PendingUpload {
  key: string
  name: string
  percent: number
  error: string | null
}

const DEFAULT_TOP_K = 5
const DEFAULT_FETCH_K = 30

/**
 * A unique-enough key for one in-flight upload.
 *
 * `crypto.randomUUID` is the obvious choice, but it is only defined in a
 * secure context. `http://127.0.0.1` counts as one, so it works here -- but a
 * server bound to `RAG_API_HOST=0.0.0.0` and reached over a LAN address is
 * *not* secure, and an unguarded call would throw where the user can least
 * afford it. A React key needs uniqueness within one session, not
 * unpredictability.
 */
let uploadSeq = 0
function uploadKey(file: File): string {
  uploadSeq += 1
  return `${file.name}:${file.size}:${uploadSeq}`
}

export default function App() {
  const stats = useStats()
  const jobs = useJobs()
  const documents = useDocuments()

  const [pending, setPending] = useState<PendingUpload[]>([])
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [answer, setAnswer] = useState<AskResponse | null>(null)
  const [asking, setAsking] = useState(false)
  const [askError, setAskError] = useState<string | null>(null)
  const [actionError, setActionError] = useState<string | null>(null)
  const [topK, setTopK] = useState<number | null>(null)
  const [highlighted, setHighlighted] = useState<string | null>(null)
  const [trace, setTrace] = useState<StageEvent[]>([])
  const [askedAt, setAskedAt] = useState<number | null>(null)
  // The web fallback is its own run with its own clock. Its stages are stamped
  // from the moment the search started, so they are kept out of the corpus
  // trace: two runs measured from two different zeros cannot share one column
  // of durations without one of the numbers becoming a fiction.
  const [web, setWeb] = useState<WebAskResponse | null>(null)
  const [webBusy, setWebBusy] = useState(false)
  const [webError, setWebError] = useState<string | null>(null)
  const [webTrace, setWebTrace] = useState<StageEvent[]>([])
  const [webAskedAt, setWebAskedAt] = useState<number | null>(null)
  const webSection = useRef<HTMLElement>(null)
  // Which web search is current. The corpus Ask button stays live while a
  // search is in flight -- it is a different question with its own panel, and
  // disabling it would punish asking twice -- so a second question can be put
  // while the first one's search is still running. Without this the earlier
  // result lands afterwards and appears under the newer question, which is
  // exactly the stale answer `handleAsk` clears the corpus answer to avoid.
  const webRun = useRef(0)

  const { refresh: refreshJobs } = jobs
  const { refresh: refreshDocuments } = documents
  const { refresh: refreshStats } = stats

  // Started as soon as the first stats response says the models are not
  // resident. The error case is excluded rather than retried: a warm-up needs
  // a server, and an offline one will be reported by the status bar already.
  const warmup = useModelWarmup(
    stats.data?.models_loaded === false,
    stats.data?.models_loaded === true,
  )

  const jobList = useMemo(() => jobs.data ?? [], [jobs.data])
  const activeJobs = jobList.filter(isActive)
  const docList = documents.data ?? []

  // --- refresh the corpus when an ingest lands -----------------------------

  // Counting terminal jobs rather than diffing ids: a document appears in
  // /api/documents only once its ingest stops running, so "one more job has
  // finished" is exactly the signal, and it survives jobs being reordered.
  const finished = jobList.filter((job) => !isActive(job)).length
  const previousFinished = useRef(finished)
  useEffect(() => {
    if (finished > previousFinished.current) {
      refreshDocuments()
      refreshStats()
    }
    previousFinished.current = finished
  }, [finished, refreshDocuments, refreshStats])

  // A document deleted in another tab, or one whose ingest was cancelled and
  // indexed nothing, must not stay selected: the scope would filter on a
  // doc_id that no longer exists and every question would come back empty.
  useEffect(() => {
    if (!selectedId || documents.loading) return
    if (!docList.some((doc) => doc.doc_id === selectedId)) setSelectedId(null)
  }, [selectedId, docList, documents.loading])

  // --- actions -------------------------------------------------------------

  const handleFiles = useCallback(
    (files: File[]) => {
      for (const file of files) {
        const key = uploadKey(file)
        setPending((current) => [...current, { key, name: file.name, percent: 0, error: null }])

        void uploadDocument(file, (percent) => {
          setPending((current) =>
            current.map((item) => (item.key === key ? { ...item, percent } : item)),
          )
        })
          .then(() => {
            setPending((current) => current.filter((item) => item.key !== key))
            // Pulled straight away rather than waiting up to 5s for the next
            // tick: the job card is the only feedback that the upload worked.
            refreshJobs()
          })
          .catch((caught: unknown) => {
            setPending((current) =>
              current.map((item) =>
                item.key === key ? { ...item, error: errorMessage(caught) } : item,
              ),
            )
          })
      }
    },
    [refreshJobs],
  )

  const handleAsk = useCallback(
    (question: string) => {
      setAsking(true)
      setAskError(null)
      setTrace([])
      setAskedAt(Date.now() / 1000)
      // The previous answer is cleared rather than left standing under the new
      // question: for the ten seconds a generation takes, a stale answer with a
      // fresh trace above it reads as the answer to what was just asked. The
      // web result goes with it, for the same reason and one more: it answered
      // the *previous* question, and leaving it on screen beside a new one
      // invites reading it as an answer to the new one.
      setAnswer(null)
      setWeb(null)
      setWebError(null)
      setWebTrace([])
      // Abandons any search still in flight, so its answer cannot arrive under
      // this question. The busy flag is cleared here rather than by that run's
      // `finally`, which no longer matches and so no longer fires.
      webRun.current += 1
      setWebBusy(false)
      const scoped = selectedId ? { doc_id: selectedId } : undefined

      askStream({ question, top_k: topK ?? undefined, filters: scoped }, (event) =>
        setTrace((current) => [...current, event]),
      )
        .then((response) => {
          setAnswer(response)
          setHighlighted(null)
        })
        .catch((caught: unknown) => {
          setAnswer(null)
          setAskError(errorMessage(caught))
        })
        .finally(() => setAsking(false))
    },
    [selectedId, topK],
  )

  /**
   * Take one question the corpus could not answer to the open web.
   *
   * The question is passed in rather than read from the ask box: the box holds
   * whatever has been typed since the answer arrived, and searching for that
   * would answer something the user never asked.
   */
  const handleSearchWeb = useCallback((question: string) => {
    const run = (webRun.current += 1)
    setWebBusy(true)
    setWebError(null)
    setWeb(null)
    setWebTrace([])
    setWebAskedAt(Date.now() / 1000)

    webSearchStream(question, (event) => {
      // A superseded run's stages would interleave with the live one's and
      // leave a trace describing neither search.
      if (webRun.current === run) setWebTrace((current) => [...current, event])
    })
      .then((response) => {
        if (webRun.current === run) setWeb(response)
      })
      .catch((caught: unknown) => {
        if (webRun.current === run) setWebError(errorMessage(caught))
      })
      .finally(() => {
        if (webRun.current === run) setWebBusy(false)
      })
  }, [])

  const handleCancel = useCallback(
    (id: string) => {
      cancelJob(id)
        .then(refreshJobs)
        .catch((caught: unknown) => setActionError(errorMessage(caught)))
    },
    [refreshJobs],
  )

  const handleDelete = useCallback(
    (docId: string) => {
      deleteDocument(docId)
        .then(() => {
          refreshDocuments()
          refreshStats()
          if (docId === selectedId) setSelectedId(null)
        })
        .catch((caught: unknown) => setActionError(errorMessage(caught)))
    },
    [refreshDocuments, refreshStats, selectedId],
  )

  // --- derived -------------------------------------------------------------

  const maxTopK = stats.data?.fetch_k ?? DEFAULT_FETCH_K
  const effectiveTopK = Math.min(topK ?? stats.data?.top_k ?? DEFAULT_TOP_K, maxTopK)
  const scopeName = docList.find((doc) => doc.doc_id === selectedId)?.source_name ?? null

  const status = stats.error
    ? { tone: 'bad', label: 'API offline' }
    : activeJobs.length
      ? { tone: 'busy', label: `${activeJobs.length} indexing` }
      : warmup.state === 'loading'
        ? { tone: 'busy', label: 'loading models' }
        : { tone: 'live', label: 'ready' }

  // A web search has been run, or is running now. It outlives the search
  // itself: a result the reader has scrolled back to should still be there.
  const webStarted = webBusy || web !== null || webError !== null || webTrace.length > 0

  // The section lands below the corpus answer and its sources, which on a long
  // answer is below the fold -- and a button whose only visible effect is
  // off-screen is indistinguishable from a button that did nothing. `start`
  // rather than `nearest` because the trace is the top of this section and the
  // trace is the feedback.
  useEffect(() => {
    if (!webStarted) return
    webSection.current?.scrollIntoView({
      block: 'start',
      behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches
        ? 'auto'
        : 'smooth',
    })
  }, [webStarted])

  return (
    <div className="app">
      <header className="app-bar">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
              <path d="M4 5.5A1.5 1.5 0 0 1 5.5 4H10a2 2 0 0 1 2 2v13a1.5 1.5 0 0 0-1.5-1.5H4z" />
              <path d="M20 5.5A1.5 1.5 0 0 0 18.5 4H14a2 2 0 0 0-2 2v13a1.5 1.5 0 0 1 1.5-1.5H20z" />
            </svg>
          </span>
          <div className="brand-text">
            <h1>Local RAG</h1>
            <p>Your documents, retrieved and cited — nothing leaves this machine</p>
          </div>
        </div>

        <div className="app-bar-spacer" />

        <span className="status">
          <span className={`dot dot-${status.tone}`} />
          <span>{status.label}</span>
        </span>
      </header>

      <main className="layout">
        {/* Spans both columns, above them: the state of the machine is not
            either column's business, and on a reload it is the first thing
            worth saying. */}
        <ModelNotice state={warmup.state} seconds={warmup.seconds} />

        {/* ---------------- left rail ---------------- */}
        <div className="rail">
          <section className="panel">
            <div className="panel-head">
              <h2 className="panel-title">Library</h2>
              {docList.length > 0 && (
                <span className="panel-sub">
                  {docList.length} document{docList.length === 1 ? '' : 's'}
                </span>
              )}
            </div>
            <div className="panel-body">
              <StatsBar stats={stats.data} error={stats.error} loading={stats.loading} />

              <Dropzone
                onFiles={handleFiles}
                maxUploadMb={stats.data?.max_upload_mb ?? 512}
                active={activeJobs.length > 0}
              />

              {pending.length > 0 && (
                <ul className="jobs">
                  {pending.map((item) => (
                    <li key={item.key} className="upload-row">
                      <div className="upload-head">
                        <span className="upload-name" title={item.name}>
                          {item.name}
                        </span>
                        <span className="muted">
                          {item.error ? 'failed' : `${item.percent.toFixed(0)}%`}
                        </span>
                      </div>
                      {item.error ? (
                        <p className="notice notice-danger">{item.error}</p>
                      ) : (
                        <div className="bar" role="progressbar" aria-label={`Uploading ${item.name}`}>
                          <div className="bar-fill" style={{ width: `${item.percent}%` }} />
                        </div>
                      )}
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </section>

          {jobList.length > 0 && (
            <section className="panel">
              <div className="panel-head">
                <h2 className="panel-title">Ingest</h2>
                {activeJobs.length > 0 && <span className="panel-sub">one at a time</span>}
              </div>
              <div className="panel-body">
                <div className="jobs">
                  {jobList.map((job) => (
                    <JobCard key={job.id} job={job} onCancel={handleCancel} />
                  ))}
                </div>
              </div>
            </section>
          )}

          <section className="panel">
            <div className="panel-head">
              <h2 className="panel-title">Documents</h2>
              <span className="panel-sub">pick one to scope the next question</span>
            </div>
            <div className="panel-body">
              {actionError && (
                <p className="notice notice-danger" role="alert">
                  {actionError}
                </p>
              )}
              <DocumentList
                documents={docList}
                loading={documents.loading}
                error={documents.error}
                selectedId={selectedId}
                onSelect={setSelectedId}
                onDelete={handleDelete}
              />
            </div>
          </section>
        </div>

        {/* ---------------- right pane ---------------- */}
        <div className="pane">
          <section className="panel">
            <div className="panel-head">
              <h2 className="panel-title">Ask</h2>
            </div>
            <div className="panel-body">
              <AskPanel
                onSubmit={handleAsk}
                busy={asking}
                topK={effectiveTopK}
                onTopK={setTopK}
                maxTopK={maxTopK}
                scopeName={scopeName}
                onClearScope={() => setSelectedId(null)}
                generationEnabled={stats.data?.generation_enabled ?? true}
              />
            </div>
          </section>

          {askError && (
            <p className="notice notice-danger" role="alert">
              {askError}
            </p>
          )}

          {/* Rendered only once the first stage lands, so an idle page is not
              carrying an empty heading. It stays after the answer arrives: how
              the answer was made is the same provenance the citations carry. */}
          {trace.length > 0 && (
            <section className="panel">
              <div className="panel-head">
                <h2 className="panel-title">Pipeline</h2>
              </div>
              <div className="panel-body">
                <PipelineTrace events={trace} startedAt={askedAt} running={asking} />
              </div>
            </section>
          )}

          {answer ? (
            <>
              <AnswerCard
                response={answer}
                onHighlight={setHighlighted}
                onSearchWeb={handleSearchWeb}
                searchBusy={webBusy}
              />
              <section className="panel">
                <div className="panel-head">
                  <h2 className="panel-title">Sources</h2>
                  <span className="panel-sub">{answer.elapsed_ms.toFixed(0)} ms</span>
                </div>
                <div className="panel-body">
                  <CitationList
                    citations={answer.citations}
                    highlighted={highlighted}
                    onHighlight={setHighlighted}
                  />
                </div>
              </section>
            </>
          ) : (
            <div className="empty">
              <span className="empty-mark" aria-hidden="true">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
                  <circle cx="11" cy="11" r="6.5" />
                  <path d="m20 20-3.6-3.6" />
                </svg>
              </span>
              <h2>{docList.length ? 'Ask your library' : 'Start with a document'}</h2>
              <p>
                {docList.length
                  ? 'Every answer points back at the passage it came from, with the page number, so you can check it.'
                  : 'Drop a PDF, an EPUB, a Word file or a folder of notes into the library. A 600-page book is fine — it is ingested in slices, and you can keep asking questions while it runs.'}
              </p>
              <ol className="empty-steps">
                <li>Upload a document</li>
                <li>Watch it index</li>
                <li>Ask</li>
              </ol>
            </div>
          )}

          {/* The web fallback, in its own section below the corpus answer: it
              is an addition to what the library said, not a replacement for
              it, and the passages the corpus did find are still worth reading
              against it. */}
          {webStarted && (
            <section className="panel" ref={webSection}>
              <div className="panel-head">
                <h2 className="panel-title">Web search</h2>
                {web && <span className="panel-sub">{web.elapsed_ms.toFixed(0)} ms</span>}
              </div>
              <div className="panel-body">
                {/* Its own clock, started at the click: these stages are timed
                    from the search, not from the question, so they cannot be
                    folded into the corpus trace above without one set of
                    durations becoming a lie. */}
                <PipelineTrace events={webTrace} startedAt={webAskedAt} running={webBusy} />

                {webError && (
                  <p className="notice notice-danger" role="alert">
                    {webError}
                  </p>
                )}

                {web && <WebSearchCard response={web} />}
              </div>
            </section>
          )}
        </div>
      </main>
    </div>
  )
}
