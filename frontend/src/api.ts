/**
 * Typed client for the local API.
 *
 * Every function here is a module-level function with a stable identity, which
 * is what lets the polling hooks take one directly without re-subscribing on
 * every render.
 */

import type {
  AskResponse,
  DocumentSummary,
  Job,
  StageEvent,
  Stats,
  StreamEvent,
  WebAskResponse,
} from './types'

const BASE = '/api'

export class ApiError extends Error {
  readonly status: number

  constructor(message: string, status: number) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

/** FastAPI puts the useful text in ``detail``; fall back to the status text. */
function detailFrom(body: string, fallback: string): string {
  try {
    const parsed = JSON.parse(body) as { detail?: unknown }
    if (typeof parsed.detail === 'string') return parsed.detail
  } catch {
    // Not JSON: the server said something unstructured. Better to show the
    // fallback than to throw a second error while reporting the first.
  }
  return fallback
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}${path}`, init)
  if (!response.ok) {
    throw new ApiError(
      detailFrom(await response.text(), response.statusText),
      response.status,
    )
  }
  return (await response.json()) as T
}

function jsonInit(method: string, body: unknown): RequestInit {
  return {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }
}

export const getStats = (): Promise<Stats> => request<Stats>('/stats')

export const listJobs = (): Promise<Job[]> => request<Job[]>('/jobs')

export const getJob = (id: string): Promise<Job> => request<Job>(`/jobs/${id}`)

export const cancelJob = (id: string): Promise<Job> =>
  request<Job>(`/jobs/${id}/cancel`, { method: 'POST' })

export const listDocuments = (): Promise<DocumentSummary[]> =>
  request<DocumentSummary[]>('/documents')

export const deleteDocument = (docId: string): Promise<{ doc_id: string }> =>
  request(`/documents/${encodeURIComponent(docId)}`, { method: 'DELETE' })

export interface AskPayload {
  question: string
  top_k?: number
  filters?: Record<string, string>
}

export const ask = (payload: AskPayload): Promise<AskResponse> =>
  request<AskResponse>('/ask', jsonInit('POST', payload))

/**
 * Read an NDJSON stream of stages, and settle on the response at the end.
 *
 * Shared by both ask paths, which differ only in their endpoint and in what
 * the result carries. NDJSON over a POST via `fetch`, not SSE over an
 * `EventSource`: the question is a JSON body and an `EventSource` can only
 * GET, so it would have to be crammed into a URL. The stages call `onStage` as
 * they arrive.
 *
 * A rejected question — a guardrail, an unknown filter key — arrives as an
 * `error` line rather than an HTTP status, because the status line was sent
 * before the pipeline ran. It is still thrown as an `ApiError` carrying the
 * server's own status, so a caller cannot tell the two paths apart.
 */
async function streamStages<T>(
  path: string,
  payload: unknown,
  onStage: (event: StageEvent) => void,
): Promise<T> {
  const response = await fetch(`${BASE}${path}`, jsonInit('POST', payload))

  // Reaching here at all non-ok means the request never got as far as the
  // pipeline: a malformed body, or a server that is not this one.
  if (!response.ok || !response.body) {
    throw new ApiError(
      detailFrom(await response.text(), response.statusText),
      response.status,
    )
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let result: T | null = null
  let failure: ApiError | null = null

  try {
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      // `stream: true` so a multi-byte character split across two chunks is
      // held back rather than decoded into a replacement character.
      buffer += decoder.decode(value, { stream: true })

      // Lines, not chunks. A chunk boundary lands wherever the network put it
      // -- often mid-object -- so only whole lines are parsed and the tail is
      // carried into the next read.
      let newline = buffer.indexOf('\n')
      while (newline !== -1) {
        const line = buffer.slice(0, newline).trim()
        buffer = buffer.slice(newline + 1)
        newline = buffer.indexOf('\n')
        if (!line) continue

        const event = JSON.parse(line) as StreamEvent<T>
        if (event.type === 'stage') onStage(event)
        else if (event.type === 'result') result = event
        else failure = new ApiError(event.detail, event.status)
      }
      if (failure) break
    }
  } finally {
    // Stops reading a body nobody is waiting for when the caller navigated
    // away or the error branch above broke early.
    void reader.cancel().catch(() => {})
  }

  if (failure) throw failure
  if (!result) {
    throw new ApiError('The answer stream ended before the answer arrived.', 0)
  }
  return result
}

/** Ask the corpus, reporting each pipeline stage as it happens. */
export const askStream = (
  payload: AskPayload,
  onStage: (event: StageEvent) => void,
): Promise<AskResponse> => streamStages<AskResponse>('/ask/stream', payload, onStage)

/**
 * Ask the open web, reporting each pipeline stage as it happens.
 *
 * The fallback for a question the corpus could not answer, and the only call
 * in this module that sends anything off the machine — which is why it takes a
 * question and nothing else. The endpoint has no reranker to narrow and no
 * index to scope, and it is never reached except by a user clicking the button
 * that offers it.
 */
export const webSearchStream = (
  question: string,
  onStage: (event: StageEvent) => void,
): Promise<WebAskResponse> =>
  streamStages<WebAskResponse>('/web/stream', { question }, onStage)

/**
 * Upload a file, reporting transfer progress.
 *
 * XMLHttpRequest rather than fetch, for one reason: fetch still cannot report
 * *upload* progress in any shipping browser, and on a 600-page scan the
 * transfer is the first thing the user waits on. The progress bar below is a
 * different, later thing (pages parsed) and is polled separately.
 *
 * No Content-Type header is set on purpose -- the browser must add the
 * multipart boundary itself.
 */
export function uploadDocument(
  file: File,
  onProgress: (percent: number) => void,
): Promise<Job> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    xhr.open('POST', `${BASE}/documents`)

    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable && event.total > 0) {
        onProgress((event.loaded / event.total) * 100)
      }
    }

    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText) as Job)
        } catch {
          reject(new ApiError('The server returned a response we could not read.', xhr.status))
        }
      } else {
        reject(new ApiError(detailFrom(xhr.responseText, xhr.statusText), xhr.status))
      }
    }
    xhr.onerror = () => reject(new ApiError('The upload could not reach the server.', 0))
    xhr.onabort = () => reject(new ApiError('The upload was cancelled.', 0))

    const form = new FormData()
    form.append('file', file)
    xhr.send(form)
  })
}

/** A short, human message from anything thrown by this module. */
export function errorMessage(error: unknown): string {
  if (error instanceof ApiError) return error.message
  if (error instanceof Error) return error.message
  return String(error)
}
