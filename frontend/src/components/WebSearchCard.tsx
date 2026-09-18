import { useState } from 'react'

import { AnswerBody } from './AnswerCard'
import type { WebAskResponse } from '../types'

interface Props {
  response: WebAskResponse
}

/** Mirrors ``_LABEL`` in rag/websearch.py, so the name on screen is the name
 *  of the service that was actually asked. An unrecognised provider falls back
 *  to its own string rather than to a guess. */
const PROVIDER: Record<string, string> = {
  tavily: 'Tavily',
  duckduckgo: 'DuckDuckGo',
}

/** Just the host, as a reader would say it: a full URL in a meta line is a
 *  wall of query string nobody reads. */
function hostOf(url: string): string {
  try {
    return new URL(url).host
  } catch {
    // Not a URL we can parse. Showing it unparsed beats showing nothing.
    return url
  }
}

/**
 * One web answer, and the pages it was written from.
 *
 * Rendered beside the corpus answer rather than in place of it: the question
 * that reaches here is one the library could not answer, and the passages that
 * came closest are still worth comparing against what the web said.
 *
 * The answer is set by :func:`AnswerBody`, the same renderer the corpus answer
 * uses, so the `[N]` marks in the prose are the same interactive elements and
 * the numbering lines up with the list below.
 */
export function WebSearchCard({ response }: Props) {
  const [highlighted, setHighlighted] = useState<string | null>(null)

  const sources = response.sources.length
  const provider = response.provider ? (PROVIDER[response.provider] ?? response.provider) : null

  return (
    <article className="answer-card">
      <header className="answer-head">
        {/* Oxblood, the accent the design spends on provenance. The one thing
            a reader must never have to work out is whether an answer came from
            their own library or from the open web, and this is that fact. */}
        <span className="badge badge-web">From the web</span>
        <span className="answer-meta">
          {provider && <span>{provider}</span>}
          {sources > 0 && (
            <span>
              {sources} source{sources === 1 ? '' : 's'}
            </span>
          )}
          <span>{response.elapsed_ms.toFixed(0)} ms</span>
        </span>
      </header>

      {/* An ordinary outcome, not a crash: no key, no results, a rate limit.
          The pages below, if there are any, are still the answer's evidence. */}
      {response.error && <p className="notice notice-warn">{response.error}</p>}

      {response.answer ? (
        <>
          <AnswerBody text={response.answer} onHighlight={setHighlighted} />
          {response.refused && (
            <p className="muted small">
              A guardrail stopped this one — the answer did not cite the pages it
              was given, or drifted from them. They are still listed below to
              read.
            </p>
          )}
        </>
      ) : (
        !response.error && (
          <p className="muted">
            {response.generation_enabled
              ? 'No written answer — every model in the fallback chain failed. '
              : 'Generation is off (RAG_ENABLE_GENERATION), so there is no written answer. '}
            {sources > 0
              ? 'The pages the search found are listed below.'
              : 'The search found nothing to read.'}
          </p>
        )
      )}

      {sources > 0 && (
        <ol className="citations">
          {response.sources.map((source) => (
            <li
              key={source.marker}
              className={
                highlighted === source.marker ? 'citation citation-active' : 'citation'
              }
              onMouseEnter={() => setHighlighted(source.marker)}
              onMouseLeave={() => setHighlighted(null)}
            >
              <span className="marker marker-strong">{source.marker}</span>

              <div className="citation-body">
                <a
                  className="citation-source web-source-link"
                  href={source.url}
                  target="_blank"
                  // `noopener` because a page opened with a handle on this one
                  // could navigate it, and this window holds the user's
                  // library; `noreferrer` because the source has no business
                  // knowing where the reader came from.
                  rel="noopener noreferrer"
                >
                  {source.title}
                </a>
                <span className="muted">{hostOf(source.url)}</span>
                {source.snippet && <p className="web-snippet">{source.snippet}</p>}
              </div>
            </li>
          ))}
        </ol>
      )}
    </article>
  )
}
