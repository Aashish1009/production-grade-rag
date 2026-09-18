import { useState } from 'react'
import katex from 'katex'

import type { AskResponse } from '../types'

interface Props {
  response: AskResponse
  onHighlight: (marker: string | null) => void
  /** Runs a web search for the question this card answers. Absent when the
   *  server has no backend to run one with — which is the same condition the
   *  server reports as `search_suggested` being false, so the two cannot
   *  disagree about whether an offer is worth making. */
  onSearchWeb?: (question: string) => void
  /** A web search is in flight. The offer is disabled rather than hidden: it
   *  is the thing the user just clicked, and a control that vanishes under
   *  the pointer reads as a failure rather than as work in progress. */
  searchBusy?: boolean
}

const MARKER = /(\[\d+\])/g

/**
 * Delimiters the model actually reaches for.
 *
 * `\[ ... \]` and `\( ... \)` are what it emits unprompted; the prompt asks
 * for `\[ ... \]` for a standalone equation. The dollar forms are accepted
 * because models produce them too, and a delimiter the renderer does not know
 * is a delimiter it prints as raw backslashes.
 *
 * Display alternatives come first so `\[ ... \]` wins over `$ ... $` at a
 * position where both could start. Inline content may not cross a newline,
 * which bounds the damage a stray `$` in prose (a price, a shell variable)
 * can do.
 */
const MATH_SOURCE =
  '\\\\\\[([\\s\\S]+?)\\\\\\]' + // \[ ... \]
  '|\\$\\$([\\s\\S]+?)\\$\\$' + // $$ ... $$
  '|\\\\\\(([\\s\\S]+?)\\\\\\)' + // \( ... \)
  '|\\$([^$\\n]+?)\\$' // $ ... $

/** A run of answer text, or a run of LaTeX. */
interface Segment {
  kind: 'text' | 'math'
  value: string
  /** Display math sits on its own line, the way the book sets an equation. */
  display: boolean
}

/** Split an answer into alternating text and math runs. */
function splitMath(text: string): Segment[] {
  const out: Segment[] = []
  const re = new RegExp(MATH_SOURCE, 'g')
  let cursor = 0
  let match: RegExpExecArray | null

  while ((match = re.exec(text)) !== null) {
    if (match.index > cursor) {
      out.push({ kind: 'text', value: text.slice(cursor, match.index), display: false })
    }
    // Groups 1-2 are the display forms, 3-4 the inline ones; a group that
    // did not participate in the match comes back undefined.
    const [square, dollars, parens, single] = match.slice(1) as (string | undefined)[]
    out.push({
      kind: 'math',
      value: (square ?? dollars ?? parens ?? single ?? '').trim(),
      display: square !== undefined || dollars !== undefined,
    })
    cursor = match.index + match[0].length
    if (match[0].length === 0) re.lastIndex += 1
  }

  if (cursor < text.length) {
    out.push({ kind: 'text', value: text.slice(cursor), display: false })
  }
  return out
}

/**
 * Render LaTeX through KaTeX.
 *
 * `throwOnError: false` keeps one malformed equation from taking the whole
 * answer down mid-render: KaTeX shows the offending source in its error
 * colour instead. `trust` stays at its default of false, so the `\href` and
 * `\htmlClass` commands cannot inject anything into the page.
 */
function renderMath(tex: string, display: boolean): string {
  return katex.renderToString(tex, { displayMode: display, throwOnError: false })
}

/** Split an answer so each `[N]` becomes an element the pointer can act on. */
function withMarkers(text: string, onHighlight: (marker: string | null) => void) {
  return text.split(MARKER).map((part, index) =>
    /^\[\d+\]$/.test(part) ? (
      <span
        key={index}
        className="marker marker-inline"
        onMouseEnter={() => onHighlight(part)}
        onMouseLeave={() => onHighlight(null)}
      >
        {part}
      </span>
    ) : (
      <span key={index}>{part}</span>
    ),
  )
}

/**
 * The answer body: paragraphs of prose with the equations set as math.
 *
 * Blank lines separate paragraphs. A block that is nothing but display math is
 * rendered as a math block rather than wrapped in a `<p>`, so KaTeX can centre
 * it and give it the vertical room an equation needs to be read.
 *
 * Exported because a web answer is set the same way, down to the `[N]` marks:
 * the numbering the guardrails verify is the numbering the reader clicks, and
 * a second renderer would be a second place for the two to drift apart.
 */
export function AnswerBody({
  text,
  onHighlight,
}: {
  text: string
  onHighlight: (marker: string | null) => void
}) {
  const blocks = text
    .split(/\n{2,}/)
    .map((block) => block.trim())
    .filter(Boolean)

  return (
    <>
      {blocks.map((block, index) => {
        const segments = splitMath(block)
        const [only] = segments
        if (segments.length === 1 && only.kind === 'math' && only.display) {
          return (
            <div
              key={index}
              className="answer-math"
              // KaTeX escapes its own input and `trust` is off; the string is
              // markup KaTeX built, not markup the model supplied.
              dangerouslySetInnerHTML={{ __html: renderMath(only.value, true) }}
            />
          )
        }
        return (
          <p className="answer-text" key={index}>
            {segments.map((segment, part) =>
              segment.kind === 'math' ? (
                <span
                  key={part}
                  dangerouslySetInnerHTML={{ __html: renderMath(segment.value, segment.display) }}
                />
              ) : (
                <span key={part}>{withMarkers(segment.value, onHighlight)}</span>
              ),
            )}
          </p>
        )
      })}
    </>
  )
}

/** A retrieved passage, split into its header line and its text. */
function ContextBlock({ block }: { block: string }) {
  const newline = block.indexOf('\n')
  const head = newline < 0 ? block : block.slice(0, newline)
  const body = newline < 0 ? '' : block.slice(newline + 1)

  return (
    <div className="context-block">
      <div className="context-head">{head}</div>
      {body && <div className="context-body">{body}</div>}
    </div>
  )
}

/**
 * The offer to take a question the corpus could not answer to the web.
 *
 * This is the one control in the app that sends anything off the machine, so
 * it says so on itself rather than trusting the reader to have found it in the
 * README. The button is filled ink -- the strongest shape the design has --
 * because the state it appears in is one where the user has nothing else to
 * press, and a hairline outline on paper reads as ornament exactly when it
 * needs to read as the next step.
 *
 * The question travels with the click rather than being read back out of the
 * ask box: the box holds whatever was typed since, and searching for that
 * instead would answer something the user never asked.
 */
function SearchOffer({
  question,
  busy,
  onSearch,
}: {
  question: string
  busy: boolean
  onSearch: (question: string) => void
}) {
  return (
    <div className="search-offer">
      <p className="search-offer-note">
        No answer in your documents. Searching the web is the one thing this app
        does that leaves this machine.
      </p>
      <button
        type="button"
        className="button button-primary"
        onClick={() => onSearch(question)}
        disabled={busy}
      >
        {busy ? 'Searching the web…' : 'Search the web'}
      </button>
    </div>
  )
}

export function AnswerCard({
  response,
  onHighlight,
  onSearchWeb,
  searchBusy = false,
}: Props) {
  const [copied, setCopied] = useState(false)

  const copy = async () => {
    if (!response.answer) return
    try {
      // Undefined outside a secure context, which a LAN-bound server is.
      await navigator.clipboard.writeText(response.answer)
      setCopied(true)
      setTimeout(() => setCopied(false), 1600)
    } catch {
      // Nothing to say: the answer is on screen and selectable either way.
    }
  }

  // The offer is drawn on all three shapes an empty-handed answer takes -- no
  // answer at all, one the guardrails withheld, and one where the model
  // reported that its passages do not cover the question -- because all three
  // leave the user with the same problem and the same one way out.
  const offer =
    response.search_suggested && onSearchWeb ? (
      <SearchOffer question={response.question} busy={searchBusy} onSearch={onSearchWeb} />
    ) : null

  // No answer, but not because of a guardrail: generation is off, or every
  // model in the fallback chain failed. Either way the passages are still
  // useful, so they are shown rather than an empty card.
  if (!response.answer) {
    return (
      <article className="answer-card">
        <header className="answer-head">
          <span className="badge badge-context">Context only</span>
          <span className="answer-meta">
            <span>{response.elapsed_ms.toFixed(0)} ms</span>
          </span>
        </header>

        {response.context ? (
          <div className="context-list">
            {response.context.split('\n\n').map((block, index) => (
              <ContextBlock key={index} block={block} />
            ))}
          </div>
        ) : (
          <p className="muted">
            Nothing in the corpus matched this question. Try different wording, or
            upload a document that covers it.
            {/* Said only when it is true, and only here: an empty answer with
                no offer above it otherwise looks like a button that failed to
                appear rather than a server that cannot make one. */}
            {!response.web_search_enabled &&
              ' This server has no web-search backend configured, so there is nowhere else to look.'}
          </p>
        )}

        {offer}
      </article>
    )
  }

  if (response.refused) {
    return (
      <article className="answer-card answer-refused">
        <header className="answer-head">
          <span className="badge badge-warn">Withheld</span>
          <span className="answer-meta">
            <span>{response.elapsed_ms.toFixed(0)} ms</span>
          </span>
        </header>
        <AnswerBody text={response.answer} onHighlight={onHighlight} />
        <p className="muted small">
          A guardrail stopped this one — the answer did not cite the passages it
          was given, or drifted from them. The passages below are still there to
          read.
        </p>
        {offer}
      </article>
    )
  }

  return (
    <article className="answer-card">
      <header className="answer-head">
        <span className="badge badge-done">Answered</span>
        <span className="answer-meta">
          <span>
            {response.citations.length} passage{response.citations.length === 1 ? '' : 's'}
          </span>
          <span>{response.elapsed_ms.toFixed(0)} ms</span>
        </span>
        <button type="button" className="button button-quiet button-small" onClick={copy}>
          {copied ? 'Copied' : 'Copy'}
        </button>
      </header>

      <AnswerBody text={response.answer} onHighlight={onHighlight} />
      {offer}
    </article>
  )
}
