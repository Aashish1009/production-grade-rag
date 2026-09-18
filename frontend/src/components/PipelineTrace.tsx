import { useEffect, useState } from 'react'

import type { QueryStage, StageEvent } from '../types'

interface Props {
  /** Stage events, in the order the server sent them. */
  events: StageEvent[]
  /**
   * Wall-clock seconds when the question was submitted; the trace's zero.
   *
   * Only the in-flight row counts against this. Every concluded row is
   * measured between two of the server's own `elapsed_ms`, so it is exact --
   * this one starts at the click, which is a few milliseconds of request
   * dispatch ahead of the pipeline's clock and no more.
   */
  startedAt: number | null
  running: boolean
}

/**
 * The pipeline's own vocabulary, not a friendlier paraphrase of it.
 *
 * The question the user asked was "what is it actually doing", and the honest
 * answer is the name of the step: a trace that said "Thinking…" would be the
 * spinner this replaced. The message beside each name is what makes it
 * legible.
 */
const STAGE_LABEL: Record<QueryStage, string> = {
  guard: 'Guard',
  retrieve: 'Retrieve',
  rerank: 'Rerank',
  generate: 'Generate',
  verify: 'Verify',
  // Reached only on a web search. Every stage in the union has a label here on
  // purpose: a missing one renders as a blank column, and a trace whose step
  // names are invisible is worse than no trace at all.
  search: 'Search',
  fetch: 'Fetch',
}

interface Row {
  stage: QueryStage
  message: string
  /** Milliseconds from the question to this stage's first event. */
  entered: number
  /** Milliseconds from the question to this stage's most recent event. */
  latest: number
}

/**
 * Collapse the event stream into one row per stage.
 *
 * A stage speaks at least twice — on entry and on leaving — and a row that
 * appeared twice would double the apparent length of the pipeline. The query
 * graph is a straight line, so a stage can never be re-entered and the fold
 * needs no revisiting logic.
 */
function fold(events: StageEvent[]): Row[] {
  const rows: Row[] = []
  for (const event of events) {
    const last = rows[rows.length - 1]
    if (last && last.stage === event.stage) {
      last.message = event.message
      last.latest = event.elapsed_ms
    } else {
      rows.push({
        stage: event.stage,
        message: event.message,
        entered: event.elapsed_ms,
        latest: event.elapsed_ms,
      })
    }
  }
  return rows
}

function seconds(ms: number): string {
  return `${(ms / 1000).toFixed(1)}s`
}

/**
 * What the pipeline is doing, stage by stage, as it does it.
 *
 * The timings are the server's own `elapsed_ms`, so a row's duration is the
 * gap to the next row's start — a fact about the pipeline rather than about
 * when a line happened to arrive. While a stage is in flight its duration is
 * counted against that same clock, which is what keeps a long generation from
 * looking like a hang.
 */
export function PipelineTrace({ events, startedAt, running }: Props) {
  const [now, setNow] = useState(() => Date.now() / 1000)

  // Ticks only while a question is in flight. A finished trace is a record,
  // and a record that keeps counting is a claim that something is still going.
  useEffect(() => {
    if (!running) return
    setNow(Date.now() / 1000)
    const timer = setInterval(() => setNow(Date.now() / 1000), 100)
    return () => clearInterval(timer)
  }, [running])

  const rows = fold(events)
  if (!rows.length) return null

  const live = startedAt === null ? 0 : (now - startedAt) * 1000

  return (
    <ol className="trace" aria-live="polite">
      {rows.map((row, index) => {
        const next = rows[index + 1]
        const active = running && index === rows.length - 1
        // The stage's cost is the wait before the next one began; the last
        // row, having no successor, is measured against its own last word.
        const span = active ? live - row.entered : (next?.entered ?? row.latest) - row.entered

        return (
          <li key={row.stage} className={active ? 'trace-row trace-row-now' : 'trace-row'}>
            <span className="trace-stage">{STAGE_LABEL[row.stage]}</span>
            <span className="trace-message">{row.message}</span>
            <span className="trace-time">{seconds(Math.max(0, span))}</span>
          </li>
        )
      })}
    </ol>
  )
}
