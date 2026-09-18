import { useEffect, useState } from 'react'

import { isActive } from '../hooks/useJobs'
import type { Job, JobStatus, Stage } from '../types'

interface Props {
  job: Job
  onCancel: (id: string) => void
}

/**
 * The stages a document passes through, in order.
 *
 * `percent` from the API is progress through *the current stage*, not the
 * whole run, so the bar resets at each boundary. Shown alone that reads as a
 * job restarting; shown beside the record below it reads as what it is. These
 * are the same keys as `rag.state.ProgressStage`.
 */
const ORDER: Stage[] = ['discover', 'load', 'clean', 'chunk', 'index']

const STAGE_LABEL: Record<Stage, string> = {
  discover: 'Scan',
  load: 'Parse',
  clean: 'Clean',
  chunk: 'Split',
  index: 'Index',
}

const STATUS_LABEL: Record<JobStatus, string> = {
  queued: 'Queued',
  running: 'Running',
  done: 'Indexed',
  failed: 'Failed',
  cancelled: 'Cancelled',
}

function duration(startedAt: number | null, finishedAt: number | null, now: number): string {
  if (startedAt === null) return ''
  const seconds = Math.max(0, Math.round((finishedAt ?? now) - startedAt))
  const minutes = Math.floor(seconds / 60)
  return minutes ? `${minutes}m ${seconds % 60}s` : `${seconds}s`
}

function span(seconds: number | null): string {
  if (seconds === null) return ''
  const minutes = Math.floor(seconds / 60)
  return minutes ? `${minutes}m ${Math.round(seconds % 60)}s` : `${seconds.toFixed(1)}s`
}

export function JobCard({ job, onCancel }: Props) {
  const active = isActive(job)

  // Only ticks while the job is live: a finished card showing "1m 12s" does not
  // need a timer, and a page full of them should not each hold one.
  const [now, setNow] = useState(() => Date.now() / 1000)
  useEffect(() => {
    if (!active) return
    const timer = setInterval(() => setNow(() => Date.now() / 1000), 1000)
    return () => clearInterval(timer)
  }, [active])

  const determinate = active && job.percent !== null

  // The record of what has happened, plus -- while the job runs -- the stages
  // it has not reached yet. One list rather than a stepper beside a history:
  // they are the same five steps, and showing them twice is how the two end up
  // disagreeing about which one is current.
  const lastIndex = job.stages.length
    ? ORDER.indexOf(job.stages[job.stages.length - 1].stage)
    : -1
  const ahead = active ? ORDER.slice(lastIndex + 1) : []

  // Durations are recorded when a stage is *left*, so the in-flight one has
  // none. It is measured against the job's own clock instead: everything this
  // job has spent so far, minus everything it spent before this stage.
  const spent = job.stages.reduce((total, record) => total + (record.seconds ?? 0), 0)
  const elapsed = job.started_at === null ? 0 : (job.finished_at ?? now) - job.started_at

  return (
    <article className={`job job-${job.status}`}>
      <header className="job-head">
        <span className="job-name" title={job.filename}>
          {job.filename}
        </span>
        <span className={`badge badge-${job.status}`}>{STATUS_LABEL[job.status]}</span>
      </header>

      {active && (
        <div
          className={`bar${determinate ? '' : ' bar-indeterminate'}`}
          role="progressbar"
          aria-label="Ingest progress"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={determinate ? (job.percent ?? 0) : undefined}
        >
          <div
            className="bar-fill"
            style={determinate ? { width: `${job.percent}%` } : undefined}
          />
        </div>
      )}

      {(job.stages.length > 0 || ahead.length > 0) && (
        <ol className="trace trace-compact">
          {job.stages.map((record) => {
            const open = record.seconds === null
            return (
              <li key={record.stage} className="trace-row">
                <span className="trace-stage">{STAGE_LABEL[record.stage]}</span>
                {/* Only a stage that has been left has an outcome to report.
                    The one still running is narrated by the live line below,
                    and saying it in both places is how the two drift apart. */}
                <span className="trace-message">{open ? '' : record.message}</span>
                <span className="trace-time">
                  {open ? span(Math.max(0, elapsed - spent)) : span(record.seconds)}
                </span>
              </li>
            )
          })}

          {ahead.map((stage) => (
            <li key={stage} className="trace-row trace-row-ahead">
              <span className="trace-stage">{STAGE_LABEL[stage]}</span>
              <span className="trace-message" />
              <span className="trace-time" />
            </li>
          ))}
        </ol>
      )}

      <p className="job-line" aria-live="polite">
        {active && (job.message ?? 'waiting for the worker…')}
        {job.status === 'done' && (
          <>
            {job.chunks.toLocaleString()} chunks indexed
            {job.cancel_requested ? ' (cancel arrived too late)' : ''}
          </>
        )}
        {job.status === 'cancelled' && (
          <>
            Stopped. Whatever had already been written stays in the index, marked
            incomplete.
          </>
        )}
        {job.status === 'failed' && (job.error ?? 'The ingest failed.')}
      </p>

      <footer className="job-foot">
        <span className="muted small">
          {duration(job.started_at, job.finished_at, now)}
          {job.started_at === null && job.status === 'queued' ? 'waiting for the current job' : ''}
        </span>
        {active && (
          <button
            type="button"
            className="button button-quiet"
            onClick={() => onCancel(job.id)}
            disabled={job.cancel_requested}
          >
            {job.cancel_requested ? 'Stopping…' : 'Cancel'}
          </button>
        )}
      </footer>
    </article>
  )
}
