import { listJobs } from '../api'
import { usePolling } from './usePolling'
import type { Job } from '../types'

/** Still queued or running: the only states where polling fast is worth it. */
export function isActive(job: Job): boolean {
  return job.status === 'queued' || job.status === 'running'
}

const ACTIVE_INTERVAL_MS = 1_500
const IDLE_INTERVAL_MS = 5_000

/**
 * Poll the ingest jobs, faster while something is actually happening.
 *
 * The halved-and-then-some tick while a job runs is what makes the page
 * counter look live on a 600-page book; the slower tick is for the rest of the
 * time, which is most of the time, when a poll returns the same list it did a
 * moment ago.
 */
export function useJobs() {
  return usePolling(listJobs, (jobs) =>
    jobs?.some(isActive) ? ACTIVE_INTERVAL_MS : IDLE_INTERVAL_MS,
  )
}
