import { listDocuments } from '../api'
import { usePolling } from './usePolling'

const INTERVAL_MS = 5_000

/**
 * Poll the indexed documents.
 *
 * The list only changes when an ingest finishes or a document is deleted, and
 * `App` refreshes it the moment a job reaches a terminal state, so this tick
 * is a safety net rather than the mechanism.
 */
export function useDocuments() {
  return usePolling(listDocuments, INTERVAL_MS)
}
