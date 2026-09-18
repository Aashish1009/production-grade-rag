import { getStats } from '../api'
import { usePolling } from './usePolling'

const INTERVAL_MS = 10_000

/**
 * Poll collection and model state.
 *
 * Slower than the other polls on purpose: reading the stats takes the store
 * guard, which an ingest is holding for a slice at a time, and the numbers
 * here change only when a job finishes.
 */
export function useStats() {
  return usePolling(getStats, INTERVAL_MS)
}
