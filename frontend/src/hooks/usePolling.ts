import { useCallback, useEffect, useRef, useState } from 'react'

import { errorMessage } from '../api'

export interface Polled<T> {
  /** The last successful response, or null before the first one. */
  data: T | null
  error: string | null
  /** True only until the first response, successful or not. */
  loading: boolean
  /** Fetch again immediately instead of waiting for the next tick. */
  refresh: () => void
}

/**
 * A tab nobody is looking at does not need 1.5s updates, but it should not go
 * fully stale either: the browser throttles background timers heavily anyway,
 * so this is a floor rather than a promise.
 */
const HIDDEN_INTERVAL_MS = 15_000

/**
 * Poll `fetcher` on an interval, re-rendering with each response.
 *
 * A recursive `setTimeout` rather than `setInterval`, so a slow response
 * cannot stack requests on top of each other, and so the interval can depend
 * on the data just fetched (see `useJobs`).
 *
 * `fetcher` and `intervalMs` are read through refs on every tick and are
 * deliberately *not* effect dependencies: that lets a caller pass an inline
 * arrow without tearing down and restarting the loop on every render, which
 * would poll continuously and never settle. Pass a module-level function where
 * you can, but the hook does not require it.
 */
export function usePolling<T>(
  fetcher: () => Promise<T>,
  intervalMs: number | ((previous: T | null) => number),
): Polled<T> {
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [nonce, setNonce] = useState(0)

  const latest = useRef<T | null>(null)
  const fetchRef = useRef(fetcher)
  const intervalRef = useRef(intervalMs)
  fetchRef.current = fetcher
  intervalRef.current = intervalMs

  useEffect(() => {
    let alive = true
    let timer: ReturnType<typeof setTimeout> | undefined

    const schedule = () => {
      const configured =
        typeof intervalRef.current === 'function'
          ? intervalRef.current(latest.current)
          : intervalRef.current
      const delay = document.hidden
        ? Math.max(configured, HIDDEN_INTERVAL_MS)
        : configured
      timer = setTimeout(run, delay)
    }

    const run = async () => {
      try {
        const next = await fetchRef.current()
        if (!alive) return
        latest.current = next
        setData(next)
        setError(null)
      } catch (caught) {
        if (!alive) return
        setError(errorMessage(caught))
      } finally {
        if (alive) {
          setLoading(false)
          schedule()
        }
      }
    }

    void run()

    return () => {
      // Anything still in flight belongs to a world that is gone; dropping its
      // response is the point of this flag.
      alive = false
      clearTimeout(timer)
    }
  }, [nonce])

  const refresh = useCallback(() => setNonce((value) => value + 1), [])

  return { data, error, loading, refresh }
}
