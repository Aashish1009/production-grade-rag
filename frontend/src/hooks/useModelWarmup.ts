import { useEffect, useRef, useState } from 'react'

import { warmModels } from '../api'

/** Where the model load has got to. */
export type WarmupState =
  /** The models are already resident, or nothing has been asked of them yet. */
  | 'idle'
  | 'loading'
  | 'ready'
  /** The request failed. Not an error state: the next question loads them. */
  | 'failed'

export interface Warmup {
  state: WarmupState
  /** Whole seconds since the load started, while it is running. */
  seconds: number
}

/**
 * Start loading the models, once, when the page finds them absent.
 *
 * The server drops its models after a spell of no use and rebuilds them on
 * demand, so the first visitor after a quiet period pays the whole reload. It
 * costs about ninety seconds, and it is paid *inside their first question* --
 * which is the worst possible place for it, because a pipeline trace with
 * "Generate" sitting still for a minute reads as a hang. Starting it on page
 * open moves it to the time the visitor spends reading and choosing a file,
 * which is a wait they were going to have anyway.
 *
 * The two facts are passed in rather than fetched here, so this hook has no
 * opinion about where they came from: they are `models_loaded` from
 * `/api/stats`, which the page already polls.
 *
 * Deliberately once per page load. The stats poll repeats every ten seconds
 * and a warm-up outlasts it several times over, so the ref guard is what keeps
 * one visitor from issuing six loads in a row -- and there is no periodic
 * re-warm either, because a tab left open in the background would then keep
 * this machine's models resident for nobody.
 */
export function useModelWarmup(needsWarm: boolean, alreadyLoaded: boolean): Warmup {
  const [state, setState] = useState<WarmupState>('idle')
  const [seconds, setSeconds] = useState(0)
  const started = useRef(false)

  useEffect(() => {
    if (!needsWarm || started.current) return
    started.current = true
    setSeconds(0)
    setState('loading')
    warmModels()
      .then(() => setState('ready'))
      .catch(() => setState('failed'))
  }, [needsWarm])

  // Someone else's load, or one already running when this page opened. Also
  // how a failed warm-up recovers: the first question loads the models, the
  // next stats poll reports them, and the notice goes.
  useEffect(() => {
    if (alreadyLoaded) setState('ready')
  }, [alreadyLoaded])

  useEffect(() => {
    if (state !== 'loading') return
    const ticker = setInterval(() => setSeconds((value) => value + 1), 1000)
    return () => clearInterval(ticker)
  }, [state])

  return { state, seconds }
}
