import type { WarmupState } from '../hooks/useModelWarmup'

interface Props {
  state: WarmupState
  seconds: number
}

/**
 * The one wait this app asks for, said out loud.
 *
 * A first visit here can land while the models are being reloaded, and every
 * other web app has taught the visitor what a page that has stopped responding
 * looks like. So the notice has to carry three facts at once: what is
 * happening, that the app is working, and that it will not happen again --
 * which is why the reassurance is in the body text rather than left to the
 * absence of an error.
 *
 * Shown on `idle` never and on `ready` not at all: the notice is for a wait
 * that is actually being served, and a bar that outlives its work is the
 * spinner this app has otherwise avoided.
 */
export function ModelNotice({ state, seconds }: Props) {
  if (state === 'idle' || state === 'ready') return null

  if (state === 'failed') {
    return (
      <p className="notice warmup-note" role="status">
        The models did not finish loading. Nothing is broken — the next question loads them
        instead, which takes about a minute.
      </p>
    )
  }

  return (
    <section className="warmup">
      {/* The live region is the words only. The clock below changes every
          second, and a screen reader inside a polite region would read the
          whole notice out again on every tick. */}
      <div className="warmup-main" role="status">
        <h2 className="warmup-title">Loading the models</h2>
        <p className="warmup-text">
          This app keeps its models out of memory while nobody is using them, so the first visit
          after a quiet spell waits while they are read back in. It is working — this is the one
          slow part. A document uploaded now is queued and starts indexing the moment they are
          ready, and after this, answers are fast.
        </p>
      </div>
      <div className="warmup-meter" aria-hidden="true">
        <span className="warmup-clock">{seconds}s</span>
        <div className="bar bar-indeterminate">
          <div className="bar-fill" />
        </div>
      </div>
    </section>
  )
}
