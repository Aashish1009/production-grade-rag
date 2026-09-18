import { useRef, useState } from 'react'

interface Props {
  onSubmit: (question: string) => void
  busy: boolean
  topK: number
  onTopK: (value: number) => void
  maxTopK: number
  scopeName: string | null
  onClearScope: () => void
  generationEnabled: boolean
}

export function AskPanel({
  onSubmit,
  busy,
  topK,
  onTopK,
  maxTopK,
  scopeName,
  onClearScope,
  generationEnabled,
}: Props) {
  const [question, setQuestion] = useState('')
  const textarea = useRef<HTMLTextAreaElement>(null)

  const submit = () => {
    if (busy) return
    const trimmed = question.trim()
    if (!trimmed) {
      // Deliberately not a disabled button. A greyed-out control next to a
      // box the user has not typed in yet reads as "there is no submit
      // button"; an enabled one that puts the cursor where the input is
      // missing reads as "type something".
      textarea.current?.focus()
      return
    }
    onSubmit(trimmed)
    // Focused on purpose: the next question is usually a follow-up, and having
    // to click back into the box after every answer gets old fast.
    textarea.current?.focus()
  }

  return (
    <section className="ask">
      {scopeName && (
        <div className="scope">
          <span className="muted small">Scoped to</span>
          <span className="scope-name">{scopeName}</span>
          <button type="button" className="button button-quiet button-small" onClick={onClearScope}>
            search everything
          </button>
        </div>
      )}

      <textarea
        ref={textarea}
        className="ask-input"
        value={question}
        onChange={(event) => setQuestion(event.target.value)}
        onKeyDown={(event) => {
          // Enter alone inserts a newline: questions about a book often run to
          // a couple of lines, and a stray Enter losing the draft is worse
          // than an extra keypress.
          if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) {
            event.preventDefault()
            submit()
          }
        }}
        placeholder="Ask something about your documents…"
        rows={3}
        disabled={busy}
      />

      <div className="ask-controls">
        <label className="slider">
          <span className="muted small">
            Passages: <strong>{topK}</strong>
          </span>
          <input
            type="range"
            min={1}
            max={maxTopK}
            value={topK}
            onChange={(event) => onTopK(Number(event.target.value))}
            disabled={busy}
            // The ceiling is fetch_k, the number of candidates retrieval
            // produces: asking the reranker to keep more than it was given
            // could not be honoured, so the UI does not offer it.
            title={`Keep between 1 and ${maxTopK} passages, out of ${maxTopK} candidates retrieved`}
          />
        </label>

        <button
          type="button"
          className="button button-primary ask-submit"
          onClick={submit}
          disabled={busy}
        >
          {busy ? 'Asking…' : 'Ask'}
        </button>
      </div>

      <p className="muted small ask-hint">
        {generationEnabled
          ? 'Ctrl/⌘ + Enter to ask. Answers cite their sources as [1], [2].'
          : 'Generation is off (RAG_ENABLE_GENERATION) — you will get the retrieved passages, not a written answer.'}
      </p>
    </section>
  )
}
