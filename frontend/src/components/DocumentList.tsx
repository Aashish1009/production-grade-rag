import { useState } from 'react'

import type { DocumentSummary } from '../types'

interface Props {
  documents: DocumentSummary[]
  loading: boolean
  error: string | null
  selectedId: string | null
  onSelect: (docId: string | null) => void
  onDelete: (docId: string) => void
}

function pageSpan(doc: DocumentSummary): string {
  if (doc.page_start === null && doc.page_end === null) return ''
  if (doc.page_start === doc.page_end) return `p. ${doc.page_start}`
  return `pp. ${doc.page_start ?? '?'}–${doc.page_end ?? '?'}`
}

/**
 * The corpus, as one row per source document.
 *
 * Selecting a row scopes the next question to it. That is a *filter on the
 * query*, not a mode: the selection is shown on the ask panel too, so it is
 * never invisible while it is changing what you get back.
 */
export function DocumentList({
  documents,
  loading,
  error,
  selectedId,
  onSelect,
  onDelete,
}: Props) {
  // Deleting is irreversible from the UI's side, so the button asks twice.
  // Local, not a window.confirm: a modal would block the poll that is telling
  // the user an ingest is still running.
  const [confirming, setConfirming] = useState<string | null>(null)

  if (error) {
    return (
      <p className="notice notice-warn" role="alert">
        Could not list documents: {error}
      </p>
    )
  }

  if (!documents.length) {
    return (
      <p className="muted small">
        {loading ? 'Loading…' : 'Nothing indexed yet. Upload a document to start.'}
      </p>
    )
  }

  return (
    <ul className="docs">
      <li>
        <button
          type="button"
          className={`doc doc-all${selectedId === null ? ' doc-selected' : ''}`}
          onClick={() => onSelect(null)}
        >
          <span className="doc-name">All documents</span>
          <span className="muted small">search everything</span>
        </button>
      </li>

      {documents.map((doc) => (
        <li key={doc.doc_id}>
          <div className={`doc${selectedId === doc.doc_id ? ' doc-selected' : ''}`}>
            <button
              type="button"
              className="doc-main"
              onClick={() => onSelect(selectedId === doc.doc_id ? null : doc.doc_id)}
              title={doc.source ?? doc.source_name}
            >
              <span className="doc-name">
                {doc.source_name}
                {!doc.complete && (
                  <span className="badge badge-partial" title="An ingest was interrupted before this finished">
                    partial
                  </span>
                )}
              </span>
              <span className="muted small">
                {doc.chunks.toLocaleString()} chunks
                {pageSpan(doc) && `, ${pageSpan(doc)}`}
              </span>
            </button>

            {confirming === doc.doc_id ? (
              <span className="doc-confirm">
                <button
                  type="button"
                  className="button button-danger button-small"
                  onClick={() => {
                    setConfirming(null)
                    onDelete(doc.doc_id)
                  }}
                >
                  Delete
                </button>
                <button
                  type="button"
                  className="button button-quiet button-small"
                  onClick={() => setConfirming(null)}
                >
                  Keep
                </button>
              </span>
            ) : (
              <button
                type="button"
                className="button button-quiet button-small"
                aria-label={`Delete ${doc.source_name}`}
                onClick={() => setConfirming(doc.doc_id)}
              >
                ✕
              </button>
            )}
          </div>
        </li>
      ))}
    </ul>
  )
}
