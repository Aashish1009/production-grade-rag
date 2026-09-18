import type { Citation } from '../types'

interface Props {
  citations: Citation[]
  highlighted: string | null
  onHighlight: (marker: string | null) => void
}

function pageSpan(citation: Citation): string {
  if (citation.page_start === null) return 'no page'
  if (citation.page_start === citation.page_end) return `p. ${citation.page_start}`
  return `pp. ${citation.page_start}–${citation.page_end ?? '?'}`
}

/**
 * The sources behind an answer, in the order they were numbered.
 *
 * The score is shown as a number rather than as a bar. A reranker's scores are
 * only meaningful relative to each other within one query, so a bar scaled
 * against the best hit renders a comparison the number itself cannot support;
 * a column of figures is the honest version of the same information, and it
 * can be read down.
 */
export function CitationList({ citations, highlighted, onHighlight }: Props) {
  if (!citations.length) {
    return <p className="muted small">No passages were retrieved.</p>
  }

  return (
    <ol className="citations">
      {citations.map((citation) => (
        <li
          key={citation.marker}
          className={
            highlighted === citation.marker
              ? 'citation citation-active'
              : 'citation'
          }
          onMouseEnter={() => onHighlight(citation.marker)}
          onMouseLeave={() => onHighlight(null)}
        >
          <span className="marker marker-strong">{citation.marker}</span>

          <div className="citation-body">
            <span className="citation-source">{citation.source}</span>
            <span className="muted">
              {pageSpan(citation)}
              {citation.section && `, ${citation.section}`}
            </span>
          </div>

          {citation.rerank_score !== null && (
            <span className="score" title="Rerank score — comparable within this result set only">
              {citation.rerank_score.toFixed(2)}
            </span>
          )}
        </li>
      ))}
    </ol>
  )
}
