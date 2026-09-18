import type { Stats } from '../types'

interface Props {
  stats: Stats | null
  error: string | null
  loading: boolean
}

function Chip({ label, value, title }: { label: string; value: string; title?: string }) {
  return (
    <div className="chip" title={title ?? value}>
      <span className="chip-label">{label}</span>
      <span className="chip-value">{value}</span>
    </div>
  )
}

/**
 * Collection and model state.
 *
 * When the stats call fails there is usually exactly one reason worth naming --
 * the server is not up -- so the error replaces the whole bar rather than
 * sitting beside stale numbers that look authoritative.
 */
export function StatsBar({ stats, error, loading }: Props) {
  if (error) {
    return (
      <div className="stats-bar stats-bar-error" role="status">
        <span className="dot dot-bad" />
        <div>
          <strong>The API is not responding.</strong>
          <p className="muted">
            Start it with <code>uv run rag-server</code>, then reload.
            <br />
            <span className="mono small">{error}</span>
          </p>
        </div>
      </div>
    )
  }

  if (!stats) {
    return (
      <div className="stats-bar" aria-busy={loading}>
        <span className="dot dot-idle" />
        <span className="muted">{loading ? 'Connecting…' : 'No stats yet.'}</span>
      </div>
    )
  }

  const pages = stats.exists ? `${stats.points.toLocaleString()} chunks` : 'not created yet'

  return (
    <div className="stats-bar">
      <Chip
        label={stats.collection}
        value={pages}
        title={stats.exists ? 'Vectors in the collection' : 'The store is created on first ingest'}
      />
      <Chip label="device" value={stats.device} />
      <Chip label="retrieval" value={stats.retrieval_mode} title={`${stats.fetch_k} fetched, ${stats.top_k} kept`} />
      <Chip label="rerank" value={stats.reranker_model} />
      <Chip
        label="generation"
        value={
          stats.generation_enabled
            ? (stats.llm_provider ?? 'on')
            : 'off (context only)'
        }
      />
    </div>
  )
}
