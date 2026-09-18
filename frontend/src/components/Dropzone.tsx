import { useRef, useState } from 'react'

import { ACCEPT_ATTR, isAccepted, suffixOf } from '../formats'

interface Props {
  onFiles: (files: File[]) => void
  maxUploadMb: number
  active: boolean
}

const MB = 1024 * 1024

/**
 * Drag-and-drop, or click to browse.
 *
 * Validation happens here rather than at the server so a mistake is answered
 * instantly. The server still validates -- this is a courtesy, not the check
 * that matters, and the two can disagree only in the direction of the server
 * being stricter.
 */
export function Dropzone({ onFiles, maxUploadMb, active }: Props) {
  const [dragging, setDragging] = useState(false)
  const [rejected, setRejected] = useState<string[]>([])
  // Drag events fire for every child element the pointer crosses, so "drag
  // left" has to be counted rather than toggled: enter and leave come in
  // unbalanced pairs and the highlight sticks without this.
  const depth = useRef(0)
  const input = useRef<HTMLInputElement>(null)

  const accept = (incoming: File[]) => {
    const problems: string[] = []
    const good: File[] = []

    for (const file of incoming) {
      if (!isAccepted(file)) {
        problems.push(`${file.name} — ${suffixOf(file.name) || 'no extension'} is not a supported format`)
      } else if (file.size > maxUploadMb * MB) {
        problems.push(
          `${file.name} — ${(file.size / MB).toFixed(0)} MB exceeds the ${maxUploadMb} MB limit`,
        )
      } else if (file.size === 0) {
        problems.push(`${file.name} — the file is empty`)
      } else {
        good.push(file)
      }
    }

    setRejected(problems)
    if (good.length) onFiles(good)
  }

  return (
    <div className="dropzone-wrap">
      <div
        className={`dropzone${dragging ? ' dropzone-active' : ''}${active ? ' dropzone-busy' : ''}`}
        onDragEnter={(event) => {
          event.preventDefault()
          depth.current += 1
          setDragging(true)
        }}
        onDragOver={(event) => event.preventDefault()}
        onDragLeave={(event) => {
          event.preventDefault()
          depth.current -= 1
          if (depth.current <= 0) {
            depth.current = 0
            setDragging(false)
          }
        }}
        onDrop={(event) => {
          event.preventDefault()
          depth.current = 0
          setDragging(false)
          const dropped = Array.from(event.dataTransfer?.files ?? [])
          if (dropped.length) accept(dropped)
        }}
        onClick={() => input.current?.click()}
        onKeyDown={(event) => {
          if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault()
            input.current?.click()
          }
        }}
        role="button"
        tabIndex={0}
        aria-label="Upload documents"
      >
        <input
          ref={input}
          type="file"
          multiple
          accept={ACCEPT_ATTR}
          hidden
          onChange={(event) => {
            const chosen = Array.from(event.target.files ?? [])
            if (chosen.length) accept(chosen)
            // Cleared so choosing the same file twice in a row still fires.
            event.target.value = ''
          }}
        />

        <svg viewBox="0 0 24 24" aria-hidden="true" className="dropzone-icon">
          <path
            d="M12 16V4m0 0L7.5 8.5M12 4l4.5 4.5M4 15v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.6"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>

        <p className="dropzone-title">
          {dragging ? 'Drop to ingest' : 'Drop a document here'}
        </p>
        <p className="muted small">
          or click to browse — PDF, Office, Markdown, text, EPUB, images
          <br />
          up to {maxUploadMb} MB per file
        </p>
      </div>

      {rejected.length > 0 && (
        <ul className="notice notice-warn" role="alert">
          {rejected.map((message) => (
            <li key={message}>{message}</li>
          ))}
        </ul>
      )}
    </div>
  )
}
