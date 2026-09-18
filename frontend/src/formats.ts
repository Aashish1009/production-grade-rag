/**
 * File types the pipeline can parse, mirroring `loaders.SUPPORTED_SUFFIXES`.
 *
 * Duplicated on purpose, and the server stays the authority: it validates the
 * suffix and answers 400 with the list it accepts. This copy exists only so
 * the picker filters and an obviously wrong file is refused *before* being
 * sent -- the FastAPI body parser spools the whole multipart body to disk
 * before the handler gets to say no, so pre-filtering is the difference
 * between an instant "not a document" and uploading 200 MB to be told the
 * same thing.
 *
 * Drift degrades gracefully for the same reason: if this list falls behind,
 * the file is uploaded and the server's own message is what the user sees.
 */

export const ACCEPTED_SUFFIXES: readonly string[] = [
  '.pdf',
  '.docx', '.doc', '.odt', '.rtf',
  '.pptx', '.ppt',
  '.xlsx', '.xls', '.csv', '.tsv',
  '.html', '.htm', '.xml',
  '.md', '.markdown', '.txt', '.rst', '.org',
  '.epub',
  '.eml', '.msg',
  '.json',
  '.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.heic',
]

export const ACCEPT_ATTR = ACCEPTED_SUFFIXES.join(',')

/** The extension of `name`, lowercased, or '' when it has none. */
export function suffixOf(name: string): string {
  const dot = name.lastIndexOf('.')
  return dot < 0 ? '' : name.slice(dot).toLowerCase()
}

export function isAccepted(file: File): boolean {
  return ACCEPTED_SUFFIXES.includes(suffixOf(file.name))
}
