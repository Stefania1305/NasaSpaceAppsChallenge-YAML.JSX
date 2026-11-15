// Calls to the Python backend (served on the same origin; proxied to :8000 in `npm run dev`).

export class ApiError extends Error {
  constructor(message: string, public status: number) { super(message); }
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export async function getJSON(url: string): Promise<any> {
  // Pipeline stages can take minutes, but a hung server must not leave the UI on "Running…" forever.
  let res: Response;
  try { res = await fetch(url, { signal: AbortSignal.timeout(15 * 60_000) }); }
  catch (err) {
    if ((err as Error).name === 'TimeoutError') throw new ApiError('The server took too long to answer.', 504);
    throw err;
  }
  let d: { error?: string } = {};
  let unreadable = false;
  try { d = await res.json(); } catch { d = {}; unreadable = true; }
  if (!res.ok) throw new ApiError(d.error || res.statusText, res.status);
  if (unreadable) throw new ApiError('The server sent an unreadable answer.', res.status);
  return d;
}
