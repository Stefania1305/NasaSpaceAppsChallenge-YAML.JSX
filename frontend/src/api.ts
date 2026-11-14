// Calls to the Python backend (served on the same origin; proxied to :8000 in `npm run dev`).

export class ApiError extends Error {
  constructor(message: string, public status: number) { super(message); }
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export async function getJSON(url: string): Promise<any> {
  const res = await fetch(url);
  let d: { error?: string } = {};
  try { d = await res.json(); } catch { d = {}; }
  if (!res.ok) throw new ApiError(d.error || res.statusText, res.status);
  return d;
}
