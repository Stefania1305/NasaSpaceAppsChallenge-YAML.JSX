// Number and date formatting used across the panel.

export const pct = (v: number | null | undefined) => v == null ? 'n/a' : `${Math.round(v * 100)}%`;
export const num = (v: number | null | undefined) =>
  v == null ? 'n/a' : (+v).toLocaleString(undefined, { maximumFractionDigits: 1 });
export const shortDate = (s: string) =>
  new Date(s + 'T00:00:00Z').toLocaleDateString(undefined, { day: 'numeric', month: 'short', timeZone: 'UTC' });
