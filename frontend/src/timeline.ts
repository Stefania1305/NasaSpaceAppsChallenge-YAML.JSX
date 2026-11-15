import type { FC } from './geo';
import type { Forecast } from './store';
import { heatExtent, type Extent, type HeatBox } from './riskHeat';

// The fire timeline: one frame per day, first the observed days (where fire actually burned, rebuilt from the
// loaded detections on the forecast's own grid), then the forecast days (+1, +2, +3). The map plays the frames
// like a short looping video; the bar under the legend shows them with checkpoints for notable moments, found
// here by following fire clusters (groups of touching burning cells) from one day to the next.

export type FrameMeta = { date: string; kind: 'observed' | 'forecast'; index: number; label: string; cells: number };
export type EventKind = 'start' | 'merge' | 'peak' | 'jump' | 'forecast' | 'risk' | 'merge-risk';
export type TLEvent = { frame: number; kind: EventKind; title: string; text: string; at: [number, number] | null; weight: number };
export type TimelineData = { frames: FrameMeta[]; boxes: HeatBox[][]; extent: Extent; events: TLEvent[] };

const short = (iso: string) => new Date(iso + 'T00:00:00Z').toLocaleDateString('en-GB', { day: 'numeric', month: 'short', timeZone: 'UTC' });
const addDays = (iso: string, n: number) => { const d = new Date(iso + 'T00:00:00Z'); d.setUTCDate(d.getUTCDate() + n); return d.toISOString().slice(0, 10); };
const key = (i: number, j: number) => `${i},${j}`;
const num = (n: number) => n.toLocaleString('en-GB');

type Cluster = { cells: string[]; at: [number, number] };

// Groups of touching cells (8 neighbours), largest first, each with its centre (lon, lat).
function clusters(cells: Set<string>, size: number): Cluster[] {
  const seen = new Set<string>(), out: Cluster[] = [];
  for (const start of cells) {
    if (seen.has(start)) continue;
    const stack = [start], group: string[] = [];
    seen.add(start);
    while (stack.length) {
      const c = stack.pop()!; group.push(c);
      const [i, j] = c.split(',').map(Number);
      for (let di = -1; di <= 1; di++) for (let dj = -1; dj <= 1; dj++) {
        const k = key(i + di, j + dj);
        if (cells.has(k) && !seen.has(k)) { seen.add(k); stack.push(k); }
      }
    }
    let x = 0, y = 0;
    for (const c of group) { const [i, j] = c.split(',').map(Number); x += i; y += j; }
    out.push({ cells: group, at: [(x / group.length + .5) * size, (y / group.length + .5) * size] });
  }
  return out.sort((a, b) => b.cells.length - a.cells.length);
}
// Which clusters of `prev` a cluster touches (within `reach` cells).
function touching(c: Cluster, prevIndex: Map<string, number>, reach: number) {
  const hit = new Set<number>();
  for (const cell of c.cells) {
    const [i, j] = cell.split(',').map(Number);
    for (let di = -reach; di <= reach; di++) for (let dj = -reach; dj <= reach; dj++) {
      const p = prevIndex.get(key(i + di, j + dj));
      if (p != null) hit.add(p);
    }
  }
  return hit;
}
const indexOf = (cs: Cluster[]) => { const m = new Map<string, number>(); cs.forEach((c, n) => c.cells.forEach(k => m.set(k, n))); return m; };

export function buildTimeline(d: Forecast, fires: FC, zoneArea: number, inside: ((lon: number, lat: number) => boolean) | null): TimelineData | null {
  // Grid cell size: the forecast's own (or, with no forecast cells, the same rule the server uses).
  const size: number = d.cells?.length ? +(d.cells[0].e - d.cells[0].w).toFixed(6) : zoneArea <= 25 ? .05 : zoneArea <= 400 ? .1 : .25;
  const [first, last] = d.range as [string, string];
  const nObs = Math.round((Date.parse(last) - Date.parse(first)) / 864e5) + 1;
  const obsDates = Array.from({ length: nObs }, (_, i) => addDays(first, i));

  // Observed days: burning cells and how many detections each had (more detections = burned harder).
  const perDay = obsDates.map(() => new Map<string, number>());
  const dayOf = new Map(obsDates.map((dt, i) => [dt, i]));
  for (const f of fires.features) {
    const t = dayOf.get(f.properties.date);
    if (t == null) continue;
    const [lon, lat] = f.geometry.coordinates;
    if (inside && !inside(lon, lat)) continue;
    const k = key(Math.floor(lon / size), Math.floor(lat / size));
    perDay[t].set(k, (perDay[t].get(k) ?? 0) + 1);
  }
  const box = (k: string, p: number): HeatBox => { const [i, j] = k.split(',').map(Number); return { w: i * size, s: j * size, e: (i + 1) * size, n: (j + 1) * size, p }; };
  // A lone detection glows yellow-orange; cells with many detections burn deeper red.
  const boxes: HeatBox[][] = perDay.map(m => [...m].map(([k, count]) => box(k, Math.min(.92, .35 + .09 * count))));
  const frames: FrameMeta[] = obsDates.map((dt, i) => ({ date: dt, kind: 'observed', index: i, label: `${short(dt)} · observed fire`, cells: perDay[i].size }));

  // Forecast days: the chance of fire in each cell.
  const fc = d.forecast as { date: string; expected_cells: number }[];
  fc.forEach((f, k) => {
    boxes.push((d.cells ?? []).filter((c: { p: number[] }) => (c.p[k] ?? 0) >= .1)
      .map((c: { w: number; s: number; e: number; n: number; p: number[] }) => ({ w: c.w, s: c.s, e: c.e, n: c.n, p: c.p[k] })));
    frames.push({ date: f.date, kind: 'forecast', index: k, label: `${short(f.date)} · forecast +${k + 1} day${k ? 's' : ''}`, cells: Math.round(f.expected_cells) });
  });
  const extent = heatExtent(boxes.flat());
  if (!extent) return null;

  // ---- Checkpoints ----
  const events: TLEvent[] = [];
  const cl = perDay.map(m => clusters(new Set(m.keys()), size));
  const counts = perDay.map(m => m.size);
  for (let t = 1; t < nObs; t++) {
    const prevIdx = indexOf(cl[t - 1]);
    // New fires: clusters at least two cells away from any fire the day before.
    const fresh = cl[t].filter(c => touching(c, prevIdx, 2).size === 0);
    if (fresh.length >= 2 || (fresh.length === 1 && fresh[0].cells.length >= 3))
      events.push({ frame: t, kind: 'start', title: `${fresh.length} new fire${fresh.length === 1 ? '' : 's'} started`, weight: 3 + fresh.length / 20,
        at: fresh[0].at, text: `Fire appeared away from the previous day's fires; the largest new one covers ${num(fresh[0].cells.length)} cell${fresh[0].cells.length === 1 ? '' : 's'}.` });
    // Merges: a cluster touching two or more of the previous day's clusters.
    // (only fronts of a few cells or more count, so two touching detections aren't called a merge)
    const merged = cl[t].map(c => ({ c, n: touching(c, prevIdx, 1).size })).filter(x => x.n >= 2 && x.c.cells.length >= 4)
      .sort((a, b) => b.c.cells.length - a.c.cells.length);
    if (merged.length)
      events.push({ frame: t, kind: 'merge', title: 'Fires merged', weight: 6 + merged[0].c.cells.length / 10,
        at: merged[0].c.at, text: `${merged[0].n} separate fires joined into one front of ${num(merged[0].c.cells.length)} cells` +
          (merged.length > 1 ? ` (and ${merged.length - 1} smaller merge${merged.length > 2 ? 's' : ''} that day).` : '.') });
  }
  // Peak and biggest jump of the observed days.
  const peak = counts.indexOf(Math.max(...counts));
  if (nObs > 1 && counts[peak] > 0)
    events.push({ frame: peak, kind: 'peak', title: 'Peak activity', weight: 5, at: cl[peak][0]?.at ?? null,
      text: `${num(counts[peak])} cells burning, the most of the observed days.` });
  let jump = -1, rise = .25;
  for (let t = 1; t < nObs; t++) if (counts[t - 1] >= 5 && (counts[t] - counts[t - 1]) / counts[t - 1] > rise) { rise = (counts[t] - counts[t - 1]) / counts[t - 1]; jump = t; }
  if (jump > 0 && jump !== peak)
    events.push({ frame: jump, kind: 'jump', title: `Activity jumped +${Math.round(rise * 100)}%`, weight: 4, at: cl[jump][0]?.at ?? null,
      text: `From ${num(counts[jump - 1])} to ${num(counts[jump])} burning cells in a day.` });
  // Forecast: where it begins, its busiest day, and fronts likely to merge.
  if (fc.length) {
    events.push({ frame: nObs, kind: 'forecast', title: 'Forecast begins', weight: 7, at: null,
      text: `From here on the map shows the predicted chance of fire, not observed fire.` });
    const best = fc.reduce((b, f, k) => f.expected_cells > fc[b].expected_cells ? k : b, 0);
    events.push({ frame: nObs + best, kind: 'risk', title: 'Most predicted fire', weight: 4, at: null,
      text: `About ${num(Math.round(fc[best].expected_cells))} cells expected to burn on ${short(fc[best].date)}.` });
    const lastIdx = indexOf(cl[nObs - 1]);
    for (let k = 0; k < fc.length; k++) {
      const likely = new Set(boxes[nObs + k].filter(b => b.p >= .4).map(b => key(Math.round(b.w / size), Math.round(b.s / size))));
      const fronts = clusters(likely, size).map(c => ({ c, n: touching(c, lastIdx, 1).size })).filter(x => x.n >= 2);
      if (fronts.length) {
        events.push({ frame: nObs + k, kind: 'merge-risk', title: 'Fires likely to merge', weight: 6, at: fronts[0].c.at,
          text: `High risk links ${fronts[0].n} of today's fires into one front of about ${num(fronts[0].c.cells.length)} cells.` });
        break;
      }
    }
  }
  // Keep only the strongest moment of each kind (e.g. the day with the most new fires, the biggest merge), at most
  // 2 per day, in time order, so the bar shows a handful of real highlights instead of a marker on every day.
  const bestOfKind = new Map<EventKind, TLEvent>();
  for (const e of events) if (!bestOfKind.has(e.kind) || e.weight > bestOfKind.get(e.kind)!.weight) bestOfKind.set(e.kind, e);
  const perFrame = new Map<number, number>();
  const kept = [...bestOfKind.values()].sort((a, b) => b.weight - a.weight).filter(e => {
    const n = perFrame.get(e.frame) ?? 0;
    if (n >= 2) return false;
    perFrame.set(e.frame, n + 1); return true;
  }).sort((a, b) => a.frame - b.frame || b.weight - a.weight);
  return { frames, boxes, extent, events: kept };
}
