import type { FC } from './geo';
import type { Forecast, Pipeline } from './store';
import { heatExtent, type Extent, type HeatBox } from './riskHeat';

// The fire timeline: one frame per day, first the observed days (where fire actually burned, rebuilt from the
// loaded detections on the forecast's own grid), then the forecast days (+1, +2, +3). The map plays the frames
// like a short looping video; the bar under the legend shows them with checkpoints for notable moments, found
// here by following fire clusters (groups of touching burning cells) from one day to the next.

// `unit`: what `cells` counts on this day (shown beside the number); burning cells when not given.
export type FrameMeta = { date: string; kind: 'observed' | 'forecast'; index: number; label: string; cells: number; unit?: string };
export type EventKind = 'start' | 'merge' | 'peak' | 'jump' | 'forecast' | 'risk' | 'merge-risk' | 'out';
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

// ---------------------------------------------------------------------------------------------------------------
// The prediction timeline (pipeline results): the observed days (the correlated detections, day by day), then the
// model's three predicted days (its spread map: the chance that each 1 km cell burns, gathered onto the same grid).
// Checkpoints mark the most important changes, from the workers' and the model's real outputs:
//   observed   new ignitions and merges seen by the expansion worker, the day of peak activity
//   predicted  where the prediction begins, the fire that grows most, the one that moves farthest, a likely
//              merge, and the day most fires are expected to have gone out
// ---------------------------------------------------------------------------------------------------------------
const COMPASS = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'];
const km = (v: number, d = 2) => v.toLocaleString('en-GB', { maximumFractionDigits: d });

export function buildPipelineTimeline(p: Pipeline, fires: FC, zoneArea: number, first: string,
                                      inside: ((lon: number, lat: number) => boolean) | null): TimelineData | null {
  const size = zoneArea <= 25 ? .05 : zoneArea <= 400 ? .1 : .25;
  // observed days run to the last day with real data: the window, plus the days after it already fetched
  const realDays = p.actual?.days ?? [];
  const last: string = realDays.length ? realDays[realDays.length - 1] : p.prediction?.from_day ?? first;
  const nObs = Math.max(1, Math.round((Date.parse(last) - Date.parse(first)) / 864e5) + 1);
  const obsDates = Array.from({ length: nObs }, (_, i) => addDays(first, i));

  // Observed days: burning grid cells and how many detections each had.
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
  const box = (k: string, v: number): HeatBox => { const [i, j] = k.split(',').map(Number); return { w: i * size, s: j * size, e: (i + 1) * size, n: (j + 1) * size, p: v }; };
  const boxes: HeatBox[][] = perDay.map(m => [...m].map(([k, count]) => box(k, Math.min(.92, .35 + .09 * count))));
  const frames: FrameMeta[] = obsDates.map((dt, i) => ({ date: dt, kind: 'observed', index: i,
    label: `${short(dt)} · ${realDays.includes(dt) ? 'real data (after the window)' : 'observed fire'}`, cells: perDay[i].size, unit: 'cells burning' }));

  // Predicted days: the model's spread map on the same grid (each grid cell takes its highest burn chance), and
  // how many fires are still likely to be burning that day.
  const rows = p.predictions ?? [];
  const ks = [...new Set(rows.map(r => r.day_ahead as number))].sort();     // predicted days (no real data)
  const frameOf = (k: number) => nObs + Math.max(0, ks.indexOf(k));
  // The active fires stay on the predicted days: the cells burning on the last observed day, each matched to the
  // nearest fire the model predicts, keep their heat scaled by the chance that fire is still burning (1 - p_out).
  const lastCells = [...perDay[nObs - 1]].map(([k, count]) => {
    const [i, j] = k.split(',').map(Number);
    return { k, v: Math.min(.92, .35 + .09 * count), lon: (i + .5) * size, lat: (j + .5) * size };
  });
  for (const k of ks) {
    const cells = new Map<string, number>();
    const active = rows.filter(r => r.day_ahead === k);
    if (active.length) for (const c of lastCells) {
      let best: (typeof active)[number] | null = null, bd = Infinity;
      for (const r of active) {
        const d = (r.lon0 - c.lon) ** 2 + (r.lat0 - c.lat) ** 2;
        if (d < bd) { bd = d; best = r; }
      }
      if (!best || bd > (size * 1.5) ** 2) continue;      // no predicted fire there: that cell's fire is over
      cells.set(c.k, Math.max(cells.get(c.k) ?? 0, c.v * (1 - (best.p_out ?? 0))));
    }
    for (const c of p.spread ?? []) {
      if (c.day_ahead !== k || (inside && !inside(c.lon, c.lat))) continue;
      const g = key(Math.floor(c.lon / size), Math.floor(c.lat / size));
      // a burn chance of a few % per 1 km cell still means fire in a grid cell: lift it onto the heat scale
      cells.set(g, Math.max(cells.get(g) ?? 0, Math.min(.9, .2 + c.p * 1.6)));
    }
    boxes.push([...cells].filter(([, v]) => v >= .05).map(([g, v]) => box(g, v)));
    const dayRows = rows.filter(r => r.day_ahead === k);
    const date: string = dayRows[0]?.date ?? addDays(p.prediction?.from_day ?? last, k);
    frames.push({ date, kind: 'forecast', index: k - 1, label: `${short(date)} · predicted +${k} day${k > 1 ? 's' : ''}`,
      cells: dayRows.filter(r => (r.p_out ?? 0) < .5).length, unit: 'fires still burning' });
  }
  const extent = heatExtent(boxes.flat());
  if (!extent) return null;

  const events: TLEvent[] = [];
  // Observed: the day with the most new ignitions (after the first day, when every fire "ignites"), merges.
  const evs = p.events ?? [];
  const ign = new Map<number, typeof evs>();
  for (const e of evs) {
    const t = dayOf.get(String(e.time).slice(0, 10));
    if (e.event !== 'ignition' || t == null || t === 0) continue;
    ign.set(t, [...(ign.get(t) ?? []), e]);
  }
  const busiest = [...ign].sort((a, b) => b[1].length - a[1].length)[0];
  if (busiest)
    events.push({ frame: busiest[0], kind: 'start', title: `${num(busiest[1].length)} new fire${busiest[1].length === 1 ? '' : 's'} ignited`, weight: 4,
      at: [busiest[1][0].longitude, busiest[1][0].latitude], text: 'The fire expansion worker saw them ignite on this day.' });
  const merges = evs.filter(e => e.event === 'merge' && dayOf.has(String(e.time).slice(0, 10)));
  if (merges.length)
    events.push({ frame: dayOf.get(String(merges[0].time).slice(0, 10))!, kind: 'merge', title: 'Fires merged', weight: 6,
      at: [merges[0].longitude, merges[0].latitude],
      text: `Fire ${merges[0].fire_id}: separate burning areas joined${merges.length > 1 ? ` (${merges.length} merges in all)` : ''}.` });
  const counts = perDay.map(m => m.size), peak = counts.indexOf(Math.max(...counts));
  if (nObs > 1 && counts[peak] > 0)
    events.push({ frame: peak, kind: 'peak', title: 'Peak activity', weight: 5, at: null,
      text: `${num(counts[peak])} cells burning, the most of the observed days.` });

  // Predicted (the model's output for the fires burning on the last observed day).
  if (rows.length) {
    events.push({ frame: nObs, kind: 'forecast', title: 'Prediction begins', weight: 9, at: null,
      text: `The model predicts the ${num(rows.filter(r => r.day_ahead === ks[0]).length)} fires still burning on ${short(p.prediction?.from_day ?? last)}, ${p.prediction?.weather === 'provided' ? 'with the weather forecast' : 'without weather'}; days with real data are shown as observed.` });
    const byFire = new Map<number, typeof rows>();
    for (const r of rows) byFire.set(r.fire_id, [...(byFire.get(r.fire_id) ?? []), r].sort((a, b) => a.day_ahead - b.day_ahead));
    const kLast = ks[ks.length - 1], day3 = rows.filter(r => r.day_ahead === kLast);
    // the fire that grows most, placed on the day it grows fastest
    const grow = [...day3].sort((a, b) => (b.growth_likely ?? 0) - (a.growth_likely ?? 0))[0];
    if (grow && grow.growth_likely > 0.01) {
      const fr = byFire.get(grow.fire_id)!, g = fr.map(r => r.growth_likely ?? 0), inc = g.map((v, i) => v - (i ? g[i - 1] : 0));
      events.push({ frame: frameOf(fr[inc.indexOf(Math.max(...inc))].day_ahead), kind: 'risk', title: `Fire ${grow.fire_id} grows the most`, weight: 7,
        at: [grow.centre_lon, grow.centre_lat],
        text: `+${km(grow.growth_likely)} km² by day +${kLast} (${km(grow.growth_low)}–${km(grow.growth_high)}), to ${km(grow.footprint_likely_km2)} km².` });
    }
    // the fire that moves farthest by day +3
    const mover = [...day3].sort((a, b) => (b.move_km ?? 0) - (a.move_km ?? 0))[0];
    if (mover && mover.move_km >= 0.1)
      events.push({ frame: frameOf(kLast), kind: 'jump', title: `Fire ${mover.fire_id} moves farthest`, weight: 5,
        at: [mover.centre_lon, mover.centre_lat],
        text: `Its centre shifts ${km(mover.move_km)} km ${COMPASS[Math.round((mover.direction_deg ?? 0) / 45) % 8]} by day +${kLast}.` });
    // a likely merge with a neighbouring fire
    const merger = [...rows].sort((a, b) => (b.p_merge ?? 0) - (a.p_merge ?? 0))[0];
    if (merger && merger.p_merge >= 0.1) {
      const firstHigh = byFire.get(merger.fire_id)!.find(r => r.p_merge >= merger.p_merge * 0.8) ?? merger;
      events.push({ frame: frameOf(firstHigh.day_ahead), kind: 'merge-risk', title: `Fire ${merger.fire_id} may merge`, weight: 8,
        at: [merger.centre_lon, merger.centre_lat],
        text: `${Math.round(merger.p_merge * 100)}% chance it joins a neighbouring fire by day +${merger.day_ahead}.` });
    }
    // the day most fires are likely out
    const active = rows.filter(r => r.day_ahead === ks[0]).length;
    const outs = ks.map(k => rows.filter(r => r.day_ahead === k && r.p_out >= .5).length);
    const half = outs.findIndex(n => n >= active / 2), kk = half >= 0 ? half : outs.indexOf(Math.max(...outs));
    if (outs[kk] > 0)
      events.push({ frame: frameOf(ks[kk]), kind: 'out', title: `${num(outs[kk])} of ${num(active)} fires likely out`, weight: 6, at: null,
        text: `By day +${ks[kk]}, ${num(outs[kk])} fires have at least a 50% chance of having burned out.` });
  }
  const perFrame = new Map<number, number>();
  const kept = [...events].sort((a, b) => b.weight - a.weight).filter(e => {
    const n = perFrame.get(e.frame) ?? 0;
    if (n >= 2) return false;
    perFrame.set(e.frame, n + 1); return true;
  }).sort((a, b) => a.frame - b.frame || b.weight - a.weight);
  return { frames, boxes, extent, events: kept };
}
