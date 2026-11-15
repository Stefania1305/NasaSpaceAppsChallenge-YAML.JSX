import type { PipeFire, Pipeline } from './store';
import type { FrameMeta } from './timeline';

// The card shown when a fire is clicked on the map after a pipeline run: everything the workers and the model
// produced for that one fire. Observed: where it is and how big (localisation, epicenter), when it started and how
// long it was seen (expansion), its power (fire power), where it started (epicenters). Predicted: the model's
// values for days +1 to +3 (likely, with the low-high range), when the fire was still burning on the last day.

const COMPASS = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'];
const esc = (v: unknown) => String(v ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]!));
const ok = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);
const n = (v: unknown, d = 1) => (ok(v) ? v.toLocaleString('en-GB', { maximumFractionDigits: d }) : '—');
const pct = (v: unknown) => (ok(v) ? `${Math.round(v * 100)}%` : '—');
const dir = (deg: unknown) => (ok(deg) ? COMPASS[Math.round(deg / 45) % 8] : '');
const when = (t: unknown) => (t ? esc(String(t).replace('T', ' ').replace('Z', ' UTC')) : '—');
const coord = (lat: number, lon: number) =>
  `${Math.abs(lat).toFixed(3)}° ${lat >= 0 ? 'N' : 'S'}, ${Math.abs(lon).toFixed(3)}° ${lon >= 0 ? 'E' : 'W'}`;
const row = (label: string, value: string) => `<div><dt>${label}</dt><dd>${value}</dd></div>`;

// `day`: the timeline's day on screen; the card then opens with that fire's state on that day.
export function fireCardHTML(p: Pipeline, f: PipeFire, day: FrameMeta | null = null): string {
  const out = f.expansion_status === 'extinguished';
  const epis = p.epicenters.filter(e => e.fire_id === f.fire_id);
  const preds = p.predictions.filter(r => r.fire_id === f.fire_id).sort((a, b) => a.day_ahead - b.day_ahead);
  const spread = ok(f.spread_rate_km_h) && ok(f.spread_direction_deg) ? `${n(f.spread_rate_km_h, 2)} km/h ${dir(f.spread_direction_deg)}` : '—';
  const epi = epis[0];

  let html = `<div class="fc-head"><b>Fire ${esc(f.fire_id)}</b>`
    + `<span class="fc-status${out ? ' out' : ''}">${out ? 'Extinguished' : 'Burning'}</span>`
    + (f.peak_class ? `<span class="fc-class ${esc(f.peak_class)}">${esc(f.peak_class)}</span>` : '') + '</div>'
    + `<div class="fc-where">${coord(f.latitude, f.longitude)}</div>`;

  if (day) html += dayBlock(p, f, day, preds);

  html += '<div class="fc-sec">Observed · whole period</div><dl class="fc-grid">'
    + row('Burst', when(f.burst ?? f.first_seen))
    + row('Last seen', when(f.expansion_last_seen ?? f.last_seen))
    + row('Seen for', `${n(f.seen_for_h)} h`)
    + row('Detections', `${n(f.detections, 0)} · ${n(f.observations, 0)} passes`)
    + row('Footprint', `${n(f.final_footprint_km2, 2)} km²`)
    + row('Extent (hull)', `${n(f.hull_area_km2, 2)} km²`)
    + row('Peak FRP', `${n(f.peak_frp_mw)} MW`)
    + row('Biomass burned', ok(f.biomass_burned_t) ? `${n(f.biomass_burned_t, 0)} t` : '—')
    + row('Spread', spread)
    + row('Merges · splits', `${n(f.merges ?? 0, 0)} · ${n(f.splits ?? 0, 0)}`)
    + '</dl>';
  if (epi)
    html += `<p class="fc-note">${epis.length > 1 ? `${epis.length} epicenters; the first` : 'Epicenter'} at ${coord(epi.latitude, epi.longitude)}`
      + ` (± ${n(epi.uncertainty_km, 2)} km), first seen by ${esc(epi.first_satellite ?? '—')}.</p>`;

  const cur = (r: PipeFire) => (day?.kind === 'forecast' && r.day_ahead === day.index + 1 ? ' class="cur"' : '');
  if (preds.length) {
    html += `<div class="fc-sec">Predicted · from the model</div>`
      + '<table class="fc-pred"><thead><tr><th></th>' + preds.map(r => `<th${cur(r)}>+${r.day_ahead} day${r.day_ahead > 1 ? 's' : ''}</th>`).join('') + '</tr></thead><tbody>'
      + '<tr><td>Footprint, km²</td>' + preds.map(r => `<td${cur(r)}><b>${n(r.footprint_likely_km2, 2)}</b><small>${n(r.footprint_low_km2, 2)}–${n(r.footprint_high_km2, 2)}</small></td>`).join('') + '</tr>'
      + '<tr><td>Moves</td>' + preds.map(r => `<td${cur(r)}><b>${ok(r.move_km) && r.move_km >= 0.05 ? `${n(r.move_km, 2)} km ${dir(r.direction_deg)}` : 'stays'}</b></td>`).join('') + '</tr>'
      + '<tr><td>FRP, MW</td>' + preds.map(r => `<td${cur(r)}><b>${n(r.frp_likely, 0)}</b><small>${n(r.frp_low, 0)}–${n(r.frp_high, 0)}</small></td>`).join('') + '</tr>'
      + '<tr><td>Chance out</td>' + preds.map(r => `<td${cur(r)}><b>${pct(r.p_out)}</b></td>`).join('') + '</tr>'
      + '<tr><td>Chance of merging</td>' + preds.map(r => `<td${cur(r)}><b>${pct(r.p_merge)}</b></td>`).join('') + '</tr>'
      + '</tbody></table>';
  } else {
    html += `<p class="fc-note">${out ? 'Extinguished before the last day of the data' : 'Not seen on the last day of the data'}, so the model has no prediction for it.</p>`;
  }
  return html;
}

// The fire on the timeline's day: observed days from the expansion worker's pass-by-pass history, predicted days
// from the model's row for that day.
function dayBlock(p: Pipeline, f: PipeFire, day: FrameMeta, preds: PipeFire[]): string {
  const label = esc(day.label.replace(' · ', ' · '));
  if (day.kind === 'forecast') {
    const r = preds.find(x => x.day_ahead === day.index + 1);
    if (!r) return `<div class="fc-now pred"><span class="fc-now-label">${label}</span>`
      + '<p class="fc-now-note">Not burning on the last day of the data, so the model has no prediction for it.</p></div>';
    const out = (r.p_out ?? 0) >= .5;
    return `<div class="fc-now pred"><span class="fc-now-label">${label}${out ? '<i class="fc-pill out">Likely out</i>' : ''}`
      + `${(r.p_merge ?? 0) >= .2 ? '<i class="fc-pill merge">May merge</i>' : ''}</span><dl class="fc-now-grid">`
      + row('Footprint', `${n(r.footprint_likely_km2, 2)} km²<small>${n(r.footprint_low_km2, 2)}–${n(r.footprint_high_km2, 2)}</small>`)
      + row('Moves', ok(r.move_km) && r.move_km >= 0.05 ? `${n(r.move_km, 2)} km ${dir(r.direction_deg)}` : 'stays put')
      + row('FRP', `${n(r.frp_likely, 0)} MW<small>${n(r.frp_low, 0)}–${n(r.frp_high, 0)}</small>`)
      + row('Out · merge', `${pct(r.p_out)} · ${pct(r.p_merge)}`)
      + '</dl></div>';
  }
  // a day after the window with real data: the real detections in and around this fire's extent
  if (p.actual?.days?.includes(day.date)) return `<div class="fc-now"><span class="fc-now-label">${label}</span>${realDay(p, f, day.date)}</div>`;
  // observed day
  const hist = (p.history ?? []).filter(h => h.fire_id === f.fire_id).sort((a, b) => String(a.time).localeCompare(String(b.time)));
  const today = hist.filter(h => String(h.time).slice(0, 10) === day.date);
  const before = hist.filter(h => String(h.time).slice(0, 10) <= day.date);
  const burst = String(f.burst ?? f.first_seen ?? '').slice(0, 10), last = String(f.expansion_last_seen ?? f.last_seen ?? '').slice(0, 10);
  const shortD = (d: string) => new Date(d + 'T00:00:00Z').toLocaleDateString('en-GB', { day: 'numeric', month: 'short', timeZone: 'UTC' });
  let body: string;
  if (!before.length) body = `<p class="fc-now-note">Not burning yet: this fire bursts on ${shortD(burst)}.</p>`;
  else {
    const fp = Math.max(...before.map(h => h.cumulative_footprint_km2 ?? 0));
    if (!today.length)
      body = day.date > last
        ? `<p class="fc-now-note">${f.expansion_status === 'extinguished' ? 'Out' : 'Not seen'} since ${shortD(last)}; footprint ${n(fp, 2)} km².</p>`
        : `<p class="fc-now-note">No satellite pass saw it on this day; footprint so far ${n(fp, 2)} km².</p>`;
    else {
      const evs = [...new Set(today.map(h => h.events).filter(Boolean).flatMap(e => String(e).split(/[;,]\s*/)))];
      body = '<dl class="fc-now-grid">'
        + row('Footprint so far', `${n(fp, 2)} km²`)
        + row('Burning area', `${n(Math.max(...today.map(h => h.burning_area_km2 ?? 0)), 2)} km²`)
        + row('FRP (strongest pass)', `${n(Math.max(...today.map(h => h.frp_sum_mw ?? 0)), 1)} MW`)
        + row('Passes · pixels', `${today.length} · ${today.reduce((a, h) => a + (h.pixels ?? 0), 0)}`)
        + '</dl>' + (evs.length ? `<p class="fc-now-note">Events: ${esc(evs.join(', '))}.</p>` : '');
    }
  }
  return `<div class="fc-now"><span class="fc-now-label">${label}</span>${body}</div>`;
}

// Real detections of a fetched day (after the window) within 2 km of this fire's extent.
function realDay(p: Pipeline, f: PipeFire, date: string): string {
  const a = p.actual!, c = a.columns, ix = (k: string) => c.indexOf(k);
  const la = ix('latitude'), lo = ix('longitude'), di = ix('acq_date'), fr = ix('frp_mw') >= 0 ? ix('frp_mw') : ix('frp');
  const pad = 2 / 110.574, padx = 2 / (111.32 * Math.cos(f.latitude * Math.PI / 180));
  const w = (f.west ?? f.longitude) - padx, e = (f.east ?? f.longitude) + padx, s = (f.south ?? f.latitude) - pad, n2 = (f.north ?? f.latitude) + pad;
  const hits = a.rows.filter(r => r[di] === date && +r[la] >= s && +r[la] <= n2 && +r[lo] >= w && +r[lo] <= e);
  if (!hits.length) return '<p class="fc-now-note">Real data: no detection in or near this fire on this day (out, or not seen by any satellite).</p>';
  const frp = hits.map(r => parseFloat(r[fr]) || 0);
  return '<dl class="fc-now-grid">'
    + row('Real detections', n(hits.length, 0))
    + row('Strongest FRP', `${n(Math.max(...frp), 1)} MW`)
    + row('Total FRP', `${n(frp.reduce((x, y) => x + y, 0), 1)} MW`)
    + row('Status', 'Still burning')
    + '</dl><p class="fc-now-note">From the real satellite data of this day (fetched, not predicted).</p>';
}
