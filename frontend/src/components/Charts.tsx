// Small charts for the forecast panel. Shared rules (see the dataviz guide): thin marks growing from one
// hairline baseline, 4px rounded data ends, labels sparingly and in text colours (never the data colour),
// a tooltip on every mark (hover or keyboard focus), and values repeated in the aria-label.

import type { ReactNode } from 'react';
import { num, pct, shortDate } from '../format';

const FIRE = '#ea580c';
// The fire heatmap colours by chance band (same ramp as riskHeat.ts).
export const RISK_BANDS = [
  { from: .10, to: .25, color: '#fde68a', label: '10–25%' },
  { from: .25, to: .40, color: '#fbbf24', label: '25–40%' },
  { from: .40, to: .60, color: '#f97316', label: '40–60%' },
  { from: .60, to: .85, color: '#dc2626', label: '60–85%' },
  { from: .85, to: 1.01, color: '#7f1d1d', label: '85%+' },
];

const Tip = ({ children }: { children: ReactNode }) => <span className="tip" role="tooltip">{children}</span>;

// ---------------- Past + forecast timeline ----------------
// Observed burning cells per day (solid) followed by the forecast days (striped texture, same hue),
// separated by a "today" divider. The selected forecast day is highlighted; clicking a forecast column
// selects that day. Labels: the last observed value and the selected forecast value only.
export function Timeline({ observed, start, forecast, day, onPick }: {
  observed: number[]; start: string;
  forecast: { date: string; expected_cells: number; expected_new_elsewhere: number | null }[];
  day: number; onPick: (i: number) => void;
}) {
  const fc = forecast.map(f => ({ ...f, total: f.expected_cells + (f.expected_new_elsewhere ?? 0) }));
  const max = Math.max(1, ...observed, ...fc.map(f => f.total));
  const date = (i: number) => new Date(Date.parse(start + 'T00:00:00Z') + i * 864e5).toISOString().slice(0, 10);
  const label = 'Burning cells per day. Observed: ' + observed.map((v, i) => `${shortDate(date(i))} ${num(v)}`).join(', ') +
    '. Forecast: ' + fc.map(f => `${shortDate(f.date)} about ${num(Math.round(f.total))}`).join(', ') + '.';
  const h = (v: number) => `${Math.max(2, (v / max) * 100)}%`;
  return (
    <figure className="chart">
      <div className="legend-row" aria-hidden="true">
        <span><i className="sw obs" />Observed</span>
        <span><i className="sw fc" />Forecast</span>
      </div>
      <div className="cols timeline" role="img" aria-label={label}>
        {observed.map((v, i) => (
          <div className="col" key={'o' + i} tabIndex={0}>
            <div className="colbar-wrap">
              {i === observed.length - 1 && <span className="colcap" style={{ bottom: h(v) }}>{num(v)}</span>}
              <div className="colbar" style={{ height: h(v) }} />
            </div>
            <span className="coldate">{new Date(date(i) + 'T00:00:00Z').getUTCDate()}</span>
            <Tip><b>{num(v)}</b> burning cells<br />{shortDate(date(i))} · observed</Tip>
          </div>
        ))}
        <div className="today" aria-hidden="true"><span>today</span></div>
        {fc.map((f, i) => (
          <button type="button" className={`col fcol${i === day ? ' sel' : ''}`} key={'f' + i} onClick={() => onPick(i)}
                  aria-pressed={i === day}>
            <div className="colbar-wrap">
              {i === day && <span className="colcap" style={{ bottom: h(f.total) }}>~{num(Math.round(f.total))}</span>}
              <div className="colbar hatch" style={{ height: h(f.total) }} />
            </div>
            <span className="coldate">+{i + 1}d</span>
            <Tip><b>~{num(Math.round(f.total))}</b> burning cells expected<br />
              {shortDate(f.date)}: ~{num(Math.round(f.expected_cells))} near current fires
              {f.expected_new_elsewhere != null && <> + ~{num(Math.round(f.expected_new_elsewhere))} elsewhere</>}</Tip>
          </button>
        ))}
      </div>
      <figcaption className="dim">
        Days of {new Date(start + 'T00:00:00Z').toLocaleDateString(undefined, { month: 'long', timeZone: 'UTC' })} · click a forecast day to show it on the map
      </figcaption>
    </figure>
  );
}

// ---------------- How the forecast spots split by chance ----------------
// A histogram of forecast grid spots per chance band, coloured exactly like the risk zones on the map, so it
// doubles as their key. Each bar sits on a faint full-height track so even the darkest red stays visible.
export function RiskHistogram({ probs }: { probs: number[] }) {
  const counts = RISK_BANDS.map(b => probs.filter(p => p >= b.from && p < b.to).length);
  const max = Math.max(1, ...counts);
  const total = counts.reduce((a, b) => a + b, 0);
  const label = 'Forecast spots by chance of fire: ' + RISK_BANDS.map((b, i) => `${b.label} ${counts[i]}`).join(', ');
  const top = counts.indexOf(max);
  return (
    <figure className="chart">
      <div className="cols hist" role="img" aria-label={label}>
        {RISK_BANDS.map((b, i) => (
          <div className="col" key={b.label} tabIndex={0}>
            <div className="colbar-wrap track">
              {i === top && counts[i] > 0 && <span className="colcap" style={{ bottom: `${(counts[i] / max) * 100}%` }}>{num(counts[i])}</span>}
              <div className="colbar" style={{ height: counts[i] ? `${Math.max(3, (counts[i] / max) * 100)}%` : 0, background: b.color }} />
            </div>
            <span className="coldate">{b.label}</span>
            <Tip><b>{num(counts[i])}</b> spot{counts[i] === 1 ? '' : 's'} with a {b.label} chance
              {total ? <> ({pct(counts[i] / total)} of the map)</> : null}</Tip>
          </div>
        ))}
      </div>
      <figcaption className="dim">Forecast spots by chance of fire · colours match the map zones</figcaption>
    </figure>
  );
}

// ---------------- Spread compass ----------------
export function Compass({ bearing, compass }: { bearing: number | null; compass: string | null }) {
  const has = bearing != null;
  return (
    <svg className="compass" viewBox="0 0 64 64" width="64" height="64" role="img"
         aria-label={has ? `Fire spreading towards the ${compass} (${bearing}°)` : 'No consistent spread direction'}>
      <circle cx="32" cy="32" r="27" className="c-ring" />
      {[0, 90, 180, 270].map(a => (
        <line key={a} x1="32" y1="7" x2="32" y2="11" className="c-tick" transform={`rotate(${a} 32 32)`} />
      ))}
      <text x="32" y="20" className="c-n">N</text>
      {has
        ? <g transform={`rotate(${bearing} 32 32)`}>
            <path d="M32 12 L37 30 L32 27 L27 30 Z" fill={FIRE} />
            <line x1="32" y1="27" x2="32" y2="48" className="c-tail" />
          </g>
        : <circle cx="32" cy="32" r="3.5" className="c-dot" />}
    </svg>
  );
}

// ---------------- Small probability bar (hotspot rows) ----------------
export const MiniBar = ({ p }: { p: number }) => (
  <span className="minibar" aria-hidden="true"><span style={{ width: `${Math.min(100, p * 100)}%` }} /></span>
);

// ---------------- Ranking gauge (AUC: 0.5 = guessing → 1 = perfect) ----------------
// A semicircle meter: the fire-orange fill shows the score on a lighter track of the same arc.
export function RankingGauge({ auc, label }: { auc: number; label: string }) {
  const k = Math.max(0, Math.min(1, (auc - .5) / .5));
  const r = 30, len = Math.PI * r;
  return (
    <svg className="gauge" viewBox="0 0 80 48" width="104" height="62" role="img"
         aria-label={`Ranking score ${auc}, ${label}. 0.5 is guessing, 1 is perfect.`}>
      <path d="M10 42 A30 30 0 0 1 70 42" className="g-track" />
      <path d="M10 42 A30 30 0 0 1 70 42" className="g-fill" strokeDasharray={`${k * len} ${len}`} />
      <text x="40" y="37" className="g-val">{auc.toFixed(2)}</text>
      <text x="10" y="47.5" className="g-end">0.5</text>
      <text x="70" y="47.5" className="g-end">1</text>
    </svg>
  );
}
