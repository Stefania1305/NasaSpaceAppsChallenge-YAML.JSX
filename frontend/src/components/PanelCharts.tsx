import { useId, useState, type ReactNode } from 'react';

// Small SVG charts for the side panel (360 px wide). One accent hue for magnitude, a hatched fill for predicted
// values, muted ink for text, a recessive baseline; every mark has a hover tooltip, and every chart a text fallback
// (aria-label) so nothing is told by colour alone.

const W = 356;
export const fmtN = (v: number, d = 1) => v.toLocaleString('en-GB', { maximumFractionDigits: d });

function Tip({ x, children }: { x: number; children: ReactNode }) {
  // x: the mark's centre in chart pixels; the tooltip stays inside the chart's width
  return <div className="ctip" style={{ left: Math.min(W - 8, Math.max(8, x)) }}>{children}</div>;
}

export type Col = { label: string; sub?: string; value: number; predicted?: boolean; tip: ReactNode; active?: boolean };

// Column chart: one value per day. Predicted columns are hatched; the active one is outlined; clickable when onPick.
export function Columns({ cols, height = 120, onPick, ariaLabel }: { cols: Col[]; height?: number; onPick?: (i: number) => void; ariaLabel: string }) {
  const [hover, setHover] = useState<number | null>(null);
  const pid = useId().replace(/:/g, '');
  const max = Math.max(1e-9, ...cols.map(c => c.value));
  const top = 16, bottom = 34, ph = height - top - bottom;
  const slot = W / cols.length, bw = Math.min(34, slot * 0.62);
  const iMax = cols.findIndex(c => c.value === max);   // the tallest column keeps its value label
  return (
    <div className="chart" onMouseLeave={() => setHover(null)}>
      <svg width={W} height={height} viewBox={`0 0 ${W} ${height}`} role="img" aria-label={ariaLabel}>
        <defs>
          <pattern id={`h${pid}`} width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <rect width="6" height="6" className="c-pred-bg" /><line x1="0" y1="0" x2="0" y2="6" className="c-pred-line" />
          </pattern>
        </defs>
        <line x1="0" x2={W} y1={top + ph + 0.5} y2={top + ph + 0.5} className="c-base" />
        {cols.map((c, i) => {
          const h = Math.max(c.value > 0 ? 3 : 0, (c.value / max) * ph);
          const x = i * slot + (slot - bw) / 2, y = top + ph - h;
          const r = Math.min(4, h / 2);
          return (
            <g key={i} className={`c-col${c.active ? ' active' : ''}${hover === i ? ' hover' : ''}${onPick ? ' pick' : ''}`}
               onMouseEnter={() => setHover(i)} onClick={() => onPick?.(i)}>
              <rect x={i * slot} y={0} width={slot} height={height} fill="transparent" />
              {h > 0 && <path d={`M${x},${top + ph} V${y + r} Q${x},${y} ${x + r},${y} H${x + bw - r} Q${x + bw},${y} ${x + bw},${y + r} V${top + ph} Z`}
                    className={c.predicted ? 'c-pred' : 'c-obs'} fill={c.predicted ? `url(#h${pid})` : undefined} />}
              {(c.active || hover === i || i === iMax) && <text x={i * slot + slot / 2} y={y - 5} className="c-val" textAnchor="middle">{fmtN(c.value)}</text>}
              <text x={i * slot + slot / 2} y={top + ph + 15} className={`c-lab${c.predicted ? ' pred' : ''}`} textAnchor="middle">{c.label}</text>
              {c.sub && <text x={i * slot + slot / 2} y={top + ph + 28} className="c-sub" textAnchor="middle">{c.sub}</text>}
            </g>
          );
        })}
      </svg>
      {hover != null && <Tip x={hover * slot + slot / 2}>{cols[hover].tip}</Tip>}
    </div>
  );
}

export type Pt = { label: string; value: number; forecast?: boolean; active?: boolean; tip: ReactNode };

// Dot-line chart for a measure that doesn't start at zero (temperature, humidity, wind): its own min-max scale,
// the line dashed over forecast days, the active day's dot ringed and labelled, the lowest and highest labelled.
export function DotLine({ pts, color, height = 78, ariaLabel }: { pts: Pt[]; color: string; unit?: string; height?: number; ariaLabel: string }) {
  const [hover, setHover] = useState<number | null>(null);
  const vals = pts.map(p => p.value);
  let lo = Math.min(...vals), hi = Math.max(...vals);
  const pad = Math.max((hi - lo) * 0.15, 1); lo -= pad; hi += pad;
  const top = 16, bottom = 20, ph = height - top - bottom, slot = W / pts.length;
  const X = (i: number) => i * slot + slot / 2, Y = (v: number) => top + ph - ((v - lo) / (hi - lo)) * ph;
  const iMax = vals.indexOf(Math.max(...vals)), iMin = vals.indexOf(Math.min(...vals));
  return (
    <div className="chart" onMouseLeave={() => setHover(null)}>
      <svg width={W} height={height} viewBox={`0 0 ${W} ${height}`} role="img" aria-label={ariaLabel}>
        <line x1="0" x2={W} y1={top + ph + 0.5} y2={top + ph + 0.5} className="c-base" />
        {pts.slice(1).map((p, k) => (
          <line key={k} x1={X(k)} y1={Y(pts[k].value)} x2={X(k + 1)} y2={Y(p.value)} stroke={color} strokeWidth={2}
                strokeLinecap="round" strokeDasharray={p.forecast ? '4 4' : undefined} />
        ))}
        {pts.map((p, i) => (
          <g key={i} onMouseEnter={() => setHover(i)}>
            <rect x={i * slot} y={0} width={slot} height={height} fill="transparent" />
            <circle cx={X(i)} cy={Y(p.value)} r={p.active || hover === i ? 5 : 3.5} fill={p.forecast ? 'var(--bg)' : color} stroke={color} strokeWidth={2} />
            {p.active && <circle cx={X(i)} cy={Y(p.value)} r={9} fill="none" stroke="var(--fg)" strokeWidth={1.25} />}
            {(p.active || i === iMax || i === iMin) && hover !== i &&
              <text x={X(i)} y={Y(p.value) - 10} className="c-val" textAnchor="middle">{fmtN(p.value, 0)}</text>}
            <text x={X(i)} y={height - 4} className={`c-lab${p.active ? ' on' : ''}`} textAnchor="middle">{p.label}</text>
          </g>
        ))}
      </svg>
      {hover != null && <Tip x={X(hover)}>{pts[hover].tip}{pts[hover].forecast ? ' (forecast)' : ''}</Tip>}
    </div>
  );
}

export type Part ={ label: string; value: number; cls: string };

// Split bar: parts of a whole (e.g. fires burning vs out), each labelled with its count and share below.
export function SplitBar({ parts, ariaLabel }: { parts: Part[]; ariaLabel: string }) {
  const [hover, setHover] = useState<number | null>(null);
  const total = parts.reduce((a, p) => a + p.value, 0) || 1;
  let x = 0;
  const segs = parts.map(p => { const w = (p.value / total) * W; const s = { ...p, x, w }; x += w; return s; });
  return (
    <div className="chart" onMouseLeave={() => setHover(null)}>
      <svg width={W} height={14} viewBox={`0 0 ${W} 14`} role="img" aria-label={ariaLabel}>
        {segs.map((s, i) => s.w > 0 && (
          <rect key={i} x={s.x + (i ? 1 : 0)} y={0} width={Math.max(1, s.w - (i ? 1 : 0) - (i < segs.length - 1 ? 1 : 0))} height={14} rx={4}
                className={`${s.cls}${hover === i ? ' hover' : ''}`} onMouseEnter={() => setHover(i)} />
        ))}
      </svg>
      <ul className="c-legend">
        {segs.map((s, i) => (
          <li key={i} onMouseEnter={() => setHover(i)}><i className={s.cls} aria-hidden="true" /><b>{fmtN(s.value, 0)}</b> {s.label}
            <span> · {s.value > 0 && s.value / total < 0.005 ? '<1' : Math.round((s.value / total) * 100)}%</span></li>
        ))}
      </ul>
      {hover != null && <Tip x={segs[hover].x + segs[hover].w / 2}><b>{fmtN(segs[hover].value, 0)}</b> {segs[hover].label}</Tip>}
    </div>
  );
}

export type HBar = { label: string; value: number; color: string; tip: ReactNode };

// Horizontal bars for ordered classes (e.g. fire intensity, with the map legend's colours as the swatches).
export function HBars({ bars, ariaLabel }: { bars: HBar[]; ariaLabel: string }) {
  const [hover, setHover] = useState<number | null>(null);
  const max = Math.max(1, ...bars.map(b => b.value));
  const lw = 92, vw = 40, bwMax = W - lw - vw, rh = 24;
  return (
    <div className="chart" onMouseLeave={() => setHover(null)}>
      <svg width={W} height={bars.length * rh} viewBox={`0 0 ${W} ${bars.length * rh}`} role="img" aria-label={ariaLabel}>
        {bars.map((b, i) => {
          const w = b.value ? Math.max(3, (b.value / max) * bwMax) : 0, y = i * rh;
          return (
            <g key={i} onMouseEnter={() => setHover(i)} className={hover === i ? 'hover' : ''}>
              <rect x={0} y={y} width={W} height={rh} fill="transparent" />
              <circle cx={5} cy={y + rh / 2} r={4} fill={b.color} />
              <text x={15} y={y + rh / 2 + 4} className="c-lab left">{b.label}</text>
              {w > 0 && <rect x={lw} y={y + 6} width={w} height={rh - 12} rx={3} className="c-obs" />}
              <text x={lw + w + 6} y={y + rh / 2 + 4} className="c-val">{fmtN(b.value, 0)}</text>
            </g>
          );
        })}
      </svg>
      {hover != null && <Tip x={W / 2}>{bars[hover].tip}</Tip>}
    </div>
  );
}

// Range bar: a value now, the likely value and its low-high range, on one axis from 0.
export function RangeBar({ now, low, likely, high, unit, ariaLabel }: { now: number; low: number; likely: number; high: number; unit: string; ariaLabel: string }) {
  const [hover, setHover] = useState(false);
  const max = Math.max(now, high, likely) * 1.08 || 1;
  const X = (v: number) => (v / max) * (W - 2) + 1;
  return (
    <div className="chart" onMouseEnter={() => setHover(true)} onMouseLeave={() => setHover(false)}>
      <svg width={W} height={52} viewBox={`0 0 ${W} 52`} role="img" aria-label={ariaLabel}>
        <rect x={1} y={14} width={W - 2} height={10} rx={5} className="c-track" />
        <rect x={X(low)} y={14} width={Math.max(2, X(high) - X(low))} height={10} rx={5} className="c-range" />
        <line x1={X(now)} x2={X(now)} y1={8} y2={30} className="c-now" />
        <circle cx={X(likely)} cy={19} r={6} className="c-likely" />
        <text x={Math.min(W - 2, Math.max(2, X(now)))} y={44} className="c-sub" textAnchor={X(now) < 40 ? 'start' : X(now) > W - 40 ? 'end' : 'middle'}>now {fmtN(now)}</text>
        <text x={W - 1} y={44} className="c-sub" textAnchor="end">{X(now) > W - 90 ? '' : `${fmtN(max / 1.08, 0)} ${unit}`}</text>
      </svg>
      {hover && <Tip x={X(likely)}>Likely <b>{fmtN(likely)} {unit}</b> · range {fmtN(low)}–{fmtN(high)} · now {fmtN(now)}</Tip>}
    </div>
  );
}

export function ChartHead({ title, value, note }: { title: string; value?: ReactNode; note?: ReactNode }) {
  return (
    <div className="chead">
      <span className="ct">{title}</span>{value != null && <span className="cv">{value}</span>}
      {note && <p className="cn">{note}</p>}
    </div>
  );
}
