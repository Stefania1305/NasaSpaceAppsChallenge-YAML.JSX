import { useEffect, useRef, type InputHTMLAttributes } from 'react';
import { useApp, zoneSummary, type Method } from '../store';
import { areaKm2 } from '../geo';

// Degrees with their hemisphere, for the selected-zone readout.
const lat = (v: number, d = 2) => `${Math.abs(v).toFixed(d)}° ${v >= 0 ? 'N' : 'S'}`;
const lon = (v: number, d = 2) => `${Math.abs(v).toFixed(d)}° ${v >= 0 ? 'E' : 'W'}`;

// An input whose value is committed on the browser's native "change" event (Enter, leaving the field or the
// spinner arrows), exactly like the original app; React's onChange would fire on every keystroke instead.
function CommitInput({ onCommit, ...props }: InputHTMLAttributes<HTMLInputElement> & { onCommit: () => void }) {
  const ref = useRef<HTMLInputElement>(null);
  useEffect(() => {
    const el = ref.current!;
    el.addEventListener('change', onCommit);
    return () => el.removeEventListener('change', onCommit);
  }, [onCommit]);
  return <input ref={ref} {...props} />;
}

const PinIcon = ({ w = 11, h = 16 }) => (
  <svg viewBox="0 0 28 40" width={w} height={h} aria-hidden="true">
    <path d="M14 1C7 1 1.5 6.5 1.5 13.5 1.5 23 14 39 14 39S26.5 23 26.5 13.5C26.5 6.5 21 1 14 1Z" fill="currentColor" />
    <circle cx="14" cy="13.5" r="4.5" fill="var(--card)" />
  </svg>
);

// The four ways to set the zone, as a row of icon buttons (a radio group: arrow keys move between them).
const ico = { viewBox: '0 0 24 24', width: 18, height: 18, fill: 'none', stroke: 'currentColor', strokeWidth: 1.6,
              strokeLinecap: 'round' as const, strokeLinejoin: 'round' as const, 'aria-hidden': true };
const METHODS: { id: Method; label: string; icon: JSX.Element }[] = [
  { id: 'draw', label: 'Draw', icon: <svg {...ico}><rect x="4" y="5" width="16" height="14" rx="1.5" strokeDasharray="3 2.4" /><path d="M14 13l5 5M19 14v4h-4" /></svg> },
  { id: 'country', label: 'Country', icon: <svg {...ico}><path d="M5 6.5 9 5l6 2 4-1.5v12L15 19l-6-2-4 1.5z" /><path d="M9 5v12M15 7v12" /></svg> },
  { id: 'pin', label: 'Pin', icon: <svg {...ico}><path d="M12 21s-6.5-6.2-6.5-11a6.5 6.5 0 0 1 13 0c0 4.8-6.5 11-6.5 11z" /><circle cx="12" cy="10" r="2.3" /></svg> },
  { id: 'coords', label: 'Coordinates', icon: <svg {...ico}><circle cx="12" cy="12" r="7" /><path d="M12 2.5v4M12 17.5v4M2.5 12h4M17.5 12h4" /><circle cx="12" cy="12" r="1.2" fill="currentColor" stroke="none" /></svg> },
];

// Part 1 of the side panel: choose the prediction zone (four ways, picked from a row of icon buttons).
// "Pick a country": hovering a country on the map highlights its border; clicking it selects it.
export function ZoneSection() {
  const hidden = useApp(s => s.forecastVisible);
  const method = useApp(s => s.method);
  const drawing = useApp(s => s.drawing);
  const pinMode = useApp(s => s.pinMode);
  const pinError = useApp(s => s.pinError);
  const zone = useApp(s => s.zone);
  const pinAt = useApp(s => s.pinAt);
  const zoneCountry = useApp(s => s.zoneCountry);
  const countries = useApp(s => s.countries);
  const { setMethod, toggleDrawing, togglePinMode, coordsCommitted, applyPin, clearZone } = useApp.getState();
  const commitPin = useRef(() => applyPin(true)).current;
  const summary = zone ? zoneSummary(zone, pinAt, zoneCountry) : null;

  return (
    <section id="zoneSection" hidden={hidden}>
      <h2>Prediction zone</h2>
      <p className="methodlabel" id="zoneMethodLabel">How to set the zone</p>
      <div className="methods" id="zoneMethod" role="radiogroup" aria-labelledby="zoneMethodLabel">
        {METHODS.map(m => (
          <button key={m.id} type="button" role="radio" aria-checked={method === m.id} data-value={m.id}
                  className={`mbtn${method === m.id ? ' on' : ''}`} onClick={() => setMethod(m.id)}
                  onKeyDown={e => {
                    const k = METHODS.findIndex(x => x.id === method);
                    const step = e.key === 'ArrowRight' || e.key === 'ArrowDown' ? 1 : e.key === 'ArrowLeft' || e.key === 'ArrowUp' ? -1 : 0;
                    if (!step) return;
                    e.preventDefault();
                    const next = METHODS[(k + step + METHODS.length) % METHODS.length].id;
                    setMethod(next);
                    (e.currentTarget.parentElement?.querySelector(`[data-value="${next}"]`) as HTMLElement | null)?.focus();
                  }}
                  tabIndex={method === m.id ? 0 : -1}>
            {m.icon}<span>{m.label}</span>
          </button>
        ))}
      </div>

      <div className="method" data-method="draw" hidden={method !== 'draw'}>
        <p className="hint">Click the button, then drag a rectangle on the map.</p>
        <button type="button" id="draw" className={`secondary wide${drawing ? ' on' : ''}`} onClick={toggleDrawing}>Draw on map</button>
      </div>

      <div className="method" data-method="pin" hidden={method !== 'pin'}>
        <p className="hint" id="pinHint">
          {pinError
            ? <span className="err" style={{ color: '#dc2626' }}>Zone area must be between 10 and 5,000,000 km².</span>
            : <>Set the area, click <b>Drop a pin</b>, then click the map: a square zone of that size is centred on the pin. Drag the pin to move it.</>}
        </p>
        <div className="pinrow">
          <label>Zone area (km²)
            <CommitInput id="pinArea" type="number" min={10} max={5000000} step="any" defaultValue={10000} inputMode="decimal"
                         onCommit={commitPin} />
          </label>
          <button type="button" id="pin" className={`secondary${pinMode ? ' on' : ''}`} onClick={togglePinMode}>
            <PinIcon /> Drop a pin</button>
        </div>
      </div>

      <div className="method" data-method="coords" hidden={method !== 'coords'}>
        <p className="hint">Enter the zone's edges in degrees (longitude −180 to 180, latitude −90 to 90).</p>
        <div className="row">
          <label>West (lon)<CommitInput id="w" type="number" step={0.01} min={-180} max={180} onCommit={coordsCommitted} /></label>
          <label>East (lon)<CommitInput id="e" type="number" step={0.01} min={-180} max={180} onCommit={coordsCommitted} /></label>
          <label>South (lat)<CommitInput id="s" type="number" step={0.01} min={-90} max={90} onCommit={coordsCommitted} /></label>
          <label>North (lat)<CommitInput id="n" type="number" step={0.01} min={-90} max={90} onCommit={coordsCommitted} /></label>
        </div>
      </div>

      <div className="method" data-method="country" hidden={method !== 'country'}>
        <p className="hint" id="countryHint">
          {countries === 'loading' ? 'Loading the countries…'
            : countries === 'error' ? <span style={{ color: '#dc2626' }}>Couldn't load the countries. Choose the method again to retry.</span>
            : <>Move the mouse over the map: the country under it lights up. <b>Click it</b> to make it the zone: the forecast then covers that country only.</>}
        </p>
      </div>

      <div className="selected" id="zoneSelected" hidden={!zone}>
        <div className="mhead">Selected zone</div>
        {/* A readout: the zone's name, its four edges, its size, then the pin and country notes when they apply. */}
        <div id="zoneinfo" className="zoneinfo zcard">
          {zone && summary ? <>
            <p className="z-name">{zoneCountry ?? (pinAt ? 'Area around the pin' : 'Custom area')}</p>
            <dl className="z-bounds">
              <div><dt>North</dt><dd>{lat(zone.n)}</dd></div>
              <div><dt>South</dt><dd>{lat(zone.s)}</dd></div>
              <div><dt>West</dt><dd>{lon(zone.w)}</dd></div>
              <div><dt>East</dt><dd>{lon(zone.e)}</dd></div>
            </dl>
            <dl className="z-size">
              <div><dt>Area</dt><dd>{Math.round(areaKm2(zone)).toLocaleString()} km²</dd></div>
              <div><dt>Span</dt><dd>{(zone.e - zone.w).toFixed(2)}° × {(zone.n - zone.s).toFixed(2)}°</dd></div>
            </dl>
            {pinAt && <p className="z-note"><span>Pin</span>{lat(pinAt.lat, 3)}, {lon(pinAt.lng, 3)}</p>}
            {zoneCountry && <p className="z-cover">The forecast covers {zoneCountry} only, inside its borders.</p>}
          </> : 'No zone selected.'}
        </div>
        <button type="button" id="clear" className="secondary wide" onClick={clearZone}>
          <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" aria-hidden="true">
            <path d="M6 6l12 12M18 6L6 18" /></svg>
          Clear zone</button>
      </div>
    </section>
  );
}
