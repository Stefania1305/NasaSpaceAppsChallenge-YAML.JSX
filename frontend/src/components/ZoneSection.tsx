import { useEffect, useRef, type InputHTMLAttributes } from 'react';
import { useApp, zoneSummary, type Method } from '../store';

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

// Part 1 of the side panel: choose the prediction zone (four ways, picked from a dropdown).
export function ZoneSection() {
  const hidden = useApp(s => s.forecastVisible);
  const method = useApp(s => s.method);
  const drawing = useApp(s => s.drawing);
  const pinMode = useApp(s => s.pinMode);
  const pinError = useApp(s => s.pinError);
  const zone = useApp(s => s.zone);
  const pinAt = useApp(s => s.pinAt);
  const { setMethod, toggleDrawing, togglePinMode, coordsCommitted, useCurrentView, applyPin, clearZone } = useApp.getState();
  const commitPin = useRef(() => applyPin(true)).current;
  const summary = zone ? zoneSummary(zone, pinAt) : null;

  return (
    <section id="zoneSection" hidden={hidden}>
      <h2>1 · Prediction zone</h2>
      <label className="methodpick">How to set the zone
        <select id="zoneMethod" value={method} onChange={e => setMethod(e.target.value as Method)}>
          <option value="draw">Draw a rectangle</option>
          <option value="pin">Pin a point</option>
          <option value="coords">Type coordinates</option>
          <option value="view">Use what's on screen</option>
        </select>
      </label>

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

      <div className="method" data-method="view" hidden={method !== 'view'}>
        <p className="hint">Pan and zoom the map to the area you want, then click the button.</p>
        <button type="button" id="useView" className="secondary wide" onClick={useCurrentView}>Use current view</button>
      </div>

      <div className="selected" id="zoneSelected" hidden={!zone}>
        <div className="mhead" style={{ marginBottom: 2 }}>Selected zone</div>
        <div id="zoneinfo" className="zoneinfo">
          {summary
            ? <>Zone <b>{summary.coords}</b><br />{summary.size}{summary.pin && <><br />{summary.pin}</>}</>
            : 'No zone selected.'}
        </div>
        <button type="button" id="clear" className="secondary wide" onClick={clearZone}>
          <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" aria-hidden="true">
            <path d="M6 6l12 12M18 6L6 18" /></svg>
          Clear zone</button>
      </div>
    </section>
  );
}
