import { useApp } from '../store';
import { Forecast } from './Forecast';
import { ZoneSection } from './ZoneSection';

// Desktop: the "Fire Pattern Predictor" pill. Clicking it slides the panel in from the left.
export function Pill() {
  const open = useApp(s => s.panelOpen);
  const setPanel = useApp(s => s.setPanel);
  return (
    <button type="button" id="appToggle" aria-expanded={open} aria-controls="panel" title="Open the panel" onClick={() => setPanel(true)}>
      <span className="pt">Fire Pattern Predictor
        <svg className="chev" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round"
             strokeLinejoin="round" aria-hidden="true"><path d="M9 6l6 6-6 6" /></svg></span>
      <span className="ps">NASA FIRMS · MODIS · VIIRS · Landsat</span>
    </button>
  );
}

// The side panel. Its children stay in this order: the CSS fades them in one after another by position.
export function Panel() {
  const open = useApp(s => s.panelOpen);
  const zone = useApp(s => s.zone);
  const date = useApp(s => s.date);
  const days = useApp(s => s.days);
  const availability = useApp(s => s.availability);
  const status = useApp(s => s.status);
  const { setPanel, setDate, setDays, openModal } = useApp.getState();

  const nrt = Object.entries(availability).filter(([k]) => k.endsWith('_NRT')).map(([, v]) => v.max).sort();

  return (
    <aside id="panel">
      {/* Panel header: clicking it slides the panel back out to the left (desktop). */}
      <div className="apphead" id="appHead" role="button" tabIndex={0} aria-expanded={open} title="Close the panel"
           onClick={() => setPanel(false)}
           onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setPanel(false); } }}>
        <h1>Fire Pattern Predictor
          <svg className="chev" viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2.4"
               strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M6 9l6 6 6-6" /></svg></h1>
        <p>NASA FIRMS · MODIS · VIIRS · Landsat</p>
      </div>

      <Forecast />
      <ZoneSection />

      <section>
        <h2>2 · Time window</h2>
        <div className="row">
          <label>Start date<input type="date" id="date" value={date} onChange={e => setDate(e.target.value)} /></label>
          <label>Days of data
            <select id="days" value={days} onChange={e => setDays(e.target.value)}>
              {['1', '2', '3', '4', '5'].map(d => <option key={d}>{d}</option>)}
            </select>
          </label>
        </div>
        <p className="hint" id="avail">{nrt.length ? `Near-real-time data available up to ${nrt[nrt.length - 1]}.` : ''}</p>
      </section>

      <button type="button" id="run" disabled={!zone} hidden={!zone} onClick={openModal}>Analyse zone</button>
      <div id="status" className="hint">{status}</div>
    </aside>
  );
}
