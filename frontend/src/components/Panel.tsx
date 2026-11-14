import { useLayoutEffect, useRef } from 'react';
import { useApp } from '../store';
import { Forecast } from './Forecast';
import { ZoneSection } from './ZoneSection';

// The side panel. Its children stay in this order: the CSS fades them in one after another by position.
export function Panel() {
  const open = useApp(s => s.panelOpen);
  const zone = useApp(s => s.zone);
  const date = useApp(s => s.date);
  const days = useApp(s => s.days);
  const availability = useApp(s => s.availability);
  const status = useApp(s => s.status);
  const { setPanel, setDate, setDays, openModal } = useApp.getState();

  // The header's title row (dot · name · back arrow) and its subtitle are each sized to span the panel's full
  // width: measure at the current size, scale the font by how much room is left, and repeat once to settle.
  const headRef = useRef<HTMLDivElement>(null);
  useLayoutEffect(() => {
    const head = headRef.current;
    if (!head) return;
    const fit = () => {
      const cs = getComputedStyle(head);
      const room = head.clientWidth - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight);
      if (!(room > 0)) return;
      for (const el of head.querySelectorAll<HTMLElement>('.fit')) {
        for (let i = 0; i < 3; i++) {
          const w = el.getBoundingClientRect().width, f = parseFloat(getComputedStyle(el).fontSize);
          if (!(w > 0) || Math.abs(w - room) < .5) break;
          el.style.fontSize = `${Math.max(10, f * room / w)}px`;
        }
      }
    };
    fit();
    document.fonts?.ready.then(fit);                    // re-fit once the web font has loaded
    const ro = new ResizeObserver(fit);
    ro.observe(head);
    return () => ro.disconnect();
  }, []);

  // The panel is centred vertically, so its top moves as its content grows or the window resizes. The menu
  // planets dock at the top of the panel, so keep its top edge in a CSS variable they can follow.
  const panelRef = useRef<HTMLElement>(null);
  useLayoutEffect(() => {
    const el = panelRef.current;
    if (!el) return;
    const track = () => document.documentElement.style.setProperty('--panel-top', `${Math.round(el.getBoundingClientRect().top)}px`);
    track();
    const ro = new ResizeObserver(track);
    ro.observe(el);
    addEventListener('resize', track);
    return () => { ro.disconnect(); removeEventListener('resize', track); };
  }, []);

  const nrt = Object.entries(availability).filter(([k]) => k.endsWith('_NRT')).map(([, v]) => v.max).sort();

  return (
    <aside id="panel" ref={panelRef}>
      {/* Desktop: a fixed strip at the top where the menu planets dock; only the content below it scrolls. */}
      <div className="dock" aria-hidden="true" />
      <div className="pscroll">
      {/* Panel header: clicking it slides the panel back out to the left (desktop). */}
      <div className="apphead" id="appHead" ref={headRef} role="button" tabIndex={0} aria-expanded={open} title="Close the panel"
           onClick={() => setPanel(false)}
           onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setPanel(false); } }}>
        <h1 className="fit">Fire Pattern Predictor
          <svg className="chev" viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2.4"
               strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M6 9l6 6 6-6" /></svg></h1>
        <p className="fit">NASA FIRMS · MODIS · VIIRS · Landsat</p>
      </div>

      <Forecast />
      <ZoneSection />

      <section>
        <h2>Time window</h2>
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
      </div>
    </aside>
  );
}
