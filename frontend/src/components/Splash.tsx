import { useEffect, useState } from 'react';
import { useApp } from '../store';
import { cameraSnapshot } from '../engine';

// The opening screen, laid out like a satellite Earth viewer: the live globe in true colour fills the screen, a thin
// top bar (brand · live UTC clock), the title column on the left, a readout of the point at the centre of the view,
// and a status bar of the satellites the app reads with the date of their latest data (from /api/sensors).
// "Open the fire map" hands over to the app; the CSS stages the transition and the element is removed once faded.

// The near-real-time feeds the app merges, as the status bar names them.
const FEEDS = [
  { id: 'MODIS_NRT', name: 'MODIS', sat: 'Terra · Aqua' },
  { id: 'VIIRS_SNPP_NRT', name: 'VIIRS', sat: 'Suomi NPP' },
  { id: 'VIIRS_NOAA20_NRT', name: 'VIIRS', sat: 'NOAA-20' },
  { id: 'VIIRS_NOAA21_NRT', name: 'VIIRS', sat: 'NOAA-21' },
  { id: 'LANDSAT_NRT', name: 'OLI', sat: 'Landsat 8 · 9' },
];

const pad = (n: number) => String(n).padStart(2, '0');
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
const shortDay = (iso?: string) => {
  if (!iso) return '—';
  const [, m, d] = iso.split('-').map(Number);
  return `${d} ${MONTHS[m - 1]}`;
};
const coord = (v: number, pos: string, neg: string) => `${Math.abs(v).toFixed(2)}° ${v >= 0 ? pos : neg}`;

// A feed is "live" when its latest data is from today or yesterday (UTC), otherwise it's lagging.
function fresh(iso?: string) {
  if (!iso) return false;
  const age = (Date.now() - Date.parse(iso + 'T00:00:00Z')) / 864e5;
  return age < 2;
}

function useTelemetry(active: boolean) {
  const [now, setNow] = useState(() => new Date());
  const [centre, setCentre] = useState<[number, number] | null>(null);
  useEffect(() => {
    if (!active) return;
    const tick = () => {
      setNow(new Date());
      try { const c = cameraSnapshot().center; setCentre([c[0], c[1]]); } catch { /* map not ready yet */ }
    };
    tick();
    const t = setInterval(tick, 250);
    return () => clearInterval(t);
  }, [active]);
  return { now, centre };
}

const Mark = () => (
  <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <path d="M12 20.75c-3.75 0-6.25-2.55-6.25-6 0-3.3 2.3-5.25 3.55-7.75.5 1.9 1.55 2.9 2.5 3.3.1-2.5 1.15-4.7 2.9-6.45.2 3.25 3.55 5.5 3.55 9.75 0 3.85-2.5 7.15-6.25 7.15z" />
    <path d="M12 20.75c-1.5 0-2.55-1.05-2.55-2.55 0-1.5 1.1-2.35 1.95-3.45.45 1.05 1.2 1.5 1.8 1.7.55.85 1.3 1.3 1.3 2.25 0 1.15-1.05 2.05-2.5 2.05z" fill="currentColor" stroke="none" />
  </svg>
);

export function Splash() {
  const leaving = useApp(s => s.splashLeaving);
  const gone = useApp(s => s.splashGone);
  const returning = useApp(s => s.splashReturning);
  const start = useApp(s => s.start);
  const availability = useApp(s => s.availability) as Record<string, { min: string; max: string }>;
  const { now, centre } = useTelemetry(!gone && !leaving);
  if (gone) return null;
  // `again`: it came back via the Home button, so the whole screen fades back in (not just the text).
  const cls = leaving ? 'gone' : returning ? 'again' : undefined;
  const utc = `${pad(now.getUTCHours())}:${pad(now.getUTCMinutes())}:${pad(now.getUTCSeconds())}`;
  const day = `${now.getUTCDate()} ${MONTHS[now.getUTCMonth()]} ${now.getUTCFullYear()}`;
  return (
    <div id="splash" className={cls} role="dialog" aria-labelledby="splashTitle">
      <div className="shade" />
      <span className="sp-corner tl" aria-hidden="true" /><span className="sp-corner tr" aria-hidden="true" />
      <span className="sp-corner bl" aria-hidden="true" /><span className="sp-corner br" aria-hidden="true" />

      <header className="sp-top">
        <span className="sp-brand"><Mark />Fire Pattern Predictor</span>
        <span className="sp-clock" aria-label={`Coordinated Universal Time ${utc}, ${day}`}>
          <span className="sp-live"><i aria-hidden="true" />Live</span>
          <span className="mono">{day}</span>
          <span className="mono strong">{utc} UTC</span>
        </span>
      </header>

      <div className="content">
        <p className="eyebrow">Near-real-time fire detection · NASA FIRMS</p>
        <h1 id="splashTitle">Fire Pattern Predictor</h1>
        <p className="lead">Every active fire NASA's satellites detect, on one live globe. Pick any zone on Earth and see
          where fire is burning now, and where it is likely to spread over the next three days.</p>
        <dl className="sp-specs">
          <div><dt>Sensors merged</dt><dd>MODIS · VIIRS · Landsat</dd></div>
          <div><dt>Detection pixel</dt><dd>30 m – 1 km</dd></div>
          <div><dt>Spread forecast</dt><dd>3 days ahead</dd></div>
        </dl>
        <button type="button" id="start" autoFocus disabled={leaving} onClick={start}>
          Open the fire map<span className="arrow" aria-hidden="true">→</span>
        </button>
      </div>

      {centre && (
        <p className="sp-readout mono" aria-hidden="true">
          <span>View centre</span><b>{coord(centre[1], 'N', 'S')}</b><b>{coord(centre[0], 'E', 'W')}</b>
        </p>
      )}

      <footer className="sp-status">
        <ul className="sp-feeds" aria-label="Satellite feeds and their latest data">
          {FEEDS.map(f => {
            const max = availability?.[f.id]?.max;
            const live = fresh(max);
            return (
              <li key={f.id} className={live ? 'live' : max ? 'lag' : ''}>
                <i aria-hidden="true" />
                <span className="name">{f.name}</span>
                <span className="sat">{f.sat}</span>
                <span className="mono date" title={max ? `Latest data: ${max}` : 'Status not loaded'}>{shortDay(max)}</span>
              </li>
            );
          })}
        </ul>
        <p className="credit">Data: NASA FIRMS · Imagery © Esri · Borders: Natural Earth</p>
      </footer>
    </div>
  );
}
