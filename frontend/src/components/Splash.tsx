import { useApp } from '../store';

// The opening screen. "Get started" hands over to the app; the CSS stages the transition
// (text leaves, the fire filter dissolves) and the element is removed once it has faded out.
export function Splash() {
  const leaving = useApp(s => s.splashLeaving);
  const gone = useApp(s => s.splashGone);
  const returning = useApp(s => s.splashReturning);
  const start = useApp(s => s.start);
  if (gone) return null;
  // `again`: it came back via the Home button, so the whole screen fades back in (not just the text).
  const cls = leaving ? 'gone' : returning ? 'again' : undefined;
  return (
    <div id="splash" className={cls} role="dialog" aria-labelledby="splashTitle">
      <div className="tint" /><div className="glow" /><div className="shade" />
      <div className="content">
        <p className="eyebrow">Powered by NASA satellite data</p>
        <h1 id="splashTitle">Fire Pattern Predictor</h1>
        <p className="lead">Wildfires move fast. This app reads what NASA's satellites see from orbit and
          turns it into a picture of where fire is burning, and where it is likely to go next.</p>
        <ul>
          <li><span><b>Live detections</b> from MODIS, VIIRS and Landsat, merged into one consistent view.</span></li>
          <li><span><b>Choose any zone on Earth</b>: draw it on the map or type its coordinates.</span></li>
          <li><span><b>Fire patterns at a glance</b>: intensity, timing and spread across the days you choose.</span></li>
        </ul>
        <button type="button" id="start" autoFocus disabled={leaving} onClick={start}>Get started →</button>
      </div>
      <p className="credit">Data: NASA FIRMS · Imagery © Esri · Borders: Natural Earth</p>
    </div>
  );
}
