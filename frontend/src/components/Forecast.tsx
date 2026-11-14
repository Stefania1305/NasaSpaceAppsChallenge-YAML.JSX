import { useApp } from '../store';
import { num, pct, shortDate } from '../format';

// Part 1 of the side panel after Predict: the fire forecast (it takes the place of the zone controls).
export function Forecast() {
  const visible = useApp(s => s.forecastVisible);
  const d = useApp(s => s.forecast);
  const day = useApp(s => s.day);
  const sub = useApp(s => s.forecastSub);
  const { showDay, changeZone, zoomToZone, flyToHotspot } = useApp.getState();

  const f = d?.forecast?.[day];
  const bt = d?.backtest, m = d?.models?.[0];
  const obs: number[] = d?.observed_cells_per_day ?? [];

  return (
    <section id="forecastSection" hidden={!visible}>
      <div className="fhead">
        <h2>1 · Fire forecast</h2>
        <button type="button" className="secondary" id="changeZone" onClick={changeZone}>
          <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round"
               strokeLinejoin="round" aria-hidden="true"><path d="M19 12H5M11 6l-6 6 6 6" /></svg>
          Change zone</button>
      </div>
      <p className="fsub" id="forecastSub">{sub}</p>

      <div className="days" id="dayTabs" role="tablist" aria-label="Forecast day">
        {d?.forecast?.map((fd: { date: string }, i: number) => (
          <button key={i} type="button" role="tab" aria-selected={i === day} onClick={() => showDay(i)}>
            +{i + 1} day{i ? 's' : ''}<small>{shortDate(fd.date)}</small>
          </button>
        ))}
      </div>

      <div className="pstats" id="pstats">
        {f && <>
          <div className="pstat" title="Expected number of grid cells burning near current fires">Expected burning cells
            <b>~{num(Math.round(f.expected_cells))}</b></div>
          <div className="pstat" title="Cells with more than a 50% chance of burning">Likely to burn (&gt;50%)
            <b>{num(f.likely_cells)}</b></div>
          <div className="pstat" title="New fires expected far from any current fire, where the model can't place them">New fires elsewhere
            <b>{f.expected_new_elsewhere == null ? 'n/a' : '~' + num(Math.round(f.expected_new_elsewhere))}</b></div>
        </>}
      </div>

      <div className="scale"><span>Chance of fire</span><span>10%</span><i /><span>85%+</span></div>

      {d && (
        <ul className="facts" id="facts">
          <li>Burning {d.cell_km} km cells per day in the loaded window: {obs.map(v => num(v)).join(' → ')}.</li>
          {d.trend_last_day != null &&
            <li>Fire activity {d.trend_last_day >= 0 ? 'rose' : 'fell'} {pct(Math.abs(d.trend_last_day))} on the last observed day.</li>}
          <li>{d.spread_direction
            ? <>Fire fronts have mostly been spreading towards the <b>{d.spread_direction.compass}</b> ({d.spread_direction.bearing_deg}°).</>
            : 'No consistent spread direction in this data.'}</li>
          {!!d.detections_skipped_non_vegetation &&
            <li>{num(d.detections_skipped_non_vegetation)} detections ignored (volcanoes, industrial heat sources, offshore).</li>}
          {d.cells_truncated && <li>Only the highest-risk cells are drawn on the map.</li>}
        </ul>
      )}

      <div id="hotspots">
        {!!d?.hotspots?.length && <>
          <h4>Where new fire is most likely tomorrow</h4>
          {d.hotspots.map((h: { lat: number; lon: number; p: number }, i: number) => (
            <button key={i} type="button" className="hot" onClick={() => flyToHotspot(h.lat, h.lon)}>
              <span>{h.lat.toFixed(2)}, {h.lon.toFixed(2)}</span><b>{pct(h.p)}</b>
            </button>
          ))}
        </>}
      </div>

      {/* How much to trust it: measured on this zone by predicting the last loaded day. */}
      <p className="reliab" id="reliab">
        {!d ? null
          : !m?.fitted ? '⚠ Too few fires in this zone and time window to train the model, so it uses default settings. Treat the forecast as a rough guide.'
          : !bt ? 'Load at least 3 days of data to measure how accurate the forecast is for this zone.'
          : (!bt.model_fitted || bt.auc == null) ? `Accuracy check on ${shortDate(bt.day)}: not enough fires to score it.`
          : <><b>Accuracy check</b> (model trained without {shortDate(bt.day)}, then asked to predict it):{' '}
              {pct(bt.precision)} of the {num(bt.top_k)} cells it ranked most likely did burn, versus {pct(bt.baseline_precision)} for
              a "same as yesterday" guess. Ranking score {bt.auc} (0.5 = guessing, 1 = perfect).</>}
      </p>

      <button type="button" className="secondary small" id="zoomZone" style={{ marginTop: 10 }} onClick={() => zoomToZone(1500)}>
        Zoom to zone</button>
    </section>
  );
}
