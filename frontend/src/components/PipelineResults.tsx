import { useApp, type PipeStage } from '../store';
import { num } from '../format';
import { areaKm2 } from '../geo';
import { WeatherGroup } from './Forecast';
import { Columns, SplitBar, HBars, RangeBar, ChartHead, fmtN } from './PanelCharts';

// The pipeline results panel: about the run, an overview of what the workers found, the predicted evolution
// (day tabs and a summary per day) and the weather layers. Each fire's own details and prediction open in a card
// when the fire is clicked on the map (fireCard.ts); the stage list is shown in the Analyse window while it runs.

const fmt = (v: number | null | undefined, d = 1) => (v == null ? '—' : v.toLocaleString('en-GB', { maximumFractionDigits: d }));
const shortD = (d?: string) => (d ? new Date(d + 'T00:00:00Z').toLocaleDateString('en-GB', { day: 'numeric', month: 'short', timeZone: 'UTC' }) : '');
const dayNo = (d: string) => String(+d.slice(8, 10));
// FRP classes of the map legend (same colours as the dots on the map)
const FRP_CLASSES = [
  { label: 'Under 10 MW', lo: -Infinity, hi: 10, color: '#facc15' },
  { label: '10–50 MW', lo: 10, hi: 50, color: '#f97316' },
  { label: '50–200 MW', lo: 50, hi: 200, color: '#dc2626' },
  { label: 'Over 200 MW', lo: 200, hi: Infinity, color: '#7f1d1d' },
];

export function StageList({ stages }: { stages: PipeStage[] }) {
  // Localisation and fire power run side by side: they share one step of the list.
  const by = (id: string) => stages.find(s => s.id === id);
  const rows: PipeStage[][] = [['correlation'], ['interpreter'], ['localisation', 'fire_power'], ['epicenter'], ['expansion'], ['kg'], ['prediction']]
    .map(ids => ids.map(by).filter(Boolean) as PipeStage[]).filter(r => r.length);
  return (
    <ol className="pipe">
      {rows.map((group, k) => (
        <li key={k} className={`pipe-step${group.length > 1 ? ' pair' : ''}`}>
          {group.map(s => (
            <div key={s.id} className={`pipe-stage ${s.state}`}>
              <span className="pipe-dot" aria-hidden="true" />
              <span className="pipe-name">{s.label}</span>
              <span className="pipe-time">{s.state === 'ok' ? (s.skipped ? 'skipped' : s.reused ? 'saved' : `${s.seconds} s`) : s.state === 'run' ? 'running…' : s.state === 'error' ? 'failed' : ''}</span>
              {s.detail && <span className="pipe-detail">{s.detail}</span>}
            </div>
          ))}
        </li>
      ))}
    </ol>
  );
}


export function PipelineResults() {
  const visible = useApp(s => s.forecastVisible);
  const p = useApp(s => s.pipeline);
  const { changeZone } = useApp.getState();
  if (!p) return null;
  if (p.realOnly) return <RealOnly />;

  return (
    <section id="pipelineSection" hidden={!visible}>
      <div className="fhead">
        <h2>Pipeline results</h2>
        <button type="button" className="secondary" id="changeZone" onClick={changeZone}>
          <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round"
               strokeLinejoin="round" aria-hidden="true"><path d="M19 12H5M11 6l-6 6 6 6" /></svg>
          Change zone</button>
      </div>

      <WeatherGroup />

      <AboutRun />

      <Overview />

      <Prediction />

    </section>
  );
}


const latS = (v: number) => `${Math.abs(+v.toFixed(2))}° ${v >= 0 ? 'N' : 'S'}`;
const lonS = (v: number) => `${Math.abs(+v.toFixed(2))}° ${v >= 0 ? 'E' : 'W'}`;
const addDays = (d: string, k: number) => new Date(Date.parse(d + 'T00:00:00Z') + k * 864e5).toISOString().slice(0, 10);
const longD = (d: string) => new Date(d + 'T00:00:00Z').toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric', timeZone: 'UTC' });
// Sensor product ids -> the names people know
const SENSOR_NAMES: Record<string, string> = { MODIS_NRT: 'MODIS', MODIS_SP: 'MODIS', VIIRS_SNPP_NRT: 'VIIRS S-NPP', VIIRS_SNPP_SP: 'VIIRS S-NPP',
  VIIRS_NOAA20_NRT: 'VIIRS NOAA-20', VIIRS_NOAA20_SP: 'VIIRS NOAA-20', VIIRS_NOAA21_NRT: 'VIIRS NOAA-21', LANDSAT_NRT: 'Landsat' };

// About this run, at a glance: where (the zone's name, size and edges), when (a strip of days: the satellite data the
// pipeline read, then the days it predicts, or that already have real data), and what went in (satellites, weather, model).
function AboutRun() {
  const p = useApp(s => s.pipeline)!;
  const loaded = useApp(s => s.loaded);
  const zone = useApp(s => s.zone);
  const country = useApp(s => s.zoneCountry);
  const pin = useApp(s => s.pinAt);
  const { zoomToZone } = useApp.getState();
  const start = loaded?.params.date as string | undefined, n = +(loaded?.params.days ?? 0);
  const real = new Set(p.actual?.days ?? []);
  const win = start ? Array.from({ length: n }, (_, i) => addDays(start, i)) : [];
  const after = start ? [1, 2, 3].map(k => addDays(start, n - 1 + k)) : [];
  const predicted = new Set(p.predictions.map(r => r.date as string));
  const sensors = loaded?.mode === 'fetch' ? [SENSOR_NAMES[loaded.sensor] ?? loaded.sensor] : ['MODIS', 'VIIRS', 'Landsat'];
  const info = p.prediction;
  const model = info?.models ? String(info.models).replace(/^weather: |^no_weather: /, '') : null;
  return (
    <div className="fgroup about">
      <div className="fgt">About this run<button type="button" className="small" onClick={() => zoomToZone()}>Zoom to zone</button></div>

      {zone && <div className="ab-zone">
        <p className="ab-name">{country ?? (pin ? 'Area around the pin' : 'Custom area')}</p>
        <p className="ab-sub">{Math.round(areaKm2(zone)).toLocaleString('en-GB')} km² · {latS(zone.s)} to {latS(zone.n)}, {lonS(zone.w)} to {lonS(zone.e)}</p>
      </div>}

      {start && <div className="ab-when">
        <div className="ab-row"><span className="ab-k">Period</span>
          <span className="ab-v">{shortD(win[0])} – {longD(after[after.length - 1])}</span></div>
        <ol className="ab-days" aria-label="Days of this run">
          {win.map(d => <li key={d} className="obs" title={`${longD(d)}: satellite data`}><b>{dayNo(d)}</b></li>)}
          {after.map((d, i) => {
            const kind = real.has(d) ? 'real' : predicted.has(d) ? 'pred' : 'none';
            return <li key={d} className={kind} title={`${longD(d)}: ${kind === 'real' ? 'real satellite data' : kind === 'pred' ? 'predicted' : 'nothing to predict'}`}>
              <b>{dayNo(d)}</b><small>+{i + 1}</small></li>;
          })}
        </ol>
        <ul className="ab-key">
          <li><i className="obs" />Satellite data · {n} day{n === 1 ? '' : 's'}</li>
          {after.some(d => predicted.has(d)) && <li><i className="pred" />Predicted</li>}
          {after.some(d => real.has(d)) && <li><i className="real" />Real data, not predicted</li>}
        </ul>
      </div>}

      <dl className="ab-src">
        <div><dt>Satellites</dt><dd>{sensors.map(x => <span key={x} className="ab-chip">{x}</span>)}
          {loaded?.mode !== 'fetch' && <small>merged into one dataset</small>}</dd></div>
        <div><dt>Weather</dt><dd>{info?.weather === 'provided' ? 'Open-Meteo hourly forecast' : info ? 'Not available for this run' : 'Open-Meteo'}</dd></div>
        {model && <div><dt>Model</dt><dd>{model}</dd></div>}
      </dl>
      <p className="ab-run" title="The folder this run's files are saved in">Run {p.run}</p>
    </div>
  );
}

// What the workers found: headline counts, detections per day, how the fires ended, and how intense they were.
function Overview() {
  const p = useApp(s => s.pipeline)!;
  const fires = useApp(s => s.lastFires);
  const s = p.summary;
  const footprint = p.fires.reduce((a, f) => a + (f.final_footprint_km2 ?? 0), 0);
  const biomass = p.fires.reduce((a, f) => a + (f.biomass_burned_t ?? 0), 0);
  const peak = Math.max(0, ...p.fires.map(f => f.peak_frp_mw ?? 0));
  const perDay = new Map<string, number>();
  for (const f of fires?.features ?? []) perDay.set(f.properties.date, (perDay.get(f.properties.date) ?? 0) + 1);
  const days = [...perDay].sort((a, b) => a[0].localeCompare(b[0]));
  const burning = s.status['burning at data end'] ?? 0, out = s.status['extinguished'] ?? 0;
  const classes = FRP_CLASSES.map(c => ({ ...c, value: p.fires.filter(f => (f.peak_frp_mw ?? 0) >= c.lo && (f.peak_frp_mw ?? 0) < c.hi).length }));
  return (
    <div className="fgroup">
      <div className="fgt">Overview</div>
      <div className="pstats">
        <div className="pstat"><b>{num(s.fires)}</b>Fires</div>
        <div className="pstat"><b>{num(s.detections)}</b>Detections</div>
        <div className="pstat"><b>{fmt(peak, 0)}</b>Peak FRP, MW</div>
      </div>

      {days.length > 0 && <>
        <ChartHead title="Detections per day" note="Satellite fire pixels in the zone, each day of the window." />
        <Columns ariaLabel={`Detections per day: ${days.map(([d, n]) => `${shortD(d)} ${n}`).join(', ')}`}
                 cols={days.map(([d, n]) => ({ label: dayNo(d), value: n, tip: <><b>{num(n)}</b> detections · {shortD(d)}</> }))} />
      </>}

      {burning + out > 0 && <>
        <ChartHead title="How the fires ended" note="At the last satellite pass of the window." />
        <SplitBar ariaLabel={`${burning} fires still burning, ${out} out`}
                  parts={[{ label: 'still burning', value: burning, cls: 'c-hot' }, { label: 'went out', value: out, cls: 'c-cold' }]} />
      </>}

      <ChartHead title="How intense they were" note="Each fire's peak radiative power, in the map legend's classes." />
      <HBars ariaLabel={classes.map(c => `${c.label}: ${c.value} fires`).join(', ')}
             bars={classes.map(c => ({ label: c.label, value: c.value, color: c.color, tip: <><b>{num(c.value)}</b> fires peaked at {c.label.toLowerCase()}</> }))} />

      <p className="notes">Burnt footprint {fmt(footprint, 1)} km² · biomass burned {fmt(biomass, 0)} t · {num(s.epicenters)} epicenters
        {s.merged_fires ? ` · ${num(s.merged_fires)} fires merged` : ''}.</p>
    </div>
  );
}

// The model's output: how each active fire evolves over the next 3 days, from the pipeline's data plus the hourly
// weather forecast (our weather API). Day tabs pick the day; the map's spread heatmap and predicted extents follow.
function Prediction() {
  const p = useApp(s => s.pipeline);
  const day = useApp(s => s.pipeDay);
  const { setPipeDay } = useApp.getState();
  if (!p) return null;
  const info = p.prediction;
  const real = p.actual?.days ?? [];
  const short0 = (d: string) => new Date(d + 'T00:00:00Z').toLocaleDateString('en-GB', { day: 'numeric', month: 'short', timeZone: 'UTC' });
  const realNote = real.length
    ? <p className="gsub">Real satellite data exists for {real.map(short0).join(', ')}, so {real.length > 1 ? 'those days are' : 'that day is'} shown
        as observed (fetched, not predicted).</p> : null;
  if (!info || !p.predictions.length) {
    return (
      <div className="fgroup">
        <div className="fgt">Predicted evolution</div>
        {real.length === 3
          ? <p className="gsub">Real satellite data exists for all 3 days after your window ({real.map(short0).join(', ')}), so nothing was
              predicted: the timeline plays what really happened.</p>
          : <>{realNote}<p className="gsub">No fires were still burning on the last day of the data, so there is nothing to predict.</p></>}
      </div>
    );
  }
  const ks = [...new Set(p.predictions.map(r => r.day_ahead as number))].sort();
  const rows = p.predictions.filter(r => r.day_ahead === day).sort((a, b) => (b.footprint_likely_km2 ?? 0) - (a.footprint_likely_km2 ?? 0));
  const dates = [1, 2, 3].map(k => p.predictions.find(r => r.day_ahead === k)?.date as string | undefined);
  const sum = (key: string) => rows.reduce((a, r) => a + (r[key] ?? 0), 0);
  const now = rows.reduce((a, r) => a + (r.footprint_now_km2 ?? 0), 0);
  const out = rows.filter(r => r.p_out >= 0.5).length, merge = rows.filter(r => r.p_merge >= 0.2).length;
  const short = shortD;
  // seen: fires observed on the last day of the data; ahead: the expected number still burning (sum of 1 - chance out)
  const seen = new Set((p.history ?? []).filter(h => String(h.time).slice(0, 10) === info.from_day).map(h => h.fire_id)).size;
  const burningCols = [
    { k: 0, label: dayNo(String(info.from_day)), sub: 'seen', value: seen, tip: <><b>{num(seen)}</b> fires seen on {short(info.from_day)}</> },
    ...ks.map(k => {
      const v = p.predictions.filter(r => r.day_ahead === k).reduce((a, r) => a + (1 - (r.p_out ?? 0)), 0);
      return { k, label: dayNo(dates[k - 1] ?? ''), sub: `+${k}`, value: v, predicted: true, active: k === day,
               tip: <>About <b>{fmtN(v)}</b> fires still burning on {short(dates[k - 1])} (predicted)</> };
    }),
  ];
  return (
    <div className="fgroup">
      <div className="fgt">Predicted evolution</div>
      <p className="gsub">From the fires burning on {short(info.from_day)} · {info.weather === 'provided'
        ? 'with the hourly weather forecast (Open-Meteo)' : 'without weather (the forecast was unavailable)'}
        {info.models ? ` · model: ${String(info.models).replace(/^weather: |^no_weather: /, '')}` : ''}.</p>
      {realNote}
      <div className="days" role="tablist" aria-label="Day ahead">
        {ks.map(k => (
          <button key={k} type="button" role="tab" aria-selected={day === k} onClick={() => setPipeDay(k)}>
            +{k} day{k > 1 ? 's' : ''}<small>{short(dates[k - 1])}</small></button>
        ))}
      </div>
      <ChartHead title="Fires still burning" note="Seen on the last day, then expected each day ahead. Click a day." />
      <Columns ariaLabel={`Fires burning: ${burningCols.map(c => `${c.label} ${fmtN(c.value)}`).join(', ')}`}
               cols={burningCols} onPick={i => { const k = burningCols[i].k; if (k) setPipeDay(k); }} />

      <ChartHead title={`On ${short(dates[day - 1])}`} value={`${num(rows.length)} active fires`} />
      <SplitBar ariaLabel={`${rows.length - out} fires likely still burning, ${out} likely out`}
                parts={[{ label: 'likely still burning', value: rows.length - out, cls: 'c-hot' }, { label: 'likely out (≥ 50%)', value: out, cls: 'c-cold' }]} />

      <ChartHead title="Burnt footprint" value={<>{fmt(sum('footprint_likely_km2'), 1)} km² <small>likely</small></>}
                 note="Dot: likely total · band: low–high range · line: now." />
      <RangeBar ariaLabel={`Footprint now ${fmt(now, 1)} km², likely ${fmt(sum('footprint_likely_km2'), 1)}, range ${fmt(sum('footprint_low_km2'), 1)} to ${fmt(sum('footprint_high_km2'), 1)} km²`}
                now={now} low={sum('footprint_low_km2')} likely={sum('footprint_likely_km2')} high={sum('footprint_high_km2')} unit="km²" />

      <div className="pstats">
        <div className="pstat"><b>{fmt(sum('growth_likely'), 1)}</b>New area, km²</div>
        <div className="pstat"><b>{num(merge)}</b>May merge (≥ 20%)</div>
        <div className="pstat"><b>{fmt(Math.max(0, ...rows.map(r => r.frp_likely ?? 0)), 0)}</b>Top FRP, MW</div>
      </div>
      <p className="notes">The map shows the chance that each 1 km cell burns on this day (heatmap) and each fire's likely extent
        (orange circles). Click a fire on the map to see its own details and prediction.</p>
    </div>
  );
}

// Shown when the pipeline was skipped: every day after the window already had real data, so the app fetched it
// (window + those days) and shows it as observed fire, day by day, with nothing predicted.
function RealOnly() {
  const p = useApp(s => s.pipeline)!;
  const fires = useApp(s => s.lastFires);
  const zone = useApp(s => s.zone);
  const loaded = useApp(s => s.loaded);
  const visible = useApp(s => s.forecastVisible);
  const { changeZone, zoomToZone } = useApp.getState();
  const perDay = new Map<string, number>();
  for (const f of fires?.features ?? []) perDay.set(f.properties.date, (perDay.get(f.properties.date) ?? 0) + 1);
  const days = [...perDay].sort((a, b) => a[0].localeCompare(b[0]));
  const real = new Set(p.actual?.days ?? []);
  const short0 = (d: string) => new Date(d + 'T00:00:00Z').toLocaleDateString('en-GB', { day: 'numeric', month: 'short', timeZone: 'UTC' });
  return (
    <section id="pipelineSection" hidden={!visible}>
      <div className="fhead">
        <h2>Real data</h2>
        <button type="button" className="secondary" id="changeZone" onClick={changeZone}>
          <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round"
               strokeLinejoin="round" aria-hidden="true"><path d="M19 12H5M11 6l-6 6 6 6" /></svg>
          Change zone</button>
      </div>
      <WeatherGroup />
      <div className="fgroup">
        <div className="fgt">Pipeline skipped<button type="button" className="small" onClick={() => zoomToZone()}>Zoom to zone</button></div>
        <p className="gsub">Real satellite data already exists for all 3 days after your window
          ({[...real].map(short0).join(', ')}), so there was nothing to predict: the pipeline and the model were not run, and the
          real detections are shown instead. The timeline plays them day by day.</p>
        <dl className="meta">
          {zone && <><dt>Zone</dt><dd>{zone.w}, {zone.s} → {zone.e}, {zone.n}</dd></>}
          <dt>Window</dt><dd>{loaded ? `${loaded.params.date} · ${loaded.params.days} day${loaded.params.days === '1' ? '' : 's'}` : '—'}</dd>
          <dt>Data</dt><dd>{loaded?.mode === 'fetch' ? loaded.sensor : 'All sensors (harmonised)'}</dd>
        </dl>
      </div>
      <div className="fgroup">
        <div className="fgt">Detections per day</div>
        <Columns ariaLabel={`Detections per day: ${days.map(([d, n]) => `${short0(d)} ${n}`).join(', ')}`}
                 cols={days.map(([d, n]) => ({ label: dayNo(d), sub: real.has(d) ? 'after' : undefined, value: n,
                   tip: <><b>{num(n)}</b> detections · {short0(d)}{real.has(d) ? ' (after the window)' : ''}</> }))} />
        <p className="notes">To see the workers' fire analysis and a prediction, pick a window that ends yesterday or today.</p>
      </div>
    </section>
  );
}
