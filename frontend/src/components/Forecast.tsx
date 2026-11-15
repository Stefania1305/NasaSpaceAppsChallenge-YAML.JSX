import { useApp, type Weather, type WeatherSummary } from '../store';
import { DotLine, ChartHead, fmtN } from './PanelCharts';
import { num, pct, shortDate } from '../format';
import { Compass, MiniBar, RankingGauge, RiskHistogram, Timeline } from './Charts';
import { WEATHER_VARS, weatherRange, type WeatherVar } from '../weatherHeat';

// Part 1 of the side panel after Predict: the fire forecast (it takes the place of the zone controls).
// The information is grouped into titled cards: about the forecast, the forecast itself, recent fire
// activity, where new fire is most likely, and how reliable the forecast is.

// Plain-language label for the ranking score (AUC): 0.5 is guessing, 1 is perfect.
const aucLabel = (a: number) => a >= .85 ? 'strong' : a >= .75 ? 'good' : a >= .6 ? 'fair' : 'weak';

function Group({ title, action, children }: { title: string; action?: React.ReactNode; children: React.ReactNode }) {
  return (
    <div className="fgroup">
      <div className="fgt"><span>{title}</span>{action}</div>
      {children}
    </div>
  );
}

// Two horizontal bars: the model's hit rate against the naive "same as yesterday" guess.
function CompareBars({ rows }: { rows: { name: string; v: number; muted?: boolean }[] }) {
  const max = Math.max(.01, ...rows.map(r => r.v));
  return (
    <div className="hbars">
      {rows.map(r => (
        <div className="hbar" key={r.name} tabIndex={0}>
          <span className="hname">{r.name}</span>
          <span className="htrack"><span className={`hfill${r.muted ? ' muted' : ''}`} style={{ width: `${(r.v / max) * 100}%` }} /></span>
          <span className="hval">{pct(r.v)}</span>
          <span className="tip" role="tooltip"><b>{pct(r.v)}</b> of the cells it ranked most likely did burn</span>
        </div>
      ))}
    </div>
  );
}

// On/off switch (used for the weather layers).
function Switch({ on, onClick, label, busy, id }: { on: boolean; onClick: () => void; label: string; busy?: boolean; id?: string }) {
  return (
    <button type="button" role="switch" aria-checked={on} aria-label={label} title={label} id={id}
            className={`switch${on ? ' on' : ''}${busy ? ' busy' : ''}`} onClick={onClick}>
      <span className="knob" />
    </button>
  );
}

const RATING_TEXT = { high: 'High: hot, dry or windy, fire spreads easily',
                      moderate: 'Moderate: some fire-friendly conditions',
                      low: 'Low: cool, humid, calm or wet' } as const;

// Weather card: one switch per weather layer (temperature, humidity, wind), each drawn separately on the map.
// While any is on, the zone's weather for the selected forecast day is summarised at the top.
// The weather day by day: one small chart per measure (small multiples, each on its own scale), the shown day outlined.
function WeatherDays({ w, date }: { w: Weather; date: string }) {
  const days = w.summary.filter(x => x.available);
  const today = new Date().toISOString().slice(0, 10);
  const rows: { k: WeatherVar; title: string; get: (x: WeatherSummary) => number }[] = [
    { k: 'temp', title: 'Highest temperature', get: x => x.temp_max ?? 0 },
    { k: 'rh', title: 'Lowest humidity', get: x => x.rh_min ?? 0 },
    { k: 'wind', title: 'Strongest wind', get: x => x.wind_max ?? 0 },
  ];
  return (
    <div className="wxdays">
      {rows.map(r => {
        const unit = WEATHER_VARS[r.k].unit, cur = days.find(x => x.date === date) ?? days[0];
        return (
          <div key={r.k}>
            <ChartHead title={r.title} value={`${fmtN(r.get(cur), 0)} ${unit} on ${shortDate(cur.date)}`} />
            <DotLine color={WEATHER_VARS[r.k].ramp[2]} unit={unit}
                     ariaLabel={`${r.title}: ${days.map(x => `${shortDate(x.date)} ${fmtN(r.get(x), 0)} ${unit}`).join(', ')}`}
                     pts={days.map(x => ({ label: String(+x.date.slice(8, 10)), value: r.get(x), active: x.date === date, forecast: x.date >= today,
                       tip: <><b>{fmtN(r.get(x), 0)} {unit}</b> · {shortDate(x.date)}</> }))} />
          </div>
        );
      })}
      <p className="notes">Dashed: forecast days. Each chart has its own scale.</p>
    </div>
  );
}

export function WeatherGroup() {
  const layers = useApp(s => s.weatherLayers);
  const w = useApp(s => s.weather);
  const loading = useApp(s => s.weatherLoading);
  const error = useApp(s => s.weatherError);
  const fdate = useApp(s => (s.pipeline ? s.wxDate ?? s.weather?.dates[0] :s.forecast?.forecast?.[s.day]?.date));
  const toggle = useApp.getState().toggleWeatherLayer;
  const any = Object.values(layers).some(Boolean);
  const sm = any && w ? w.summary.find(x => x.date === fdate) : undefined;
  return (
    <Group title="Weather">
      {any && loading && <p className="gsub">Loading the weather…</p>}
      {any && error && <p className="warn">⚠ {error}</p>}
      {any && !loading && !error && sm && (sm.available ? <>
        {sm.rating && <div className={`wxrating ${sm.rating}`}>
          <span className="dot" aria-hidden="true" />Fire weather on {shortDate(sm.date)}: <b>{RATING_TEXT[sm.rating]}</b>
        </div>}
        <div className="pstats wxstats">
          <div className="pstat"><b>{Math.round(sm.temp_max!)}°C</b>Highest temperature</div>
          <div className="pstat"><b>{sm.rh_min}%</b>Lowest humidity</div>
          <div className="pstat"><b>{sm.wind_max} km/h</b>Strongest wind{sm.wind_from_compass ? `, from the ${sm.wind_from_compass}` : ''}</div>
        </div>
        <p className="gsub">Rain {sm.rain_mean} mm on average{sm.gust_max != null ? ` · gusts up to ${sm.gust_max} km/h` : ''}.</p>
        {w && w.summary.filter(x => x.available).length > 1 && <WeatherDays w={w} date={sm.date} />}
      </> :<p className="gsub">No weather forecast is available for {shortDate(sm.date)}.</p>)}
      {any && !loading && !error && w && !sm && <p className="gsub">The weather covers {shortDate(w.dates[0])} to {shortDate(w.dates[w.dates.length - 1])}: move the timeline to one of those days to see it.</p>}
      {!any && <p className="gsub">Show the weather across the zone on the map: switch on any layer below.</p>}

      {/* One row per layer: its name, its switch, and (while on) its colour key */}
      <div className="wxlayers">
        {(Object.keys(WEATHER_VARS) as WeatherVar[]).map(k => {
          const cfg = WEATHER_VARS[k], on = layers[k];
          const [lo, hi] = w ? weatherRange(w, k) : [0, 1];
          return (
            <div className={`wxlayer${on ? ' on' : ''}`} key={k}>
              <div className="wxrow">
                <span className="wxname"><i style={{ background: cfg.ramp[3] }} aria-hidden="true" />{cfg.label}</span>
                <Switch id={`wx-${k}-toggle`} on={on} busy={on && loading} onClick={() => toggle(k)}
                        label={`${on ? 'Hide' : 'Show'} ${cfg.label.toLowerCase()} on the map`} />
              </div>
              {on && w && !loading && (
                <div className="wxkey">
                  <span className="wxbar" style={{ background: `linear-gradient(90deg, ${cfg.ramp.join(', ')})` }} />
                  <span className="wxends"><span>{Math.round(lo)} {cfg.unit}</span><span>{Math.round(hi)} {cfg.unit}</span></span>
                  <p className="gsub">{cfg.note}.</p>
                </div>
              )}
            </div>
          );
        })}
      </div>
      {any && w && <p className="notes">Hover the map for the values at a spot. Weather: {w.source}.</p>}
    </Group>
  );
}

export function Forecast() {
  const visible = useApp(s => s.forecastVisible);
  const d = useApp(s => s.forecast);
  const day = useApp(s => s.day);
  const zone = useApp(s => s.zone);
  const loaded = useApp(s => s.loaded);
  const { showDay, changeZone, zoomToZone, flyToHotspot } = useApp.getState();

  if (!d) return null;                                   // (keeps a second, hidden #changeZone out of the page)
  const f = d?.forecast?.[day];
  const bt = d?.backtest, m = d?.models?.[0];
  const obs: number[] = d?.observed_cells_per_day ?? [];
  const source = loaded?.mode === 'harmonise' ? 'All sensors (harmonised)' : loaded?.sensor ?? '';
  const notes: string[] = [];
  if (d?.detections_skipped_non_vegetation) notes.push(`${num(d.detections_skipped_non_vegetation)} detections ignored (volcanoes, industrial heat sources, offshore).`);
  if (d?.cells_truncated) notes.push('Only the highest-risk cells are drawn on the map.');

  return (
    <section id="forecastSection" hidden={!visible || !d}>
      <div className="fhead">
        <h2>Fire forecast</h2>
        <button type="button" className="secondary" id="changeZone" onClick={changeZone}>
          <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round"
               strokeLinejoin="round" aria-hidden="true"><path d="M19 12H5M11 6l-6 6 6 6" /></svg>
          Change zone</button>
      </div>

      {d && <>
        {/* About this forecast */}
        <Group title="About this forecast"
               action={<button type="button" className="secondary small" id="zoomZone" onClick={() => zoomToZone(1500)}>Zoom to zone</button>}>
          <dl className="meta">
            <dt>Zone</dt><dd>{d.country ? <><b>{d.country}</b> (inside its borders)</> : zone ? `${zone.w}, ${zone.s} → ${zone.e}, ${zone.n}` : '—'}</dd>
            <dt>Data</dt><dd>{shortDate(d.range[0])} – {shortDate(d.range[1])}</dd>
            <dt>Source</dt><dd>{source}</dd>
            <dt>Grid</dt><dd>{d.cell_km} km cells</dd>
          </dl>
        </Group>

        {/* The forecast itself */}
        <Group title="Forecast">
          <div className="days" id="dayTabs" role="tablist" aria-label="Forecast day">
            {d.forecast.map((fd: { date: string }, i: number) => (
              <button key={i} type="button" role="tab" aria-selected={i === day} onClick={() => showDay(i)}>
                +{i + 1} day{i ? 's' : ''}<small>{shortDate(fd.date)}</small>
              </button>
            ))}
          </div>
          {f && (
            <div className="pstats" id="pstats">
              <div className="pstat" title="Expected number of grid cells burning near current fires">
                <b>~{num(Math.round(f.expected_cells))}</b>Expected burning cells</div>
              <div className="pstat" title="Cells with more than a 50% chance of burning">
                <b>{num(f.likely_cells)}</b>Likely to burn (&gt;50%)</div>
              <div className="pstat" title="New fires expected far from any current fire, where the model can't place them">
                <b>{f.expected_new_elsewhere == null ? 'n/a' : '~' + num(Math.round(f.expected_new_elsewhere))}</b>New fires elsewhere</div>
            </div>
          )}
          <RiskHistogram probs={(d.cells ?? []).map((c: { p: number[] }) => c.p[day] ?? 0)} />
        </Group>

        {/* Weather overlay switch + summary for the selected day */}
        <WeatherGroup />

        {/* Fire activity: the observed days flowing into the forecast */}
        <Group title="Fire activity · past and forecast">
          <p className="gsub">Burning {d.cell_km} km cells per day</p>
          <Timeline observed={obs} start={d.range[0]} forecast={d.forecast} day={day} onPick={showDay} />
          <div className="spread-row" id="facts">
            <Compass bearing={d.spread_direction?.bearing_deg ?? null} compass={d.spread_direction?.compass ?? null} />
            <dl className="meta">
              <dt>Spread</dt>
              <dd>{d.spread_direction
                ? <>Towards the <b>{d.spread_direction.compass}</b> ({d.spread_direction.bearing_deg}°)</>
                : 'No consistent direction'}</dd>
              <dt>Trend</dt>
              <dd>{d.trend_last_day == null ? 'n/a'
                : <><span className="trend" aria-hidden="true">{d.trend_last_day >= 0 ? '▲' : '▼'}</span>
                    {d.trend_last_day >= 0 ? 'Rose' : 'Fell'} {pct(Math.abs(d.trend_last_day))} on the last day</>}</dd>
            </dl>
          </div>
          {notes.length > 0 && <p className="notes">{notes.join(' ')}</p>}
        </Group>

        {/* Where new fire is most likely */}
        {!!d.hotspots?.length && (
          <Group title="Most likely new fire · tomorrow">
            <div id="hotspots">
              {d.hotspots.map((h: { lat: number; lon: number; p: number }, i: number) => (
                <button key={i} type="button" className="hot" onClick={() => flyToHotspot(h.lat, h.lon)}
                        title="Fly to this place">
                  <span>{h.lat.toFixed(2)}, {h.lon.toFixed(2)}</span><MiniBar p={h.p} /><b>{pct(h.p)}</b>
                </button>
              ))}
            </div>
          </Group>
        )}

        {/* How reliable is this? Measured on this zone by predicting the last loaded day. */}
        <Group title="How reliable is this?">
          <div id="reliab">
            {!m?.fitted
              ? <p className="warn">⚠ Too few fires in this zone and time window to train the model, so it uses default settings. Treat the forecast as a rough guide.</p>
              : !bt
              ? <p className="gsub">Load at least 3 days of data to measure how accurate the forecast is for this zone.</p>
              : (!bt.model_fitted || bt.auc == null)
              ? <p className="gsub">Accuracy check on {shortDate(bt.day)}: not enough fires to score it.</p>
              : <>
                  <p className="gsub">Trained without {shortDate(bt.day)}, then asked to predict it. Share of its top {num(bt.top_k)} cells that really burned:</p>
                  <CompareBars rows={[
                    { name: 'This model', v: bt.precision },
                    { name: 'Same as yesterday', v: bt.baseline_precision ?? 0, muted: true },
                  ]} />
                  <div className="gauge-row">
                    <RankingGauge auc={bt.auc} label={aucLabel(bt.auc)} />
                    <p className="gsub" style={{ margin: 0 }}>
                      <b className="strong">Ranking score: {aucLabel(bt.auc)}</b><br />
                      How well it ranks the cells that burned above those that didn't. 0.5 is guessing, 1 is perfect.
                    </p>
                  </div>
                </>}
          </div>
        </Group>
      </>}
    </section>
  );
}
