// App state and actions (Zustand). The React components read this state; the actions update it and drive
// the map engine. The logic follows the original app.html step for step.

import { create } from 'zustand';
import * as engine from './engine';
import { getJSON } from './api';
import { countryById, countryZone, placesFor, territoryAt } from './countries';
import { buildTimeline, type TimelineData } from './timeline';
import { featherPx, heatField, heatFrame, type HeatField } from './riskHeat';
import { shapeMask } from './mapImage';
import type { WeatherVar } from './weatherHeat';
import { areaKm2, EMPTY, normalise, r2, zoneAroundPoint, type FC, type Zone } from './geo';
import { num, shortDate } from './format';

export type Method = 'draw' | 'pin' | 'coords' | 'country';
export type Mode = 'harmonise' | 'fetch';
type Params = { date: string; days: string; area: string };
type Loaded = { mode: Mode; sensor: string; params: Params };
export type Msg = { text: string; kind: 'info' | 'ok' | 'error' } | null;
type Availability = Record<string, { min: string; max: string }>;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type Forecast = any;
export type WeatherDay = { temp: number | null; rh: number | null; wind: number | null; dir: number | null;
                           gust: number | null; rain: number | null };
export type WeatherSummary = { date: string; available: boolean; temp_max?: number; rh_min?: number; wind_max?: number;
                               gust_max?: number | null; wind_from?: number | null; wind_from_compass?: string | null;
                               rain_mean?: number; rating?: 'low' | 'moderate' | 'high' };
export type Weather = { dates: string[]; nx: number; ny: number; bbox: [number, number, number, number]; points: { lat: number; lon: number; days: WeatherDay[] }[];
                        summary: WeatherSummary[]; source: string };

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T | null;
const body = () => document.body.classList;
const COORD_KEYS = ['w', 's', 'e', 'n'] as const;
const INTRO_MS = 3000;
const afterRender = (fn: () => void) => requestAnimationFrame(() => requestAnimationFrame(fn));
let introTimer: ReturnType<typeof setTimeout> | undefined;

interface AppState {
  // Opening screen / panel
  splashLeaving: boolean;
  splashGone: boolean;
  splashReturning: boolean;     // the opening screen came back via the Home button (it fades back in)
  panelOpen: boolean;
  // Zone
  method: Method;
  drawing: boolean;
  pinMode: boolean;
  zone: Zone | null;
  pinAt: { lat: number; lng: number } | null;
  zoneCountry: string | null;                   // the country's name, when the zone was picked by clicking a country
  zoneCountryId: number | null;                 // ...and its id (the forecast is then limited to its borders)
  zoneTerritory: number | null;                 // ...and which of its territories (France: mainland, French Guiana...)
  countries: 'idle' | 'loading' | 'ready' | 'error';   // the country shapes for that method
  pinError: boolean;
  // Time window + data availability
  date: string;
  days: string;
  sensors: string[];
  availability: Availability;
  // Analyse window
  modalOpen: boolean;
  mode: Mode;
  sensor: string;
  loaded: Loaded | null;
  lastFires: FC | null;
  loading: boolean;
  loadMsg: Msg;
  predicting: boolean;
  predictMsg: Msg;
  // Forecast
  forecast: Forecast | null;
  forecastSub: string;
  forecastVisible: boolean;
  day: number;
  cameraBeforePredict: ReturnType<typeof engine.cameraSnapshot> | null;
  // Weather overlay (Open-Meteo, via /api/weather): the switch in the forecast panel, and the data for the
  // forecast days of the current prediction (`weatherKey` says which zone and days it belongs to).
  weatherLayers: Record<WeatherVar, boolean>;  // which weather layers are on (each has its own switch)
  weather: Weather | null;
  weatherKey: string | null;
  weatherLoading: boolean;
  weatherError: string | null;
  // Misc
  status: string;
  firesShown: boolean;

  start: () => void;
  setPanel: (open: boolean) => void;
  goHome: () => void;
  showGlobe: () => void;
  openFire: () => void;
  setMethod: (m: Method) => void;
  toggleDrawing: () => void;
  togglePinMode: () => void;
  selectZone: (z: Zone, fit?: boolean, fromPin?: boolean, country?: { id: number; name: string; part: number | null }) => void;
  selectCountry: (id: number, part?: number | null, at?: [number, number]) => void;
  // The territory box: go to another territory of the picked country or of its family (France → New Caledonia...);
  // with a forecast on screen, it's re-run there.
  switchTerritory: (id: number, part: number | null) => Promise<void>;
  switchingTo: string | null;                   // the territory being loaded from the box, if any
  switchError: string | null;
  clearZone: () => void;
  coordsCommitted: () => void;
  readPinArea: () => number | null;
  applyPin: (fit: boolean) => void;
  setDate: (v: string) => void;
  setDays: (v: string) => void;
  loadSensors: () => void;
  openModal: () => void;
  closeModal: () => void;
  setMode: (m: Mode) => void;
  setSensor: (s: string) => void;
  loadData: () => Promise<void>;
  predict: () => Promise<void>;
  analyse: () => Promise<void>;
  showDay: (i: number, fromTimeline?: boolean) => void;
  // Fire timeline (animated heatmap + the bar under the legend)
  timeline: Pick<TimelineData, 'frames' | 'events'> | null;
  timelineBuilding: boolean;
  tlFrame: number;                              // the frame on screen
  tlPlaying: boolean;
  tlSeek: { frame: number; n: number } | null;  // a jump requested from outside the bar (day tabs, checkpoints)
  buildTimeline: () => Promise<void>;
  setTlFrame: (i: number) => void;
  setTlPlaying: (on: boolean) => void;
  seekTimeline: (i: number) => void;
  clearTimeline: () => void;
  toggleWeatherLayer: (v: WeatherVar) => void;
  loadWeather: () => Promise<void>;
  zoomToZone: (duration?: number) => void;
  flyToHotspot: (lat: number, lon: number) => void;
  changeZone: () => void;
  invalidateData: () => void;
  clearPrediction: () => void;
  showForecastPanel: (on: boolean) => void;
}

const currentParams = (z: Zone, date: string, days: string): Params =>
  ({ date, days, area: [z.w, z.s, z.e, z.n].join(',') });
const sameParams = (a?: Params | null, b?: Params | null) =>
  !!a && !!b && a.date === b.date && a.days === b.days && a.area === b.area;

// Prepares the detections from either endpoint (/api/harmonise has `source` + `frp_mw`, /api/fires has `frp`)
// as map dots. They are only put on the map when Predict is clicked, together with the legend and the forecast.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function prepareFires(d: any, sensor: string): FC {
  const c: string[] = d.columns, col = (name: string) => c.indexOf(name);
  const li = col('latitude'), lo = col('longitude'), di = col('acq_date'), ti = col('acq_time'),
        si = col('source'), ci = col('confidence'), fi = col('frp_mw') >= 0 ? col('frp_mw') : col('frp');
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const features = d.rows.map((r: any[]) => ({ type: 'Feature',
    geometry: { type: 'Point', coordinates: [+r[lo], +r[li]] },
    properties: { frp: fi >= 0 ? parseFloat(r[fi]) || 0 : 0, source: si >= 0 ? r[si] : sensor,
                  date: r[di], time: String(r[ti]).padStart(4, '0'), conf: ci >= 0 ? r[ci] : '' } }));
  return { type: 'FeatureCollection', features };
}

export const useApp = create<AppState>((set, get) => ({
  splashLeaving: false,
  splashGone: false,
  splashReturning: false,
  panelOpen: false,
  method: 'draw',
  drawing: false,
  pinMode: false,
  zone: null,
  pinAt: null,
  zoneCountry: null,
  zoneCountryId: null,
  zoneTerritory: null,
  switchingTo: null,
  switchError: null,
  countries: 'idle',
  pinError: false,
  // Default window: the 5 days ending yesterday (UTC). Today's near-real-time data is still arriving, so a window
  // ending today sees a nearly fire-free last day and the forecast comes out almost empty.
  date: new Date(Date.now() - 5 * 864e5).toISOString().slice(0, 10),
  days: '5',
  sensors: [],
  availability: {},
  modalOpen: false,
  mode: 'harmonise',
  sensor: '',
  loaded: null,
  lastFires: null,
  loading: false,
  loadMsg: null,
  predicting: false,
  predictMsg: null,
  forecast: null,
  forecastSub: '',
  forecastVisible: false,
  day: 0,
  cameraBeforePredict: null,
  weatherLayers: { temp: false, rh: false, wind: false },
  timeline: null,
  timelineBuilding: false,
  tlFrame: 0,
  tlPlaying: false,
  tlSeek: null,
  weather: null,
  weatherKey: null,
  weatherLoading: false,
  weatherError: null,
  status: '',
  firesShown: false,

  // ---------------- Opening screen → app ----------------
  start: () => {
    const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
    // On phones the panel simply sits under the map, open.
    if (!engine.isDesktop()) { set({ panelOpen: true }); body().remove('collapsed'); }
    body().add('intro');                                   // the menu bar's first appearance waits for the opening screen
    set({ splashLeaving: true });
    body().remove('splash');
    engine.introGlide(get().panelOpen);                    // globe glides to the centre and grows
    introTimer = setTimeout(finishIntro, reduced ? 0 : INTRO_MS);
  },

  // Open: the panel slides in from the left and the pill fades away. Close: the reverse.
  setPanel: (open) => {
    if (!engine.isDesktop() || body().contains('splash')) return;
    // The first click often comes while the opening transition is still finishing (the menu appears ~1.5 s
    // after "Get started", the transition ends at 3 s). End it now, or the fading opening screen would still
    // cover the panel as it slides in.
    finishIntro();
    set({ panelOpen: open });
    body().toggle('collapsed', !open);
    // Keep keyboard focus somewhere visible.
    if (open) $('appHead')?.focus({ preventScroll: true }); else $('navFire')?.focus({ preventScroll: true });
    engine.panelChanged(open);
  },

  // ---------------- Menu bar: Home · Globe · Fire ----------------
  // Home: back to the opening screen. It fades back in (text rising as on first load), the fire tint returns,
  // the globe glides back to the right and shrinks to its opening size, and the side panel slides away.
  // Everything is reset as on first entering the app: no zone, no forecast, and the default settings.
  goHome: () => {
    if (body().contains('splash')) return;
    finishIntro();
    if (engine.isDrawing()) engine.setDrawing(false);
    if (engine.isPinMode()) engine.setPinMode(false);
    get().clearTimeline();
    get().clearZone();                                    // zone, pin, loaded data, forecast and map layers
    set({
      method: 'draw', date: new Date(Date.now() - 5 * 864e5).toISOString().slice(0, 10), days: '5',
      mode: 'harmonise', sensor: get().sensors[0] || '',
      loadMsg: null, predictMsg: null, forecast: null, forecastSub: '', day: 0, cameraBeforePredict: null,
      weatherLayers: { temp: false, rh: false, wind: false }, weatherError: null,
      switchError: null, pinError: false, status: '',
    });
    set({ modalOpen: false, splashGone: false, splashLeaving: false, splashReturning: true });
    if (engine.isDesktop()) { set({ panelOpen: false }); body().add('collapsed'); }
    body().add('splash');
    engine.goHome();
    afterRender(() => $('start')?.focus({ preventScroll: true }));
  },

  // Globe: just the globe, spinning in the middle of the screen (closes the panel; zooms back out if needed).
  showGlobe: () => {
    if (body().contains('splash')) return;
    finishIntro();
    if (engine.isDrawing()) engine.setDrawing(false);
    if (engine.isPinMode()) engine.setPinMode(false);
    set({ modalOpen: false });
    if (engine.isDesktop() && get().panelOpen) { set({ panelOpen: false }); body().add('collapsed'); }
    engine.globeView();
  },

  // Fire: open the Fire Pattern Predictor panel (on phones it's always open below the map: scroll to it).
  openFire: () => {
    if (!engine.isDesktop()) { $('panel')?.scrollIntoView({ behavior: 'smooth', block: 'start' }); return; }
    if (!get().panelOpen) get().setPanel(true);
  },

  // ---------------- Zone ----------------
  setMethod: (m) => {
    if (engine.isDrawing()) engine.setDrawing(false);     // don't leave a half-started draw/pin mode active
    if (engine.isPinMode()) engine.setPinMode(false);
    set({ method: m });
    // "Pick a country" needs the country shapes: load them the first time.
    if (m === 'country' && get().countries !== 'ready' && get().countries !== 'loading') {
      set({ countries: 'loading' });
      engine.ensureCountries().then(() => set({ countries: 'ready' }), () => set({ countries: 'error' }));
    }
  },
  toggleDrawing: () => engine.setDrawing(!engine.isDrawing()),
  togglePinMode: () => engine.setPinMode(!engine.isPinMode()),

  // `fromPin`: the zone was built around the map pin; any other way of setting the zone removes the pin.
  // `country`: the zone is a clicked country's extent (its border is shown with the zone).
  selectZone: (z, fit = false, fromPin = false, country) => {
    if (!fromPin) engine.removePin();
    engine.setSelectedCountry(country?.id ?? null);
    const zone = normalise(z);
    if (zone.w === zone.e || zone.s === zone.n) { get().clearZone(); return; }
    get().invalidateData();
    engine.showZone(zone);
    COORD_KEYS.forEach(k => { const el = $<HTMLInputElement>(k); if (el) el.value = String(zone[k]); });
    const appearing = !get().zone;
    set({ zone, pinAt: fromPin ? engine.pinLngLat() : null, zoneCountry: country?.name ?? null, zoneCountryId: country?.id ?? null,
          zoneTerritory: country?.part ?? null });
    // "Analyse zone" only exists once there is a zone; bring it into view when it first appears.
    if (appearing && document.activeElement?.tagName !== 'INPUT') {
      afterRender(() => $('run')?.scrollIntoView({ block: 'nearest', behavior: 'smooth' }));
    }
    // Flying in from the globe passes through the globe→flat transition on the way.
    // (a country with territories shows the territory box at the bottom: keep the zone clear of it)
    const hasBox = !!(country && placesFor(country.id, country.part));
    if (fit) engine.fitZone(zone, hasBox ? { top: 40, left: 40, right: 40, bottom: 160 } : 40, 2800);
  },

  // "Pick a country": the zone becomes the rectangle the country fills, and the map flies there. A country with
  // far-off territories is picked one territory at a time: the one clicked on the map (`at`), the one chosen in
  // the territory box (`part`), or else its main one.
  selectCountry: (id, part, at) => {
    const c = countryById(id);
    if (!c) return;
    const k = !c.parts?.length ? null : part ?? (at ? territoryAt(c, at[0], at[1]) : 0) ?? 0;
    const tName = k != null && c.parts![k].name !== c.name ? c.parts![k].name : null;
    get().selectZone(countryZone(c, k), true, false, { id, name: tName ? `${c.name} · ${tName}` : c.name, part: k });
  },

  switchTerritory: async (id, part) => {
    const { zoneCountryId, forecastVisible, zoneTerritory } = get();
    const c = countryById(id);
    if (!c || (part != null && !c.parts?.[part]) || get().switchingTo) return;
    if (id === zoneCountryId && part === zoneTerritory && forecastVisible) return;
    set({ switchError: null });
    get().selectCountry(id, part);                      // new zone + fly there (clears the old forecast)
    if (!forecastVisible) return;
    // A forecast was on screen: run the same analysis for this territory, so the box hops straight between them.
    set({ switchingTo: part != null ? c.parts![part].name : c.name });
    try {
      await get().analyse();
      const err = get().predictMsg?.kind === 'error' ? get().predictMsg!.text : get().loadMsg?.kind === 'error' ? get().loadMsg!.text : null;
      if (err) set({ switchError: err });
    } finally {
      set({ switchingTo: null });
    }
  },

  clearZone: () => {
    engine.removePin();
    engine.setSelectedCountry(null);
    set({ zone: null, pinAt: null, zoneCountry: null, zoneCountryId: null, zoneTerritory: null, loaded: null, lastFires: null, status: '' });
    get().clearPrediction();
    engine.showZone(null);
    engine.setFires(EMPTY);
    COORD_KEYS.forEach(k => { const el = $<HTMLInputElement>(k); if (el) el.value = ''; });
  },

  coordsCommitted: () => {
    const v = Object.fromEntries(COORD_KEYS.map(k => [k, parseFloat($<HTMLInputElement>(k)?.value ?? '')])) as Zone;
    if (Object.values(v).every(Number.isFinite)) get().selectZone(v, true);
  },

  readPinArea: () => {
    const a = parseFloat($<HTMLInputElement>('pinArea')?.value ?? '');
    return Number.isFinite(a) && a >= 10 && a <= 5e6 ? a : null;
  },

  applyPin: (fit) => {
    const area = get().readPinArea();
    // Validate under the area box itself, so the current zone's details stay visible.
    set({ pinError: area == null });
    const at = engine.pinLngLat();
    if (!at || area == null) return;
    get().selectZone(zoneAroundPoint(at, area), fit, true);
  },

  // ---------------- Time window ----------------
  setDate: (v) => { set({ date: v }); get().invalidateData(); },
  setDays: (v) => { set({ days: v }); get().invalidateData(); },

  loadSensors: () => {
    fetch('/api/sensors').then(r => r.json()).then(d => {
      const sensors: string[] = d.sensors || [];
      set({ availability: d.availability || {}, sensors, sensor: sensors[0] || '' });
    }).catch(() => {});
  },

  // ---------------- Analyse window: step 1 get data, step 2 predict ----------------
  openModal: () => {
    const { zone, date, days, loaded } = get();
    if (!zone) return;
    if (!date) { $<HTMLInputElement>('date')?.reportValidity(); return; }
    engine.showZone(zone);                                // outline back on the map (Change zone hides it)
    if (!sameParams(loaded?.params, currentParams(zone, date, days))) {
      set({ loaded: null, loadMsg: null });
      get().clearPrediction();
    }
    set({ modalOpen: true });
    afterRender(() => $('predict')?.focus());
  },
  closeModal: () => { set({ modalOpen: false }); afterRender(() => $('run')?.focus()); },

  setMode: (m) => set({ mode: m }),
  setSensor: (s) => set({ sensor: s, mode: 'fetch' }),

  loadData: async () => {
    const { mode, sensor, zone, date, days } = get();
    if (!zone) return;
    if (mode === 'fetch' && !sensor) {
      set({ loadMsg: { text: 'Sensor list not loaded yet: is the server running?', kind: 'info' } });
      return;
    }
    const params = currentParams(zone, date, days);
    set({ loading: true, loaded: null, lastFires: null });
    engine.setFires(EMPTY);
    get().clearPrediction();
    set({ loadMsg: { kind: 'info', text: mode === 'harmonise'
      ? 'Fetching every sensor and harmonising… this can take a little while.' : `Fetching ${sensor}…` } });
    try {
      const d = mode === 'harmonise'
        ? await getJSON('/api/harmonise?' + new URLSearchParams(params))
        : await getJSON('/api/fires?' + new URLSearchParams({ ...params, sensor }));
      const fires = prepareFires(d, mode === 'fetch' ? sensor : '');
      const n = fires.features.length;
      const what = mode === 'harmonise'
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        ? `${(d.sources || []).filter((s: any) => s.status === 'ok' && s.count).length} sensors harmonised`
        : sensor;
      set({
        lastFires: fires,
        loaded: { mode, sensor, params },
        loadMsg: { kind: 'ok', text: `✓ ${n.toLocaleString()} fire detections loaded (${what}).` },
        status: `${n.toLocaleString()} detections loaded (${what}).`,
      });
    } catch (err) {
      set({ loadMsg: { kind: 'error', text: (err as Error).message } });
    } finally {
      set({ loading: false });
    }
  },

  predict: async () => {
    const { loaded, zone } = get();
    if (!loaded || !zone) return;
    // Remember where the map was when Predict was clicked, so "Change zone" can fly back there.
    set({ cameraBeforePredict: engine.cameraSnapshot(), predicting: true,
          predictMsg: { kind: 'info', text: 'Running prediction…' } });
    get().clearTimeline();
    engine.setRisk(EMPTY);
    engine.setWeather(null);
    try {
      const countryId = get().zoneCountryId, part = get().zoneTerritory;
      const d = await getJSON('/api/predict?' + new URLSearchParams({ ...loaded.params, source: loaded.mode,
        ...(loaded.mode === 'fetch' ? { sensor: loaded.sensor } : {}),
        ...(countryId != null ? { country: String(countryId) } : {}),
        ...(countryId != null && part != null ? { part: String(part) } : {}) }));
      set({ predictMsg: { kind: 'ok', text: `✓ ${d.forecast.length}-day forecast ready.` } });
      // Close the window, fly to the zone (fires + forecast are on the map) and show the details in the panel.
      get().closeModal();
      // A country's forecast: fire heat, dots and weather are drawn only inside its borders.
      engine.setClipCountry(d.country && countryId != null ? countryId : null, part);
      const { lastFires } = get();
      if (lastFires) engine.setFires(lastFires);           // fire dots + legend appear only now
      // Map: one square per grid cell, carrying the probability for each forecast day.
      engine.setRisk({ type: 'FeatureCollection',
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        features: d.cells.map((c: any) => ({ type: 'Feature',
          properties: { p1: c.p[0], p2: c.p[1] ?? 0, p3: c.p[2] ?? 0, now: c.burning_now },
          geometry: { type: 'Polygon', coordinates: [[[c.w, c.s], [c.e, c.s], [c.e, c.n], [c.w, c.n], [c.w, c.s]]] } })) });
      const src = loaded.mode === 'harmonise' ? 'all sensors harmonised' : loaded.sensor;
      set({ forecast: d, forecastSub: `Zone ${zone.w}, ${zone.s} → ${zone.e}, ${zone.n} · data ${shortDate(d.range[0])}–${shortDate(d.range[1])} · ${src} · ${d.cell_km} km grid` });
      get().showDay(0);
      get().showForecastPanel(true);
      get().zoomToZone();
      get().buildTimeline();                             // the animated heatmap + its bar
      if (weatherOn(get())) get().loadWeather();         // weather layers are on: bring them for the new forecast
    } catch (err) {
      set({ predictMsg: { kind: 'error', text: (err as Error).message } });
    } finally {
      set({ predicting: false });
    }
  },

  // The single "Predict" button: load the data first unless exactly this data (zone, dates, source) is
  // already loaded, then run the prediction.
  analyse: async () => {
    const { loaded, mode, sensor, zone, date, days } = get();
    if (!zone) return;
    const needLoad = !loaded || !sameParams(loaded.params, currentParams(zone, date, days)) ||
                     loaded.mode !== mode || (mode === 'fetch' && loaded.sensor !== sensor);
    if (needLoad) {
      await get().loadData();
      if (!get().loaded) return;                         // loading failed: its error message stays visible
    }
    await get().predict();
  },

  showDay: (i, fromTimeline = false) => {
    const f = get().forecast?.forecast?.[i];
    if (!f) return;
    // Picking a day tab moves the timeline to that forecast day (and pauses it).
    const tl = get().timeline;
    if (!fromTimeline && tl) {
      const k = tl.frames.findIndex(fr => fr.kind === 'forecast' && fr.index === i);
      if (k >= 0) { get().seekTimeline(k); return; }
    }
    if (get().day === i && fromTimeline) return;
    engine.setRiskDay(i);
    const { weather, weatherLayers } = get();
    if (weatherOn(get()) && weather) engine.setWeather(weather, i, weatherLayers);
    set({ day: i, status: `Showing the fire forecast for ${shortDate(f.date)} (+${i + 1} day${i ? 's' : ''}) on the map.` });
  },

  // ---------------- Fire timeline ----------------
  // Built after each prediction: one heatmap image per day (observed days, then forecast days) and checkpoints.
  buildTimeline: async () => {
    const { forecast, lastFires, zone } = get();
    if (!forecast || !zone) return;
    set({ timelineBuilding: true });
    const data = buildTimeline(forecast, lastFires ?? EMPTY, (zone.e - zone.w) * (zone.n - zone.s), engine.getClipInside());
    if (!data) { set({ timelineBuilding: false }); return; }
    // One grid of heat values per day on a shared frame (and the country's soft-edged mask, if limited to one).
    // The longer side is 720 px: plenty after the map's smoothing, and quick to recolour on every animation tick.
    const frame = heatFrame(data.boxes.flat(), data.extent, 720);
    const fields: HeatField[] = [];
    for (const b of data.boxes) {
      fields.push(heatField(b, frame));
      await new Promise(r => setTimeout(r, 0));       // keep the page responsive while the days are computed
      if (get().forecast !== forecast) return;         // a newer prediction (or Change zone) took over
    }
    const clipGeom = engine.getClipGeometry();
    const mask = clipGeom ? shapeMask(frame, clipGeom, featherPx(data.boxes.flat(), frame)) : null;
    engine.setTimeline({ fields, frame, mask });
    const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
    set({ timeline: { frames: data.frames, events: data.events }, timelineBuilding: false, tlFrame: 0,
          tlPlaying: !reduced, tlSeek: { frame: 0, n: (get().tlSeek?.n ?? 0) + 1 } });
    get().setTlFrame(0);
  },
  // The bar reports each new frame: show that day's fire dots, and follow forecast days in the panel too.
  setTlFrame: (i) => {
    const fr = get().timeline?.frames[i];
    if (!fr) return;
    set({ tlFrame: i });                                // (the bar fades the fire dots between days itself)
    if (fr.kind === 'forecast') get().showDay(fr.index, true);
  },
  setTlPlaying: (on) => set({ tlPlaying: on }),
  seekTimeline: (i) => set({ tlPlaying: false, tlSeek: { frame: i, n: (get().tlSeek?.n ?? 0) + 1 } }),
  clearTimeline: () => {
    engine.setTimeline(null);
    engine.setFireDay(null);
    set({ timeline: null, timelineBuilding: false, tlPlaying: false, tlFrame: 0 });
  },

  // Weather switch in the forecast panel: shows / hides the weather markers for the forecast day on the map.
  // A weather layer's switch in the forecast panel: shows / hides that layer for the forecast day on the map.
  toggleWeatherLayer: (v) => {
    const layers = { ...get().weatherLayers, [v]: !get().weatherLayers[v] };
    set({ weatherLayers: layers, weatherError: null });
    if (!Object.values(layers).some(Boolean)) engine.setWeather(null);
    else get().loadWeather();                          // reuses the loaded weather when it's for this forecast
  },
  // Weather for the current zone on the forecast days (reused if already loaded for the same zone and days).
  loadWeather: async () => {
    const { forecast, zone } = get();
    const first = forecast?.forecast?.[0]?.date;
    if (!zone || !first) return;
    const params = { area: [zone.w, zone.s, zone.e, zone.n].join(','), start: first, days: String(forecast.forecast.length) };
    const key = new URLSearchParams(params).toString();
    if (get().weather && get().weatherKey === key) { engine.setWeather(get().weather, get().day, get().weatherLayers); return; }
    set({ weatherLoading: true, weatherError: null });
    try {
      const w: Weather = await getJSON('/api/weather?' + key);
      set({ weather: w, weatherKey: key });
      // Only draw it if it's still wanted and still for the forecast on screen.
      const now = get();
      if (weatherOn(now) && now.forecastVisible && now.forecast?.forecast?.[0]?.date === first) engine.setWeather(w, now.day, now.weatherLayers);
    } catch (err) {
      set({ weatherError: (err as Error).message });
    } finally {
      set({ weatherLoading: false });
    }
  },

  // Zoom in until the zone fills the map edge to edge (a camera move only: you can still zoom out or pan).
  zoomToZone: (duration = 2600) => {
    const { zone, zoneCountryId, zoneTerritory } = get();
    // (with the territory box at the bottom, keep the zone clear of it)
    const hasBox = zoneCountryId != null && !!placesFor(zoneCountryId, zoneTerritory);
    if (zone) engine.fitZone(zone, hasBox ? { top: 0, left: 0, right: 0, bottom: 140 } : 0, duration);
  },
  flyToHotspot: (lat, lon) => { engine.stopSpin(); engine.flyTo({ center: [lon, lat], zoom: 9, duration: 2400 }); },

  // Back to the zone controls: clear the map (fire dots, legend, forecast squares and the zone outline), clear the
  // selected zone so a new one can be picked straight away, and fly back to where the map was when Predict was clicked.
  changeZone: () => {
    get().showForecastPanel(false);
    get().clearTimeline();
    get().clearZone();                                  // also drops the loaded data and the forecast
    set({ status: '' });
    const cam = get().cameraBeforePredict;
    if (cam) engine.flyTo({ ...cam, duration: 2600, essential: true });
  },

  // Loaded data no longer matches once the zone or time window changes.
  invalidateData: () => {
    if (!get().loaded) return;
    set({ loaded: null, lastFires: null, status: '' });
    engine.setFires(EMPTY);
    get().clearPrediction();
  },
  clearPrediction: () => {
    get().clearTimeline();
    engine.setClipCountry(null);
    engine.setRisk(EMPTY);
    engine.setWeather(null);
    get().showForecastPanel(false);
    set({ predictMsg: null });
  },
  // The forecast takes the place of part 1 (Prediction zone) in the side panel.
  showForecastPanel: (on) => {
    set({ forecastVisible: on });
    if (on) afterRender(() => $('forecastSection')?.scrollIntoView({ block: 'start', behavior: 'smooth' }));
  },
}));

// Is any weather layer switched on?
export const weatherOn = (s: { weatherLayers: Record<WeatherVar, boolean> }) => Object.values(s.weatherLayers).some(Boolean);

// End of the opening transition: remove the (now invisible) opening screen and the intro-only CSS state.
function finishIntro() {
  clearTimeout(introTimer);
  if (!useApp.getState().splashLeaving || useApp.getState().splashGone) return;
  useApp.setState({ splashGone: true });
  body().remove('intro');
}

// Text for the "Selected zone" box.
export function zoneSummary(zone: Zone, pinAt: { lat: number; lng: number } | null, country: string | null = null) {
  return {
    coords: `${zone.w}, ${zone.s} → ${zone.e}, ${zone.n}`,
    size: `≈ ${Math.round(areaKm2(zone)).toLocaleString()} km² · ${r2(zone.e - zone.w)}° × ${r2(zone.n - zone.s)}°`,
    pin: pinAt ? `Centred on the pin at ${pinAt.lat.toFixed(3)}, ${pinAt.lng.toFixed(3)}` : null,
    country: country ? `The forecast covers ${country} only, inside its borders.` : null,
  };
}

export { num };
