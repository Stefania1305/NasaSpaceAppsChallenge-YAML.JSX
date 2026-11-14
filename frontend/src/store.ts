// App state and actions (Zustand). The React components read this state; the actions update it and drive
// the map engine. The logic follows the original app.html step for step.

import { create } from 'zustand';
import * as engine from './engine';
import { getJSON } from './api';
import { areaKm2, EMPTY, normalise, r2, zoneAroundPoint, type FC, type Zone } from './geo';
import { num, shortDate } from './format';

export type Method = 'draw' | 'pin' | 'coords' | 'view';
export type Mode = 'harmonise' | 'fetch';
type Params = { date: string; days: string; area: string };
type Loaded = { mode: Mode; sensor: string; params: Params };
export type Msg = { text: string; kind: 'info' | 'ok' | 'error' } | null;
type Availability = Record<string, { min: string; max: string }>;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type Forecast = any;

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
  selectZone: (z: Zone, fit?: boolean, fromPin?: boolean) => void;
  clearZone: () => void;
  coordsCommitted: () => void;
  useCurrentView: () => void;
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
  showDay: (i: number) => void;
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
  pinError: false,
  date: new Date(Date.now() - 4 * 864e5).toISOString().slice(0, 10),
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
  goHome: () => {
    if (body().contains('splash')) return;
    finishIntro();
    if (engine.isDrawing()) engine.setDrawing(false);
    if (engine.isPinMode()) engine.setPinMode(false);
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
  },
  toggleDrawing: () => engine.setDrawing(!engine.isDrawing()),
  togglePinMode: () => engine.setPinMode(!engine.isPinMode()),

  // `fromPin`: the zone was built around the map pin; any other way of setting the zone removes the pin.
  selectZone: (z, fit = false, fromPin = false) => {
    if (!fromPin) engine.removePin();
    const zone = normalise(z);
    if (zone.w === zone.e || zone.s === zone.n) { get().clearZone(); return; }
    get().invalidateData();
    engine.showZone(zone);
    COORD_KEYS.forEach(k => { const el = $<HTMLInputElement>(k); if (el) el.value = String(zone[k]); });
    const appearing = !get().zone;
    set({ zone, pinAt: fromPin ? engine.pinLngLat() : null });
    // "Analyse zone" only exists once there is a zone; bring it into view when it first appears.
    if (appearing && document.activeElement?.tagName !== 'INPUT') {
      afterRender(() => $('run')?.scrollIntoView({ block: 'nearest', behavior: 'smooth' }));
    }
    // Flying in from the globe passes through the globe→flat transition on the way.
    if (fit) engine.fitZone(zone, 40, 2500);
  },

  clearZone: () => {
    engine.removePin();
    set({ zone: null, pinAt: null, loaded: null, lastFires: null, status: '' });
    get().clearPrediction();
    engine.showZone(null);
    engine.setFires(EMPTY);
    COORD_KEYS.forEach(k => { const el = $<HTMLInputElement>(k); if (el) el.value = ''; });
  },

  coordsCommitted: () => {
    const v = Object.fromEntries(COORD_KEYS.map(k => [k, parseFloat($<HTMLInputElement>(k)?.value ?? '')])) as Zone;
    if (Object.values(v).every(Number.isFinite)) get().selectZone(v, true);
  },

  useCurrentView: () => get().selectZone(engine.currentViewZone()),

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
    engine.setRisk(EMPTY);
    try {
      const d = await getJSON('/api/predict?' + new URLSearchParams({ ...loaded.params, source: loaded.mode,
        ...(loaded.mode === 'fetch' ? { sensor: loaded.sensor } : {}) }));
      set({ predictMsg: { kind: 'ok', text: `✓ ${d.forecast.length}-day forecast ready.` } });
      // Close the window, fly to the zone (fires + forecast are on the map) and show the details in the panel.
      get().closeModal();
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

  showDay: (i) => {
    const f = get().forecast?.forecast?.[i];
    if (!f) return;
    engine.setRiskDay(i);
    set({ day: i, status: `Showing the fire forecast for ${shortDate(f.date)} (+${i + 1} day${i ? 's' : ''}) on the map.` });
  },

  // Zoom in until the zone fills the map edge to edge (a camera move only: you can still zoom out or pan).
  zoomToZone: (duration = 2000) => {
    const { zone } = get();
    if (zone) engine.fitZone(zone, 0, duration);
  },
  flyToHotspot: (lat, lon) => { engine.stopSpin(); engine.flyTo({ center: [lon, lat], zoom: 9, duration: 2000 }); },

  // Back to the zone controls: clear the map (fire dots, legend, forecast squares and the zone outline) and fly back
  // to where the map was when Predict was clicked. The zone stays selected and the loaded data is kept, so
  // "Analyse zone" brings the outline back and predicting again redraws the rest.
  changeZone: () => {
    get().showForecastPanel(false);
    engine.setFires(EMPTY);
    engine.setRisk(EMPTY);
    engine.showZone(null);
    set({ status: '' });
    const cam = get().cameraBeforePredict;
    if (cam) engine.flyTo({ ...cam, duration: 2000, essential: true });
  },

  // Loaded data no longer matches once the zone or time window changes.
  invalidateData: () => {
    if (!get().loaded) return;
    set({ loaded: null, lastFires: null, status: '' });
    engine.setFires(EMPTY);
    get().clearPrediction();
  },
  clearPrediction: () => {
    engine.setRisk(EMPTY);
    get().showForecastPanel(false);
    set({ predictMsg: null });
  },
  // The forecast takes the place of part 1 (Prediction zone) in the side panel.
  showForecastPanel: (on) => {
    set({ forecastVisible: on });
    if (on) afterRender(() => $('forecastSection')?.scrollIntoView({ block: 'start', behavior: 'smooth' }));
  },
}));

// End of the opening transition: remove the (now invisible) opening screen and the intro-only CSS state.
function finishIntro() {
  clearTimeout(introTimer);
  if (!useApp.getState().splashLeaving || useApp.getState().splashGone) return;
  useApp.setState({ splashGone: true });
  body().remove('intro');
}

// Text for the "Selected zone" box.
export function zoneSummary(zone: Zone, pinAt: { lat: number; lng: number } | null) {
  return {
    coords: `${zone.w}, ${zone.s} → ${zone.e}, ${zone.n}`,
    size: `≈ ${Math.round(areaKm2(zone)).toLocaleString()} km² · ${r2(zone.e - zone.w)}° × ${r2(zone.n - zone.s)}°`,
    pin: pinAt ? `Centred on the pin at ${pinAt.lat.toFixed(3)}, ${pinAt.lng.toFixed(3)}` : null,
  };
}

export { num };
