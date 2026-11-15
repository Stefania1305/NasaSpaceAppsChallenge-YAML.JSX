// The map engine: everything imperative about the globe and the map, ported from the original app.html.
// React renders the page; this module owns MapLibre, the space backdrop (stars, embers, halo), the spin,
// the glide between layouts, globe sizing, rectangle drawing, the pin and the map popups. It talks to the
// UI through the app store (src/store.ts).

import maplibregl from 'maplibre-gl';
import type { GeoJSONSource } from 'maplibre-gl';
import { useApp } from './store';
import { EMPTY, ZONE_COLOR, boxPolygon, lngLatOk, normalise, wrapLng, zoneAroundPoint,
         type FC, type LngLat, type Zone } from './geo';
import { pct } from './format';
import { colorize, riskImage, type HeatField } from './riskHeat';
import { WindFlow } from './windFlow';
import { countryById, loadCountries, territoryGeometry } from './countries';
import { shapeTest, type Frame, type Geometry } from './mapImage';
import type { Weather } from './store';
import { WEATHER_VARS, weatherAt, weatherImage, type WeatherVar } from './weatherHeat';

let map!: maplibregl.Map;
let mapReady!: Promise<void>;
let started = false;

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const esc = (s: unknown) => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]!));
const reducedMotion = () => matchMedia('(prefers-reduced-motion: reduce)').matches;
const isSplash = () => document.body.classList.contains('splash');

// ---------------- Layout padding ----------------
// On desktop the side panel floats over the full-screen map, so the camera keeps its content
// clear of it with padding (centring, fitBounds and flyTo all respect this).
export const PANEL_W = 444;            // 12px margin + 420px glass card + 12px gap (see the Glass CSS)
export const isDesktop = () => innerWidth > 800;
const pad = (left = 0, top = 0) => ({ left, top, right: 0, bottom: 0 });
const panelOpen = () => useApp.getState().panelOpen;
// The menu sits at the left edge, so nothing is reserved at the top: the globe is centred in the free space.
export const appPadding = () => pad(isDesktop() && panelOpen() ? PANEL_W : 0);

// Run map updates only once the style has loaded.
const setData = (src: string, data: FC) => mapReady.then(() => (map.getSource(src) as GeoJSONSource).setData(data as never));

// ---------------- Spin / glide state ----------------
let spinning = true, paused = false, gliding = false, glideId = 0;
let resumeTimer: ReturnType<typeof setTimeout> | undefined;
let drawing = false, pinMode = false, drawStart: LngLat | null = null;
let pin: maplibregl.Marker | null = null;

export function init() {
  if (started) return;                  // React may mount twice in development; the map must exist once
  started = true;

  map = new maplibregl.Map({
    container: 'map',
    center: [15, 25],
    zoom: 1.6,
    maxPitch: 0,
    attributionControl: { compact: true },
    // Keep tiles for every zoom level in memory (the default keeps about 5 levels). Otherwise, after browsing a
    // zone up close, the low-detail tiles the whole globe needs have been dropped, and zooming back out shows
    // black holes (at the edges and over the poles) for a moment while they reload.
    maxTileCacheZoomLevels: 22,
    style: {
      version: 8,
      // Globe below zoom 2.8, fully flat (Web Mercator) from zoom 5.6; MapLibre morphs between them. The morph is
      // spread over a wide zoom range and eased in and out (slow start, slow finish) so the globe unrolls into the
      // flat map gradually; the atmosphere and the orange halo fade on the same curve (see MORPH_* below).
      projection: { type: ['interpolate', ['cubic-bezier', .45, 0, .55, 1], ['zoom'], 2.8, 'vertical-perspective', 5.6, 'mercator'] },
      sky: { 'atmosphere-blend': ['interpolate', ['cubic-bezier', .45, 0, .55, 1], ['zoom'], 0, 1, 2.8, 1, 5.6, 0] },
      sources: {
        imagery: { type: 'raster', tileSize: 256, maxzoom: 19,
          tiles: ['https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'],
          attribution: 'Imagery &copy; Esri' },
        // Country borders on land only (no sea / maritime boundaries): Natural Earth 1:10m, public domain,
        // shipped with the app in public/borders.geojson.
        borders: { type: 'geojson', data: `${import.meta.env.BASE_URL}borders.geojson`,
          attribution: 'Borders: Natural Earth' },
        // Country shapes for the "Pick a country" zone method (loaded on first use, see countries.ts).
        countries: { type: 'geojson', data: EMPTY as never },
        zone: { type: 'geojson', data: EMPTY as never },
        draft: { type: 'geojson', data: EMPTY as never },
        fires: { type: 'geojson', data: EMPTY as never },
        risk: { type: 'geojson', data: EMPTY as never },
      },
      layers: [
        // Satellite imagery at every zoom, globe and flat map alike.
        // No tile fade: when zooming out quickly, each coarser set of tiles would fade in from transparent while the
        // detailed ones are dropped, leaving black gaps (and a black polar cap) for a moment. Shown at once instead.
        { id: 'imagery', type: 'raster', source: 'imagery', paint: { 'raster-fade-duration': 0 } },
        // Land borders: a thin dark casing under a soft white line, so they read on bright desert and dark forest
        // alike; they get a little bolder as you zoom in.
        { id: 'borders-casing', type: 'line', source: 'borders', layout: { 'line-join': 'round', 'line-cap': 'round' },
          paint: { 'line-color': 'rgba(0, 0, 0, .45)',
                   'line-width': ['interpolate', ['linear'], ['zoom'], 1, 1.2, 5, 2.6, 10, 4] } },
        { id: 'borders', type: 'line', source: 'borders', layout: { 'line-join': 'round', 'line-cap': 'round' },
          paint: { 'line-color': 'rgba(255, 255, 255, .78)',
                   'line-width': ['interpolate', ['linear'], ['zoom'], 1, .5, 5, 1.1, 10, 1.8] } },
        // "Pick a country": an invisible fill to find the country under the mouse, then the hovered and the
        // selected country's border as a soft orange glow under a crisp line (filters pick the country by id).
        { id: 'country-hit', type: 'fill', source: 'countries', paint: { 'fill-color': '#000', 'fill-opacity': 0 } },
        { id: 'country-hover-fill', type: 'fill', source: 'countries', filter: ['==', ['get', 'id'], -1],
          paint: { 'fill-color': '#fb923c', 'fill-opacity': .12 } },
        ...(['hover', 'sel'] as const).flatMap(k => [
          { id: `country-${k}-glow`, type: 'line' as const, source: 'countries', filter: ['==', ['get', 'id'], -1],
            layout: { 'line-join': 'round' as const },
            paint: { 'line-color': '#f97316', 'line-opacity': .55, 'line-blur': 4,
                     'line-width': ['interpolate', ['linear'], ['zoom'], 1, 4, 6, 9] } },
          { id: `country-${k}`, type: 'line' as const, source: 'countries', filter: ['==', ['get', 'id'], -1],
            layout: { 'line-join': 'round' as const },
            paint: { 'line-color': '#fdba74', 'line-width': ['interpolate', ['linear'], ['zoom'], 1, 1.2, 6, 2.4] } },
        ]) as maplibregl.LayerSpecification[],
        { id: 'zone-fill', type: 'fill', source: 'zone', paint: { 'fill-color': ZONE_COLOR, 'fill-opacity': .1 } },
        { id: 'zone-line', type: 'line', source: 'zone',
          paint: { 'line-color': ZONE_COLOR, 'line-width': 2, 'line-dasharray': [3, 2] } },
        { id: 'draft-fill', type: 'fill', source: 'draft', paint: { 'fill-color': ZONE_COLOR, 'fill-opacity': .15 } },
        { id: 'draft-line', type: 'line', source: 'draft', paint: { 'line-color': ZONE_COLOR, 'line-width': 2 } },
        // The predicted fire risk is drawn as a soft heatmap image (added when there's a forecast, see setRisk).
        // The grid cells themselves stay as an invisible layer, so clicking a spot still shows its chances.
        { id: 'risk-fill', type: 'fill', source: 'risk', paint: { 'fill-color': '#000', 'fill-opacity': 0 } },
        { id: 'fires', type: 'circle', source: 'fires', paint: {
          'circle-color': ['step', ['get', 'frp'], '#facc15', 10, '#f97316', 50, '#dc2626', 200, '#7f1d1d'],
          'circle-radius': ['interpolate', ['linear'], ['zoom'],
            2, ['+', 1.5, ['min', 3, ['/', ['sqrt', ['get', 'frp']], 4]]],
            8, ['+', 3, ['min', 7, ['/', ['sqrt', ['get', 'frp']], 2]]]],
          'circle-opacity': .8, 'circle-stroke-width': .5, 'circle-stroke-color': '#00000066' } },
      ],
    },
  });
  map.addControl(new maplibregl.NavigationControl({ showCompass: false }), 'top-right');
  mapReady = new Promise(res => map.on('load', () => res()));
  // Handy for debugging from the browser console.
  (window as unknown as { map: maplibregl.Map }).map = map;

  setupBackdrop();
  setupHalo();
  setupSpin();
  setupWeatherTip();
  setupCountryPick();
  // Redrawing a heatmap image cancels the previous image's load; MapLibre reports that cancellation as an error.
  // It's harmless, so drop just that case and log everything else as usual.
  map.on('error', e => {
    const err = (e as { error?: Error }).error;
    if (err && /aborted|AbortError/i.test(`${err.name} ${err.message}`)) return;
    console.error(err ?? e);
  });
  setupInteractions();

  addEventListener('resize', () => {
    if (!isSplash() && !gliding) map.setPadding(appPadding());
  });
  // The map container only changes size on narrow screens (stacked layout); keep the canvas in step.
  new ResizeObserver(() => map.resize()).observe($('map'));

  // ---------------- Opening screen ----------------
  // On wide screens the text sits on the left, so push the globe to the right.
  if (isDesktop()) map.setPadding(pad(innerWidth * 0.42));
}

// ---------------- Space backdrop: star tiles + rising embers ----------------
function starTile(size: number, count: number, seed: number) {
  const c = document.createElement('canvas');
  c.width = c.height = size;
  const g = c.getContext('2d')!;
  let s = seed;
  const rnd = () => (s = (s * 16807) % 2147483647) / 2147483647;
  for (let i = 0; i < count; i++) {
    const x = rnd() * size, y = rnd() * size, big = rnd() < .08, r = big ? .9 + rnd() * .8 : .3 + rnd() * .55;
    const warm = rnd() < .18;                              // a few stars glow ember-orange
    g.fillStyle = warm ? `rgba(253, ${170 + rnd() * 60 | 0}, 110, ${.5 + rnd() * .5})`
                       : `rgba(255, ${235 + rnd() * 20 | 0}, ${215 + rnd() * 40 | 0}, ${.35 + rnd() * .6})`;
    g.beginPath(); g.arc(x, y, r, 0, Math.PI * 2); g.fill();
    if (big) {   // soft glow fading out from the star, not a hard ring
      const glow = g.createRadialGradient(x, y, 0, x, y, r * 5);
      glow.addColorStop(0, warm ? 'rgba(251,146,60,.35)' : 'rgba(255,240,220,.25)');
      glow.addColorStop(1, 'rgba(0,0,0,0)');
      g.fillStyle = glow; g.beginPath(); g.arc(x, y, r * 5, 0, Math.PI * 2); g.fill();
    }
  }
  return `url(${c.toDataURL()})`;
}

function setupBackdrop() {
  document.documentElement.style.setProperty('--stars-far', starTile(520, 260, 7));
  document.documentElement.style.setProperty('--stars-near', starTile(760, 90, 42));

  // Embers: a few dozen glowing sparks drifting up from the bottom, flickering as they fade.
  const cv = $<HTMLCanvasElement>('embers'), g = cv.getContext('2d')!;
  if (reducedMotion()) return;
  let W = 0, H = 0, dpr = 1, last = performance.now(), cleared = false;
  const fit = () => {
    dpr = Math.min(2, devicePixelRatio || 1);
    W = cv.clientWidth; H = cv.clientHeight;
    cv.width = W * dpr; cv.height = H * dpr;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
  };
  new ResizeObserver(fit).observe(cv);
  fit();
  type Spark = { x: number; y: number; vy: number; sway: number; phase: number; freq: number; r: number;
                 life: number; max: number; hue: number };
  const spawn = (initial: boolean): Spark => ({
    x: Math.random() * W,
    y: initial ? H * (.3 + Math.random() * .7) : H + 10 + Math.random() * 40,
    vy: 12 + Math.random() * 30,                 // px/s upward
    sway: 8 + Math.random() * 18, phase: Math.random() * 6.28, freq: .4 + Math.random() * .9,
    r: 1 + Math.random() * 1.8,
    life: 0, max: 7 + Math.random() * 9,         // seconds
    hue: 18 + Math.random() * 28,                // red-orange → amber
  });
  const N = Math.round(Math.min(70, Math.max(30, innerWidth / 22)));
  const sparks = Array.from({ length: N }, () => spawn(true));
  const frame = (now: number) => {
    const dt = Math.min(.05, (now - last) / 1000);
    last = now;
    // Nothing to see once the flat map covers the backdrop, or when the tab is hidden.
    if (document.hidden || map.getZoom() > MORPH_END) {
      if (!cleared) { g.clearRect(0, 0, W, H); cleared = true; }
      requestAnimationFrame(frame);
      return;
    }
    cleared = false;
    g.clearRect(0, 0, W, H);
    g.globalCompositeOperation = 'lighter';
    for (let i = 0; i < sparks.length; i++) {
      const p = sparks[i];
      p.life += dt;
      p.y -= p.vy * dt;
      p.phase += p.freq * dt;
      if (p.life > p.max || p.y < -20) { sparks[i] = spawn(false); continue; }
      const x = p.x + Math.sin(p.phase) * p.sway;
      const fade = Math.min(1, p.life / 1.2) * Math.min(1, (p.max - p.life) / 2.5);   // fade in, fade out
      const flicker = .7 + .3 * Math.sin(now / 90 + p.phase * 7);
      const heightFade = Math.min(1, .25 + p.y / H);                                  // dimmer as they rise
      const a = fade * flicker * heightFade;
      // Soft glow around the spark.
      const glow = g.createRadialGradient(x, p.y, 0, x, p.y, p.r * 9);
      glow.addColorStop(0, `hsla(${p.hue}, 100%, 60%, ${a * .45})`);
      glow.addColorStop(.35, `hsla(${p.hue - 8}, 100%, 50%, ${a * .14})`);
      glow.addColorStop(1, 'hsla(10, 100%, 40%, 0)');
      g.fillStyle = glow;
      g.beginPath(); g.arc(x, p.y, p.r * 9, 0, 6.283); g.fill();
      // Short tail trailing below the spark as it rises.
      const tx = x - Math.cos(p.phase) * p.sway * p.freq * .25, ty = p.y + p.vy * .22;
      const tail = g.createLinearGradient(x, p.y, tx, ty);
      tail.addColorStop(0, `hsla(${Math.min(44, p.hue + 8)}, 100%, 62%, ${a * .55})`);
      tail.addColorStop(1, `hsla(${p.hue}, 100%, 50%, 0)`);
      g.strokeStyle = tail; g.lineWidth = p.r * .9; g.lineCap = 'round';
      g.beginPath(); g.moveTo(x, p.y); g.lineTo(tx, ty); g.stroke();
      // Hot core: white-yellow centre.
      g.fillStyle = `hsla(${Math.min(48, p.hue + 14)}, 100%, 80%, ${a})`;   // capped so it never tips into green
      g.beginPath(); g.arc(x, p.y, p.r * .75, 0, 6.283); g.fill();
    }
    g.globalCompositeOperation = 'source-over';
    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
}

// ---------------- Globe halo ----------------
// Sits exactly around the globe *as drawn*, and fades out between zoom 2.5 and 4.5 as the globe flattens into
// the map. Its rim is measured on screen in four directions: walk away from the map centre along a great
// circle and keep the furthest on-screen point; the halo is centred between opposite rim points.
function globeRim(center: maplibregl.LngLat, c: maplibregl.Point, bearingDeg: number) {
  const rad = Math.PI / 180, φ1 = center.lat * rad, λ1 = center.lng * rad, θ = bearingDeg * rad;
  let best: maplibregl.Point | null = null, bestD = -1;
  for (let deg = 40; deg <= 90; deg += 2) {
    const δ = deg * rad;
    const φ2 = Math.asin(Math.sin(φ1) * Math.cos(δ) + Math.cos(φ1) * Math.sin(δ) * Math.cos(θ));
    const λ2 = λ1 + Math.atan2(Math.sin(θ) * Math.sin(δ) * Math.cos(φ1), Math.cos(δ) - Math.sin(φ1) * Math.sin(φ2));
    const p = map.project([λ2 / rad, φ2 / rad]);
    if (!Number.isFinite(p.x) || !Number.isFinite(p.y)) continue;
    const d = Math.hypot(p.x - c.x, p.y - c.y);
    if (d > bestD) { bestD = d; best = p; }
  }
  return best;
}

// The globe ↔ flat-map morph range (must match the projection in the style above) and a soft 0..1 ramp.
const MORPH_START = 2.8, MORPH_END = 5.6;
const smoothstep = (a: number, b: number, x: number) => { const t = Math.max(0, Math.min(1, (x - a) / (b - a))); return t * t * (3 - 2 * t); };
// Camera flights between the globe and a zone: ease in and out (slow start, slow landing).
export const easeInOut = (t: number) => t < .5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;

function setupHalo() {
  const halo = document.querySelector<HTMLElement>('.space .halo')!;
  const placeHalo = () => {
    const z = map.getZoom();
    if (z > MORPH_END - .6) { halo.style.opacity = '0'; return; }
    const center = map.getCenter(), c = map.project(center);
    const E = globeRim(center, c, 90), W = globeRim(center, c, 270), N = globeRim(center, c, 0), S = globeRim(center, c, 180);
    let cx = c.x, cy = c.y, r = 512 * Math.pow(2, z) / (2 * Math.PI);
    if (E && W && N && S) {
      cx = (E.x + W.x) / 2;
      cy = (N.y + S.y) / 2;
      r = (Math.hypot(E.x - W.x, E.y - W.y) + Math.hypot(N.x - S.x, N.y - S.y)) / 4;
    }
    const D = r * 2 / 0.6;                                                        // 60% stop = globe edge
    halo.style.width = halo.style.height = `${D}px`;
    halo.style.transform = `translate(${cx - D / 2}px, ${cy - D / 2}px)`;
    // Fades out as the globe unrolls, on an eased curve, gone a little before the map is fully flat.
    halo.style.opacity = String(1 - smoothstep(MORPH_START, MORPH_END - .6, z));
  };
  // Never let a decoration throw inside the map's render loop (an exception there freezes the map).
  const safePlaceHalo = () => { try { placeHalo(); } catch (err) { console.warn('halo', err); } };
  // Update on 'render' (when the globe is actually drawn) so the halo never trails a moving globe.
  map.on('render', safePlaceHalo);
  map.on('resize', safePlaceHalo);
  safePlaceHalo();
}

// ---------------- Globe size ----------------
// The globe's on-screen radius doubles with each zoom level, so the zoom that gives a wanted radius follows
// from the radius measured right now. Closed panel: 84% of the smaller side; open panel: 72% of the space beside it.
export function globeRadiusPx() {
  const center = map.getCenter(), c = map.project(center);
  const E = globeRim(center, c, 90), W = globeRim(center, c, 270);
  return E && W ? Math.hypot(E.x - W.x, E.y - W.y) / 2 : 512 * Math.pow(2, map.getZoom()) / (2 * Math.PI);
}
function globeZoomFor(open: boolean): number | null {
  const cv = map.getCanvas();                                   // the map's area (shorter on phones)
  const w = cv.clientWidth - (isDesktop() && open ? PANEL_W : 0), h = cv.clientHeight;
  const want = Math.min(w, h) * (open ? 0.36 : 0.42);          // radius = 72% / 84% of the side, halved
  // Perspective makes the size not quite double per zoom level, so refine: try the zoom, measure the
  // real radius there, correct, and repeat. The camera is put back before anything is drawn.
  const z0 = map.getZoom(), min0 = map.getMinZoom();
  let z = z0;
  map.setMinZoom(null);                                         // the trial zooms must not be clamped by the limit
  try {
    for (let i = 0; i < 3; i++) {
      if (i) map.jumpTo({ zoom: z });
      const r = globeRadiusPx();
      if (!(r > 0) || !Number.isFinite(r)) break;
      z += Math.log2(want / r);
      if (!Number.isFinite(z)) { z = z0; break; }
      z = Math.max(1, Math.min(2.8, z));                        // stay in globe view (flattening starts at 3)
    }
  } finally {
    map.jumpTo({ zoom: z0 });
    setMinZoomQuiet(min0);
  }
  return Number.isFinite(z) ? z : null;                         // null = leave the size alone
}

// ---------------- Zoom-out limit ----------------
// With the panel closed the globe can't be zoomed out further than the size it has right after "Get started"
// (`closedZoom`). The limit is lifted while the panel is open (the globe is smaller there), on the opening
// screen, and during every glide (so glides can pass through any size); it's put back once the camera rests.
// On the globe MapLibre applies minZoom to the latitude-adjusted zoom (which tracks the globe's drawn size:
// at the same zoom it looks bigger further from the equator), so the limit is kept in that unit too.
let closedZoom: number | null = null;
const sizeZoom = (z = map.getZoom(), lat = map.getCenter().lat) =>
  z + Math.log2(1 / Math.max(0.05, Math.cos(lat * Math.PI / 180)));
// map.setMinZoom() also snaps the camera by comparing the *raw* zoom, which on the globe would wrongly enlarge
// it; setting the limit on the transform only changes the clamp that scrolling and pinching respect.
function setMinZoomQuiet(z: number | null) {
  const t = (map as unknown as { transform?: { setMinZoom?: (z: number) => void } }).transform;
  if (z == null || !t?.setMinZoom) { map.setMinZoom(z); return; }
  t.setMinZoom(z);
}
let limitTimer: ReturnType<typeof setTimeout> | undefined;
function syncZoomLimit() {
  const want = panelClosed() && closedZoom != null && !gliding ? closedZoom : null;
  if (want == null) { if (map.getMinZoom() > 0) map.setMinZoom(null); return; }
  if (Math.abs(map.getMinZoom() - want) < 1e-6) return;
  // Only set it when the map is at (or inside) the limit: setting it below would make the map jump.
  const now = sizeZoom();
  if (now >= want - 0.01) { setMinZoomQuiet(Math.min(want, now)); return; }
  // Smaller than the limit (the user scrolled out while a glide was still bringing the globe to its size):
  // stop it shrinking any further right away, then, once the map is at rest, ease it back up to the limit
  // (the 'moveend' at the end of that ease sets the full limit, and the spin resumes there).
  setMinZoomQuiet(Math.min(map.getMinZoom() > 0 ? map.getMinZoom() : now, now));
  if (map.isMoving()) return;                                   // still being moved: try again on 'moveend'
  map.easeTo({ zoom: map.getZoom() + (want - now), duration: reducedMotion() ? 0 : 700, essential: true });
}
// Outside map events and gestures (camera changes there can break MapLibre's render loop).
const syncZoomLimitSoon = () => { clearTimeout(limitTimer); limitTimer = setTimeout(syncZoomLimit, 0); };
const atZoomLimit = () => closedZoom != null && sizeZoom() <= closedZoom + 0.05;
// Only resize while the whole globe is in view; never undo a zoom the user made into a region.
const showingWholeGlobe = () => map.getZoom() < 3;

// ---------------- Spin ----------------
// While the panel is closed (desktop) the globe always spins: touching the map only pauses it, and it resumes
// shortly after the map comes to rest. With the panel open, the first interaction stops it. The spin is one
// continuous frame-by-frame rotation (6°/s); while running it also eases the framing back to where it belongs
// (beside the panel, or centred), so the globe always settles in the right place.
function setupSpin() {
  if (reducedMotion()) spinning = false;
  let lastSpinT: number | null = null;
  const spinFrame = (now: number) => {
    requestAnimationFrame(spinFrame);
    const dt = lastSpinT == null ? 0 : Math.min(.1, (now - lastSpinT) / 1000);
    lastSpinT = now;
    if (paused || gliding || document.hidden || map.isMoving()) return;
    const next: maplibregl.JumpToOptions = {};
    // Spin (only in globe view) — on the opening screen too.
    if (spinning && map.getZoom() <= 3) {
      const c = map.getCenter();
      next.center = [c.lng - 6 * dt, c.lat];
    }
    // Keep the framing right: ease the globe back beside the open panel, or to the centre when it's closed.
    // Not on the opening screen, where the globe deliberately sits to the right of the title text.
    const cur = map.getPadding(), want = appPadding();
    const haveL = cur.left ?? 0, haveT = cur.top ?? 0;
    if (!isSplash() && (Math.abs(haveL - want.left) > .5 || Math.abs(haveT - want.top) > .5)) {
      const k = Math.min(1, dt * 3), step = (a: number, b: number) => Math.max(0, Math.round((a + (b - a) * k) * 100) / 100);
      next.padding = { left: step(haveL, want.left), top: step(haveT, want.top), right: 0, bottom: 0 };
    }
    if (next.center || next.padding) map.jumpTo(next);
  };
  mapReady.then(() => requestAnimationFrame(spinFrame));

  // Used when the user touches the map: only end the spin (map.stop() here would cancel their own gesture).
  const userStopsSpin = () => { spinning = false; clearTimeout(resumeTimer); };
  ['mousedown', 'touchstart', 'wheel'].forEach(ev => map.getCanvasContainer().addEventListener(ev, userStopsSpin, { passive: true }));
  // Scrolling out while already at the limit moves nothing (no 'moveend'), so the wheel schedules the resume
  // itself; each new wheel tick pushes it back, so it starts ~0.6 s after the scrolling stops.
  map.getCanvasContainer().addEventListener('wheel', () => resumeSpinSoon(), { passive: true });
  // A click or tap that doesn't move the map fires no 'moveend', so also schedule the resume on release.
  ['mouseup', 'touchend'].forEach(ev => map.getCanvasContainer().addEventListener(ev, () => { if (!spinning) resumeSpinSoon(); }, { passive: true }));
  // The zoom buttons and keyboard zoom don't touch the canvas, so end the spin for those too.
  map.on('zoomstart', e => { if (e.originalEvent) userStopsSpin(); });
  map.on('moveend', () => { syncZoomLimitSoon(); if (!spinning) resumeSpinSoon(); });
}

// Before the app moves the camera itself (fly to a zone, etc.): end the spin and its animation.
export function stopSpin() { if (spinning) { spinning = false; map.stop(); } }

// Panel closed → resume the spin shortly after the user lets go, once the map is at rest and zoomed all the
// way out (to the limit). Zoomed in to a region, the globe stays still.
const panelClosed = () => isDesktop() && !panelOpen() && !isSplash();
function resumeSpinSoon() {
  clearTimeout(resumeTimer);
  if (reducedMotion() || !panelClosed()) return;
  resumeTimer = setTimeout(() => {
    if (!panelClosed() || drawing || pinMode) return;
    if (map.isMoving()) { resumeSpinSoon(); return; }             // still settling: try again a bit later
    if (!atZoomLimit()) return;                                   // zoomed in: stay still
    spinning = true;
  }, 600);
}

// ---------------- Glide ----------------
// Move the globe to its place frame by frame: the padding (and optionally the zoom) ease in and out while
// the globe keeps turning at its spin speed. Every glide has an id; its "user grabbed the map" listeners are
// removed when it ends and only act on that same glide. If the user grabs the map mid-glide, control is handed
// over straight away and the frame loop finishes the framing once the map is at rest.
function glideTo(target: { left: number; top: number; right: number; bottom: number }, ms: number, toZoom: number | null = null,
                 onDone?: () => void) {
  const id = ++glideId;
  if (toZoom != null && !Number.isFinite(toZoom)) toZoom = null;   // never feed the map an invalid zoom
  const from = map.getPadding(), fromZoom = map.getZoom(), t0 = performance.now();
  let fromLeft = Number.isFinite(from.left) ? (from.left as number) : target.left;
  fromLeft = Math.max(0, fromLeft);
  const fromTop = Math.max(0, Number.isFinite(from.top) ? (from.top as number) : target.top);
  // MapLibre rejects any negative padding (even -0.0000014 px) and then every camera move fails.
  const safeLeft = (v: number) => Math.max(0, Math.round(v * 100) / 100);
  const ease = (t: number) => t < .5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
  const zoomAt = (e: number) => toZoom == null ? {} : { zoom: fromZoom + (toZoom - fromZoom) * e };
  let last = t0;
  gliding = paused = true;
  map.stop();
  if (map.getMinZoom() > 0) map.setMinZoom(null);               // glides may pass below the limit
  clearTimeout(limitTimer);
  const grabEvents = ['mousedown', 'touchstart', 'wheel'];
  const box = map.getCanvasContainer();
  const cleanup = () => grabEvents.forEach(ev => box.removeEventListener(ev, abort));
  const finish = () => {
    cleanup();
    if (id !== glideId) return;                                 // a newer glide has taken over
    gliding = paused = false;
    map.jumpTo({ padding: target, ...zoomAt(1) });
    syncZoomLimit();
    onDone?.();
  };
  const frame = (now: number) => {
    if (!gliding || id !== glideId) return;
    const k = Math.min(1, (now - t0) / ms), e = ease(k), c = map.getCenter();
    const turn = spinning ? 6 * (now - last) / 1000 : 0;     // same 6°/s as the spin
    last = now;
    map.jumpTo({ center: [c.lng - turn, c.lat], ...zoomAt(e), padding: {
      left: safeLeft(fromLeft + (target.left - fromLeft) * e), top: safeLeft(fromTop + (target.top - fromTop) * e),
      right: 0, bottom: 0 } });
    if (k < 1) requestAnimationFrame(frame); else finish();
  };
  function abort() {
    cleanup();
    if (id !== glideId || !gliding) return;
    gliding = false; paused = false;
    // The user took over mid-glide: put the zoom-out limit back straight away (only the limit, no camera move,
    // which would fight their gesture), so scrolling out can't shrink the centred globe past where it is.
    if (panelClosed() && closedZoom != null && sizeZoom() < closedZoom - 0.01)
      setMinZoomQuiet(Math.min(map.getMinZoom() > 0 ? map.getMinZoom() : sizeZoom(), sizeZoom()));
    else syncZoomLimit();
  }
  grabEvents.forEach(ev => box.addEventListener(ev, abort, { passive: true }));
  if (ms > 0) requestAnimationFrame(frame); else finish();
}

// After "Get started": the globe glides from the right-hand side to the centre and grows.
export function introGlide(open: boolean) {
  const z = globeZoomFor(open);
  if (!open && z != null) closedZoom = sizeZoom(z);             // this size becomes the zoom-out limit
  glideTo(appPadding(), reducedMotion() ? 0 : 2400, z);
}

// Home button: back to the opening-screen framing — the globe glides back to the right of the title text and
// shrinks to its opening size (zoom 1.6), still spinning.
export function goHome() {
  if (!reducedMotion()) spinning = true;
  glideTo(pad(isDesktop() ? innerWidth * 0.42 : 0), reducedMotion() ? 0 : 1800, 1.6);
}

// Globe button: just the globe, centred and spinning. From a zoomed-in map this flies back out to the whole globe.
export function globeView() {
  if (!reducedMotion()) spinning = true;
  if (showingWholeGlobe()) { glideTo(appPadding(), reducedMotion() ? 0 : 1100, closedZoomNow()); return; }
  // From a flat (zoomed-in) map the globe can't be measured (the map only becomes a globe when it next draws),
  // but its size is already known (`closedZoom`, measured after "Get started", in latitude-adjusted units), so
  // glide straight out to it in one continuous move. Only if that's unknown, glide out to an approximate size
  // and settle once the globe is drawn.
  if (closedZoom != null) {
    const z = closedZoom - sizeZoom(0);                          // back to a plain zoom at the current latitude
    glideTo(appPadding(), reducedMotion() ? 0 : 2400, Math.max(1, Math.min(MORPH_START, z)));
    return;
  }
  glideTo(appPadding(), reducedMotion() ? 0 : 2300, 2.2, () => {
    requestAnimationFrame(() => requestAnimationFrame(() => {
      if (!gliding && showingWholeGlobe()) glideTo(appPadding(), reducedMotion() ? 0 : 500, closedZoomNow());
    }));
  });
}

// Panel opened/closed: the globe glides aside to make room (or back to the centre) and is resized to suit.
export function panelChanged(open: boolean) {
  if (!open) {
    if (drawing) setDrawing(false);               // nothing half-started behind a closed panel
    if (pinMode) setPinMode(false);
    if (!reducedMotion()) spinning = true;        // closed panel: the globe always spins
  }
  glideTo(appPadding(), reducedMotion() ? 0 : open ? 1200 : 1000,
          showingWholeGlobe() ? (open ? globeZoomFor(true) : closedZoomNow()) : null);
}
// The closed-panel globe size, measured now (it also refreshes the zoom-out limit, e.g. after a window resize).
function closedZoomNow() {
  const z = globeZoomFor(false);
  if (z != null) closedZoom = sizeZoom(z);
  return z;
}

// ---------------- Map layers ----------------
export const showZone = (z: Zone | null) => {
  // A zone picked by clicking a country is that country (its border is shown below), not the dashed rectangle.
  setData('zone', z && selectedCountry == null ? boxPolygon(z) : EMPTY);
  // A zone picked by clicking a country also shows that country's border (hidden along with the zone).
  const id = z && selectedCountry != null ? selectedCountry : -1;
  mapReady.then(() => ['country-sel-glow', 'country-sel'].forEach(l => map.setFilter(l, ['==', ['get', 'id'], id])));
};

// ---------------- "Pick a country" ----------------
let selectedCountry: number | null = null;
let hoveredCountry = -1;
let countryTip: maplibregl.Popup | null = null;
// Load the country shapes onto the map (the first time the method is chosen).
export const ensureCountries = () => loadCountries().then(fc => setData('countries', fc));
export const setSelectedCountry = (id: number | null) => { selectedCountry = id; };
// The method is live while it's chosen and its controls are on screen (not during a forecast or another mode).
function countryMode() {
  const s = useApp.getState();
  return s.method === 'country' && (s.panelOpen || !isDesktop()) && !s.forecastVisible && !drawing && !pinMode && !isSplash();
}
function setHover(id: number, at?: maplibregl.LngLat) {
  if (id !== hoveredCountry) {
    hoveredCountry = id;
    ['country-hover-glow', 'country-hover', 'country-hover-fill'].forEach(l => map.setFilter(l, ['==', ['get', 'id'], id]));
    map.getCanvas().style.cursor = id >= 0 ? 'pointer' : '';
  }
  const c = id >= 0 ? countryById(id) : null;
  if (!c || !at) { countryTip?.remove(); return; }
  countryTip!.setLngLat(at).setHTML(`<b>${c.name.replace(/[<>&]/g, '')}</b> · click to select`);
  if (!countryTip!.isOpen()) countryTip!.addTo(map);
}
function setupCountryPick() {
  countryTip = new maplibregl.Popup({ closeButton: false, closeOnClick: false, className: 'wx-tip', offset: 14, maxWidth: 'none' });
  // The country under a screen point, or -1. Beside the globe (in space) MapLibre snaps the point to the horizon,
  // which would pick a country at the globe's edge, so the point must map back onto itself to count.
  const under = (p: maplibregl.Point, at: maplibregl.LngLat) => {
    if (!lngLatOk(at)) return -1;
    const back = map.project(at);
    if (Math.hypot(back.x - p.x, back.y - p.y) > 2) return -1;
    const f = map.queryRenderedFeatures(p, { layers: ['country-hit'] })[0];
    return f ? Number(f.properties?.id) : -1;
  };
  map.on('mousemove', e => {
    if (!countryMode()) { if (hoveredCountry >= 0 || countryTip?.isOpen()) setHover(-1); return; }
    setHover(under(e.point, e.lngLat), e.lngLat);
  });
  map.getCanvas().addEventListener('mouseleave', () => { if (hoveredCountry >= 0) setHover(-1); });
  map.on('click', e => {
    if (!countryMode()) return;
    const id = under(e.point, e.lngLat);
    if (id < 0) return;
    setHover(-1);
    // Outside the map's own event handling (the selection flies the camera).
    const at: [number, number] = [wrapLng(e.lngLat.lng), e.lngLat.lat];     // the territory clicked comes first
    setTimeout(() => useApp.getState().selectCountry(id, null, at), 0);
  });
}
// Fire dots + their legend always appear and disappear together.
// The fire detections as loaded (for the zone's rectangle); with a country limit only those inside it are drawn.
let firesAll: FC = EMPTY;
export function setFires(data: FC) {
  firesAll = data;
  drawFires();
}
function drawFires() {
  const shown = clip
    ? { type: 'FeatureCollection' as const, features: firesAll.features.filter(f => clip!.inside(f.geometry.coordinates[0], f.geometry.coordinates[1])) }
    : firesAll;
  setData('fires', shown);
  useApp.setState({ firesShown: shown.features.length > 0 });
}
// ---------------- Fire-risk heatmap ----------------
// The forecast squares (one per grid cell) and the day shown; the map draws them as a soft heatmap (riskHeat.ts)
// just under the fire dots, with the invisible squares kept for click-to-inspect.
let riskSquares: FC = EMPTY, riskProp = 'p1';
// Put an image layer on the map (or update it), inserted under `before`; remove it when `img` is null.
function setImageLayer(id: string, img: ReturnType<typeof riskImage>, opacity: number, before: string) {
  const src = map.getSource(id) as maplibregl.ImageSource | undefined;
  if (!img) {
    if (map.getLayer(id)) map.removeLayer(id);
    if (src) map.removeSource(id);
    return;
  }
  if (src) { src.updateImage(img); return; }
  map.addSource(id, { type: 'image', ...img });
  map.addLayer({ id, type: 'raster', source: id,
                 paint: { 'raster-opacity': opacity, 'raster-fade-duration': 250, 'raster-resampling': 'linear' } }, before);
}
// While the timeline is on, it draws the fire heat (one image per day); the static day image stays off.
const drawRisk = () => mapReady.then(() =>
  setImageLayer('risk-heat', timelineImages ? null : riskImage(riskSquares, riskProp, clip?.geometry ?? null), 1, 'risk-fill'));

// ---------------- Fire timeline (animated heatmap) ----------------
// Each day is kept as a grid of heat values (observed days, then forecast days; see timeline.ts), all on one
// shared frame. Every animation tick blends two days' values and recolours them into a canvas that the map shows
// (a canvas source), so fires grow, fade and spread continuously instead of one picture cross-fading into the next.
export type TimelineFields = { fields: HeatField[]; frame: Frame; mask: Uint8ClampedArray | null };
let tl: (TimelineFields & { canvas: HTMLCanvasElement; ctx: CanvasRenderingContext2D; img: ImageData; last: number }) | null = null;
let timelineImages: boolean = false;                     // the static day heatmap stays off while this is on
export const getClipInside = () => clip?.inside ?? null;
export const getClipGeometry = () => clip?.geometry ?? null;
export function setTimeline(data: TimelineFields | null) {
  timelineImages = !!data?.fields.length;
  mapReady.then(() => {
    if (map.getLayer('tl-heat')) map.removeLayer('tl-heat');
    if (map.getSource('tl-heat')) map.removeSource('tl-heat');
    if (!data?.fields.length) { tl = null; drawRisk(); return; }
    const canvas = document.createElement('canvas');
    canvas.width = data.frame.width; canvas.height = data.frame.height;
    const ctx = canvas.getContext('2d')!;
    tl = { ...data, canvas, ctx, img: ctx.createImageData(canvas.width, canvas.height), last: -1 };
    setImageLayer('risk-heat', null, 1, 'risk-fill');
    map.addSource('tl-heat', { type: 'canvas', canvas, coordinates: data.frame.coordinates, animate: false });
    map.addLayer({ id: 'tl-heat', type: 'raster', source: 'tl-heat',
                   paint: { 'raster-fade-duration': 0, 'raster-resampling': 'linear' } }, 'risk-fill');
    showTimeline(0);
  });
}
// Which two days a timeline position lies between, and how far the morph from the first to the second has got
// (eased: it lingers briefly on each day, then moves smoothly to the next).
export function timelineBlend(pos: number, n: number) {
  const i = Math.max(0, Math.min(n - 1, Math.floor(pos))), j = Math.min(n - 1, i + 1);
  const u = Math.max(0, Math.min(1, (pos - i - .08) / .84));
  return { i, j, t: j === i ? 0 : u * u * (3 - 2 * u) };
}
// Show position `pos` (in days; the fraction morphs into the next day, easing in and out around each day).
export function showTimeline(pos: number) {
  if (!tl) return;
  const n = tl.fields.length, { i, j, t } = timelineBlend(pos, n);
  const key = i + t;
  if (Math.abs(key - tl.last) < 1e-4) return;           // nothing changed (paused)
  tl.last = key;
  colorize(tl.fields[i], tl.img.data, t > 0 ? { to: tl.fields[j], t } : undefined, tl.mask ?? undefined);
  tl.ctx.putImageData(tl.img, 0, 0);
  // Upload the new pixels once: a canvas source re-reads its canvas only while "playing".
  const src = map.getSource('tl-heat') as maplibregl.CanvasSource | undefined;
  if (src) { src.play(); src.pause(); map.triggerRepaint(); }
}
// Fire dots for one observed day only (null: all; 'none': hide them, e.g. on forecast days).
export function setFireDay(date: string | null) {
  fireBlendKey = '';
  mapReady.then(() => {
    map.setFilter('fires', date ? ['==', ['get', 'date'], date] : null);
    map.setPaintProperty('fires', 'circle-opacity', .8);
    map.setPaintProperty('fires', 'circle-stroke-opacity', 1);
  });
}
// During the timeline: the dots of day `a` fade out while those of day `b` fade in (t = 0..1, the same eased
// morph as the heatmap). null for a forecast day, which has no detections. Updated in small steps (12 per
// change of day) so the map isn't re-styled on every animation tick.
let fireBlendKey = '';
export function setFireBlend(a: string | null, b: string | null, t: number) {
  const q = Math.round(t * 12) / 12;
  const key = `${a}|${b}|${q}`;
  if (key === fireBlendKey || !map.getLayer('fires')) return;
  fireBlendKey = key;
  const days = [...new Set([a, b].filter((d): d is string => !!d))];
  map.setFilter('fires', days.length ? ['in', ['get', 'date'], ['literal', days]] : ['==', ['get', 'date'], '-']);
  const op = (scale: number) => (a && b && a !== b
    ? ['case', ['==', ['get', 'date'], b], scale * q, scale * (1 - q)]
    : scale * (a && a === b ? 1 : a ? 1 - q : q)) as never;
  map.setPaintProperty('fires', 'circle-opacity', op(.8));
  map.setPaintProperty('fires', 'circle-stroke-opacity', op(1));
}
// Pan (without zooming) to a timeline checkpoint.
export function panToEvent(at: [number, number]) {
  map.easeTo({ center: at, duration: 1200, easing: easeInOut, essential: true });
}

// Forecast limited to a country (zone picked by clicking it): the fire heatmap, fire dots, weather layers and
// wind streaks are all cut to its borders. null lifts the limit.
let clip: { geometry: Geometry; inside: (lon: number, lat: number) => boolean } | null = null;
export function setClipCountry(id: number | null, part: number | null = null) {
  const c = id == null ? null : countryById(id);
  const geometry = c ? territoryGeometry(c, part) : null;     // one territory (e.g. French Guiana) or the whole country
  clip = geometry ? { geometry, inside: shapeTest(geometry) } : null;
  drawFires();
}
export function setRisk(data: FC) {
  riskSquares = data;
  setData('risk', data);
  drawRisk();
}
export function setRiskDay(i: number) {
  riskProp = `p${i + 1}`;
  drawRisk();
  mapReady.then(() => map.setFilter('risk-fill', ['>=', ['get', riskProp], 0.1]));
}

// ---------------- Weather layers ----------------
// Each weather variable is its own layer, switched on and off separately: temperature and humidity as heatmaps,
// wind as a speed heatmap with animated streaks flowing the way it blows (windFlow.ts). They sit under the fire
// heatmap and dots, so those stay readable, and fade out towards the zone's border. Hovering the zone shows the
// values of the layers that are on.
export type WeatherLayers = Record<WeatherVar, boolean>;
const NO_LAYERS: WeatherLayers = { temp: false, rh: false, wind: false };
let wxShown: { w: Weather; day: number; layers: WeatherLayers } | null = null;
let wxTip: maplibregl.Popup | null = null;
let windFlow: WindFlow | null = null;
export function setWeather(w: Weather | null, dayIdx = 0, layers: WeatherLayers = NO_LAYERS) {
  const any = !!w && Object.values(layers).some(Boolean);
  wxShown = any ? { w: w!, day: dayIdx, layers } : null;
  mapReady.then(() => {
    // Under the fire heatmap if it's there (so fire stays on top), otherwise under the fire-cell layer.
    const before = map.getLayer('risk-heat') ? 'risk-heat' : 'risk-fill';
    for (const v of ['temp', 'rh', 'wind'] as WeatherVar[])
      setImageLayer(`wx-${v}`, any && layers[v] ? weatherImage(w!, dayIdx, v, clip?.geometry ?? null) : null, v === 'wind' ? .45 : .55, before);
    windFlow ??= new WindFlow(map);
    windFlow.show(any && layers.wind ? w : null, dayIdx, clip?.inside ?? null);
    if (!any) wxTip?.remove();
  });
}
function setupWeatherTip() {
  wxTip = new maplibregl.Popup({ closeButton: false, closeOnClick: false, className: 'wx-tip', offset: 14, maxWidth: 'none' });
  map.on('mousemove', e => {
    if (!wxShown || drawing || pinMode) { wxTip?.remove(); return; }
    const { w, day, layers } = wxShown;
    const parts: string[] = [];
    if (clip && !clip.inside(e.lngLat.lng, e.lngLat.lat)) { wxTip?.remove(); return; }   // outside the country
    for (const v of ['temp', 'rh', 'wind'] as WeatherVar[]) {
      if (!layers[v]) continue;
      const x = weatherAt(w, day, v, e.lngLat.lng, e.lngLat.lat);
      if (x != null) parts.push(`${WEATHER_VARS[v].label} <b>${Math.round(x)} ${WEATHER_VARS[v].unit}</b>`);
    }
    if (!parts.length) { wxTip?.remove(); return; }
    wxTip!.setLngLat(e.lngLat).setHTML(parts.join(' · '));
    if (!wxTip!.isOpen()) wxTip!.addTo(map);
  });
  map.getCanvas().addEventListener('mouseleave', () => wxTip?.remove());
}

// ---------------- Camera helpers ----------------
export function fitZone(z: Zone, padding: number | maplibregl.PaddingOptions, duration: number) {
  stopSpin();
  // A gentle arc (curve < 1.42 means less zooming out mid-flight) with an eased start and landing.
  map.fitBounds([[z.w, z.s], [z.e, z.n]], { padding, duration, essential: true, curve: 1.2, easing: easeInOut });
}
export function flyTo(opts: maplibregl.FlyToOptions) { map.flyTo({ curve: 1.2, easing: easeInOut, ...opts }); }
export function cameraSnapshot() {
  const c = map.getCenter();
  return { center: [c.lng, c.lat] as [number, number], zoom: map.getZoom(), bearing: map.getBearing(), pitch: map.getPitch() };
}
// ---------------- Drawing and pin modes ----------------
export function setDrawing(on: boolean) {
  if (on && pinMode) setPinMode(false);
  drawing = on;
  stopSpin();
  $('map').classList.toggle('drawing', on);
  $('maptip').textContent = 'Drag to draw the prediction zone · Esc to cancel';
  $('maptip').style.display = on ? 'block' : 'none';
  if (on) map.dragPan.disable(); else map.dragPan.enable();
  if (!on) setData('draft', EMPTY);
  drawStart = null;
  useApp.setState({ drawing: on });
}
export function setPinMode(on: boolean) {
  if (on && drawing) setDrawing(false);
  pinMode = on;
  stopSpin();
  $('map').classList.toggle('drawing', on);          // crosshair cursor
  $('maptip').textContent = 'Click on the map to drop the pin · Esc to cancel';
  $('maptip').style.display = on ? 'block' : 'none';
  useApp.setState({ pinMode: on });
}
export const isDrawing = () => drawing;
export const isPinMode = () => pinMode;

function makePinElement() {
  const el = document.createElement('div');
  el.className = 'map-pin';
  el.innerHTML = `<svg viewBox="0 0 28 40" width="28" height="40" aria-hidden="true">
    <defs><linearGradient id="pinGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#fbbf24"/><stop offset=".55" stop-color="#f97316"/><stop offset="1" stop-color="#dc2626"/></linearGradient></defs>
    <path d="M14 1C7 1 1.5 6.5 1.5 13.5 1.5 23 14 39 14 39S26.5 23 26.5 13.5C26.5 6.5 21 1 14 1Z" fill="url(#pinGrad)" stroke="#fff" stroke-width="2"/>
    <circle cx="14" cy="13.5" r="4.5" fill="#fff"/></svg>`;
  el.title = 'Drag to move the zone';
  return el;
}
function placePin(lngLat: [number, number]) {
  if (!pin) {
    pin = new maplibregl.Marker({ element: makePinElement(), anchor: 'bottom', draggable: true });
    // While dragging, preview the zone outline; apply it when the pin is dropped.
    pin.on('drag', () => {
      const a = useApp.getState().readPinArea();
      if (a && pin) setData('draft', boxPolygon(normalise(zoneAroundPoint(pin.getLngLat(), a))));
    });
    pin.on('dragend', () => { setData('draft', EMPTY); useApp.getState().applyPin(false); });
  }
  pin.setLngLat(lngLat).addTo(map);
  useApp.getState().applyPin(true);
}
export function removePin() { if (pin) { pin.remove(); pin = null; } }
export const pinLngLat = (): LngLat | null => pin ? pin.getLngLat() : null;

// ---------------- Map interactions ----------------
function setupInteractions() {
  map.on('mousedown', e => {
    if (!drawing || !lngLatOk(e.lngLat)) return;   // ignore clicks in space beside the globe
    drawStart = { lng: wrapLng(e.lngLat.lng), lat: e.lngLat.lat };
  });
  map.on('mousemove', e => {
    if (!drawing || !drawStart || !lngLatOk(e.lngLat)) return;
    const b = { lng: wrapLng(e.lngLat.lng), lat: e.lngLat.lat };
    setData('draft', boxPolygon({ w: Math.min(drawStart.lng, b.lng), e: Math.max(drawStart.lng, b.lng),
                                  s: Math.min(drawStart.lat, b.lat), n: Math.max(drawStart.lat, b.lat) }));
  });
  map.on('mouseup', e => {
    if (!drawing || !drawStart) return;
    const a = drawStart, b = lngLatOk(e.lngLat) ? { lng: wrapLng(e.lngLat.lng), lat: e.lngLat.lat } : drawStart;
    setDrawing(false);
    useApp.getState().selectZone({ w: a.lng, s: a.lat, e: b.lng, n: b.lat });
  });

  // Pin mode: the click drops the pin (and flies to the zone, outside the map's own event handling).
  map.on('click', e => {
    if (!pinMode || !lngLatOk(e.lngLat)) return;
    (e.originalEvent as MouseEvent & { pinPlaced?: boolean }).pinPlaced = true;   // so popups ignore this click
    setPinMode(false);
    const at: [number, number] = [wrapLng(e.lngLat.lng), e.lngLat.lat];
    setTimeout(() => placePin(at), 0);
  });

  // Fire popups.
  map.on('click', 'fires', e => {
    if (drawing || pinMode || (e.originalEvent as MouseEvent & { pinPlaced?: boolean }).pinPlaced) return;
    const f = e.features![0];
    const p = f.properties as Record<string, string | number>;
    const [lon, lat] = (f.geometry as GeoJSON.Point).coordinates;
    new maplibregl.Popup({ offset: 6 }).setLngLat([lon, lat]).setHTML(
      `<b>${esc(p.source)}</b><br>${esc(p.date)} ${esc(p.time)} UTC<br>` +
      `FRP: ${p.frp ? (+p.frp).toFixed(1) + ' MW' : 'n/a'}<br>Confidence: ${esc(p.conf || 'n/a')}<br>${lat.toFixed(3)}, ${lon.toFixed(3)}`
    ).addTo(map);
  });
  map.on('mouseenter', 'fires', () => { if (!drawing && !pinMode) map.getCanvas().style.cursor = 'pointer'; });
  map.on('mouseleave', 'fires', () => { map.getCanvas().style.cursor = ''; });

  // Forecast square popups (when no fire dot is under the cursor).
  map.on('click', 'risk-fill', e => {
    if (drawing || pinMode || (e.originalEvent as MouseEvent & { pinPlaced?: boolean }).pinPlaced ||
        map.queryRenderedFeatures(e.point, { layers: ['fires'] }).length) return;
    const p = e.features![0].properties as Record<string, number | boolean | string>;
    new maplibregl.Popup({ offset: 4 }).setLngLat(e.lngLat).setHTML(
      `<b>Chance of fire</b><br>+1 day: ${pct(+p.p1)}<br>+2 days: ${pct(+p.p2)}<br>+3 days: ${pct(+p.p3)}` +
      (p.now === true || p.now === 'true' ? '<br><i>Burning on the last observed day</i>' : '')).addTo(map);
  });
}
