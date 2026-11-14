// The map engine: everything imperative about the globe and the map, ported from the original app.html.
// React renders the page; this module owns MapLibre, the space backdrop (stars, embers, halo), the spin,
// the glide between layouts, globe sizing, rectangle drawing, the pin and the map popups. It talks to the
// UI through the app store (src/store.ts).

import maplibregl from 'maplibre-gl';
import type { ExpressionSpecification, GeoJSONSource } from 'maplibre-gl';
import { useApp } from './store';
import { EMPTY, ZONE_COLOR, boxPolygon, lngLatOk, normalise, riskColor, wrapLng, zoneAroundPoint,
         type FC, type LngLat, type Zone } from './geo';
import { pct } from './format';

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
const pad = (left = 0) => ({ left, top: 0, right: 0, bottom: 0 });
const panelOpen = () => useApp.getState().panelOpen;
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
    style: {
      version: 8,
      // Globe below zoom 3, fully flat (Web Mercator) from zoom 5; MapLibre morphs between them.
      projection: { type: ['interpolate', ['linear'], ['zoom'], 3, 'vertical-perspective', 5, 'mercator'] },
      sky: { 'atmosphere-blend': ['interpolate', ['linear'], ['zoom'], 0, 1, 3, 1, 5, 0] },
      sources: {
        imagery: { type: 'raster', tileSize: 256, maxzoom: 19,
          tiles: ['https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'],
          attribution: 'Imagery &copy; Esri' },
        osm: { type: 'raster', tileSize: 256, maxzoom: 19,
          tiles: ['https://tile.openstreetmap.org/{z}/{x}/{y}.png'],
          attribution: '&copy; OpenStreetMap contributors' },
        zone: { type: 'geojson', data: EMPTY as never },
        draft: { type: 'geojson', data: EMPTY as never },
        fires: { type: 'geojson', data: EMPTY as never },
        risk: { type: 'geojson', data: EMPTY as never },
      },
      layers: [
        // Street map underneath; satellite imagery on top that fades out as you zoom in.
        { id: 'osm', type: 'raster', source: 'osm' },
        { id: 'imagery', type: 'raster', source: 'imagery',
          paint: { 'raster-opacity': ['interpolate', ['linear'], ['zoom'], 3, 1, 5.5, 0] } },
        { id: 'zone-fill', type: 'fill', source: 'zone', paint: { 'fill-color': ZONE_COLOR, 'fill-opacity': .1 } },
        { id: 'zone-line', type: 'line', source: 'zone',
          paint: { 'line-color': ZONE_COLOR, 'line-width': 2, 'line-dasharray': [3, 2] } },
        { id: 'draft-fill', type: 'fill', source: 'draft', paint: { 'fill-color': ZONE_COLOR, 'fill-opacity': .15 } },
        { id: 'draft-line', type: 'line', source: 'draft', paint: { 'line-color': ZONE_COLOR, 'line-width': 2 } },
        // Predicted fire probability per grid cell (drawn under the detections).
        { id: 'risk-fill', type: 'fill', source: 'risk', paint: {
          'fill-color': riskColor('p1') as ExpressionSpecification,
          'fill-opacity': ['interpolate', ['linear'], ['get', 'p1'], 0.1, .35, 0.6, .7] } },
        { id: 'risk-line', type: 'line', source: 'risk', minzoom: 6,
          paint: { 'line-color': '#7f1d1d', 'line-width': .4, 'line-opacity': .5 } },
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
    if (document.hidden || map.getZoom() > 5.2) {
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

function setupHalo() {
  const halo = document.querySelector<HTMLElement>('.space .halo')!;
  const placeHalo = () => {
    const z = map.getZoom();
    if (z > 4.6) { halo.style.opacity = '0'; return; }
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
    halo.style.opacity = String(Math.max(0, Math.min(1, (4.5 - z) / 2)));
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
  const z0 = map.getZoom();
  let z = z0;
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
  }
  return Number.isFinite(z) ? z : null;                         // null = leave the size alone
}
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
    const have = map.getPadding().left ?? 0, want = appPadding().left;
    if (!isSplash() && Math.abs(have - want) > .5) {
      next.padding = { left: Math.max(0, Math.round((have + (want - have) * Math.min(1, dt * 3)) * 100) / 100),
                       top: 0, right: 0, bottom: 0 };
    }
    if (next.center || next.padding) map.jumpTo(next);
  };
  mapReady.then(() => requestAnimationFrame(spinFrame));

  // Used when the user touches the map: only end the spin (map.stop() here would cancel their own gesture).
  const userStopsSpin = () => { spinning = false; clearTimeout(resumeTimer); };
  ['mousedown', 'touchstart', 'wheel'].forEach(ev => map.getCanvasContainer().addEventListener(ev, userStopsSpin, { passive: true }));
  // A click or tap that doesn't move the map fires no 'moveend', so also schedule the resume on release.
  ['mouseup', 'touchend'].forEach(ev => map.getCanvasContainer().addEventListener(ev, () => { if (!spinning) resumeSpinSoon(); }, { passive: true }));
  // The zoom buttons and keyboard zoom don't touch the canvas, so end the spin for those too.
  map.on('zoomstart', e => { if (e.originalEvent) userStopsSpin(); });
  map.on('moveend', () => { if (!spinning) resumeSpinSoon(); });
}

// Before the app moves the camera itself (fly to a zone, etc.): end the spin and its animation.
export function stopSpin() { if (spinning) { spinning = false; map.stop(); } }

// Panel closed → resume the spin ~1.5 s after the user lets go and the map is at rest.
const panelClosed = () => isDesktop() && !panelOpen() && !isSplash();
function resumeSpinSoon() {
  clearTimeout(resumeTimer);
  if (reducedMotion() || !panelClosed()) return;
  resumeTimer = setTimeout(() => {
    if (!panelClosed() || drawing || pinMode) return;
    if (map.isMoving()) { resumeSpinSoon(); return; }             // still settling: try again a bit later
    spinning = true;                                              // (the frame loop does nothing while zoomed in)
  }, 1500);
}

// ---------------- Glide ----------------
// Move the globe to its place frame by frame: the padding (and optionally the zoom) ease in and out while
// the globe keeps turning at its spin speed. Every glide has an id; its "user grabbed the map" listeners are
// removed when it ends and only act on that same glide. If the user grabs the map mid-glide, control is handed
// over straight away and the frame loop finishes the framing once the map is at rest.
function glideTo(target: { left: number; top: number; right: number; bottom: number }, ms: number, toZoom: number | null = null) {
  const id = ++glideId;
  if (toZoom != null && !Number.isFinite(toZoom)) toZoom = null;   // never feed the map an invalid zoom
  const from = map.getPadding(), fromZoom = map.getZoom(), t0 = performance.now();
  let fromLeft = Number.isFinite(from.left) ? (from.left as number) : target.left;
  fromLeft = Math.max(0, fromLeft);
  // MapLibre rejects any negative padding (even -0.0000014 px) and then every camera move fails.
  const safeLeft = (v: number) => Math.max(0, Math.round(v * 100) / 100);
  const ease = (t: number) => t < .5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
  const zoomAt = (e: number) => toZoom == null ? {} : { zoom: fromZoom + (toZoom - fromZoom) * e };
  let last = t0;
  gliding = paused = true;
  map.stop();
  const grabEvents = ['mousedown', 'touchstart', 'wheel'];
  const box = map.getCanvasContainer();
  const cleanup = () => grabEvents.forEach(ev => box.removeEventListener(ev, abort));
  const finish = () => {
    cleanup();
    if (id !== glideId) return;                                 // a newer glide has taken over
    gliding = paused = false;
    map.jumpTo({ padding: target, ...zoomAt(1) });
  };
  const frame = (now: number) => {
    if (!gliding || id !== glideId) return;
    const k = Math.min(1, (now - t0) / ms), e = ease(k), c = map.getCenter();
    const turn = spinning ? 6 * (now - last) / 1000 : 0;     // same 6°/s as the spin
    last = now;
    map.jumpTo({ center: [c.lng - turn, c.lat], ...zoomAt(e), padding: {
      left: safeLeft(fromLeft + (target.left - fromLeft) * e), top: 0, right: 0, bottom: 0 } });
    if (k < 1) requestAnimationFrame(frame); else finish();
  };
  function abort() {
    cleanup();
    if (id !== glideId || !gliding) return;
    gliding = false; paused = false;
  }
  grabEvents.forEach(ev => box.addEventListener(ev, abort, { passive: true }));
  if (ms > 0) requestAnimationFrame(frame); else finish();
}

// After "Get started": the globe glides from the right-hand side to the centre and grows.
export function introGlide(open: boolean) {
  glideTo(appPadding(), reducedMotion() ? 0 : 2400, globeZoomFor(open));
}

// Panel opened/closed: the globe glides aside to make room (or back to the centre) and is resized to suit.
export function panelChanged(open: boolean) {
  if (!open) {
    if (drawing) setDrawing(false);               // nothing half-started behind a closed panel
    if (pinMode) setPinMode(false);
    if (!reducedMotion()) spinning = true;        // closed panel: the globe always spins
  }
  glideTo(appPadding(), reducedMotion() ? 0 : open ? 1200 : 1000, showingWholeGlobe() ? globeZoomFor(open) : null);
}

// ---------------- Map layers ----------------
export const showZone = (z: Zone | null) => setData('zone', z ? boxPolygon(z) : EMPTY);
// Fire dots + their legend always appear and disappear together.
export function setFires(data: FC) {
  setData('fires', data);
  useApp.setState({ firesShown: data.features.length > 0 });
}
export const setRisk = (data: FC) => setData('risk', data);
export function setRiskDay(i: number) {
  const prop = `p${i + 1}`;
  mapReady.then(() => {
    map.setPaintProperty('risk-fill', 'fill-color', riskColor(prop));
    map.setPaintProperty('risk-fill', 'fill-opacity', ['interpolate', ['linear'], ['get', prop], 0.1, .35, 0.6, .7]);
    map.setFilter('risk-fill', ['>=', ['get', prop], 0.1]);
  });
}

// ---------------- Camera helpers ----------------
export function fitZone(z: Zone, padding: number, duration: number) {
  stopSpin();
  map.fitBounds([[z.w, z.s], [z.e, z.n]], { padding, duration, essential: true });
}
export function flyTo(opts: maplibregl.FlyToOptions) { map.flyTo(opts); }
export function cameraSnapshot() {
  const c = map.getCenter();
  return { center: [c.lng, c.lat] as [number, number], zoom: map.getZoom(), bearing: map.getBearing(), pitch: map.getPitch() };
}
// "Use current view": only the part of the map that isn't behind the side panel (on the globe, the whole view).
export function currentViewZone(): Zone {
  stopSpin();
  let b = map.getBounds();
  if (map.getZoom() >= 5) {
    const c = map.getCanvas(), left = map.getPadding().left ?? 0;
    b = new maplibregl.LngLatBounds(map.unproject([left, c.clientHeight]), map.unproject([c.clientWidth, 0]));
  }
  let w = b.getWest(), e = b.getEast();
  if (e - w >= 360) { w = -180; e = 180; } else { w = wrapLng(w); e = wrapLng(e); if (e < w) { w = -180; e = 180; } }
  return { w, s: b.getSouth(), e, n: b.getNorth() };
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
