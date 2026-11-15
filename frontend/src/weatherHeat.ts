import type { Weather } from './store';
import { clipToShape, frame, type Geometry } from './mapImage';

// Weather heatmaps: turn the zone's grid of weather samples (nx × ny points, row by row from the south-west)
// into smooth coloured images that cover the zone. Values between the sample points are blended with
// Catmull-Rom (bicubic) interpolation, so the map shows soft gradients rather than blocks, and the image fades
// to transparent towards the zone's border so it blends into the map. Wind also gets a vector field, which the
// moving wind streaks follow (see windFlow.ts).

export type WeatherVar = 'temp' | 'rh' | 'wind';

// Each variable gets one sequential colour ramp, light (low) → dark (high).
export const WEATHER_VARS: Record<WeatherVar, { label: string; unit: string; ramp: string[]; note: string }> = {
  temp: { label: 'Temperature', unit: '°C', ramp: ['#fff7ed', '#fdba74', '#f97316', '#c2410c', '#7c2d12'],
          note: 'Highest temperature of the day' },
  rh:   { label: 'Humidity', unit: '%', ramp: ['#f0f9ff', '#7dd3fc', '#0ea5e9', '#0369a1', '#0c4a6e'],
          note: 'Lowest humidity of the day (drier = lighter)' },
  wind: { label: 'Wind', unit: 'km/h', ramp: ['#f5f3ff', '#c4b5fd', '#8b5cf6', '#6d28d9', '#3b0764'],
          note: 'Strongest wind of the day; the streaks flow the way it blows' },
};

const value = (w: Weather, k: number, day: number, v: WeatherVar) => w.points[k]?.days[day]?.[v] ?? null;

// The colour scale's range: the lowest and highest value over all forecast days, so switching days shows the
// change instead of rescaling every time.
export function weatherRange(w: Weather, v: WeatherVar): [number, number] {
  let lo = Infinity, hi = -Infinity;
  for (const p of w.points) for (const d of p.days) { const x = d[v]; if (x != null) { lo = Math.min(lo, x); hi = Math.max(hi, x); } }
  if (!Number.isFinite(lo)) return [0, 1];
  if (hi - lo < 1e-6) { lo -= 1; hi += 1; }
  return [lo, hi];
}

// The day's values as an ny × nx table (missing samples filled with the day's average).
function table(w: Weather, day: number, v: WeatherVar): number[][] {
  const vals = w.points.map((_, k) => value(w, k, day, v));
  const ok = vals.filter((x): x is number => x != null);
  const mean = ok.length ? ok.reduce((a, b) => a + b, 0) / ok.length : 0;
  return Array.from({ length: w.ny }, (_, j) => Array.from({ length: w.nx }, (_, i) => vals[j * w.nx + i] ?? mean));
}

// Catmull-Rom through four neighbours (the samples off the grid edge repeat the edge one).
const cr = (p0: number, p1: number, p2: number, p3: number, t: number) =>
  .5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t + (-p0 + 3 * p1 - 3 * p2 + p3) * t * t * t);
function sampleTable(g: number[][], u: number, v: number) {
  const ny = g.length, nx = g[0].length;
  u = Math.max(0, Math.min(nx - 1, u)); v = Math.max(0, Math.min(ny - 1, v));
  const i = Math.min(nx - 2, Math.floor(u)), j = Math.min(ny - 2, Math.floor(v));
  const tx = u - i, ty = v - j;
  const at = (a: number, b: number) => g[Math.max(0, Math.min(ny - 1, b))][Math.max(0, Math.min(nx - 1, a))];
  const row = (b: number) => cr(at(i - 1, b), at(i, b), at(i + 1, b), at(i + 2, b), tx);
  return cr(row(j - 1), row(j), row(j + 1), row(j + 2), ty);
}

// Grid coordinates of a point in the zone (sample k sits at the centre of grid cell k).
function gridUV(w: Weather, lon: number, lat: number) {
  const [W, S, E, N] = w.bbox;
  return { u: (lon - W) / (E - W) * w.nx - .5, v: (lat - S) / (N - S) * w.ny - .5,
           inside: lon >= W && lon <= E && lat >= S && lat <= N };
}

// The interpolated value at a point (null outside the zone), for the hover read-out.
export function weatherAt(w: Weather, day: number, v: WeatherVar, lon: number, lat: number): number | null {
  const { u, v: vv, inside } = gridUV(w, lon, lat);
  return inside ? sampleTable(table(w, day, v), u, vv) : null;
}

// 1 inside the zone, easing to 0 over the outer 16% towards each edge (so the heatmap has no hard border).
const ease = (t: number) => { t = Math.max(0, Math.min(1, t)); return t * t * (3 - 2 * t); };
export const edgeFade = (fx: number, fy: number, band = .16) =>
  ease(Math.min(fx, 1 - fx) / band) * ease(Math.min(fy, 1 - fy) / band);

// Wind as a vector field for the day: a function giving the wind at a point as (east, north) components in
// km/h, or null outside the zone. The daily dominant direction is where the wind comes FROM, so it's reversed.
export function windField(w: Weather, day: number) {
  const vals = w.points.map(p => p.days[day]);
  const comp = (f: (sp: number, rad: number) => number) => Array.from({ length: w.ny }, (_, j) => Array.from({ length: w.nx }, (_, i) => {
    const d = vals[j * w.nx + i];
    return d?.wind == null || d.dir == null ? 0 : f(d.wind, d.dir * Math.PI / 180);
  }));
  const U = comp((sp, r) => -sp * Math.sin(r)), V = comp((sp, r) => -sp * Math.cos(r));
  return (lon: number, lat: number): [number, number] | null => {
    const { u, v, inside } = gridUV(w, lon, lat);
    return inside ? [sampleTable(U, u, v), sampleTable(V, u, v)] : null;
  };
}

function hexRGB(h: string) { const n = parseInt(h.slice(1), 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; }
function rampColor(ramp: number[][], t: number) {
  t = Math.max(0, Math.min(1, t)) * (ramp.length - 1);
  const i = Math.min(ramp.length - 2, Math.floor(t)), f = t - i, a = ramp[i], b = ramp[i + 1];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

// The heatmap image (as a data URL) and the zone corners it covers (top-left, top-right, bottom-right, bottom-left).
// Rows are spaced in Mercator (see mapImage.ts). `clip`: a country's shape; then the image is cut to its borders
// (softly) instead of fading towards the zone's rectangle.
export function weatherImage(w: Weather, day: number, v: WeatherVar, clip: Geometry | null = null) {
  const [W, S, E, N] = w.bbox;
  const f = frame(W, S, E, N, 256), { width, height } = f;
  const g = table(w, day, v), [lo, hi] = weatherRange(w, v), ramp = WEATHER_VARS[v].ramp.map(hexRGB);
  const cv = document.createElement('canvas');
  cv.width = width; cv.height = height;
  const ctx = cv.getContext('2d')!, img = ctx.createImageData(width, height);
  for (let y = 0; y < height; y++) {
    const lat = f.lat(y + .5), fy = (lat - S) / (N - S);
    const vv = fy * w.ny - .5;
    for (let x = 0; x < width; x++) {
      const fx = (x + .5) / width, u = fx * w.nx - .5;
      const c = rampColor(ramp, (sampleTable(g, u, vv) - lo) / (hi - lo));
      const o = (y * width + x) * 4;
      img.data[o] = c[0]; img.data[o + 1] = c[1]; img.data[o + 2] = c[2];
      img.data[o + 3] = clip ? 255 : Math.round(255 * edgeFade(fx, 1 - fy));
    }
  }
  ctx.putImageData(img, 0, 0);
  if (clip) clipToShape(ctx, f, clip, 4);
  return { url: cv.toDataURL('image/png'), coordinates: f.coordinates };
}
