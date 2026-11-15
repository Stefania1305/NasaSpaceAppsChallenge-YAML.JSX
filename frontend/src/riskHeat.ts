import type { FC } from './geo';
import { clipToShape, frame, type Frame, type Geometry } from './mapImage';

// The fire forecast as a soft heatmap. Every grid cell spreads its chance of fire outwards with a Gaussian
// fall-off, and where cells overlap the higher value wins (so a lone high-risk cell keeps its colour instead of
// being averaged away). The result is coloured with the same ramp as before (pale yellow → dark red) and its
// opacity follows the value, so zones blend into the map with gradient edges instead of hard outlines.

const STOPS: [number, string][] = [[.10, '#fde68a'], [.25, '#fbbf24'], [.40, '#f97316'], [.60, '#dc2626'], [.85, '#7f1d1d']];
const RGB = STOPS.map(([p, h]) => { const n = parseInt(h.slice(1), 16); return [p, (n >> 16) & 255, (n >> 8) & 255, n & 255]; });

function color(p: number) {
  if (p <= RGB[0][0]) return RGB[0].slice(1);
  for (let i = 1; i < RGB.length; i++) {
    if (p <= RGB[i][0]) {
      const a = RGB[i - 1], b = RGB[i], f = (p - a[0]) / (b[0] - a[0]);
      return [a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f, a[3] + (b[3] - a[3]) * f];
    }
  }
  return RGB[RGB.length - 1].slice(1);
}
const smooth = (a: number, b: number, x: number) => { const t = Math.max(0, Math.min(1, (x - a) / (b - a))); return t * t * (3 - 2 * t); };

type Square = { properties: Record<string, unknown>; geometry: { coordinates: number[][][] } };
// One grid cell with a value 0..1 (a chance of fire, or how strongly it burned).
export type HeatBox = { w: number; s: number; e: number; n: number; p: number };
export type Extent = [number, number, number, number];

// The cells' extent with a margin of 5 cells, so the wide soft edges aren't cut off.
export function heatExtent(boxes: HeatBox[]): Extent | null {
  if (!boxes.length) return null;
  const size = boxes[0].e - boxes[0].w, PAD = 5 * size;
  let W = Infinity, S = Infinity, E = -Infinity, N = -Infinity;
  for (const b of boxes) { W = Math.min(W, b.w); S = Math.min(S, b.s); E = Math.max(E, b.e); N = Math.max(N, b.n); }
  return [W - PAD, S - PAD, E + PAD, N + PAD];
}

// The heatmap image for one forecast day (`prop` = p1, p2 or p3) and the corners it covers, or null if empty.
// `clip`: a country's shape; then only what lies inside its borders is drawn (with a soft edge).
export function riskImage(squares: FC, prop: string, clip: Geometry | null = null) {
  const feats = squares.features as Square[];
  const boxes = feats.map(f => { const c = f.geometry.coordinates[0]; return { w: c[0][0], s: c[0][1], e: c[2][0], n: c[2][1], p: Number(f.properties[prop]) || 0 }; });
  const ext = heatExtent(boxes);
  return ext ? heatImage(boxes, ext, clip) : null;
}

// The soft heatmap of `boxes` over `extent` (frames of an animation share one extent so they line up).
export function heatImage(boxes: HeatBox[], extent: Extent, clip: Geometry | null = null, longest = 1600) {
  const f = heatFrame(boxes, extent, longest);
  const field = heatField(boxes, f);
  const cv = document.createElement('canvas');
  cv.width = f.width; cv.height = f.height;
  const ctx = cv.getContext('2d')!, img = ctx.createImageData(f.width, f.height);
  colorize(field, img.data);
  ctx.putImageData(img, 0, 0);
  if (clip) clipToShape(ctx, f, clip, featherPx(boxes, f));
  return { url: cv.toDataURL('image/png'), coordinates: f.coordinates };
}

// The image frame for `boxes` over `extent`: up to 5 pixels per cell, at most `longest` pixels on the longer
// side, rows spaced in Mercator (mapImage.ts).
export function heatFrame(boxes: HeatBox[], extent: Extent, longest = 1600): Frame {
  const [W, S, E, N] = extent;
  const size = boxes.length ? boxes[0].e - boxes[0].w : (E - W) / 100;
  return frame(W, S, E, N, Math.min(longest, Math.max(64, Math.round((E - W) / size) * 5)));
}
const cellPxOf = (boxes: HeatBox[], f: Frame) => (boxes.length ? boxes[0].e - boxes[0].w : (f.E - f.W) / 100) * f.pxPerDeg;
export const featherPx = (boxes: HeatBox[], f: Frame) => Math.max(1.5, cellPxOf(boxes, f) * .6);

// The heat on the frame's pixels, as two grids:
//  - `v`, the value (0..1: a chance of fire, or how strongly it burned): a smooth weighted average of the nearby
//    cells, so a zone of 15% cells stays 15% and a real hotspot stays hot (values are never inflated);
//  - `c`, the coverage (0..1): how much fire area is around, which sets how solid the zone looks. Neighbouring
//    cells' wide soft footprints add up (1 - (1-a)(1-b)), so they melt into one continuous zone with soft edges.
// A final light blur smooths what's left of the seams.
export type HeatField = { v: Float32Array; c: Float32Array };
export function heatField(boxes: HeatBox[], f: Frame): HeatField {
  const { width, height } = f;
  const sumPW = new Float32Array(width * height), sumW = new Float32Array(width * height), c = new Float32Array(width * height);
  const cellPx = cellPxOf(boxes, f);                      // a cell's width in pixels
  for (const b of boxes) {
    if (b.p < .05) continue;
    const latC = (b.s + b.n) / 2, stretch = 1 / Math.max(.05, Math.cos(latC * Math.PI / 180));
    // On a Mercator image a cell is 1/cos(latitude) taller than wide, so the spread is too.
    // The Gaussian is separable: weight = gx(x) * gy(y), so each row and column factor is computed once.
    // It's cut at 2.5 sigma, where the weight is under 5% and invisible after colouring.
    const sx = Math.max(1, 1.4 * cellPx), sy = sx * stretch, rx = Math.ceil(2.5 * sx), ry = Math.ceil(2.5 * sy);
    const ix = 1 / (2 * sx * sx), iy = 1 / (2 * sy * sy);
    const cx = f.x((b.w + b.e) / 2), cy = f.y(latC);
    const x0 = Math.max(0, Math.floor(cx - rx)), x1 = Math.min(width - 1, Math.ceil(cx + rx));
    const y0 = Math.max(0, Math.floor(cy - ry)), y1 = Math.min(height - 1, Math.ceil(cy + ry));
    const gx = new Float32Array(x1 - x0 + 1);
    for (let x = x0; x <= x1; x++) { const dx = x + .5 - cx; gx[x - x0] = Math.exp(-dx * dx * ix); }
    for (let y = y0; y <= y1; y++) {
      const dy = y + .5 - cy, gy = Math.exp(-dy * dy * iy), row = y * width;
      for (let x = x0; x <= x1; x++) {
        const w = gx[x - x0] * gy, k = row + x;
        sumPW[k] += b.p * w; sumW[k] += w;
        c[k] += (1 - c[k]) * w;                          // = 1 - (1 - c)(1 - w)
      }
    }
  }
  const v = sumPW;
  for (let k = 0; k < v.length; k++) v[k] = sumW[k] > 1e-6 ? sumPW[k] / sumW[k] : 0;
  const r = Math.max(1, Math.round(cellPx * .7));
  blur(v, width, height, r); blur(c, width, height, r);
  return { v, c };
}

// Two passes of a box blur along each axis (a cheap near-Gaussian), in place.
function blur(field: Float32Array, w: number, h: number, r: number) {
  const tmp = new Float32Array(Math.max(w, h)), win = 2 * r + 1;
  const pass = (len: number, at: (i: number) => number) => {
    let sum = 0;
    for (let i = -r; i <= r; i++) sum += field[at(Math.max(0, Math.min(len - 1, i)))];
    for (let i = 0; i < len; i++) {
      tmp[i] = sum / win;
      sum += field[at(Math.min(len - 1, i + r + 1))] - field[at(Math.max(0, i - r))];
    }
    for (let i = 0; i < len; i++) field[at(i)] = tmp[i];
  };
  for (let round = 0; round < 2; round++) {
    for (let y = 0; y < h; y++) pass(w, x => y * w + x);
    for (let x = 0; x < w; x++) pass(h, y => y * w + x);
  }
}

// Colour and opacity for every heat value, as a lookup table (fast enough to recolour a frame every animation tick).
const LUT_N = 512;
const LUT = (() => {
  const t = new Uint8ClampedArray(LUT_N * 4);
  for (let i = 0; i < LUT_N; i++) {
    const p = i / (LUT_N - 1);
    if (p < .03) continue;
    const c = color(p);
    t[i * 4] = c[0]; t[i * 4 + 1] = c[1]; t[i * 4 + 2] = c[2];
    // Low values are faint, higher ones denser (capped so the map still shows through).
    t[i * 4 + 3] = Math.round(255 * smooth(.03, .12, p) * (.42 + .4 * smooth(.1, .7, p)));
  }
  return t;
})();
// Write the coloured heat of `field` into RGBA pixels: the colour follows the value, the opacity the value's density
// times the coverage. With `blend`, the field is `field` → `blend.to` at `blend.t` (0..1), which is how the timeline
// morphs one day into the next; `mask` (0..255) limits it to a shape.
export function colorize(field: HeatField, out: Uint8ClampedArray, blend?: { to: HeatField; t: number }, mask?: Uint8ClampedArray) {
  const to = blend?.to, t = blend?.t ?? 0, s = LUT_N - 1, V = field.v, C = field.c;
  for (let k = 0; k < V.length; k++) {
    let v = V[k], c = C[k];
    if (to) { v += (to.v[k] - v) * t; c += (to.c[k] - c) * t; }
    const o = k * 4;
    if (c < .02 || v < .03 || (mask && mask[k] === 0)) { out[o + 3] = 0; continue; }
    const i = Math.min(s, Math.round(v * s)) * 4;
    // Coverage eases in, so zone edges fade softly instead of ending at a line.
    const a = LUT[i + 3] * Math.min(1, c * c * (3 - 2 * c) * 1.15);
    out[o] = LUT[i]; out[o + 1] = LUT[i + 1]; out[o + 2] = LUT[i + 2];
    out[o + 3] = mask ? (a * mask[k]) / 255 : a;
  }
}
