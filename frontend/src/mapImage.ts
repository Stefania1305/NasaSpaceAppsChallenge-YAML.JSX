// Helpers for the heatmap images laid on the map (fire risk, weather).
//
// MapLibre stretches an image source between its four corners in Web Mercator, so the image's rows must be
// spaced in Mercator, not evenly in latitude; otherwise large or far-north zones (Russia, Canada) would have their
// heat drawn in the wrong places. `frame` gives the image size and the conversions between lon/lat and pixels.
// `clipToShape` keeps only what lies inside a country, with a softly feathered edge.

const RAD = Math.PI / 180;
const mercY = (lat: number) => { const p = Math.max(-85, Math.min(85, lat)) * RAD; return Math.log(Math.tan(Math.PI / 4 + p / 2)); };
const latOf = (y: number) => (2 * Math.atan(Math.exp(y)) - Math.PI / 2) / RAD;

export type Frame = ReturnType<typeof frame>;
export type Geometry = { type: 'Polygon'; coordinates: number[][][] } | { type: 'MultiPolygon'; coordinates: number[][][][] };

// An image covering W..E × S..N whose longer side is `longest` pixels (the shorter one follows the true shape).
export function frame(W: number, S: number, E: number, N: number, longest: number) {
  const yT = mercY(N), yB = mercY(S), spanX = (E - W) * RAD, spanY = Math.max(1e-9, yT - yB);
  const aspect = spanX / spanY;
  const width = Math.max(16, Math.round(aspect >= 1 ? longest : longest * aspect));
  const height = Math.max(16, Math.round(aspect >= 1 ? longest / aspect : longest));
  return {
    W, S, E, N, width, height,
    x: (lon: number) => (lon - W) / (E - W) * width,
    y: (lat: number) => (yT - mercY(lat)) / spanY * height,
    lon: (px: number) => W + px / width * (E - W),
    lat: (py: number) => latOf(yT - py / height * spanY),
    // Pixels per degree of longitude (the map is conformal, so a degree of latitude at `lat` is this / cos(lat)).
    pxPerDeg: width / (E - W),
    coordinates: [[W, N], [E, N], [E, S], [W, S]] as [[number, number], [number, number], [number, number], [number, number]],
  };
}

// Keep only the part of the canvas inside `geometry`, fading out over about `feather` pixels at its border.
export function clipToShape(ctx: CanvasRenderingContext2D, f: Frame, geometry: Geometry, feather = 3) {
  const mask = document.createElement('canvas');
  mask.width = f.width; mask.height = f.height;
  const m = mask.getContext('2d')!;
  if (feather > 0) m.filter = `blur(${feather}px)`;
  m.fillStyle = '#fff';
  m.beginPath();
  const polys = geometry.type === 'Polygon' ? [geometry.coordinates] : geometry.coordinates;
  for (const poly of polys) for (const ring of poly) {
    ring.forEach(([lon, lat], i) => i ? m.lineTo(f.x(lon), f.y(lat)) : m.moveTo(f.x(lon), f.y(lat)));
    m.closePath();
  }
  m.fill('evenodd');
  ctx.save();
  ctx.globalCompositeOperation = 'destination-in';
  ctx.drawImage(mask, 0, 0);
  ctx.restore();
}

// The same soft-edged shape as a per-pixel mask (0..255), for clipping images that are redrawn many times.
export function shapeMask(f: Frame, geometry: Geometry, feather = 3): Uint8ClampedArray {
  const cv = document.createElement('canvas');
  cv.width = f.width; cv.height = f.height;
  const ctx = cv.getContext('2d')!;
  ctx.fillStyle = '#fff';
  ctx.fillRect(0, 0, f.width, f.height);
  clipToShape(ctx, f, geometry, feather);
  const px = ctx.getImageData(0, 0, f.width, f.height).data, out = new Uint8ClampedArray(f.width * f.height);
  for (let k = 0; k < out.length; k++) out[k] = px[k * 4 + 3];
  return out;
}

// A fast point-in-shape test (even-odd rule; edges grouped into latitude bands so only nearby edges are checked).
export function shapeTest(geometry: Geometry) {
  const BAND = .25, bands = new Map<number, number[][]>();
  let w = Infinity, s = Infinity, e = -Infinity, n = -Infinity;
  const polys = geometry.type === 'Polygon' ? [geometry.coordinates] : geometry.coordinates;
  for (const poly of polys) for (const ring of poly) for (let i = 0; i < ring.length; i++) {
    const [x1, y1] = ring[i], [x2, y2] = ring[(i + 1) % ring.length];
    w = Math.min(w, x1); e = Math.max(e, x1); s = Math.min(s, y1); n = Math.max(n, y1);
    if (y1 === y2) continue;
    for (let b = Math.floor(Math.min(y1, y2) / BAND); b <= Math.floor(Math.max(y1, y2) / BAND); b++) {
      if (!bands.has(b)) bands.set(b, []);
      bands.get(b)!.push([x1, y1, x2, y2]);
    }
  }
  return (lon: number, lat: number) => {
    if (lon < w || lon > e || lat < s || lat > n) return false;
    let inside = false;
    for (const [x1, y1, x2, y2] of bands.get(Math.floor(lat / BAND)) ?? [])
      if ((y1 > lat) !== (y2 > lat) && lon < x1 + (lat - y1) * (x2 - x1) / (y2 - y1)) inside = !inside;
    return inside;
  };
}
