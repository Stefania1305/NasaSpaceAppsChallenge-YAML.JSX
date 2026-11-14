import type { FC } from './geo';

// Turns the forecast grid (one square per cell) into smooth, rounded risk zones for the map.
// For each chance band, the cells at or above the band's lower bound are merged into regions, each region's
// outline is traced (holes included), and the outline is rounded off with Chaikin corner-cutting. The bands
// are stacked, so a zone's core shows the deeper colour of its higher-risk cells.

export const ZONE_BANDS = [
  { from: .10, color: '#fde68a', edge: '#e0a00b' },     // edge: a darker outline so pale zones stand out on the map
  { from: .25, color: '#fbbf24', edge: '#d97706' },
  { from: .40, color: '#f97316', edge: '#c2410c' },
  { from: .60, color: '#dc2626', edge: '#991b1b' },
  { from: .85, color: '#7f1d1d', edge: '#450a0a' },
];

type Pt = [number, number];
const key = (x: number, y: number) => `${x},${y}`;

// Outlines of the union of the given grid cells (integer cell indices), as closed rings of grid corners.
// Each cell contributes its edges that face a cell outside the set, oriented counter-clockwise (inside on the
// left), so outer rings come out counter-clockwise and holes clockwise.
function traceRings(cells: Set<string>): Pt[][] {
  const out = new Map<string, Pt[]>();                    // start corner → edge end corners
  const add = (a: Pt, b: Pt) => { const k = key(a[0], a[1]); (out.get(k) ?? out.set(k, []).get(k)!).push(b); };
  for (const c of cells) {
    const [i, j] = c.split(',').map(Number);
    if (!cells.has(key(i, j - 1))) add([i, j], [i + 1, j]);             // bottom, going east
    if (!cells.has(key(i + 1, j))) add([i + 1, j], [i + 1, j + 1]);     // right, going north
    if (!cells.has(key(i, j + 1))) add([i + 1, j + 1], [i, j + 1]);     // top, going west
    if (!cells.has(key(i - 1, j))) add([i, j + 1], [i, j]);             // left, going south
  }
  const rings: Pt[][] = [];
  for (const [startKey, ends] of out) {
    while (ends.length) {
      const start = startKey.split(',').map(Number) as Pt;
      const ring: Pt[] = [start];
      let prev = start, cur = ends.pop()!;
      for (let guard = 0; guard < 1e6 && key(cur[0], cur[1]) !== startKey; guard++) {
        ring.push(cur);
        const options = out.get(key(cur[0], cur[1]))!;
        // Where two regions touch only at a corner, take the sharpest left turn: it keeps them separate.
        const dx = cur[0] - prev[0], dy = cur[1] - prev[1];
        const rank = (p: Pt) => { const ex = p[0] - cur[0], ey = p[1] - cur[1], cross = dx * ey - dy * ex;
          return cross > 0 ? 0 : cross === 0 ? 1 : 2; };
        options.sort((a, b) => rank(b) - rank(a));
        prev = cur; cur = options.pop()!;
      }
      rings.push(ring);
    }
  }
  return rings;
}

// Drop corners that lie on a straight run, so long straight sides stay straight and only the turns get rounded.
function corners(ring: Pt[]): Pt[] {
  return ring.filter((p, i) => {
    const a = ring[(i - 1 + ring.length) % ring.length], b = ring[(i + 1) % ring.length];
    return (p[0] - a[0]) * (b[1] - p[1]) - (p[1] - a[1]) * (b[0] - p[0]) !== 0;
  });
}

// Chaikin corner-cutting on a closed ring: every pass replaces each corner with two points at 1/4 and 3/4 of
// its edges. A few passes turn a square into a smooth, round blob.
function chaikin(ring: Pt[], passes = 4): Pt[] {
  let r = ring;
  for (let n = 0; n < passes; n++) {
    const next: Pt[] = [];
    for (let i = 0; i < r.length; i++) {
      const a = r[i], b = r[(i + 1) % r.length];
      next.push([a[0] * .75 + b[0] * .25, a[1] * .75 + b[1] * .25], [a[0] * .25 + b[0] * .75, a[1] * .25 + b[1] * .75]);
    }
    r = next;
  }
  return r;
}

const area = (r: Pt[]) => r.reduce((s, p, i) => { const q = r[(i + 1) % r.length]; return s + p[0] * q[1] - q[0] * p[1]; }, 0) / 2;
function inside(p: Pt, r: Pt[]) {
  let hit = false;
  for (let i = 0, j = r.length - 1; i < r.length; j = i++) {
    if ((r[i][1] > p[1]) !== (r[j][1] > p[1]) &&
        p[0] < (r[j][0] - r[i][0]) * (p[1] - r[i][1]) / (r[j][1] - r[i][1]) + r[i][0]) hit = !hit;
  }
  return hit;
}

// The rounded zones for one forecast day (`prop` = p1, p2 or p3), from the grid-cell squares.
export function riskZones(squares: FC, prop: string): FC {
  const feats = squares.features as { properties: Record<string, unknown>; geometry: { coordinates: number[][][] } }[];
  if (!feats.length) return { type: 'FeatureCollection', features: [] };
  // Cell size and integer indices from the squares' corners (w, s, e, n).
  const box = (f: typeof feats[number]) => { const c = f.geometry.coordinates[0]; return { w: c[0][0], s: c[0][1], e: c[2][0], n: c[2][1] }; };
  const b0 = box(feats[0]), size = b0.e - b0.w;
  const cells = feats.map(f => { const b = box(f); return { i: Math.round(b.w / size), j: Math.round(b.s / size), p: Number(f.properties[prop]) || 0 }; });

  const features: FC['features'] = [];
  ZONE_BANDS.forEach((band, level) => {
    const set = new Set(cells.filter(c => c.p >= band.from).map(c => key(c.i, c.j)));
    if (!set.size) return;
    const rings = traceRings(set).map(corners).filter(r => r.length >= 3);
    const outers = rings.filter(r => area(r) > 0), holes = rings.filter(r => area(r) < 0);
    for (const outer of outers) {
      // A hole belongs to the region whose cells border it: test a point just on the filled (left) side of its first edge.
      const own = holes.filter(h => { const [a, b] = h, dx = b[0] - a[0], dy = b[1] - a[1], len = Math.hypot(dx, dy);
        return inside([(a[0] + b[0]) / 2 - dy / len * .25, (a[1] + b[1]) / 2 + dx / len * .25], outer); });
      const toLngLat = (r: Pt[]) => { const s = chaikin(r).map(([x, y]) => [x * size, y * size]); return [...s, s[0]]; };
      features.push({ type: 'Feature', properties: { level, color: band.color, edge: band.edge },
        geometry: { type: 'Polygon', coordinates: [toLngLat(outer), ...own.map(toLngLat)] } } as FC['features'][number]);
    }
  });
  return { type: 'FeatureCollection', features };
}
