import type { FC, Zone } from './geo';
import { shapeTest } from './mapImage';

// Country shapes for the "Pick a country" zone method: Natural Earth 1:50m admin-0 countries (public domain),
// shipped with the app in public/countries.geojson, each with a numeric `id` and its English `name`.
// They're loaded the first time the method is used.

type Geometry = { type: 'Polygon'; coordinates: number[][][] } | { type: 'MultiPolygon'; coordinates: number[][][][] };
// Countries with far-off pieces (France's overseas departments, Alaska, Hawaii...) list them as `parts`:
// territories of nearby pieces, each with a name and the indices of its polygons in the country's shape.
export type Territory = { name: string; polys: number[] };
export type Country = { id: number; name: string; geometry: Geometry; parts?: Territory[] };

let loading: Promise<FC> | null = null;
const byId = new Map<number, Country>();

export function loadCountries(): Promise<FC> {
  loading ??= fetch(`${import.meta.env.BASE_URL}countries.geojson`, { cache: 'no-cache' })   // always the current shapes, never a stale cached copy
    .then(r => { if (!r.ok) throw new Error(`Could not load the country shapes (${r.status}).`); return r.json(); })
    .then((fc: FC) => {
      for (const f of fc.features)
        byId.set(f.properties.id, { id: f.properties.id, name: f.properties.name, geometry: f.geometry, parts: f.properties.parts });
      return fc;
    })
    .catch(err => { loading = null; throw err; });
  return loading;
}
export const countryById = (id: number) => byId.get(id) ?? null;

// One entry of the territory picker: one territory of the picked country.
export type Place = { id: number; part: number | null; name: string };

// A territory's land area in km² (outer rings, equirectangular: plenty for telling islets from islands) and how many
// pieces of land it has.
function landOf(c: Country, part: number | null): { km2: number; pieces: number } {
  const g = territoryGeometry(c, part);
  const polys = g.type === 'Polygon' ? [g.coordinates] : g.coordinates;
  let km2 = 0;
  for (const [ring] of polys) {
    const lat = ring.reduce((s, p) => s + p[1], 0) / ring.length;
    const a = Math.abs(ring.reduce((s, p, i) => { const q = ring[(i + 1) % ring.length]; return s + p[0] * q[1] - q[0] * p[1]; }, 0)) / 2;
    km2 += a * 111.32 * 110.57 * Math.cos(lat * Math.PI / 180);
  }
  return { km2, pieces: polys.length };
}
// Worth a place in the picker: a big territory (≥ 5000 km²: Corsica, French Guiana, Alaska, Hawaii, the Canary
// Islands...). Small islands and island groups (Guadeloupe, Réunion, the Azores...) are left out.
const bigEnough = (c: Country, part: number | null) => landOf(c, part).km2 >= 5000;

// What the territory picker offers for a picked country: its big far-apart territories, one by one (France:
// mainland, Corsica, French Guiana). The main territory and the one currently picked (`current`) are always listed.
// Null when there's only the one.
export function placesFor(id: number, current?: number | null): { title: string; places: Place[] } | null {
  const c = byId.get(id);
  if (!c?.parts?.length) return null;
  const places = c.parts.map((t, k): Place => ({ id, part: k, name: t.name }))
    .filter(p => p.part === 0 || p.part === current || bigEnough(c, p.part));
  return places.length < 2 ? null : { title: c.name, places };
}

// The rectangle a country fills, as a zone. A country that crosses the 180° meridian (Russia's far east, Fiji,
// Alaska's Aleutians) would otherwise span the whole world, so then the side holding most of it is kept and
// clipped at ±180°.
export function countryZone(c: Country, part: number | null = null): Zone {
  const g = territoryGeometry(c, part);
  const polys = g.type === 'Polygon' ? [g.coordinates] : g.coordinates;
  const pts = polys.flatMap(p => p[0]);                         // outer rings are enough for the extent
  const box = (list: number[][]) => ({
    w: Math.min(...list.map(p => p[0])), e: Math.max(...list.map(p => p[0])),
    s: Math.min(...list.map(p => p[1])), n: Math.max(...list.map(p => p[1])),
  });
  const all = box(pts);
  if (all.e - all.w <= 180) return all;
  const east = pts.filter(p => p[0] >= 0), west = pts.filter(p => p[0] < 0);
  return box(east.length >= west.length ? east : west);
}

// The shape of one territory of a country (null: the whole country).
export function territoryGeometry(c: Country, part: number | null): Geometry {
  const t = part != null ? c.parts?.[part] : null;
  if (!t || c.geometry.type !== 'MultiPolygon') return c.geometry;
  const all = c.geometry.coordinates;
  return { type: 'MultiPolygon', coordinates: t.polys.map(k => all[k]) };
}

// Which territory a point lies in (or is nearest to), for a click on the map; null if the country has none.
export function territoryAt(c: Country, lon: number, lat: number): number | null {
  if (!c.parts?.length || c.geometry.type !== 'MultiPolygon') return null;
  for (let k = 0; k < c.parts.length; k++) if (shapeTest(territoryGeometry(c, k))(lon, lat)) return k;
  const all = c.geometry.coordinates;                     // just off a small island: the nearest territory
  let best = 0, bestD = Infinity;
  c.parts.forEach((t, k) => {
    for (const i of t.polys) for (const [x, y] of all[i][0]) {
      const dx = ((x - lon + 540) % 360) - 180, d = dx * dx + (y - lat) ** 2;
      if (d < bestD) { bestD = d; best = k; }
    }
  });
  return best;
}

// A territory's outline as an SVG path in a size × size box (for the territory picker's icons). Tiny islets are
// left out so the shape stays readable; longitudes are unwrapped across the 180° line.
export function territoryIcon(c: Country, part: number | null, size = 40, pad = 3): string {
  const g = territoryGeometry(c, part);
  const polys = g.type === 'Polygon' ? [g.coordinates] : g.coordinates;
  const rings = polys.map(p => p[0]);
  const ref = rings.reduce((a, r) => (r.length > a.length ? r : a), rings[0])[0][0];
  const pts = rings.map(r => r.map(([x, y]) => [ref + ((x - ref + 540) % 360) - 180, y]));
  const areaOf = (r: number[][]) => Math.abs(r.reduce((s2, p, i) => { const q = r[(i + 1) % r.length]; return s2 + p[0] * q[1] - q[0] * p[1]; }, 0));
  const biggest = Math.max(...pts.map(areaOf));
  const keep = pts.filter(r => areaOf(r) >= biggest * .004);
  const latC = keep.flat().reduce((s2, p) => s2 + p[1], 0) / keep.flat().length, k = Math.cos(latC * Math.PI / 180);
  const xs = keep.flat().map(p => p[0] * k), ys = keep.flat().map(p => -p[1]);
  const x0 = Math.min(...xs), y0 = Math.min(...ys), span = Math.max(Math.max(...xs) - x0, Math.max(...ys) - y0, 1e-6);
  const sc = (size - 2 * pad) / span, ox = pad + ((size - 2 * pad) - (Math.max(...xs) - x0) * sc) / 2, oy = pad + ((size - 2 * pad) - (Math.max(...ys) - y0) * sc) / 2;
  return keep.map(r => {
    const step = Math.max(1, Math.floor(r.length / 160));              // at most ~160 points per outline
    return r.filter((_, i) => i % step === 0).map(([x, y], i) =>
      `${i ? 'L' : 'M'}${(ox + (x * k - x0) * sc).toFixed(1)} ${(oy + (-y - y0) * sc).toFixed(1)}`).join('') + 'Z';
  }).join('');
}
