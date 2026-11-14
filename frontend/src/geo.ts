// Geometry helpers for zones (lat/lon boxes) and map layers.

export type Zone = { w: number; s: number; e: number; n: number };
export type LngLat = { lng: number; lat: number };

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type FC = { type: 'FeatureCollection'; features: any[] };
export const EMPTY: FC = { type: 'FeatureCollection', features: [] };

export const ZONE_COLOR = '#ea580c';

export const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
export const r2 = (v: number) => Math.round(v * 100) / 100;
export const r3 = (v: number) => Math.round(v * 1000) / 1000;      // 0.001° ≈ 100 m: keeps small pin zones accurate

export function normalise(z: Zone): Zone {
  const w = clamp(Math.min(z.w, z.e), -180, 180), e = clamp(Math.max(z.w, z.e), -180, 180);
  const s = clamp(Math.min(z.s, z.n), -90, 90), n = clamp(Math.max(z.s, z.n), -90, 90);
  return { w: r3(w), s: r3(s), e: r3(e), n: r3(n) };
}

export function areaKm2({ w, s, e, n }: Zone) {
  const R = 6371, rad = Math.PI / 180;
  return Math.abs(R * R * (e - w) * rad * (Math.sin(n * rad) - Math.sin(s * rad)));
}

// A lat/lon box as a polygon, densified so its edges follow parallels/meridians on the globe.
export function boxRing({ w, s, e, n }: Zone, N = 32): [number, number][] {
  const pts: [number, number][] = [];
  for (let i = 0; i <= N; i++) pts.push([w + (e - w) * i / N, s]);
  for (let i = 1; i <= N; i++) pts.push([e, s + (n - s) * i / N]);
  for (let i = 1; i <= N; i++) pts.push([e - (e - w) * i / N, n]);
  for (let i = 1; i <= N; i++) pts.push([w, n - (n - s) * i / N]);
  return pts;
}
export function boxPolygon(z: Zone): FC {
  return { type: 'FeatureCollection', features: [{ type: 'Feature', properties: {},
    geometry: { type: 'Polygon', coordinates: [boxRing(z)] } }] };
}

// Square of the given area (km²) centred on a point. 1° latitude ≈ 111.32 km; longitude shrinks with cos(lat).
export function zoneAroundPoint({ lng, lat }: LngLat, km2: number): Zone {
  const half = Math.sqrt(km2) / 2;
  const dLat = half / 111.32;
  const dLng = Math.min(180, half / (111.32 * Math.max(.01, Math.cos(lat * Math.PI / 180))));
  return { w: lng - dLng, e: lng + dLng, s: lat - dLat, n: lat + dLat };
}

export const wrapLng = (lng: number) => ((lng + 540) % 360) - 180;
export const lngLatOk = (ll: LngLat | null | undefined): ll is LngLat =>
  !!ll && Number.isFinite(ll.lng) && Number.isFinite(ll.lat);

