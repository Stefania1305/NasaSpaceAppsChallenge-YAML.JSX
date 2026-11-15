import type maplibregl from 'maplibre-gl';
import type { Weather } from './store';
import { edgeFade, windField } from './weatherHeat';

// Animated wind streaks over the zone, like a TV weather map: particles drift with the wind (interpolated from
// the zone's weather samples) and leave fading trails. Each particle remembers its last few positions on the
// ground (longitude/latitude), and every frame the whole trail is projected to the screen and redrawn. So the
// streaks keep flowing without a break while you zoom or pan, and stay attached to the map as it moves.

const COUNT = 600;            // particles
const TRAIL = 12;             // positions remembered per particle (the streak's length)
const LEVELS = 6;             // opacity steps used to batch the drawing into a few strokes per frame

type P = { pts: [number, number][]; age: number; life: number };

export class WindFlow {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private parts: P[] = [];
  private field: ((lon: number, lat: number) => [number, number] | null) | null = null;
  private bbox: [number, number, number, number] = [0, 0, 1, 1];
  // Optional limit (a country): streaks start only inside it and end when they leave it.
  private inside: ((lon: number, lat: number) => boolean) | null = null;
  private raf = 0;
  private readonly still = matchMedia('(prefers-reduced-motion: reduce)').matches;

  constructor(private map: maplibregl.Map) {
    this.canvas = document.createElement('canvas');
    this.canvas.className = 'windflow';
    map.getContainer().appendChild(this.canvas);
    this.ctx = this.canvas.getContext('2d')!;
    map.on('resize', this.fit);
    // Reduced motion: no animation, but the still streaks are redrawn as the map moves so they stay in place.
    map.on('move', () => { if (this.still && this.field) this.draw(false); });
    this.fit();
  }

  private fit = () => {
    const c = this.map.getCanvas(), r = devicePixelRatio || 1;
    this.canvas.width = c.clientWidth * r; this.canvas.height = c.clientHeight * r;
    this.canvas.style.width = `${c.clientWidth}px`; this.canvas.style.height = `${c.clientHeight}px`;
    this.ctx.setTransform(r, 0, 0, r, 0, 0);
    if (this.field) this.draw(false);
  };

  private spawn(p?: P): P {
    const [W, S, E, N] = this.bbox;
    const q = p ?? { pts: [], age: 0, life: 0 };
    let lon = 0, lat = 0;
    for (let tries = 0; tries < 30; tries++) {              // inside the country, when there is one
      lon = W + Math.random() * (E - W); lat = S + Math.random() * (N - S);
      if (!this.inside || this.inside(lon, lat)) break;
    }
    q.pts = [[lon, lat]];
    q.age = 0; q.life = 50 + Math.random() * 70;
    return q;
  }
  // Fade factor for a point: low near the zone border, so the streaks blend out instead of stopping at a line.
  // Limited to a country, the streaks simply end at its border (the trail already fades towards its tail).
  private fade(lon: number, lat: number) {
    if (this.inside) return this.inside(lon, lat) ? 1 : 0;
    const [W, S, E, N] = this.bbox;
    return edgeFade((lon - W) / (E - W), (lat - S) / (N - S), .14);
  }
  // How far a particle moves per frame for a given wind: a fixed share of the zone, so it reads at any zoom.
  private scale() { const [W, S, E, N] = this.bbox; return Math.max(E - W, N - S) / 9000; }

  // Show the wind for one forecast day (null hides it), optionally limited to a country (`inside`).
  show(w: Weather | null, day = 0, inside: ((lon: number, lat: number) => boolean) | null = null) {
    this.inside = inside;
    cancelAnimationFrame(this.raf);
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    if (!w) { this.field = null; this.canvas.style.display = 'none'; return; }
    this.canvas.style.display = 'block';
    this.field = windField(w, day);
    this.bbox = w.bbox;
    this.parts = Array.from({ length: COUNT }, () => { const p = this.spawn(); p.age = Math.random() * p.life; return p; });
    if (this.still) {
      // One still picture: walk each particle a few steps along the wind so it shows as a short streak.
      for (let i = 0; i < TRAIL; i++) this.advance(false);
      this.draw(false);
      return;
    }
    const frame = () => { this.advance(true); this.draw(true); this.raf = requestAnimationFrame(frame); };
    this.raf = requestAnimationFrame(frame);
  }

  // Move every particle one step along the wind; respawn those that left the zone or reached the end of their life.
  private advance(aging: boolean) {
    if (!this.field) return;
    const k = this.scale();
    for (const p of this.parts) {
      const [lon, lat] = p.pts[p.pts.length - 1];
      const v = this.field(lon, lat);
      if (!v || (aging && ++p.age > p.life) || (this.inside && !this.inside(lon, lat))) { this.spawn(p); continue; }
      const cosLat = Math.max(.2, Math.cos(lat * Math.PI / 180));
      p.pts.push([lon + v[0] * k / cosLat, lat + v[1] * k]);
      if (p.pts.length > TRAIL) p.pts.shift();
    }
  }

  // Redraw all trails at their current place on screen: older parts fainter, the head brightest.
  private draw(animated: boolean) {
    const ctx = this.ctx, map = this.map;
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    if (!this.field) return;
    const paths = Array.from({ length: LEVELS }, () => new Path2D());
    let any = false;
    for (const p of this.parts) {
      if (p.pts.length < 2) continue;
      const head = p.pts[p.pts.length - 1];
      const life = animated ? Math.min(1, p.age / 10, (p.life - p.age) / 10) : 1;   // ease in and out
      const a = life * this.fade(head[0], head[1]);
      if (a < .03) continue;
      let prev = map.project(p.pts[0]);
      for (let i = 1; i < p.pts.length; i++) {
        const cur = map.project(p.pts[i]);
        // Segment opacity: rises towards the head of the trail, times the particle's own fade.
        const level = Math.min(LEVELS - 1, Math.floor(a * (i / (p.pts.length - 1)) * LEVELS));
        if (level > 0) { paths[level].moveTo(prev.x, prev.y); paths[level].lineTo(cur.x, cur.y); any = true; }
        prev = cur;
      }
    }
    if (!any) return;
    ctx.lineWidth = 1.4; ctx.lineCap = 'round'; ctx.lineJoin = 'round';
    for (let l = 1; l < LEVELS; l++) {
      ctx.strokeStyle = `rgba(255, 255, 255, ${(.9 * (l + 1) / LEVELS).toFixed(3)})`;
      ctx.stroke(paths[l]);
    }
  }
}
