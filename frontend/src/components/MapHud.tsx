import { useEffect, useState } from 'react';
import { currentZoom, groundAt } from '../engine';

// The map's status line (bottom-left), as in satellite viewers: a live UTC clock, the ground point under the
// mouse and the zoom level. Read-only: it never catches the mouse, so the map below works as before.
const pad = (n: number) => String(n).padStart(2, '0');
const coord = (v: number, pos: string, neg: string) => `${Math.abs(v).toFixed(2)}° ${v >= 0 ? pos : neg}`;

export function MapHud() {
  const [now, setNow] = useState(() => new Date());
  const [at, setAt] = useState<[number, number] | null>(null);
  const [zoom, setZoom] = useState(0);
  useEffect(() => {
    const t = setInterval(() => { setNow(new Date()); setZoom(currentZoom()); }, 500);
    const el = document.getElementById('map');
    let raf = 0;
    const move = (e: MouseEvent) => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(() => {
        const r = el!.getBoundingClientRect();
        setAt(groundAt(e.clientX - r.left, e.clientY - r.top));
        setZoom(currentZoom());
      });
    };
    const leave = () => setAt(null);
    el?.addEventListener('mousemove', move);
    el?.addEventListener('mouseleave', leave);
    return () => { clearInterval(t); cancelAnimationFrame(raf); el?.removeEventListener('mousemove', move); el?.removeEventListener('mouseleave', leave); };
  }, []);
  const utc = `${pad(now.getUTCHours())}:${pad(now.getUTCMinutes())}:${pad(now.getUTCSeconds())}`;
  return (
    <div className="maphud" aria-hidden="true">
      <span className="hud-live"><i />Live</span>
      <span className="hud-cell"><em>UTC</em>{utc}</span>
      <span className="hud-cell"><em>Cursor</em>{at ? `${coord(at[1], 'N', 'S')}  ${coord(at[0], 'E', 'W')}` : '—'}</span>
      <span className="hud-cell"><em>Zoom</em>{zoom.toFixed(1)}</span>
    </div>
  );
}
