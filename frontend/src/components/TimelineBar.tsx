import { useEffect, useRef, type ReactNode } from 'react';
import { useApp } from '../store';
import * as engine from '../engine';
import type { EventKind, TLEvent } from '../timeline';

// The fire timeline bar (bottom-right, under the legend): play / pause, the current day, and a track with one
// stop per day (observed days, then the forecast) and checkpoints for notable moments. Playback runs on
// requestAnimationFrame and moves the map and the playhead directly (no React re-render per frame).

const STEP_S = 1.6;         // seconds per day
const HOLD_END_S = 1.6;     // pause on the last day before looping

const ICONS: Record<EventKind, ReactNode> = {
  start: <path d="M12 3v5M12 16v5M3 12h5M16 12h5M6 6l3 3M15 15l3 3M18 6l-3 3M9 15l-3 3" />,
  merge: <><path d="M5 5c0 6 7 6 7 12M19 5c0 6-7 6-7 12" /><path d="M12 17v4" /></>,
  peak: <path d="M4 19l8-14 8 14z" />,
  jump: <><path d="M5 19L19 5" /><path d="M9 5h10v10" /></>,
  forecast: <><circle cx="12" cy="12" r="7" strokeDasharray="3 3" /><circle cx="12" cy="12" r="2" /></>,
  risk: <path d="M12 21c-3.9 0-6.5-2.6-6.5-6.2 0-3.4 2.4-5.4 3.7-8 .5 2 1.6 3 2.6 3.4.1-2.6 1.2-4.9 3-6.7.2 3.4 3.7 5.7 3.7 10.1 0 4-2.6 7.4-6.5 7.4z" />,
  'merge-risk': <><path d="M5 5c0 6 7 6 7 12M19 5c0 6-7 6-7 12" strokeDasharray="2.5 2.5" /><path d="M12 17v4" /></>,
};

export function TimelineBar() {
  const visible = useApp(s => s.forecastVisible);
  const tl = useApp(s => s.timeline);
  const building = useApp(s => s.timelineBuilding);
  const playing = useApp(s => s.tlPlaying);
  const frame = useApp(s => s.tlFrame);
  const seek = useApp(s => s.tlSeek);
  const { setTlFrame, setTlPlaying, seekTimeline } = useApp.getState();
  const n = tl?.frames.length ?? 0;
  const pos = useRef(0), hold = useRef(0), shown = useRef(-1);
  const knob = useRef<HTMLSpanElement>(null), fill = useRef<HTMLSpanElement>(null), track = useRef<HTMLDivElement>(null);

  // Put the map and the playhead at position p (frames; fractions blend into the next day).
  const apply = (p: number) => {
    pos.current = p;
    engine.showTimeline(p);
    // Fire dots fade from one day's detections to the next on the same eased curve (none on forecast days).
    const { i, j, t } = engine.timelineBlend(p, n), fr = tl!.frames;
    engine.setFireBlend(fr[i].kind === 'observed' ? fr[i].date : null, fr[j].kind === 'observed' ? fr[j].date : null, t);
    const x = n > 1 ? (p / (n - 1)) * 100 : 0;
    if (knob.current) knob.current.style.left = `${x}%`;
    if (fill.current) fill.current.style.width = `${x}%`;
    const f = Math.min(n - 1, Math.round(p));                 // the day that's (mostly) on screen
    if (f !== shown.current) { shown.current = f; setTlFrame(f); }
  };

  // A jump requested from elsewhere (new timeline, day tabs, checkpoints).
  useEffect(() => { if (seek && n) { hold.current = 0; shown.current = -1; apply(Math.min(n - 1, seek.frame)); } }, [seek, n]);

  // Playback loop.
  useEffect(() => {
    if (!playing || n < 2) return;
    let last = performance.now(), raf = 0;
    if (pos.current >= n - 1) { pos.current = 0; hold.current = 0; }   // play again from the start
    const tick = (now: number) => {
      const dt = Math.min(.1, (now - last) / 1000); last = now;
      let p = pos.current;
      if (p >= n - 1) {
        hold.current += dt;
        if (hold.current >= HOLD_END_S) { hold.current = 0; p = 0; }
      } else p = Math.min(n - 1, p + dt / STEP_S);
      apply(p);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing, n]);

  // Scrubbing: click or drag along the track (pauses; snaps to the nearest day on release).
  const fromPointer = (clientX: number) => {
    const r = track.current!.getBoundingClientRect();
    return Math.max(0, Math.min(1, (clientX - r.left) / r.width)) * (n - 1);
  };
  const onPointerDown = (e: React.PointerEvent) => {
    if (n < 2) return;
    setTlPlaying(false);
    (e.target as HTMLElement).setPointerCapture?.(e.pointerId);
    apply(fromPointer(e.clientX));
    const move = (ev: PointerEvent) => apply(fromPointer(ev.clientX));
    const up = (ev: PointerEvent) => {
      apply(Math.round(fromPointer(ev.clientX)));
      removeEventListener('pointermove', move); removeEventListener('pointerup', up);
    };
    addEventListener('pointermove', move); addEventListener('pointerup', up);
  };
  const onKey = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
      e.preventDefault();
      seekTimeline(Math.max(0, Math.min(n - 1, Math.round(pos.current) + (e.key === 'ArrowRight' ? 1 : -1))));
    } else if (e.key === ' ' || e.key === 'Enter') { e.preventDefault(); setTlPlaying(!playing); }
  };
  const jumpTo = (ev: TLEvent) => { seekTimeline(ev.frame); if (ev.at) engine.panToEvent(ev.at); };

  if (!visible || (!tl && !building)) return null;
  if (!tl) return <div className="firebar glass building">Preparing the fire timeline…</div>;

  const fr = tl.frames[frame];
  const firstForecast = tl.frames.findIndex(f => f.kind === 'forecast');
  const at = (i: number) => (n > 1 ? (i / (n - 1)) * 100 : 0);
  return (
    <div className="firebar glass" role="group" aria-label="Fire timeline">
      <div className="tl-head">
        <button type="button" className="tl-play" onClick={() => setTlPlaying(!playing)}
                aria-label={playing ? 'Pause' : 'Play'} title={playing ? 'Pause' : 'Play'}>
          {playing
            ? <svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true"><rect x="6" y="5" width="4" height="14" rx="1.2" /><rect x="14" y="5" width="4" height="14" rx="1.2" /></svg>
            : <svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true"><path d="M8 5.5v13a1 1 0 0 0 1.5.9l10-6.5a1 1 0 0 0 0-1.7l-10-6.5A1 1 0 0 0 8 5.5z" /></svg>}
        </button>
        <div className="tl-now">
          <b>{fr?.label.split(' · ')[0]}</b>
          <span className={fr?.kind === 'forecast' ? 'fc' : ''}>{fr?.label.split(' · ')[1]}</span>
        </div>
        <span className="tl-count">{fr ? (fr.kind === 'forecast' ? `~${fr.cells.toLocaleString()} expected` : `${fr.cells.toLocaleString()} burning`) : ''}<br />cells</span>
      </div>

      {/* Checkpoints above the track */}
      <div className="tl-events">
        {tl.events.map((ev, k) => {
          const twin = tl.events.filter(e => e.frame === ev.frame);
          const offset = twin.length > 1 ? (twin.indexOf(ev) ? 11 : -11) : 0;   // two on one day sit side by side
          return (
            <button key={k} type="button" className={`tl-ev ${ev.kind}`} style={{ left: `calc(${at(ev.frame)}% + ${offset}px)` }}
                    onClick={() => jumpTo(ev)} aria-label={`${ev.title}: ${ev.text}`}>
              <svg viewBox="0 0 24 24" width="11" height="11" fill="none" stroke="currentColor" strokeWidth="2.6"
                   strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{ICONS[ev.kind]}</svg>
              <span className="tl-tip" role="tooltip"><b>{ev.title}</b>{tl.frames[ev.frame]?.label.split(' · ')[0]}<br />{ev.text}
                {ev.at && <em>Click to go there</em>}</span>
            </button>
          );
        })}
      </div>

      {/* The track: observed days solid, forecast days hatched; drag or click to scrub */}
      <div className="tl-track" ref={track} role="slider" tabIndex={0} aria-label="Day" aria-valuemin={0} aria-valuemax={n - 1}
           aria-valuenow={frame} aria-valuetext={fr?.label} onPointerDown={onPointerDown} onKeyDown={onKey}>
        <span className="tl-rail" />
        {firstForecast > 0 && <span className="tl-fc" style={{ left: `${at(firstForecast - .5)}%` }} />}
        <span className="tl-fill" ref={fill} />
        {tl.frames.map((_, i) => <span key={i} className={`tl-tick${i <= frame ? ' past' : ''}`} style={{ left: `${at(i)}%` }} />)}
        <span className="tl-knob" ref={knob} />
      </div>
      <div className="tl-days" aria-hidden="true">
        {tl.frames.map((f, i) => (
          <span key={i} style={{ left: `${at(i)}%` }} className={`${i === frame ? 'on' : ''}${f.kind === 'forecast' ? ' fc' : ''}`}>
            {f.kind === 'forecast' ? `+${f.index + 1}` : Number(f.date.slice(8))}
          </span>
        ))}
      </div>
      <div className="tl-key"><span><i className="obs" />Observed</span><span><i className="fcst" />Forecast</span></div>
    </div>
  );
}
