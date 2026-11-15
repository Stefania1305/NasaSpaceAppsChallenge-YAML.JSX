import { useLayoutEffect, useRef } from 'react';
import { useApp } from '../store';
import { isDesktop } from '../engine';

// The menu: three icons, each its own little planet, stacked at the left middle of the screen; while the
// Fire Pattern Predictor is open they dock in a small row above the panel's title.
//   House → back to the opening screen · Globe → just the spinning globe · Fire → the Fire Pattern Predictor panel.
// The active one glows like a small sun.

// Drawn on one 24 px grid with a single stroke weight, rounded joins and a faint two-tone fill inside each shape
// (the `.tone` layer), in the manner of system symbols.
const icon = { viewBox: '0 0 24 24', width: 24, height: 24, fill: 'none', stroke: 'currentColor', strokeWidth: 1.6,
               strokeLinecap: 'round' as const, strokeLinejoin: 'round' as const, 'aria-hidden': true };
const House = () => (
  <svg {...icon}>
    <path className="tone" d="M5.75 10.1 12 5l6.25 5.1V19a1 1 0 0 1-1 1H14.4v-4.6a2.4 2.4 0 0 0-4.8 0V20H6.75a1 1 0 0 1-1-1z" stroke="none" />
    <path d="M3.75 11.1 11.37 4.5a1 1 0 0 1 1.26 0l7.62 6.6" />
    <path d="M5.75 9.6V19a1 1 0 0 0 1 1H9.6v-4.6a2.4 2.4 0 0 1 4.8 0V20h2.85a1 1 0 0 0 1-1V9.6" />
  </svg>
);
const Globe = () => (
  <svg {...icon}>
    <circle className="tone" cx="12" cy="12" r="8.25" stroke="none" />
    <circle cx="12" cy="12" r="8.25" />
    <path d="M3.75 12h16.5" />
    <path d="M12 3.75c2.45 2.3 3.7 5.05 3.7 8.25s-1.25 5.95-3.7 8.25c-2.45-2.3-3.7-5.05-3.7-8.25S9.55 6.05 12 3.75z" />
  </svg>
);
const Fire = () => (
  <svg className="flame" {...icon}>
    <path className="tone" d="M12 20.75c-3.75 0-6.25-2.55-6.25-6 0-3.3 2.3-5.25 3.55-7.75.5 1.9 1.55 2.9 2.5 3.3.1-2.5 1.15-4.7 2.9-6.45.2 3.25 3.55 5.5 3.55 9.75 0 3.85-2.5 7.15-6.25 7.15z" stroke="none" />
    <path d="M12 20.75c-3.75 0-6.25-2.55-6.25-6 0-3.3 2.3-5.25 3.55-7.75.5 1.9 1.55 2.9 2.5 3.3.1-2.5 1.15-4.7 2.9-6.45.2 3.25 3.55 5.5 3.55 9.75 0 3.85-2.5 7.15-6.25 7.15z" />
    <path className="core" d="M12 20.75c-1.5 0-2.55-1.05-2.55-2.55 0-1.5 1.1-2.35 1.95-3.45.45 1.05 1.2 1.5 1.8 1.7.55.85 1.3 1.3 1.3 2.25 0 1.15-1.05 2.05-2.5 2.05z" fill="currentColor" stroke="none" />
  </svg>
);

export function NavBar() {
  const panelOpen = useApp(s => s.panelOpen);
  const navRef = useRef<HTMLElement>(null);
  const before = useRef<DOMRect[] | null>(null);

  // Docking animation. Opening or closing the panel switches the menu instantly between its two layouts (a column
  // at the left edge, a row at the top of the panel). Each planet is then animated from where it was to where it
  // is now ("FLIP"), one after another, so they glide into the row and back out. Where they were is noted just
  // before any click or key press, since the layout has already switched by the time this effect runs.
  const rects = () => [...(navRef.current?.querySelectorAll<HTMLElement>('.nav-btn') ?? [])].map(b => b.getBoundingClientRect());
  useLayoutEffect(() => {
    const note = () => { before.current = rects(); };
    note();
    document.addEventListener('pointerdown', note, true);
    document.addEventListener('keydown', note, true);
    return () => { document.removeEventListener('pointerdown', note, true); document.removeEventListener('keydown', note, true); };
  }, []);
  useLayoutEffect(() => {
    const was = before.current, btns = [...(navRef.current?.querySelectorAll<HTMLElement>('.nav-btn') ?? [])];
    const now = btns.map(b => b.getBoundingClientRect());
    before.current = now;
    if (!was || matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    btns.forEach((b, i) => {
      const a = was[i], c = now[i];
      if (!a || !c.width) return;
      const dx = a.left + a.width / 2 - (c.left + c.width / 2), dy = a.top + a.height / 2 - (c.top + c.height / 2);
      if (Math.abs(dx) + Math.abs(dy) < 1) return;
      b.animate([{ translate: `${dx}px ${dy}px`, scale: `${a.width / c.width}` }, { translate: '0px 0px', scale: '1' }],
                { duration: 700, delay: i * 55, easing: 'cubic-bezier(.16, 1, .3, 1)', fill: 'backwards' });
    });
  }, [panelOpen]);
  const { goHome, showGlobe, openFire } = useApp.getState();
  // Which icon is active: Fire while the panel is open (desktop), otherwise Globe. (Home is the opening screen,
  // where the bar is hidden.)
  const active = panelOpen && isDesktop() ? 2 : 1;
  const items = [
    { id: 'navHome', name: 'Home', label: 'Home: back to the start page', icon: <House />, onClick: goHome },
    { id: 'navGlobe', name: 'Globe', label: 'Globe: just the spinning globe', icon: <Globe />, onClick: showGlobe },
    { id: 'navFire', name: 'Fire Pattern Predictor', label: 'Fire Pattern Predictor', icon: <Fire />, onClick: openFire },
  ];
  return (
    <nav id="nav" ref={navRef} aria-label="Main">
      {items.map((it, i) => (
        <button key={it.id} id={it.id} type="button" className={`nav-btn${i === active ? ' active' : ''}`}
                aria-label={it.label} aria-current={i === active ? 'page' : undefined} onClick={it.onClick}>
          {it.icon}
          <span className="nav-label" aria-hidden="true">{it.name}</span>
        </button>
      ))}
    </nav>
  );
}
