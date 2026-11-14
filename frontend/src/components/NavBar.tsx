import { useApp } from '../store';
import { isDesktop } from '../engine';

// The menu bar: three icons stacked in a glass capsule at the left middle of the screen.
//   House → back to the opening screen · Globe → just the spinning globe · Fire → the Fire Pattern Predictor panel.
// An orange highlight slides behind the active icon.

const House = () => (
  <svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <path d="M3.5 11 12 4l8.5 7" /><path d="M6 9.5V20h4.5v-5.5h3V20H18V9.5" />
  </svg>
);
const Globe = () => (
  <svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <circle cx="12" cy="12" r="8.5" /><path d="M3.5 12h17" /><path d="M12 3.5c2.6 2.4 3.9 5.2 3.9 8.5s-1.3 6.1-3.9 8.5c-2.6-2.4-3.9-5.2-3.9-8.5s1.3-6.1 3.9-8.5z" />
  </svg>
);
const Fire = () => (
  <svg className="flame" viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <path d="M12 21c-3.9 0-6.5-2.6-6.5-6.2 0-3.4 2.4-5.4 3.7-8 .5 2 1.6 3 2.6 3.4.1-2.6 1.2-4.9 3-6.7.2 3.4 3.7 5.7 3.7 10.1 0 4-2.6 7.4-6.5 7.4z" />
    <path d="M12 21c-1.6 0-2.7-1.1-2.7-2.7 0-1.6 1.2-2.5 2.1-3.7.5 1.1 1.3 1.6 1.9 1.8.6.9 1.4 1.4 1.4 2.4 0 1.2-1.1 2.2-2.7 2.2z" />
  </svg>
);

export function NavBar() {
  const panelOpen = useApp(s => s.panelOpen);
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
    <nav id="nav" aria-label="Main">
      <span className="nav-ind" style={{ transform: `translateY(${active * 60}px)` }} aria-hidden="true" />
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
