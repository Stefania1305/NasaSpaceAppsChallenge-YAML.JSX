import { useEffect, useMemo, useRef } from 'react';
import { useApp } from '../store';
import { countryById, placesFor, territoryIcon } from '../countries';

// Territory picker (bottom centre of the map): shown when the picked country has big far-apart territories, e.g.
// France (mainland, Corsica, French Guiana). Each button shows the territory's outline; choosing one makes it the
// zone (the analysis then covers only it) and flies the map there. With a forecast on screen it also runs the
// forecast for that territory, so you can hop between them in one click each.
export function TerritoryBar() {
  const id = useApp(s => s.zoneCountryId);
  const part = useApp(s => s.zoneTerritory);
  const switchingTo = useApp(s => s.switchingTo);
  const switchError = useApp(s => s.switchError);
  const switchTerritory = useApp.getState().switchTerritory;
  const besideTimeline = useApp(s => s.forecastVisible && (!!s.timeline || s.timelineBuilding));
  const group = useMemo(() => (id != null ? placesFor(id, part) : null), [id, part]);
  const icons = useMemo(() => group?.places.map(p => territoryIcon(countryById(p.id)!, p.part)) ?? [], [group]);
  const listRef = useRef<HTMLDivElement>(null);
  useEffect(() => { listRef.current?.querySelector('.tr-item.on')?.scrollIntoView({ block: 'nearest', inline: 'nearest' }); }, [id, part, group]);
  if (!group) return null;
  return (
    <div className={`territories glass${besideTimeline ? ' beside-tl' : ''}`} role="group" aria-label={`Territories of ${group.title}`}>
      <span className="tr-title">{switchingTo ? `Loading the forecast for ${switchingTo}…` : `${group.title} · territories`}</span>
      <div className="tr-list" ref={listRef} onWheel={e => { if (!e.deltaX) e.currentTarget.scrollLeft += e.deltaY; }}>
        {group.places.map((p, k) => {
          const on = p.id === id && p.part === part;
          return (
            <button key={`${p.id}-${p.part}`} type="button" className={`tr-item${on ? ' on' : ''}${switchingTo === p.name ? ' busy' : ''}`}
                    aria-pressed={on} disabled={!!switchingTo} onClick={() => switchTerritory(p.id, p.part)}
                    title={`Go to ${p.name}`}>
              <svg viewBox="0 0 40 40" width="40" height="40" aria-hidden="true">
                <path d={icons[k]} fillRule="evenodd" />
              </svg>
              <span>{p.name}</span>
            </button>
          );
        })}
      </div>
      {switchError && <span className="tr-error">⚠ {switchError}</span>}
    </div>
  );
}
