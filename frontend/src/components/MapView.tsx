import type { CSSProperties } from 'react';
import { useApp } from '../store';
import { TimelineBar } from './TimelineBar';
import { TerritoryBar } from './TerritoryBar';

// The map area: the space backdrop (nebula, stars, halo, embers) behind the MapLibre map, the drawing tip,
// and in the bottom-right corner the fire legend with the fire timeline bar under it. The engine (src/engine.ts) attaches to these elements by id/class.
export function MapView() {
  const firesShown = useApp(s => s.firesShown);
  const dot = (c: string, r: string) => ({ '--c': c, '--r': r }) as CSSProperties;
  return (
    <div className="mapwrap">
      <div className="space" aria-hidden="true">
        <div className="nebula" />
        <div className="stars far" />
        <div className="stars near" />
        <div className="halo" />
        <canvas id="embers" />
      </div>
      <div id="map" />
      <div id="maptip" className="maptip">Drag to draw the prediction zone · Esc to cancel</div>
      <TerritoryBar />
      <div className="mapcorner">
      <div id="legend" className="legend" hidden={!firesShown}>
        <b>Fire detections</b>
        <small>Fire radiative power (FRP): how much heat the fire gives off</small>
        <ul>
          <li><i style={dot('#facc15', '6px')} />Under 10 MW</li>
          <li><i style={dot('#f97316', '8px')} />10–50 MW</li>
          <li><i style={dot('#dc2626', '10px')} />50–200 MW</li>
          <li><i style={dot('#7f1d1d', '12px')} />Over 200 MW</li>
        </ul>
      </div>
      <TimelineBar />
      </div>
    </div>
  );
}
