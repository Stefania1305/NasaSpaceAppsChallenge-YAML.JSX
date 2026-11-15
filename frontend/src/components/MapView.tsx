import type { CSSProperties } from 'react';
import { useApp } from '../store';
import { TimelineBar } from './TimelineBar';
import { TerritoryBar } from './TerritoryBar';
import { MapHud } from './MapHud';
import { RISK_BANDS } from './Charts';

// The map area: the space backdrop (nebula, stars, halo, embers) behind the MapLibre map, the drawing tip,
// and in the bottom-right corner the fire legend with the fire timeline bar under it. The engine (src/engine.ts) attaches to these elements by id/class.
export function MapView() {
  const firesShown = useApp(s => s.firesShown);
  const heat = useApp(s => !!s.pipeline && !s.pipeline.realOnly);
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
      <MapHud />
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
      <div className="legend" hidden={!heat}>
        <b>Fire heatmap</b>
        <small>Observed fire, then the predicted chance of fire per cell</small>
        <span aria-hidden="true" style={{ display: 'block', height: 6, borderRadius: 3,
          background: `linear-gradient(90deg, ${RISK_BANDS.map(b => b.color).join(', ')})` }} />
        <small style={{ display: 'flex', justifyContent: 'space-between', margin: '4px 0 0' }}><span>low</span><span>high</span></small>
      </div>
      <TimelineBar />
      </div>
    </div>
  );
}
