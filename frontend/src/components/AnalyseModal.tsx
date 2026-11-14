import { useApp, type Msg } from '../store';

const Message = ({ id, msg }: { id: string; msg: Msg }) => (
  <p className={`msg${msg?.kind === 'ok' ? ' ok' : ''}`} id={id} aria-live="polite">
    {msg?.kind === 'error' ? <span className="error">{msg.text}</span> : msg?.text}
  </p>
);

// The Analyse window: choose the data (harmonised or one sensor), then one "Predict" button loads the data
// (if it isn't loaded already) and runs the prediction.
export function AnalyseModal() {
  const open = useApp(s => s.modalOpen);
  const zone = useApp(s => s.zone);
  const date = useApp(s => s.date);
  const days = useApp(s => s.days);
  const mode = useApp(s => s.mode);
  const sensor = useApp(s => s.sensor);
  const sensors = useApp(s => s.sensors);
  const availability = useApp(s => s.availability);
  const loading = useApp(s => s.loading);
  const loadMsg = useApp(s => s.loadMsg);
  const predicting = useApp(s => s.predicting);
  const predictMsg = useApp(s => s.predictMsg);
  const { closeModal, setMode, setSensor, analyse } = useApp.getState();

  const n = +days;
  const end = date ? new Date(Date.parse(date) + (n - 1) * 864e5).toISOString().slice(0, 10) : '';
  const sub = zone ? `Zone ${zone.w}, ${zone.s} → ${zone.e}, ${zone.n} · ${date} to ${end} (${n} day${n > 1 ? 's' : ''})` : '';
  const a = availability[sensor];
  const sensorAvail = a ? `Available ${a.min} to ${a.max}.` + (sensor === 'LANDSAT_NRT' ? ' Covers the US and Canada only.' : '') : '';
  const busy = loading || predicting;
  // One progress line: the prediction's message once it has started, otherwise the loading message.
  const msg = predictMsg ?? loadMsg;

  return (
    <div id="modal" hidden={!open} onClick={e => { if (e.target === e.currentTarget && !busy) closeModal(); }}>
      <div className="dialog" role="dialog" aria-modal="true" aria-labelledby="modalTitle">
        <header>
          <h2 id="modalTitle">Analyse zone</h2>
          <button type="button" className="x" id="modalClose" aria-label="Close" onClick={closeModal}>×</button>
        </header>
        <p className="sub" id="modalSub">{sub}</p>

        <div className="step">
          <h3>Data to use</h3>
          <label className="choice">
            <input type="radio" name="mode" value="harmonise" checked={mode === 'harmonise'} disabled={busy} onChange={() => setMode('harmonise')} />
            <span><b>Harmonise all sensors</b>
              <small>Fetch every satellite (MODIS, VIIRS, Landsat) and merge them into one consistent dataset.</small></span>
          </label>
          <label className="choice">
            <input type="radio" name="mode" value="fetch" checked={mode === 'fetch'} disabled={busy} onChange={() => setMode('fetch')} />
            <span style={{ flex: 1 }}><b>Fetch one sensor</b>
              <small>Raw detections from a single satellite product.</small>
              <select id="sensor" aria-label="Sensor" value={sensor} disabled={busy} onChange={e => setSensor(e.target.value)}>
                {sensors.map(s => <option key={s}>{s}</option>)}
              </select>
              <small id="sensorAvail">{sensorAvail}</small></span>
          </label>
        </div>

        <div className="step" id="predictStep">
          <p className="hint" style={{ margin: '0 0 6px' }}>Loads the chosen data and predicts how fire will develop in the zone.</p>
          <button type="button" className="go" id="predict" disabled={busy} onClick={analyse}>
            {loading ? 'Loading data…' : predicting ? 'Predicting…' : 'Predict'}
          </button>
          <Message id="predictMsg" msg={msg} />
        </div>
      </div>
    </div>
  );
}
