import { useApp, type Msg } from '../store';
import { StageList } from './PipelineResults';

const Message = ({ id, msg }: { id: string; msg: Msg }) => (
  <p className={`msg${msg?.kind === 'ok' ? ' ok' : ''}`} id={id} aria-live="polite">
    {msg?.kind === 'error' ? <span className="error">{msg.text}</span> : msg?.text}
  </p>
);

// The Analyse window: choose the data going in (harmonised or one sensor), then "Run pipeline" sends it through the
// Python pipeline stage by stage, with each stage's progress listed as it finishes.
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
  const running = useApp(s => s.pipelineRunning);
  const stages = useApp(s => s.pipeStages);
  const pipeMsg = useApp(s => s.pipeMsg);
  const { closeModal, setMode, setSensor, runPipeline, cancelPipeline } = useApp.getState();

  const n = +days;
  const end = date ? new Date(Date.parse(date) + (n - 1) * 864e5).toISOString().slice(0, 10) : '';
  const sub = zone ? `Zone ${zone.w}, ${zone.s} → ${zone.e}, ${zone.n} · ${date} to ${end} (${n} day${n > 1 ? 's' : ''})` : '';
  const a = availability[sensor];
  const sensorAvail = a ? `Available ${a.min} to ${a.max}.` + (sensor === 'LANDSAT_NRT' ? ' Covers the US and Canada only.' : '') : '';
  const busy = loading || predicting || running;
  void loadMsg; void predictMsg;               // (the prediction flow is kept in the store, not used here)
  const msg = pipeMsg;

  return (
    <div id="modal" hidden={!open}
         onKeyDown={e => {                                      // keep Tab inside the window
           if (e.key !== 'Tab') return;
           const f = [...e.currentTarget.querySelectorAll<HTMLElement>('button, input, select, [tabindex]')].filter(x => !(x as HTMLInputElement).disabled && x.tabIndex >= 0);
           if (!f.length) return;
           const first = f[0], last = f[f.length - 1];
           if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
           else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
           else if (!e.currentTarget.contains(document.activeElement)) { e.preventDefault(); first.focus(); }
         }}
         onClick={e => { if (e.target === e.currentTarget && !busy) closeModal(); }}>
      <div className="dialog" role="dialog" aria-modal="true" aria-labelledby="modalTitle">
        <header>
          <h2 id="modalTitle">Analyse zone</h2>
          <button type="button" className="x" id="modalClose" aria-label="Close" onClick={() => { if (!busy) closeModal(); }}>×</button>
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
          <h3>Pipeline</h3>
          <p className="hint" style={{ margin: '0 0 10px' }}>The data goes through each stage in turn: correlation, the interpreter,
            localisation and fire power side by side, the epicenters, the fire expansion and the knowledge graph; then the
            prediction model, with the weather forecast, predicts how each fire evolves over the next 3 days.</p>
          {stages.length > 0 && <StageList stages={stages} />}
          <button type="button" className="go" id="predict" disabled={busy} onClick={runPipeline}>
            {running ? 'Running the pipeline…' : 'Run pipeline'}
          </button>
          {running && <button type="button" className="secondary wide" style={{ marginTop: 8 }} onClick={cancelPipeline}>Cancel</button>}
          <Message id="predictMsg" msg={msg} />
        </div>
      </div>
    </div>
  );
}
