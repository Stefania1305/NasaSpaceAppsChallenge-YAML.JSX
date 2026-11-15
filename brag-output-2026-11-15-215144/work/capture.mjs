// Captures the real app (1920x1080) in every state the video tours: splash, country picking (hover + picked),
// pipeline results with the panel, the fire spread on days +1..+3, humidity, wind, a fire's card, the panel's charts.
import { spawn } from 'child_process';
import { mkdirSync, writeFileSync } from 'fs';
import { resolve } from 'path';
const CHROME = 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const OUT = resolve(process.argv[2]), PORT = 9401;
const profile = resolve(OUT, '../chrome-capture'); mkdirSync(profile, { recursive: true }); mkdirSync(OUT, { recursive: true });
const chrome = spawn(CHROME, ['--headless=new', `--remote-debugging-port=${PORT}`, `--user-data-dir=${profile}`, '--no-first-run',
  '--window-size=1920,1080', '--enable-unsafe-swiftshader', '--hide-scrollbars', 'about:blank'], { stdio: 'ignore' });
const sleep = ms => new Promise(r => setTimeout(r, ms));
let page;
for (let i = 0; i < 100 && !page; i++) { try { page = (await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json()).find(t => t.type === 'page'); } catch {} await sleep(200); }
const ws = new WebSocket(page.webSocketDebuggerUrl); await new Promise(r => ws.onopen = r);
let n = 0; const wait = new Map(); const logs = [];
ws.onmessage = e => { const m = JSON.parse(e.data); if (m.id && wait.has(m.id)) { wait.get(m.id)(m); wait.delete(m.id); }
  if (m.method === 'Runtime.exceptionThrown') logs.push('EXC: ' + m.params.exceptionDetails.exception?.description); };
const send = (method, params = {}) => new Promise(r => { const id = ++n; wait.set(id, r); ws.send(JSON.stringify({ id, method, params })); });
const ev = async expr => { const r = await send('Runtime.evaluate', { expression: expr, awaitPromise: true, returnByValue: true }); if (r.result?.exceptionDetails) logs.push('EV: ' + r.result.exceptionDetails.exception?.description); return r.result?.result?.value; };
const shot = async (name, clip) => { const r = await send('Page.captureScreenshot', { format: 'png', ...(clip ? { clip: { ...clip, scale: 1 } } : {}) }); writeFileSync(`${OUT}/${name}.png`, Buffer.from(r.result.data, 'base64')); console.log('shot', name); };
const move = (x, y) => send('Input.dispatchMouseEvent', { type: 'mouseMoved', x, y });
const click = async (x, y) => { await move(x, y); await send('Input.dispatchMouseEvent', { type: 'mousePressed', x, y, button: 'left', clickCount: 1 }); await send('Input.dispatchMouseEvent', { type: 'mouseReleased', x, y, button: 'left', clickCount: 1 }); };
const S = `__app.getState()`;
const fFrame = k => ev(`(() => { const s = ${S}; const f = s.timeline.frames.findIndex(x => x.kind === 'forecast'); s.seekTimeline(f + ${k - 1}); })()`);
const panelTo = sel => ev(`(() => { const el = ${sel}; el?.scrollIntoView({ block: 'start' }); })()`);
const fgt = t => `[...document.querySelectorAll('#pipelineSection .fgt')].find(e => e.textContent.startsWith('${t}'))?.closest('.fgroup')`;

await send('Runtime.enable');
await send('Emulation.setDeviceMetricsOverride', { width: 1920, height: 1080, deviceScaleFactor: 1, mobile: false });
await send('Page.navigate', { url: 'http://127.0.0.1:5173/app/' }); await sleep(6500);
await shot('splash');
await ev(`document.getElementById('start').click()`); await sleep(5000);
await ev(`document.getElementById('navFire').click()`); await sleep(2500);

// 1 country picking: centre the map on Paraguay, hover it (lights up), then click it
await ev(`${S}.flyToHotspot(-23.6, -58.4, 4.6)`); await sleep(3500);
await ev(`document.querySelector('#zoneMethod [data-value="country"]').click()`); await sleep(3000);
// the screen pixel over central Paraguay
const [px, py] = await ev(`(() => { let best = [1160, 560], bd = 1e9; for (let x = 500; x < 1900; x += 10) for (let y = 100; y < 1000; y += 10) {
  const g = __engine.groundAt(x, y); if (!g) continue; const d = (g[0] + 58.4) ** 2 + (g[1] + 23.6) ** 2; if (d < bd) { bd = d; best = [x, y]; } } return best; })()`);
console.log('paraguay at', px, py);
await move(px - 10, py); await sleep(600); await move(px, py); await sleep(900);
await shot('country_hover');
await click(px, py); await sleep(4500);
console.log('country:', await ev(`${S}.zoneCountry`));
if (!(await ev(`${S}.zoneCountry`))) {
  await ev(`(async () => { const fc = await __countries.loadCountries(); const f = fc.features.find(x => x.properties.name === 'Paraguay');
    const c = __countries.countryById(f.properties.id); ${S}.selectZone(__countries.countryZone(c), true, false, { id: c.id, name: c.name, part: null }); })()`);
  await sleep(4000);
}
await move(1500, 300); await sleep(500);
await shot('country_zone');

// 2 the pipeline (saved run: fast) -> results
await ev(`document.getElementById('run').click()`); await sleep(1200);
await shot('analyse');
await ev(`document.getElementById('predict').click()`);
for (let i = 0; i < 400; i++) { await sleep(1000); if (await ev(`document.getElementById('modal').hidden`)) break; }
for (let i = 0; i < 40; i++) { await sleep(500); if (await ev(`!!document.querySelector('.firebar:not(.building)')`)) break; }
await sleep(4000); await move(1500, 250); await sleep(400);
await panelTo(`document.querySelector('#pipelineSection')`); await sleep(500);
await shot('results');

// 3 the spread, day by day: zoom on the fire that grows most by day +3
const fire = JSON.parse(await ev(`(() => { const p = ${S}.pipeline; const d3 = p.predictions.filter(r => r.day_ahead === 3).sort((a, b) => b.footprint_likely_km2 - a.footprint_likely_km2)[0]; return JSON.stringify(d3); })()`));
console.log('fire', fire.fire_id, fire.centre_lat, fire.centre_lon, fire.footprint_likely_km2);
await panelTo(fgt('Predicted')); await sleep(400);
await ev(`${S}.flyToHotspot(${fire.centre_lat}, ${fire.centre_lon}, 7.4)`); await sleep(4000);
for (const k of [0, 1, 2, 3]) {
  if (k === 0) await ev(`(() => { const s = ${S}; const f = s.timeline.frames.findIndex(x => x.kind === 'forecast'); s.seekTimeline(f - 1); })()`);
  else await fFrame(k);
  await sleep(2200); await move(1880, 200); await sleep(300);
  await shot(`spread${k}`);
}

// 4 humidity, then wind (zone view, predicted day +1)
await ev(`${S}.zoomToZone()`); await sleep(3500);
await ev(`document.getElementById('wx-rh-toggle').click()`);
for (let i = 0; i < 40; i++) { await sleep(500); if (!(await ev(`!!document.querySelector('.switch.busy')`))) break; }
await fFrame(1); await sleep(2500);
await panelTo(`document.querySelector('.wxdays')?.closest('.fgroup')`); await sleep(500);
await move(1200, 600); await sleep(600);
await shot('humidity');
await ev(`document.getElementById('wx-rh-toggle').click()`); await sleep(400);
await ev(`document.getElementById('wx-wind-toggle').click()`);
for (let i = 0; i < 40; i++) { await sleep(500); if (!(await ev(`!!document.querySelector('.switch.busy')`))) break; }
await sleep(2500); await panelTo(`document.querySelector('.wxdays')?.closest('.fgroup')`); await sleep(400);
await move(1200, 600); await sleep(400);
await shot('wind');
await sleep(700); await shot('wind_b');
await ev(`document.getElementById('wx-wind-toggle').click()`); await sleep(600);

// 5 a fire's card on day +2
await ev(`${S}.flyToHotspot(${fire.centre_lat}, ${fire.centre_lon}, 11.5)`); await sleep(4000);
await fFrame(2); await sleep(1500);
await move(1880, 200); await sleep(300); await shot('firecard_pre');
await ev(`__engine.openFireCard(${fire.fire_id})`); await sleep(900);
await fFrame(2); await sleep(1800);
await move(1880, 200); await sleep(400);
await shot('firecard');
await ev(`__engine.closeFireCard?.()`); await sleep(300);

// 6 the panel's charts (left 460 px)
await ev(`${S}.zoomToZone()`); await sleep(3000); await fFrame(1); await sleep(1500);
await panelTo(`document.querySelector('.about')`); await sleep(500); await shot('panel_about', { x: 0, y: 0, width: 460, height: 1080 });
await panelTo(fgt('Overview')); await sleep(500); await shot('panel_overview', { x: 0, y: 0, width: 460, height: 1080 });
await panelTo(fgt('Predicted')); await sleep(500); await shot('panel_pred', { x: 0, y: 0, width: 460, height: 1080 });
await shot('panel_pred_full');
console.log(logs.join(' | ') || 'no errors');
chrome.kill(); process.exit(0);
