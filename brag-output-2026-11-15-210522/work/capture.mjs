// Captures the real app at 1920x1080 in the states the video tours: splash, zone picked, pipeline running,
// predicted evolution, timeline with the weather heatmap, one fire's card on day +3.
import { spawn } from 'child_process';
import { mkdirSync, writeFileSync } from 'fs';
const CHROME = 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const OUT = process.argv[2], PORT = 9381;
const profile = OUT + '/../chrome-capture'; mkdirSync(profile, { recursive: true }); mkdirSync(OUT, { recursive: true });
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
const ev = async expr => (await send('Runtime.evaluate', { expression: expr, awaitPromise: true, returnByValue: true })).result?.result?.value;
const shot = async name => { const r = await send('Page.captureScreenshot', { format: 'png' }); writeFileSync(`${OUT}/${name}.png`, Buffer.from(r.result.data, 'base64')); console.log('shot', name); };
const move = (x, y) => send('Input.dispatchMouseEvent', { type: 'mouseMoved', x, y });
const type = async (sel, text) => { await ev(`(() => { const el = document.querySelector('${sel}'); el.focus(); el.select?.(); })()`); await send('Input.insertText', { text }); };
await send('Runtime.enable');
await send('Emulation.setDeviceMetricsOverride', { width: 1920, height: 1080, deviceScaleFactor: 1, mobile: false });
await send('Page.navigate', { url: 'http://127.0.0.1:5173/app/' }); await sleep(6000);
await shot('splash');
await ev(`document.getElementById('start').click()`); await sleep(5000);
await ev(`document.getElementById('navFire').click()`); await sleep(2500);
await ev(`document.querySelector('#zoneMethod [data-value="coords"]').click()`); await sleep(500);
// a zone not analysed before, so the pipeline really runs
await type('#w', '-61.5'); await type('#e', '-55.5'); await type('#s', '-15.5'); await type('#n', '-9.5'); await ev('document.activeElement.blur()');
await sleep(4500); await move(1300, 560); await sleep(400);
await shot('zone');
await ev(`document.getElementById('run').click()`); await sleep(1000);
await ev(`document.getElementById('predict').click()`);
for (let i = 0; i < 60; i++) { await sleep(250); if (await ev(`document.querySelectorAll('.dialog .pipe-stage.ok').length >= 3`)) break; }
await shot('pipeline');
for (let i = 0; i < 240; i++) { await sleep(500); if (await ev(`document.getElementById('modal').hidden`)) break; }
for (let i = 0; i < 40; i++) { await sleep(500); if (await ev(`!!document.querySelector('.firebar:not(.building)')`)) break; }
await sleep(3500); await move(1300, 560); await sleep(400);
await ev(`[...document.querySelectorAll('#pipelineSection .fgt')].find(e => e.textContent.startsWith('Predicted')).scrollIntoView()`); await sleep(600);
await shot('results');
// weather on, timeline on a predicted day
await ev(`document.getElementById('wx-temp-toggle').click()`); await sleep(300);
await ev(`document.getElementById('wx-wind-toggle').click()`);
for (let i = 0; i < 40; i++) { await sleep(500); if (!(await ev(`!!document.querySelector('.switch.busy')`))) break; }
await ev(`(() => { const s = __app.getState(); const f = s.timeline.frames.findIndex(x => x.kind === 'forecast'); s.seekTimeline(f + 1); })()`); await sleep(2500);
await ev(`[...document.querySelectorAll('#pipelineSection .fgroup')].find(g => g.querySelector('.fgt')?.textContent.startsWith('Weather'))?.scrollIntoView()`); await sleep(600);
await move(1300, 560); await sleep(400);
await shot('timeline');
// one fire's card: the fire that grows most, zoomed in, on day +3
const fid = await ev(`(() => { const p = __app.getState().pipeline; const d3 = p.predictions.filter(r => r.day_ahead === 3).sort((a, b) => b.growth_likely - a.growth_likely)[0]; return d3.fire_id; })()`);
await ev(`(() => { const s = __app.getState(); const r = s.pipeline.predictions.find(x => x.fire_id === ${fid}); s.flyToHotspot(r.centre_lat, r.centre_lon, 12.8); })()`); await sleep(3500);
await ev(`(() => { const s = __app.getState(); const f = s.timeline.frames.findIndex(x => x.kind === 'forecast'); s.seekTimeline(f + 2); })()`); await sleep(1200);
await ev(`__engine.openFireCard(${fid})`); await sleep(800);
await ev(`(() => { const s = __app.getState(); const f = s.timeline.frames.findIndex(x => x.kind === 'forecast'); s.seekTimeline(f + 2); })()`); await sleep(1500);
await move(1600, 900); await sleep(400);
await shot('firecard');
console.log('fire', fid, '|', logs.join(' | ') || 'no errors');
chrome.kill(); process.exit(0);
