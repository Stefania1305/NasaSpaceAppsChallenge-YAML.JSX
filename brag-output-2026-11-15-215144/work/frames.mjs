// Renders video.html frame by frame: for each frame time, render(t), let the page paint, screenshot.
//   node frames.mjs <outDir> [fps] [only=t1,t2,...]   (only: just those times, for stills)
import { spawn } from 'child_process';
import { mkdirSync, writeFileSync } from 'fs';
import { pathToFileURL } from 'url';
import { resolve } from 'path';
const CHROME = 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const OUT = process.argv[2], FPS = +(process.argv[3] || 30), ONLY = process.argv[4]?.replace('only=', '');
mkdirSync(OUT, { recursive: true });
const PORT = 9411, profile = resolve('chrome-frames'); mkdirSync(profile, { recursive: true });
const chrome = spawn(CHROME, ['--headless=new', `--remote-debugging-port=${PORT}`, `--user-data-dir=${profile}`, '--no-first-run',
  '--window-size=1920,1080', '--hide-scrollbars', '--allow-file-access-from-files', 'about:blank'], { stdio: 'ignore' });
const sleep = ms => new Promise(r => setTimeout(r, ms));
let page;
for (let i = 0; i < 100 && !page; i++) { try { page = (await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json()).find(t => t.type === 'page'); } catch {} await sleep(200); }
const ws = new WebSocket(page.webSocketDebuggerUrl); await new Promise(r => ws.onopen = r);
let n = 0; const wait = new Map(); const errs = [];
ws.onmessage = e => { const m = JSON.parse(e.data); if (m.id && wait.has(m.id)) { wait.get(m.id)(m); wait.delete(m.id); }
  if (m.method === 'Runtime.exceptionThrown') errs.push(m.params.exceptionDetails.exception?.description); };
const send = (method, params = {}) => new Promise(r => { const id = ++n; wait.set(id, r); ws.send(JSON.stringify({ id, method, params })); });
const ev = async expr => (await send('Runtime.evaluate', { expression: expr, awaitPromise: true, returnByValue: true })).result?.result?.value;
await send('Runtime.enable');
await send('Emulation.setDeviceMetricsOverride', { width: 1920, height: 1080, deviceScaleFactor: 1, mobile: false });
await send('Page.navigate', { url: pathToFileURL(resolve('video.html')).href });
for (let i = 0; i < 50; i++) { await sleep(200); if (await ev('window.__ready ? window.__ready : false')) break; }
const times = ONLY ? ONLY.split(',').map(Number) : Array.from({ length: Math.round(30 * FPS) }, (_, i) => i / FPS);
for (const [i, t] of times.entries()) {
  await ev(`render(${t}); new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))`);
  const r = await send('Page.captureScreenshot', { format: ONLY ? 'png' : 'jpeg', quality: ONLY ? undefined : 94 });
  const name = ONLY ? `still_${t.toFixed(2)}.png` : `f_${String(i).padStart(4, '0')}.jpg`;
  writeFileSync(`${OUT}/${name}`, Buffer.from(r.result.data, 'base64'));
  if (!ONLY && i % 60 === 0) console.log('frame', i, '/', times.length);
}
console.log(errs.join(' | ') || 'no errors');
chrome.kill(); process.exit(0);
