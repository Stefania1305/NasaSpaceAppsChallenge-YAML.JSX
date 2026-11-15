// Browser run of the manual test plan: drives the real app (http://localhost:8000)
// and the test bench in headless Chrome, clicking buttons, sorting columns,
// using Copy all rows, the Pixel calculator and the Cell aggregator.
//   bun run_browser_tests.mjs [path-to-test-bench.html]
// The server must be running (python server.py).
import { readFileSync, mkdirSync, existsSync } from "fs";
import { resolve } from "path";

const APP = "http://localhost:8000";
const DAY = "2026-10-13";
const AREA_A = "-44.6,-8.7,-43.9,-8.1";
const BENCH = resolve(process.argv[2] || "test.html");
const CHROME = ["C:/Program Files/Google/Chrome/Application/chrome.exe",
  "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"].find(p => existsSync(p));
const KEY = (readFileSync(".env", "utf8").match(/FIRMS_MAP_KEY=(\S+)/) || [])[1];
const PORT = 9333;
const PROFILE = resolve(process.env.TEMP || ".", "firms-browser-test-profile");
mkdirSync(PROFILE, { recursive: true });

// ------------------------------------------------------------------ CDP plumbing
const chrome = Bun.spawn([CHROME, "--headless=new", `--remote-debugging-port=${PORT}`, `--user-data-dir=${PROFILE}`,
  "--no-first-run", "--no-default-browser-check", "--window-size=1400,1000", "about:blank"], { stdout: "ignore", stderr: "ignore" });
let targets;
for (let i = 0; i < 100; i++) {
  try { targets = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json(); if (targets.some(t => t.type === "page")) break; } catch {}
  await Bun.sleep(200);
}
const version = await (await fetch(`http://127.0.0.1:${PORT}/json/version`)).json();
function connect(url) {
  const ws = new WebSocket(url); let n = 0; const wait = new Map();
  ws.onmessage = e => { const m = JSON.parse(e.data); if (m.id && wait.has(m.id)) { const [ok, bad] = wait.get(m.id); wait.delete(m.id); m.error ? bad(new Error(m.error.message)) : ok(m.result); } };
  const ready = new Promise(r => ws.onopen = r);
  return { ready, send: (method, params = {}) => new Promise((ok, bad) => { const id = ++n; wait.set(id, [ok, bad]); ws.send(JSON.stringify({ id, method, params })); }), close: () => ws.close() };
}
const browser = connect(version.webSocketDebuggerUrl);
const page = connect(targets.find(t => t.type === "page").webSocketDebuggerUrl);
await browser.ready; await page.ready;
await browser.send("Browser.grantPermissions", { origin: APP, permissions: ["clipboardReadWrite", "clipboardSanitizedWrite"] });
await page.send("Emulation.setFocusEmulationEnabled", { enabled: true });
await page.send("Page.enable");

async function ev(expr) {
  const r = await page.send("Runtime.evaluate", { expression: `(async () => { ${expr} })()`, awaitPromise: true, returnByValue: true, userGesture: true });
  if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description?.split("\n")[0] || r.exceptionDetails.text);
  return r.result.value;
}
async function nav(url, ready = "true") {
  await page.send("Page.navigate", { url });
  for (let i = 0; i < 300; i++) {
    try { if (await ev(`return location.href.startsWith(${JSON.stringify(url.split("#")[0])}) && document.readyState === "complete" && (${ready});`)) break; } catch {}
    await Bun.sleep(100);
  }
  await ev(HELPERS);
}

// Helpers installed in the page: they only click, type and read what is on screen.
const HELPERS = `
window.T = {
  sleep: ms => new Promise(r => setTimeout(r, ms)),
  el: id => document.getElementById(id),
  set(id, v) { const e = T.el(id); e.value = v; e.dispatchEvent(new Event('input', { bubbles: true })); e.dispatchEvent(new Event('change', { bubbles: true })); },
  async run(btn) { T.el(btn).click(); await T.sleep(50); while (T.el('go').disabled) await T.sleep(200); await T.sleep(50); },
  async query(btn, o) { for (const [k, v] of Object.entries(o)) T.set(k, v); await T.run(btn); },
  heads() { return [...document.querySelectorAll('#out th')].map(t => t.textContent.replace(/ [▲▼]$/, '')); },
  col(name) { const k = T.heads().indexOf(name); if (k < 0) throw new Error('no column ' + name); return [...document.querySelectorAll('#out tbody tr')].map(tr => tr.children[k].textContent); },
  table(sel) { const t = document.querySelector(sel); const h = [...t.querySelectorAll('th')].map(x => x.textContent); return [...t.querySelectorAll('tbody tr')].map(tr => Object.fromEntries([...tr.children].map((td, i) => [h[i], td.textContent]))); },
  async sort(name, desc = false) { const k = T.heads().indexOf(name); document.querySelectorAll('#out th')[k].click(); if (desc) document.querySelectorAll('#out th')[k].click(); await T.sleep(20); },
  async copy() {
    T.el('copyRows').click();
    for (let i = 0; i < 100 && !T.el('copyMsg').textContent; i++) await T.sleep(100);
    if (T.el('copyMsg').textContent !== 'Copied.') throw new Error('copy button said: ' + T.el('copyMsg').textContent);
    const text = (await navigator.clipboard.readText()).replace(/\\r/g, '');
    const lines = text.split('\\n').map(l => l.split('\\t'));
    T.text = text;
    return T.rows = lines.slice(1).map(r => Object.fromEntries(lines[0].map((h, i) => [h, r[i]])));
  },
  stats() { return Object.fromEntries([...document.querySelectorAll('#stats .stat')].map(s => [s.firstChild.textContent.trim(), s.querySelector('b').textContent])); },
  status() { return T.el('status').textContent; },
  assert(c, msg) { if (!c) throw new Error(msg); },
};`;

// ------------------------------------------------------------------ tests
const results = [];
async function test(id, title, fn) {
  const t0 = Date.now();
  try { const info = await fn(); results.push([id, true]); console.log(`PASS ${id.padEnd(4)} ${title}  (${((Date.now() - t0) / 1000).toFixed(1)}s)${info ? "\n          " + info : ""}`); }
  catch (e) { results.push([id, false]); console.log(`FAIL ${id.padEnd(4)} ${title}\n          ${e.message}`); }
}
const app = () => nav(APP + "/", "document.getElementById('sensor').options.length > 0");
const q = (btn, o) => ev(`await T.query(${JSON.stringify(btn)}, ${JSON.stringify(o)});`);
const base = (area = "world", date = DAY, days = "1") => ({ area, date, days });

// Raw FIRMS CSV for A2/A5 (the file a tester would open in a browser tab).
const rawCsv = await (await fetch(`https://firms.modaps.eosdis.nasa.gov/api/area/csv/${KEY}/MODIS_NRT/${AREA_A}/1/${DAY}`)).text();
const rawLines = rawCsv.trim().split("\n").map(l => l.split(","));
const raw = rawLines.slice(1).map(r => Object.fromEntries(rawLines[0].map((h, i) => [h, r[i]])));

// ---------- A: pixel level
await app();
await test("A1", "Raw scan and track are gone, pixel area is present", async () => {
  const bad = ["scan", "track", "scan_km", "track_km"];
  await q("harm", base(AREA_A));
  const h1 = await ev("return T.heads();");
  await q("go", { ...base(AREA_A), sensor: "MODIS_NRT" });
  const h2 = await ev("return T.heads();");
  for (const h of [h1, h2]) {
    if (bad.some(b => h.includes(b))) throw new Error("raw column still shown: " + h.join(","));
    if (!h.includes("pixel_area_km2")) throw new Error("pixel_area_km2 missing");
  }
  return `harmonised table ${h1.length} columns, single-sensor table ${h2.length} columns; both have pixel_area_km2 only`;
});

const pixelCalc = async (o) => ev(`
  for (const [k, v] of Object.entries(${JSON.stringify(o)})) T.set('p_' + k, v);
  return Object.fromEntries([...document.querySelectorAll('#pixelOut tr')].map(tr => [tr.children[0].textContent, tr.children[1].textContent.replace('(empty)', '')]));`);

await test("A2", "Pixel area = scan × track (calculator vs raw CSV vs app table)", async () => {
  const picks = [raw[0], raw[Math.floor(raw.length / 2)], raw[raw.length - 1]];
  await nav("file:///" + BENCH.replace(/\\/g, "/") + "#pixel");
  const calc = [];
  for (const r of picks) calc.push((await pixelCalc({ inst: "MODIS", scan: r.scan, track: r.track }))["pixel_area_km2"]);
  if (calc[0] === undefined) throw new Error("calculator output missing");
  const demo = (await pixelCalc({ scan: "1.26", track: "1.11" }))["pixel_area_km2"];
  if (demo !== "1.3986") throw new Error("1.26 × 1.11 gave " + demo);
  await app();
  await q("go", { ...base(AREA_A), sensor: "MODIS_NRT" });
  await ev("await T.sort('latitude');");
  const lat = await ev("return T.col('latitude').map(Number);");
  if (lat.some((v, i) => i && v < lat[i - 1])) throw new Error("latitude sort not ascending");
  const appRows = await ev("const h = T.heads(); return [...document.querySelectorAll('#out tbody tr')].map(tr => Object.fromEntries([...tr.children].map((td, i) => [h[i], td.textContent])));");
  picks.forEach((r, i) => {
    const a = appRows.find(x => x.latitude === r.latitude && x.longitude === r.longitude && (x.acq_time === r.acq_time || x.acq_time === r.acq_time.padStart(4, "0")));
    if (!a) throw new Error(`row ${r.latitude},${r.longitude} not in app table`);
    const exp = (parseFloat(r.scan) * parseFloat(r.track)).toFixed(4);
    if (a.pixel_area_km2 !== exp || calc[i] !== exp) throw new Error(`${r.scan}×${r.track}: app ${a.pixel_area_km2}, calculator ${calc[i]}, expected ${exp}`);
  });
  return picks.map((r, i) => `${r.scan}×${r.track}=${calc[i]}`).join(", ") + "; app, calculator and raw CSV agree; 1.26×1.11 = 1.3986";
});

// One world harmonise, copied with the button, reused by A3–A9.
await app();
await q("harm", base());
const worldInfo = await ev("const r = await T.copy(); return [r.length, T.status()];");
console.log(`     (world harmonise: ${worldInfo[0].toLocaleString()} rows copied with "Copy all rows")`);

await test("A3", "MODIS confidence boundaries 29/30/79/80", async () => ev(`
  await T.sort('confidence_raw');
  const shown = T.col('confidence_raw'), conf = T.col('confidence');
  const nums = shown.map(Number).filter(v => !isNaN(v));
  T.assert(nums.every((v, i) => !i || v >= nums[i - 1]), 'confidence_raw sort not ascending');
  const want = { 29: 'low', 30: 'nominal', 79: 'nominal', 80: 'high' }, seen = {};
  for (const r of T.rows) if (r.instrument === 'MODIS' && want[r.confidence_raw]) {
    T.assert(r.confidence === want[r.confidence_raw], r.confidence_raw + ' -> ' + r.confidence); seen[r.confidence_raw] = (seen[r.confidence_raw] || 0) + 1; }
  T.assert(Object.keys(seen).length === 4, 'boundaries present: ' + Object.keys(seen));
  return 'sorted view ascending; ' + Object.entries(seen).map(([k, n]) => k + '→' + want[k] + ' (' + n + ' rows)').join(', ');`));

await test("A4", "VIIRS and Landsat confidence letters", async () => ev(`
  const m = { l: 'low', n: 'nominal', m: 'nominal', h: 'high' }, seen = {};
  for (const r of T.rows) if (r.instrument !== 'MODIS') {
    T.assert(r.confidence && r.confidence === m[r.confidence_raw.toLowerCase()], r.instrument + ' ' + r.confidence_raw + ' -> ' + r.confidence);
    seen[r.confidence_raw] = r.confidence; }
  return Object.entries(seen).map(([k, v]) => k + '→' + v).join(', ');`));

await test("A5", "Brightness kept at full precision (table + Hot MODIS preset)", async () => {
  const r = await ev(`
    await T.sort('bright_main_k', true);
    T.assert(!T.heads().includes('bright_saturated'), 'bright_saturated column still shown');
    const b = T.col('bright_main_k').map(Number);
    T.assert(b.every((v, i) => !i || v <= b[i - 1]), 'descending sort out of order');
    const viirsMax = Math.max(...T.rows.filter(r => r.instrument === 'VIIRS').map(r => +r.bright_main_k));
    const above = T.rows.filter(r => +r.bright_main_k > 367);
    T.assert(above.every(r => r.instrument === 'MODIS'), 'non-MODIS row above 367 K');
    return [b[0], above.length, viirsMax];`);
  if (!(r[0] > 367)) throw new Error("top of descending sort is " + r[0] + " - still capped?");
  await nav("file:///" + BENCH.replace(/\\/g, "/") + "#pixel");
  const p = await ev(`document.querySelector('[data-preset="sat"]').click(); return Object.fromEntries([...document.querySelectorAll('#pixelOut tr')].map(tr => [tr.children[0].textContent, tr.children[1].textContent]));`);
  if (p.bright_main_k !== "412.6" || "bright_saturated" in p) throw new Error(`preset gave ${p.bright_main_k} / ${p.bright_saturated}`);
  return `top of sorted column ${r[0]} K; ${r[1].toLocaleString()} MODIS rows above 367 K kept; VIIRS max ${r[2]} K; Hot preset shows 412.6`;
});

await app(); await q("harm", base()); await ev("await T.copy();");

await test("A6", "Landsat rows have no brightness, FRP or area", async () => ev(`
  await T.sort('source');
  const ls = T.rows.filter(r => r.source === 'LANDSAT_NRT');
  T.assert(ls.length > 0, 'no Landsat rows');
  for (const r of ls) T.assert(r.instrument === 'OLI' && !r.bright_main_k && !r.bright_secondary_k && !r.frp_mw && !r.pixel_area_km2, JSON.stringify(r));
  return ls.length.toLocaleString() + ' Landsat rows: OLI, all four fields empty';`));

await test("A7", "Satellite names", async () => ev(`
  const want = { MODIS_NRT: ['Terra', 'Aqua'], VIIRS_SNPP_NRT: ['Suomi NPP'], VIIRS_NOAA20_NRT: ['NOAA-20'], VIIRS_NOAA21_NRT: ['NOAA-21'], LANDSAT_NRT: ['Landsat 8', 'Landsat 9'] };
  const seen = {};
  for (const r of T.rows) { (seen[r.source] ||= new Set()).add(r.satellite); }
  for (const [s, names] of Object.entries(seen)) if (want[s]) T.assert([...names].every(n => want[s].includes(n)), s + ': ' + [...names]);
  return Object.entries(seen).map(([s, n]) => s + '=' + [...n].join('/')).join(', ');`));

await test("A8", "Acquisition time always has 4 digits", async () => ev(`
  await T.sort('acq_time');
  const first = T.col('acq_time').slice(0, 3);
  T.assert(T.rows.every(r => /^\\d{4}$/.test(r.acq_time)), 'non 4-digit time found');
  T.assert(first.every(t => t.length === 4), 'shown: ' + first);
  return 'first rows after sort: ' + first.join(', ') + '; all ' + T.rows.length.toLocaleString() + ' rows HHMM';`));

await test("A9", "Overpass slot (calculator vs app, 3 sensors)", async () => {
  const picks = await ev(`
    const out = []; for (const s of ['MODIS_NRT', 'VIIRS_NOAA20_NRT', 'LANDSAT_NRT']) { const r = T.rows.find(x => x.source === s); if (r) out.push(r); } return out;`);
  await nav("file:///" + BENCH.replace(/\\/g, "/") + "#pixel");
  const out = [];
  for (const r of picks) {
    const c = await pixelCalc({ inst: r.instrument === "OLI" ? "LANDSAT" : r.instrument, time: r.acq_time, lon: r.longitude });
    if (c.overpass_slot !== r.overpass_slot) throw new Error(`${r.source} ${r.acq_time} ${r.longitude}: calculator ${c.overpass_slot}, app ${r.overpass_slot}`);
    out.push(`${r.source} ${r.acq_time}@${r.longitude}→${c.overpass_slot}`);
  }
  return out.join("; ");
});

await test("A10", "NRT rows overlapping SP are dropped", async () => {
  await app();
  await q("harm", base("20,40,30,50", "2026-07-30", "3"));
  return ev(`
    const src = T.table('#sources table');
    const note = src.find(s => s.Source === 'VIIRS_NOAA21_NRT' && /overlapping VIIRS_NOAA21_SP/.test(s.Note));
    T.assert(note, 'no overlap note in sources table');
    [...document.querySelectorAll('#chips .chip')].find(b => b.dataset.s === 'VIIRS_NOAA21_NRT').click();
    await T.sleep(50);
    await T.sort('acq_date');
    const d = T.col('acq_date');
    T.assert(d.length && d.every(x => x > '2026-07-31'), 'early NRT date shown: ' + d[0]);
    return 'note: "' + note.Note + '"; NOAA21_NRT chip earliest date ' + d[0];`);
});

// ---------- B: type
let refBefore;
await test("B1", "SP files keep their own type", async () => {
  await app();
  refBefore = await ev("return document.getElementById('typeref').textContent;");
  await q("go", { ...base("world", "2026-08-30"), sensor: "MODIS_SP" });
  const s = await ev("return T.stats();");
  const want = { "Type from file / inferred": "15,041 / 0", "Type 0: vegetation fire": "14,178", "Type 1: active volcano": "68", "Type 2: static land source": "666", "Type 3: offshore": "129" };
  for (const [k, v] of Object.entries(want)) if (s[k] !== v) throw new Error(`${k}: ${s[k]} != ${v}`);
  return Object.entries(want).map(([k, v]) => `${k.split(":")[0]} ${v}`).join(" · ");
});

await test("B2", "NRT rows get an inferred type", async () => {
  await q("go", { ...base(), sensor: "MODIS_NRT" });
  return ev(`
    const s = T.stats(); T.assert(s['Type from file / inferred'] === '0 / 11,168', s['Type from file / inferred']);
    const rows = await T.copy();
    for (const r of rows) T.assert(r.type_existed === 'no' && +r.type_confidence >= 0 && +r.type_confidence <= 1 && r.type_basis, JSON.stringify(r));
    return 'stat 0 / 11,168; every copied row no, confidence 0–1, basis set';`);
});

await test("B3", "Semeru recognised as volcano", async () => ev(`
  await T.sort('type', true);
  T.assert(T.col('type')[0] === '1' || T.col('type')[0] === '3' || T.col('type')[0] === '2', 'descending type sort top ' + T.col('type')[0]);
  const lat = T.col('latitude'), lon = T.col('longitude'), k = lat.findIndex((v, i) => v === '-8.11577' && lon[i] === '112.93331');
  T.assert(k >= 0, 'Semeru row not in table');
  const t = T.col('type')[k], l = T.col('type_label')[k];
  T.assert(t === '1' && l === 'active volcano', t + ' ' + l);
  return 'row ' + (k + 1) + ' of sorted table: type 1, active volcano';`));

await test("B4", "Repeating a query does not double-count", async () => {
  await q("go", { ...base("world", "2026-08-30"), sensor: "MODIS_SP" });
  await app();
  const after = await ev("for (let i = 0; i < 50 && !document.getElementById('typeref').textContent; i++) await T.sleep(100); return document.getElementById('typeref').textContent;");
  if (after !== refBefore) throw new Error(`before: ${refBefore}\nafter: ${after}`);
  return after.match(/[\d,]+ labelled/)[0] + " before and after reload";
});

await test("B5", "Mixed files: kept and inferred side by side", async () => {
  await q("harm", base("20,40,30,50", "2026-07-30", "3"));
  return ev(`
    const rows = await T.copy(), c = {};
    for (const r of rows) { const exp = r.source.endsWith('_SP') ? 'yes' : 'no'; T.assert(r.type_existed === exp, r.source + ' ' + r.type_existed); c[exp] = (c[exp] || 0) + 1; }
    T.assert(c.yes && c.no, JSON.stringify(c));
    return 'yes ' + c.yes.toLocaleString() + ' (SP), no ' + c.no.toLocaleString() + ' (NRT/Landsat)';`);
});

// ---------- C: comparison
const readCompare = () => ev(`
  const panel = document.querySelector('#sources .panel');
  const sum = Object.fromEntries([...panel.querySelectorAll('tbody tr')].map(tr => [tr.children[0].textContent.replace(/ \\(.*grid\\)/, ''), [...tr.children].slice(1).map(td => td.textContent)]));
  const hint = [...document.querySelectorAll('#sources .hintline')].map(p => p.textContent);
  const agree = hint[0].match(/both: ([\\d,]+) · MODIS only: ([\\d,]+) · VIIRS only: ([\\d,]+)/).slice(1).map(x => +x.replace(/,/g, ''));
  const removed = hint.find(h => h.startsWith('Removed'));
  const minFrp = document.querySelector('#sources').textContent.match(/minimum fire size of ([\\d.]+) MW/)[1];
  return { sum, agree, removed, minFrp };`);
const chip = name => ev(`[...document.querySelectorAll('#chips .chip')].find(b => b.dataset.s === ${JSON.stringify(name)}).click(); await T.sleep(50);`);

let c1cells;
await test("C1", "Known answer for test area A", async () => {
  await app();
  await q("cmp", { ...base(AREA_A), viirs: "NOAA20", cell: "0.05", maxgap: "" });
  const c = await readCompare();
  const want = { "Raw pixels (detections)": ["49", "156"], "Raw total FRP (MW)": ["6,286", "12,258"], "Fire cells": ["15", "10"] };
  for (const [k, v] of Object.entries(want)) if (c.sum[k]?.[0] !== v[0] || c.sum[k]?.[1] !== v[1]) throw new Error(`${k}: ${c.sum[k]}`);
  if (c.agree.join() !== "8,7,2") throw new Error("agreement " + c.agree);
  if (c.minFrp !== "10") throw new Error("min FRP " + c.minFrp);
  if (/capped/.test(c.removed)) throw new Error("cap still listed: " + c.removed);
  await chip("All cells");
  c1cells = await ev("return await T.copy();");
  const top = c1cells.find(r => r.cell_lat === "-8.3750" && r.cell_lon === "-44.1750" && r.daynight === "D");
  const got = [top.modis_pixels, top.modis_frp_mw, top.viirs_pixels, top.viirs_frp_mw, top.overpass_gap_min].join(" ");
  if (got !== "5 3329.12 21 3486.85 103") throw new Error("top cell " + got);
  return "summary, agreement 8/7/2, 10 MW and top cell match on screen; no cap listed";
});

await test("C2", "Cell aggregator matches the app (Copy all rows → paste → Recompute)", async () => {
  await q("harm", base(AREA_A));
  const tsv = await ev("await T.copy(); return T.text;");
  await nav("file:///" + BENCH.replace(/\\/g, "/") + "#cells");
  const agg = await ev(`
    T.set('a_viirs', 'NOAA-20'); T.set('a_cell', '0.05'); T.set('a_min', '10');
    const ta = document.getElementById('a_rows'); ta.value = ${JSON.stringify(tsv)}; ta.dispatchEvent(new Event('input', { bubbles: true }));
    document.getElementById('a_run').click(); await T.sleep(50);
    T.assert(!document.getElementById('a_err').textContent, document.getElementById('a_err').textContent);
    const tables = document.querySelectorAll('#a_out table');
    const sum = [...tables[0].querySelectorAll('tbody tr')].map(tr => [...tr.children].map(td => td.textContent));
    const h = [...tables[1].querySelectorAll('th')].map(x => x.textContent);
    const cells = [...tables[1].querySelectorAll('tbody tr')].map(tr => Object.fromEntries([...tr.children].map((td, i) => [h[i], td.textContent])));
    const kv = document.querySelector('#a_out .kv').textContent;
    return { sum, cells, kv, note: document.getElementById('sampleNote').textContent };`);
  if (agg.note !== "Using your pasted rows.") throw new Error("sample note: " + agg.note);
  const key = r => [r.cell_lat, r.cell_lon, r.acq_date, r.daynight].join("|");
  const app_ = new Map(c1cells.map(r => [key(r), r]));
  if (agg.cells.length !== app_.size) throw new Error(`aggregator ${agg.cells.length} cells, app ${app_.size}`);
  for (const r of agg.cells) {
    const a = app_.get(key(r)); if (!a) throw new Error("cell not in app: " + key(r));
    for (const f of ["detected_by", "modis_pixels", "viirs_pixels", "modis_frp_mw", "viirs_frp_mw", "modis_confidence", "viirs_confidence"])
      if ((a[f] || "") !== (r[f] || "")) throw new Error(`${key(r)} ${f}: app ${a[f]}, aggregator ${r[f]}`);
    if ((a.comparable === "yes") !== (r.comparable === "yes")) throw new Error(`${key(r)} comparable differs`);
  }
  const s = Object.fromEntries(agg.sum.map(r => [r[0].replace(/ \(.*grid\)/, ""), r.slice(1, 3).join("/")]));
  const want = { "Raw pixels (detections)": "49/156", "Raw total FRP (MW)": "6,286/12,258", "Fire cells": "15/10" };
  for (const [k, v] of Object.entries(want)) if (s[k] !== v) throw new Error(`aggregator ${k} ${s[k]}`);
  if (!/Both 8.*MODIS only 7.*VIIRS only 2/.test(agg.kv)) throw new Error(agg.kv);
  return `${agg.cells.length} cells: pixels, FRP, confidence, detected_by and comparable identical; summary 49/156, 15/10, 8/7/2`;
});

let world;
await test("C3", "Terra is excluded", async () => {
  await app();
  await q("cmp", { ...base(), viirs: "NOAA20", cell: "0.05", maxgap: "" });
  world = await readCompare();
  if (!/Terra MODIS pixels[^:]*: 4,161/.test(world.removed)) throw new Error(world.removed);
  if (world.sum["Raw pixels (detections)"][0] !== "7,007") throw new Error("raw MODIS " + world.sum["Raw pixels (detections)"][0]);
  return "Terra 4,161 removed, raw MODIS 7,007 (4,161 + 7,007 = 11,168)";
});

await test("C4", "Day and night never mixed in one cell", async () => ev(`
  [...document.querySelectorAll('#chips .chip')].find(b => b.dataset.s === 'All cells').click(); await T.sleep(50);
  await T.sort('cell_lat');
  const lat = T.col('cell_lat').map(Number); T.assert(lat.every((v, i) => !i || v >= lat[i - 1]), 'cell_lat sort not ascending');
  const rows = await T.copy(), seen = new Map(); let twice = 0;
  for (const r of rows) { T.assert(r.daynight === 'D' || r.daynight === 'N', 'daynight ' + r.daynight); const k = r.cell_lat + '|' + r.cell_lon + '|' + r.acq_date; if (seen.has(k)) { T.assert(seen.get(k) !== r.daynight, 'duplicate ' + k); twice++; } else seen.set(k, r.daynight); }
  return rows.length.toLocaleString() + ' cells, each D or N; ' + twice + ' squares appear once as D and once as N';`));

await test("C5", "Detection limit applied to both sensors", async () => ev(`
  const lim = ${JSON.stringify(+world.minFrp)};
  let below = 0, comp = 0;
  for (const r of T.rows) {
    const m = +r.modis_frp_mw || 0, v = +r.viirs_frp_mw || 0;
    if (/below common detection limit/.test(r.reason)) { T.assert(m < lim && v < lim, 'below-limit row ' + m + '/' + v); below++; }
  }
  T.assert(below > 0, 'no below-limit rows found');
  for (const r of T.rows) { const m = +r.modis_frp_mw || 0, v = +r.viirs_frp_mw || 0;
    if (r.comparable === 'yes') { T.assert(Math.max(m, v) >= lim, 'comparable row ' + m + '/' + v); comp++; }
  }
  [...document.querySelectorAll('#chips .chip')].find(b => b.dataset.s === 'Excluded').click(); await T.sleep(50);
  await T.sort('reason'); T.assert(T.col('comparable').every(x => x === 'no'), 'Excluded chip shows comparable rows');
  return 'limit ' + lim + ' MW; ' + below.toLocaleString() + ' below-limit rows under it on both sensors; ' + comp.toLocaleString() + ' comparable rows reach it';`));

await test("C6", "Swath exclusions are plausible", async () => ev(`
  const sw = T.rows.filter(r => /outside Aqua MODIS swath/.test(r.reason));
  T.assert(sw.length > 0, 'no swath rows');
  for (const r of sw) T.assert(r.modis_pixels === '0' && +r.viirs_pixels > 0, JSON.stringify(r));
  const inBand = sw.filter(r => Math.abs(+r.cell_lat) <= 35).length;
  T.assert(inBand / sw.length > 0.9, 'only ' + inBand + ' of ' + sw.length + ' within ±35°');
  return sw.length.toLocaleString() + ' rows, all MODIS 0 / VIIRS > 0, ' + (100 * inBand / sw.length).toFixed(1) + '% within ±35°';`));

await test("C8", "Cell brightness is uncapped (sorted view)", async () => ev(`
  [...document.querySelectorAll('#chips .chip')].find(b => b.dataset.s === 'All cells').click(); await T.sleep(50);
  const out = [];
  for (const c of ['modis_bt_max_k', 'viirs_bt_max_k']) {
    await T.sort(c, true); const top = T.col(c)[0]; out.push(c + ' top ' + top);
    for (const r of T.rows) if (r[c]) T.assert(!r[c].includes('(sat)'), r[c]);
    if (c === 'modis_bt_max_k') T.assert(+top > 367, 'MODIS top ' + top + ' - still capped?');
  }
  return out.join(', ');`));

await test("C9", "Summary adds up", async () => {
  const [b, m, v] = world.agree, s = world.sum["Comparable fire cells"].map(x => +x.replace(/,/g, ""));
  if (s[0] !== b + m || s[1] !== b + v) throw new Error(`${s} vs ${b}/${m}/${v}`);
  return `${b.toLocaleString()} + ${m.toLocaleString()} = ${s[0].toLocaleString()}; ${b.toLocaleString()} + ${v.toLocaleString()} = ${s[1].toLocaleString()}`;
});

await test("C10", "Harmonising narrows the gap", async () => {
  const r = k => parseFloat(world.sum[k][2]);
  const raw_ = r("Raw pixels (detections)"), cells = r("Comparable fire cells"), frp = r("Comparable total FRP (MW)");
  if (!(raw_ > 7 && cells < 2.5 && frp < 2.5)) throw new Error(`${raw_} / ${cells} / ${frp}`);
  return `on screen: raw ${raw_}×, comparable cells ${cells}×, comparable FRP ${frp}×`;
});

await test("C7", "Max pass gap filter 120", async () => {
  await q("cmp", { ...base(), viirs: "NOAA20", cell: "0.05", maxgap: "120" });
  const c = await readCompare();
  const n = c.removed.match(/passes too far apart in time: ([\d,]+)/)[1];
  return ev(`
    await T.sort('overpass_gap_min', true);
    const top = T.col('overpass_gap_min').find(x => x !== '');
    T.assert(+top <= 120, 'top comparable gap ' + top);
    return 'Comparable view sorted descending: top gap ' + top + ' min; removed ${n} cells';`);
});

await test("C11", "Raw counts match the single-sensor data", async () => {
  await q("go", { ...base(), sensor: "VIIRS_NOAA20_NRT" });
  const det = (await ev("return T.stats();"))["Detections"];
  if (det !== world.sum["Raw pixels (detections)"][1] || det !== "58,943") throw new Error(`single ${det}, compare ${world.sum["Raw pixels (detections)"][1]}`);
  return `Detections stat ${det} = VIIRS raw pixels in C10`;
});

// ---------- D: input checks (typed into the browser / form)
const openJson = async url => { await nav(url); return ev("return document.body.innerText;"); };
await test("D1", "Day range limited to 1–5", async () => {
  const t = await openJson(`${APP}/api/fires?sensor=MODIS_NRT&date=${DAY}&days=9`);
  if (!t.includes("Day range must be 1-5.")) throw new Error(t); return t.trim();
});
await test("D2", "Invalid grid refused", async () => {
  const t = await openJson(`${APP}/api/compare?date=${DAY}&days=1&cell=0.3`);
  if (!["0.01", "0.05", "0.1", "0.25"].every(x => t.includes(x))) throw new Error(t); return t.trim();
});
await test("D3", "Non-numeric max pass gap refused", async () => {
  const t = await openJson(`${APP}/api/compare?date=${DAY}&days=1&maxgap=abc`);
  await app();
  const typed = await ev("const e = T.el('maxgap'); e.value = 'abc'; return e.value;");
  if (!t.includes("Max overpass gap must be a number of minutes.")) throw new Error(t);
  if (typed !== "") throw new Error("number field accepted abc");
  return t.trim() + " · the form field rejects typed abc";
});
await test("D4", "Malformed area refused", async () => {
  await q("go", { ...base("abc"), sensor: "MODIS_NRT" });
  const r = await ev("const e = document.querySelector('#status .error'); return e ? [e.textContent, getComputedStyle(e).color] : null;");
  if (!r || !/west,south,east,north/.test(r[0])) throw new Error("no red message: " + JSON.stringify(r));
  return `"${r[0]}" in ${r[1]}`;
});
await test("D5", "Dates outside a sensor's range", async () => {
  await q("go", { ...base(), sensor: "MODIS_SP" });
  const s1 = await ev("return T.status();");
  if (!/No detections/.test(s1)) throw new Error(s1);
  await q("harm", base(AREA_A));
  const sk = await ev("return T.table('#sources table').filter(s => s.Status === 'skipped');");
  if (!sk.length || !sk.every(s => s.Source.endsWith("_SP") && /available \d{4}-\d\d-\d\d to/.test(s.Note))) throw new Error(JSON.stringify(sk));
  return `"${s1.trim()}"; skipped: ${sk.map(s => s.Source).join(", ")}`;
});
await test("D6", "Area with no fires", async () => {
  const out = [];
  for (const [btn, extra] of [["go", { sensor: "MODIS_NRT" }], ["harm", {}], ["cmp", { maxgap: "" }]]) {
    await q(btn, { ...base("-150,-40,-149,-39"), ...extra });
    const s = await ev("return [T.status(), !!document.querySelector('#status .error')];");
    if (!/No detections for this selection/.test(s[0]) || s[1]) throw new Error(`${btn}: ${s[0]}`);
    out.push(btn);
  }
  return `Fetch, Harmonise, Compare all say "No detections for this selection", no error`;
});

// ---------- E: the test bench page itself
const benchUrl = "file:///" + BENCH.replace(/\\/g, "/");
await test("E1", "Bench: verdict buttons, progress and saving across reload", async () => {
  await nav(benchUrl, "document.querySelectorAll('.test').length > 0");
  await ev("try { localStorage.clear(); } catch {}"); await nav(benchUrl, "document.querySelectorAll('.test').length > 0");
  const r1 = await ev(`
    const n = document.querySelectorAll('.test').length; T.assert(n === 32, n + ' tests rendered');
    document.querySelector('[data-id="A1"] [data-v="pass"]').click();
    document.querySelector('[data-id="A2"] [data-v="fail"]').click();
    document.querySelector('[data-id="A3"] [data-v="skip"]').click();
    const note = T.el('note-A2'); note.value = 'wrong area'; note.dispatchEvent(new Event('input', { bubbles: true }));
    return [T.el('progressText').textContent, document.getElementById('t-A1').dataset.s];`);
  if (r1[0] !== "3 / 32 tested · 1 pass · 1 fail" || r1[1] !== "pass") throw new Error(r1.join(" | "));
  await nav(benchUrl, "document.querySelectorAll('.test').length > 0");
  const r2 = await ev("return [T.el('progressText').textContent, document.getElementById('t-A2').dataset.s, T.el('note-A2').value];");
  if (r2.join("|") !== "3 / 32 tested · 1 pass · 1 fail|fail|wrong area") throw new Error(r2.join(" | "));
  const r3 = await ev(`document.querySelector('[data-id="A1"] [data-v="pass"]').click(); return T.el('progressText').textContent;`);
  if (r3 !== "2 / 32 tested · 0 pass · 1 fail") throw new Error("toggle off: " + r3);
  return `"${r1[0]}" survives reload with note; clicking Pass again clears it`;
});
await test("E2", "Bench: filters, copy results, reset (two clicks)", async () => ev(`
  const count = () => document.querySelectorAll('.test').length;
  document.querySelector('[data-filter="todo"]').click(); const todo = count();
  document.querySelector('[data-filter="fail"]').click(); const fail = count();
  document.querySelector('[data-filter="all"]').click(); const all = count();
  T.assert(todo === 30 && fail === 1 && all === 32, todo + '/' + fail + '/' + all);
  T.el('copyResults').click(); await T.sleep(300);
  const msg = T.el('planMsg').textContent;
  T.assert(/copied|blocked/.test(msg), 'copy message: ' + msg);
  T.el('resetBtn').click(); const armed = T.el('resetBtn').textContent; T.assert(T.el('progressText').textContent.startsWith('2 /'), 'first click cleared');
  T.el('resetBtn').click();
  T.assert(T.el('progressText').textContent === '0 / 32 tested · 0 pass · 0 fail', T.el('progressText').textContent);
  return 'To do 30, Failed 1, All 32; copy: "' + msg + '"; reset armed ("' + armed + '") then cleared';`));
await test("E3", "Bench: all six calculator presets", async () => {
  await nav(benchUrl + "#pixel", "true");
  const want = {
    modis: { satellite: "Terra", acq_time: "0033", confidence: "high", bright_main_k: "325.3", pixel_area_km2: "1.3986" },
    viirs: { satellite: "NOAA-20", acq_time: "0128", confidence: "nominal", pixel_area_km2: "0.1978" },
    landsat: { instrument: "OLI", satellite: "Landsat 8", confidence: "nominal", bright_main_k: "", frp_mw: "", pixel_area_km2: "" },
    sat: { bright_main_k: "412.6", confidence: "high", pixel_area_km2: "3.6250" },
    b29: { confidence: "low" }, b80: { confidence: "high" },
  };
  const out = [];
  for (const [p, exp] of Object.entries(want)) {
    const r = await ev(`document.querySelector('[data-preset="${p}"]').click(); return Object.fromEntries([...document.querySelectorAll('#pixelOut tr')].map(tr => [tr.children[0].textContent, tr.children[1].textContent.replace('(empty)', '')]));`);
    for (const [k, v] of Object.entries(exp)) if (r[k] !== v) throw new Error(`${p} ${k}: ${r[k]} != ${v}`);
    out.push(`${p}: ${r.confidence}, slot ${r.overpass_slot}`);
  }
  const b79 = await pixelCalc({ conf: "79" }), b30 = await pixelCalc({ conf: "30" });
  if (b79.confidence !== "nominal" || b30.confidence !== "nominal") throw new Error("30/79 boundaries");
  return out.join("; ") + "; typed 30/79 → nominal";
});
await test("E4", "Bench: embedded sample, Reload sample, bad paste message, tabs", async () => {
  await nav(benchUrl + "#cells", "true");
  return ev(`
    const s = [...document.querySelectorAll('#a_out table')[0].querySelectorAll('tbody tr')].map(tr => tr.children[1].textContent + '/' + tr.children[2].textContent);
    T.assert(s[0] === '49/156' && s[2] === '15/10', 'sample summary ' + s);
    const ta = T.el('a_rows'); ta.value = 'foo\\tbar\\n1\\t2'; ta.dispatchEvent(new Event('input', { bubbles: true })); T.el('a_run').click();
    const err = T.el('a_err').textContent; T.assert(/Missing columns/.test(err), 'bad paste: ' + err);
    T.el('a_sample').click(); T.assert(!T.el('a_err').textContent && /468 rows/.test(T.el('sampleNote').textContent), 'reload sample');
    T.set('a_cell', '0.25'); const big = [...document.querySelectorAll('#a_out table')[0].querySelectorAll('tbody tr')][2].children[1].textContent;
    document.getElementById('tab-plan').click(); const vis = !T.el('view-plan').hidden && T.el('view-cells').hidden && location.hash === '#plan';
    T.assert(vis, 'tab switch');
    return 'sample gives 49/156 and 15/10; bad paste names missing columns; Reload sample restores; grid 0.25° recomputes (' + big + ' MODIS cells); tabs + hash work';`);
});

// ------------------------------------------------------------------ summary
page.close(); browser.close(); chrome.kill();
const groups = {};
for (const [id, ok] of results) { const g = (groups[id[0]] ||= [0, 0]); g[1]++; g[0] += ok; }
const passed = results.filter(r => r[1]).length;
console.log("\n" + Object.entries(groups).map(([g, [p, n]]) => `${g}: ${p}/${n}`).join("   "));
console.log(`TOTAL: ${passed} / ${results.length} passed = ${(100 * passed / results.length).toFixed(1)}%`);
process.exit(passed === results.length ? 0 : 1);
