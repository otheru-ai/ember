// Render docs/perf/index.html's script against the shipped data.json with a
// stubbed DOM, for every view the page offers.
//
//     node tools/check_perf_page.js
//
// `node --check` only parses, and exercising the chart helpers directly never
// runs render(). That gap let a temporal-dead-zone ReferenceError -- the
// workload chart reading `palette` above its own const -- reach the published
// page, where it threw on load and left every card empty.
'use strict';
const fs = require('fs');
const path = require('path');

const root = path.resolve(__dirname, '..');
const page = path.join(root, 'docs/perf/index.html');
const html = fs.readFileSync(page, 'utf8');

const open = html.indexOf('<script>');
const close = html.lastIndexOf('</script>');
if (open < 0 || close < 0) {
  console.error('no <script> block in ' + page);
  process.exit(2);
}
// Drop the bootstrap: the data is supplied directly rather than fetched.
const js = html.slice(open + '<script>'.length, close).split('fetch("data.json")')[0];

function makeEl(id) {
  return {
    id, innerHTML: '', textContent: '', hidden: false, className: '', type: '',
    children: [], clientWidth: 1136,
    appendChild(c) { this.children.push(c); },
    setAttribute() {}, addEventListener() {},
    querySelector() { return null; }, querySelectorAll() { return []; },
  };
}
const store = new Map();
const byId = (id) => {
  if (!store.has(id)) store.set(id, makeEl(id));
  return store.get(id);
};

global.document = {
  getElementById: byId,
  querySelector: (sel) => byId('sel:' + sel),
  querySelectorAll: () => [],
  createElement: (tag) => makeEl(tag),
};
global.location = { hash: '' };
global.history = { replaceState() {} };
global.addEventListener = () => {};

const api = new Function(js + '\nreturn { render, S, setData: (d) => { DATA = d; } };')();

const data = JSON.parse(fs.readFileSync(path.join(root, 'docs/perf/data.json'), 'utf8'));
api.setData(data);
data.workloads.forEach((w) => api.S.w.add(w));
data.releases.forEach((r) => api.S.r.add(r.id));

const cards = ['wl-chart', 'd-chart', 'h-chart', 'table'];
let failures = 0;
let checked = 0;

function sweep(label, requiredCards) {
  for (const view of ['absolute', 'vs-ar', 'vs-baseline']) {
    for (const metric of ['decode', 'prefill', 'ttft']) {
      {
        api.S.view = view; api.S.metric = metric;
        const where = `${label} view=${view} metric=${metric}`;
        checked++;
        cards.forEach((c) => { byId(c).innerHTML = ''; });
        try {
          api.render();
        } catch (e) {
          console.log(`  THREW  ${where}: ${e.message}`);
          failures++;
          continue;
        }
        // A card that renders nothing at all is the visible symptom of render()
        // dying partway, so treat empty as a failure rather than a blank state.
        const empty = requiredCards.filter((c) => !byId(c).innerHTML);
        if (empty.length) {
          console.log(`  EMPTY  ${where} -> ${empty.join(', ')}`);
          failures++;
        }
      }
    }
  }
}

sweep('all-releases', cards);

// One release selected exercises paths the full selection never reaches: the
// history card shows a single value instead of a line, and the depth chart
// gains its autoregressive baseline.
const newest = data.releases[data.releases.length - 1];
api.S.r.clear();
api.S.r.add(newest.id);
api.S.baseline = null;
sweep('one-release', ['wl-chart', 'h-chart']);

api.render();
const hist = byId('h-chart').innerHTML;
// Both measures are shown together, so one release means two stat values.
const stats = (hist.match(/stat-value/g) || []).length;
if (stats !== 2) {
  console.log(`  MISSING  single release should render 2 stat values, got ${stats}`);
  failures++;
}
if (/Select a second release/.test(hist)) {
  console.log('  STALE    single release still renders the explanatory sentence');
  failures++;
}
// And more than one release means two charts, not one behind a toggle.
data.releases.forEach((r) => api.S.r.add(r.id));
api.render();
const svgs = (byId('h-chart').innerHTML.match(/<svg /g) || []).length;
if (svgs !== 2) {
  console.log(`  MISSING  history should render 2 charts side by side, got ${svgs}`);
  failures++;
}

console.log(failures
  ? `  ${failures} of ${checked} combinations failed`
  : `  all ${checked} combinations rendered`);
process.exit(failures ? 1 : 0);
