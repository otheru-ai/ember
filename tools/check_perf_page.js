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

for (const view of ['absolute', 'vs-ar', 'vs-baseline']) {
  for (const metric of ['decode', 'prefill', 'ttft']) {
    for (const hist of ['decode', 'prefill']) {
      api.S.view = view; api.S.metric = metric; api.S.hist = hist;
      const where = `view=${view} metric=${metric} hist=${hist}`;
      checked++;
      try {
        api.render();
      } catch (e) {
        console.log(`  THREW  ${where}: ${e.message}`);
        failures++;
        continue;
      }
      // A card that renders nothing at all is the visible symptom of render()
      // dying partway, so treat empty as a failure rather than as a blank state.
      const empty = cards.filter((c) => !byId(c).innerHTML);
      if (empty.length) {
        console.log(`  EMPTY  ${where} -> ${empty.join(', ')}`);
        failures++;
      }
    }
  }
}

console.log(failures
  ? `  ${failures} of ${checked} combinations failed`
  : `  all ${checked} view/metric combinations rendered`);
process.exit(failures ? 1 : 0);
