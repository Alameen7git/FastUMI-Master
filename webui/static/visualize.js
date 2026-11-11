// View page: the operator types a dataset name (or a Hugging Face repo id); the
// visualize_dataset.py viewer is launched as a tracked process and embedded.
// A Hugging Face dataset is downloaded first (fetch_lerobot.py, job vizfetch).
'use strict';

const $ = id => document.getElementById(id);
let TASKS = {}, HF_READY = {}, loadedFor = null;
const isRepo = v => /^[\w.-]+\/[\w.-]+$/.test(v.trim());
const valid = attachNameValidation($('viz-task'), $('viz-name-msg'), () => update(), undefined, isRepo);

async function refresh() {
  // Used to check the typed name; no names are suggested.
  const [r, h] = await Promise.all([api('/api/tasks'), api('/api/viz/hf')]);
  TASKS = r.tasks || {};
  HF_READY = h.ready || {};
  update();
}

function typed() { return $('viz-task').value.trim(); }

function dashUrl(s, task) {
  return s.viz_url + (task ? '?task=' + encodeURIComponent(task) : '');
}

function instruct(kind, text) {
  $('viz-instr').className = 'instruction' + (kind ? ' ' + kind : '');
  $('viz-instr').querySelector('.txt').textContent = text;
}

function update() {
  const s = FUI.status;
  if (!s) return;
  const job = s.jobs.viz, up = job.state === 'running';
  const task = typed(), info = TASKS[task];
  const prep = s.jobs.vizprep;
  const hf = isRepo(task), fetch = s.jobs.vizfetch, ready = hf && HF_READY[task] !== undefined;
  const fetching = hf && fetch.alive && fetch.meta.repo === task;
  const fetchFailed = hf && !fetch.alive && fetch.state === 'failed' && fetch.meta.repo === task;
  if (fetchedFor && !fetch.alive && fetch.meta.repo === fetchedFor) {   // download just finished
    const repo = fetchedFor;
    fetchedFor = null;
    if (fetch.state === 'done') refresh().then(() => { if (typed() === repo) view(); });
  }
  $('viz-info').textContent = hf
    ? (fetching ? `Hugging Face dataset: ${fetch.progress.stage || 'starting'} ${fetch.progress.step || '…'}`
       : ready ? `Hugging Face dataset · ${HF_READY[task]} episode(s) downloaded`
       : 'Hugging Face dataset · not downloaded yet')
    : info && info.synced
    ? `${info.synced} synced episode(s)` + (info.lerobot.length ? ' · converted ✔' : '') +
      (prep && prep.alive && prep.meta.task === task
        ? ` · preparing videos ${prep.progress.step || '…'} (ready ones open instantly)` : '')
    : '';
  $('viz-fetch').hidden = !(hf && ready) || fetching;

  setState($('viz-panel'), job.state);
  $('viz-state').textContent = up ? 'Open' : STATE_LABEL[job.state];
  $('viz-stop').disabled = !job.alive;
  let reason = '';
  if (!s.scripts.viz) reason = 'visualize_dataset.py not found in this checkout.';
  else if (!task) reason = 'Type a dataset name.';
  else if (fetching) reason = 'Downloading from Hugging Face…';
  else if (hf) reason = '';
  else if (!valid()) reason = 'Fix the dataset name first.';
  else if (!info) reason = `No dataset called "${task}". A Hugging Face dataset is typed as user/name.`;
  else if (!info.synced) reason = `"${task}" is not synced yet. Do Sync first.`;
  $('viz-start').disabled = !!reason || job.state === 'starting';
  $('viz-start').textContent = hf && !ready ? '⤓ Download & view' : job.alive ? '▶ Show this one' : '▶ View';
  $('viz-reason').textContent = reason;
  $('viz-reason').classList.toggle('blocked', !!reason);
  $('viz-detail').textContent = job.reason || '';

  if (fetchFailed) instruct('bad', `Could not download "${task}": ` +
    (fetch.errors.slice(-1)[0] || fetch.reason) + '. Check the name, and that this NUC is logged in to ' +
    'Hugging Face for a private dataset. Open LOG for details.');
  else if (fetching) instruct('', `Downloading "${task}" from Hugging Face… ${fetch.progress.step || ''}`);
  else if (job.state === 'failed') instruct('bad', 'The viewer had a problem: ' + job.reason + '. Open LOG for details.');
  else if (job.state === 'starting') instruct('', 'Opening the viewer… wait a moment.');
  else if (up && loadedFor) instruct('good', `Showing "${loadedFor}". Scroll down to watch it.`);
  else if (reason) instruct('', reason === 'Type a dataset name.' ? 'Type the dataset name you want to see, then press ▶ VIEW.' : reason);
  else instruct('good', 'Ready. Press ▶ VIEW.');

  $('viz-open').hidden = !(up && loadedFor);
  $('viz-open').href = dashUrl(s, loadedFor);
  const frame = $('viz-frame');
  frame.hidden = !(up && loadedFor);
  if (up && wanted && loadedFor !== wanted) {
    loadedFor = wanted;
    frame.src = dashUrl(s, wanted);
  }
  if (!up) { loadedFor = null; if (frame.src) frame.removeAttribute('src'); }
}

let wanted = null;   // the name the operator last pressed VIEW for
let fetchedFor = null;   // Hugging Face dataset being downloaded, viewed once it is ready
async function download(task) {
  const r = await api('/api/viz/fetch', {repo: task});
  if (!r.ok) return showProblem($('viz-msg'), r);
  fetchedFor = task;
  update();
}
async function view() {
  $('viz-msg').innerHTML = '';
  const task = typed();
  if (isRepo(task) && HF_READY[task] === undefined) return download(task);
  const r = await api('/api/viz/start', {task});
  if (!r.ok) return showProblem($('viz-msg'), r);
  wanted = task;
  loadedFor = null;
  update();
}
$('viz-start').onclick = view;
$('viz-fetch').onclick = () => { $('viz-msg').innerHTML = ''; download(typed()); };
$('viz-form').addEventListener('submit', ev => { ev.preventDefault(); if (!$('viz-start').disabled) view(); });
$('viz-stop').onclick = () => { wanted = null; api('/api/viz/stop', {}); };
document.addEventListener('fui:status', update);

// Viewer hot keys (Space, arrows) work from this page too: they are passed to
// the embedded viewer, which handles them (see visualize_dataset.py).
const VIEWER_KEYS = [' ', 'ArrowRight', 'ArrowLeft', 'ArrowDown', 'ArrowUp'];
document.addEventListener('keydown', ev => {
  const frame = $('viz-frame');
  if (frame.hidden || !frame.contentWindow || !VIEWER_KEYS.includes(ev.key)) return;
  if (ev.ctrlKey || ev.metaKey || ev.altKey) return;
  if (ev.target.matches('input, textarea, select')) return;   // typing a name
  ev.preventDefault();   // no page scroll
  frame.contentWindow.postMessage({type: 'fastumi-viewer-key', key: ev.key}, '*');
});

prefillRecorded($('viz-task'));
refresh();
setInterval(refresh, 10000);
