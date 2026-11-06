// Robot page: replay one episode of a LeRobot joint dataset on the UR7e, in three
// steps: dry run (no robot) -> read-only robot check -> run. The server refuses
// the run until a dry run and a check of the same episode / speed / IP passed.
'use strict';

const $ = id => document.getElementById(id);
const JOINTS = ['Base', 'Shoulder', 'Elbow', 'Wrist 1', 'Wrist 2', 'Wrist 3'];
const MODE_WORDS = {dry: 'Dry run', check: 'Robot check', run: 'Run on robot'};
let OPTS = null;
let INFO = null;          // /api/replay/info for the typed dataset
let infoFor = '';         // the text INFO was looked up for
let looking = false;
const results = {};       // mode -> {key, state, reason, progress} of the last finished run

function store(k, v) { try { if (v === undefined) return localStorage.getItem(k); localStorage.setItem(k, v); } catch (e) { return null; } }

function instruct(kind, text) {
  $('r-instr').className = 'instruction' + (kind ? ' ' + kind : '');
  $('r-instr').querySelector('.txt').textContent = text;
}

function fmtDur(s) {
  if (s == null) return '—';
  if (s < 90) return Math.round(s) + ' s';
  if (s < 5400) return Math.round(s / 60) + ' min';
  return (s / 3600).toFixed(1) + ' h';
}

const sameKey = (a, b) => !!a && !!b && JSON.stringify(a) === JSON.stringify(b);
function form() {
  return {
    dataset: INFO ? INFO.dataset : null, episode: $('r-episode').value === '' ? null : Number($('r-episode').value),
    speed: Number($('r-speed').value), ip: $('r-ip').value.trim(),
  };
}
function keys() {
  const f = form();
  return {dry: [f.dataset, f.episode, f.speed], check: [f.dataset, f.episode, f.ip]};
}

// ── options, dataset lookup ────────────────────────────────────────────────
async function loadOptions() {
  OPTS = await api('/api/replay/options');
  const dl = $('r-local');
  dl.innerHTML = '';
  for (const d of OPTS.datasets || []) {
    dl.append(el('option', {value: d.label, label: `${d.episodes} episodes · ${d.fps} fps (on this NUC)`}));
  }
  const sp = $('r-speed');
  sp.innerHTML = '';
  for (const v of OPTS.speeds || [0.1]) sp.append(el('option', {value: String(v), text: speedLabel(v)}));
  sp.value = store('fui-replay-speed') || '0.25';
  if (!sp.value) sp.value = String(OPTS.speeds[0]);
  $('r-ip').value = store('fui-robot-ip') || OPTS.robot_ip || '';
  if (!$('r-dataset').value) $('r-dataset').value = store('fui-replay-dataset') || '';
  lookup();
}

function speedLabel(v) {
  // After a dry run the duration at the other speeds follows from it (time scales with 1/speed).
  const r = results.dry, p = r && r.progress;
  let words = `${Math.round(v * 100)}%`;
  if (r && r.state === 'done' && p.duration && r.key[0] === form().dataset && r.key[1] === form().episode) {
    words += ` (about ${fmtDur(p.duration * r.key[2] / v)})`;
  }
  if (OPTS && v > OPTS.fast) words += ' needs CONFIRM_FAST';
  return words;
}
function relabelSpeeds() {
  for (const o of $('r-speed').options) o.textContent = speedLabel(Number(o.value));
}

let lookupTimer = null;
function lookup() {
  clearTimeout(lookupTimer);
  lookupTimer = setTimeout(async () => {
    const text = $('r-dataset').value.trim();
    if (text === infoFor) return;
    infoFor = text; INFO = null;
    const sel = $('r-episode');
    sel.innerHTML = '<option value="">—</option>'; sel.disabled = true;
    if (!text) { $('r-info').textContent = ''; render(); return; }
    looking = true;
    $('r-info').textContent = text.includes('/') && !text.startsWith('/') ? 'Looking it up on Hugging Face…' : 'Reading…';
    render();
    const r = await api('/api/replay/info?dataset=' + encodeURIComponent(text));
    if ($('r-dataset').value.trim() !== text) return;
    looking = false;
    if (!r.ok) {
      $('r-info').innerHTML = '';
      $('r-info').append(el('span', {class: 'bad', text: r.error}));
      render();
      return;
    }
    INFO = r;
    store('fui-replay-dataset', text);
    $('r-info').textContent = `${r.episodes} episodes · ${r.frames} frames · ${r.fps} fps · joint angles` +
      (r.dataset.startsWith('/') ? ` · ${r.dataset}` : ' · on Hugging Face');
    sel.innerHTML = '';
    const want = store('fui-replay-episode');
    for (let i = 0; i < r.episodes; i++) sel.append(el('option', {value: String(i), text: `Episode ${i}`}));
    if (want !== null && Number(want) < r.episodes) sel.value = want;
    sel.disabled = false;
    render();
  }, 500);
}

// ── follow the job ─────────────────────────────────────────────────────────
function capture(job) {
  // Keep each mode's last finished result, so all three steps show theirs.
  if (job.alive || !job.meta.mode || !job.started_at) return;
  if (!['done', 'failed', 'stopped'].includes(job.state)) return;
  const prev = results[job.meta.mode];
  if (prev && prev.started_at === job.started_at) return;
  results[job.meta.mode] = {key: job.meta.key, state: job.state, reason: job.reason, started_at: job.started_at,
                            progress: Object.assign({}, job.progress), errors: job.errors.slice()};
  if (job.meta.mode === 'dry') relabelSpeeds();
}

function jointTable(p) {
  const t = el('table', {class: 'keys'});
  t.append(el('tr', {}, el('th', {text: ''}), ...JOINTS.map(j => el('th', {text: j}))));
  if (p.current) t.append(el('tr', {}, el('th', {text: 'Robot now'}), ...p.current.map(v => el('td', {text: v.toFixed(1) + '°'}))));
  if (p.start) t.append(el('tr', {}, el('th', {text: 'Episode start'}), ...p.start.map(v => el('td', {text: v.toFixed(1) + '°'}))));
  return t;
}

function renderResult(mode, box, passedKey) {
  box.innerHTML = '';
  const r = results[mode], k = keys()[mode];
  if (mode === 'run') {
    if (!r) return;
    const p = r.progress;
    if (r.state === 'done' && p.phase === 'complete') box.append(el('span', {class: 'ok', text: '✔ Playback complete.'}));
    else if (r.state === 'stopped' || p.phase === 'interrupted') box.append(el('span', {class: 'bad', text: '■ Stopped.'}));
    else box.append(el('span', {class: 'bad', text: '✖ ' + (r.errors.slice(-1)[0] || r.reason)}));
    return;
  }
  if (!r || !sameKey(r.key, k)) {
    if (sameKey(passedKey, k)) box.append(el('span', {class: 'ok', text: '✔ Passed.'}));
    return;
  }
  const p = r.progress;
  if (r.state !== 'done') {
    box.append(el('span', {class: 'bad', text: r.state === 'stopped' ? '■ Stopped.' : '✖ ' + (r.errors.slice(-1)[0] || r.reason)}));
    return;
  }
  if (mode === 'dry') {
    box.append(el('span', {class: 'ok', text: '✔ OK. '}),
      `${p.frames} frames (${p.secs}s recorded), ${p.outside || 0} outside the joint limits. ` +
      `At ${Math.round(form().speed * 100)}% it takes about ${fmtDur(p.duration)}, peak ${p.peak_vel}°/s.`);
  } else {
    const safe = p.safety === 'normal' && p.robot_mode === 'running';
    box.append(el('span', {class: safe ? 'ok' : 'bad', text: safe ? '✔ Connected. ' : '▲ '}),
      `Robot ${p.robot_mode}, safety ${p.safety}. Largest difference to the episode start: ${p.diff}°.`);
    if (!safe) box.append(el('div', {text: 'The robot must be powered on, brakes released (running), safety normal.'}));
    box.append(jointTable(p));
    if (p.diff > 20) box.append(el('div', {text: `The arm is ${p.diff}° from the start: jog it closer with the pendant (the move there is very slow), then check again.`}));
  }
}

function render() {
  const s = FUI.status;
  if (!s || !OPTS) { instruct('', 'Connecting…'); return; }
  const job = s.jobs.replay, p = job.progress || {};
  capture(job);
  const f = form(), k = keys(), passed = s.replay_passed || {};
  const dryOk = sameKey(passed.dry, k.dry), checkOk = sameKey(passed.check, k.check);
  const ipOk = /^\d{1,3}(\.\d{1,3}){3}$/.test(f.ip);
  const ready = !!INFO && f.episode !== null;
  const busy = job.alive;
  if (!checkOk) $('r-match').checked = false;
  $('r-match-box').hidden = !checkOk;

  $('r-dry').disabled = busy || !ready || !OPTS.script;
  $('r-check').disabled = busy || !ready || !ipOk || !OPTS.script;
  $('r-run').disabled = busy || !ready || !dryOk || !checkOk || !$('r-match').checked;
  $('r-stop').disabled = !busy;
  $('r-step-dry').classList.toggle('passed', dryOk);
  $('r-step-check').classList.toggle('passed', checkOk);
  $('r-step-check').classList.toggle('locked', !dryOk);
  $('r-step-run').classList.toggle('locked', !(dryOk && checkOk));
  renderResult('dry', $('r-dry-res'), passed.dry);
  renderResult('check', $('r-check-res'), passed.check);
  renderResult('run', $('r-run-res'), null);

  renderJob(s, job, p);

  if (busy) {
    const what = MODE_WORDS[job.meta.mode] + ` of episode ${job.meta.episode}`;
    if (job.meta.mode === 'run') instruct('bad', `The robot is moving (${what}). Hand on the e-stop. ■ Stop asks it to stop.`);
    else instruct('', `${what}…`);
  } else if (!OPTS.script) instruct('bad', 'replay/replay_lerobot.py is not in this checkout.');
  else if (looking) instruct('', 'Looking up the dataset…');
  else if (!$('r-dataset').value.trim()) instruct('', 'Type a joint-angle dataset: a Hugging Face repo id or a folder from the list.');
  else if (!INFO) instruct('bad', 'This dataset cannot be replayed (see under the dataset box).');
  else if (!dryOk) instruct('', 'Step 1: press DRY RUN (the robot does not move).');
  else if (!ipOk) instruct('', 'Type the robot IP from the pendant.');
  else if (!checkOk) instruct('', 'Step 2: put only this robot in Remote Control, then press CHECK ROBOT (read-only).');
  else if (!$('r-match').checked) instruct('', 'Compare the joints in step 2 with this robot\'s pendant, then tick the box.');
  else instruct('good', 'Ready. Workspace clear, hand on the e-stop, then press ▶ RUN ON ROBOT.');
}

function renderJob(s, job, p) {
  const box = $('r-job');
  const st = job.state;
  setState(box, st === 'running' ? 'starting' : st);
  box.querySelector('.state-word').textContent = st === 'running' ? 'Working' : STATE_LABEL[st];
  let stage = job.meta.mode ? `${MODE_WORDS[job.meta.mode]} · episode ${job.meta.episode} · ` +
    `${Math.round(job.meta.speed * 100)}%` + (job.meta.mode !== 'dry' && job.meta.ip ? ` · ${job.meta.ip}` : '') : '';
  let step = '', percent = null;
  if (job.alive) {
    const el_s = p.phase_t ? s.now - p.phase_t : 0;
    if (p.phase === 'approach') {
      step = `Moving slowly to the episode start… ${fmtDur(el_s)} of about ${fmtDur(p.phase_s)}`;
      percent = Math.min(99, 100 * el_s / Math.max(p.phase_s, 1));
    } else if (p.phase === 'reached') {
      step = 'At the start. Starting playback…';
    } else if (p.phase === 'playing') {
      step = `Playing… ${fmtDur(el_s)} of ${fmtDur(p.phase_s)}`;
      percent = Math.min(99, 100 * el_s / Math.max(p.phase_s, 1));
    } else if (job.meta.mode === 'run' && p.frames) {
      step = 'Connecting to the robot…';
    } else {
      step = p.frames ? 'Working…' : 'Loading the dataset…';
    }
  } else if (st === 'done') {
    step = job.meta.mode === 'run' ? (p.phase === 'complete' ? '✔ Playback complete.' : '✔ Finished.') : '✔ Done.';
  } else if (st === 'failed') {
    step = '✖ ' + (job.errors.slice(-1)[0] || job.reason);
  } else if (st === 'stopped' && job.meta.mode) {
    step = 'Stopped.';
  }
  const bar = $('r-bar');
  bar.classList.toggle('indeterminate', job.alive && percent === null);
  bar.classList.toggle('failed', st === 'failed');
  bar.firstElementChild.style.width = st === 'done' ? '100%' : (percent !== null ? percent.toFixed(1) + '%' : (job.alive ? '' : '0'));
  $('r-stage').textContent = stage;
  $('r-step').textContent = step;

  const ro = $('r-readouts');
  ro.innerHTML = '';
  const items = [];
  if (p.secs) items.push(['Recorded', `${p.secs} s`]);
  if (p.outside != null) items.push(['Outside joint limits', String(p.outside)]);
  if (p.duration) items.push(['Replay takes', fmtDur(p.duration)]);
  if (p.peak_vel) items.push(['Peak joint speed', p.peak_vel + '°/s']);
  if (p.robot_mode) items.push(['Robot', `${p.robot_mode} · ${p.safety}`]);
  if (p.diff != null) items.push(['Away from start', p.diff + '°']);
  for (const [k, v] of items) ro.append(el('div', {class: 'readout'}, el('div', {class: 'k', text: k}), el('div', {class: 'v', text: v})));
  ro.hidden = !items.length;

  $('r-errors').innerHTML = '';
  if (job.errors.length) {
    $('r-errors').append(el('div', {class: 'notice error'}, el('pre', {class: 'errors', text: job.errors.slice(-20).join('\n')})));
  }
  const plot = results.dry && results.dry.state === 'done' && results.dry.progress.plot;
  $('r-plot-box').hidden = !plot;
  if (plot) {
    const src = `/api/replay/plot/${encodeURIComponent(plot)}?t=${results.dry.started_at}`;
    if ($('r-plot').getAttribute('src') !== src) $('r-plot').src = src;
  }
}

// ── actions ────────────────────────────────────────────────────────────────
async function start(mode) {
  const f = form();
  const body = {mode, dataset: f.dataset, episode: f.episode, speed: f.speed, robot_ip: f.ip};
  if (mode === 'run') {
    const typed = prompt(`This moves the PHYSICAL UR7e at ${f.ip}:\n` +
      `episode ${f.episode} at ${Math.round(f.speed * 100)}% speed.\n\n` +
      'Workspace clear? Hand on the e-stop?\nType RUN to start.');
    if (typed === null) return;
    body.confirm = typed.trim();
    if (f.speed > OPTS.fast) {
      const fast = prompt(`${Math.round(f.speed * 100)}% is above the careful ${Math.round(OPTS.fast * 100)}%.\nType CONFIRM_FAST to run at this speed.`);
      if (fast === null) return;
      body.confirm_fast = fast.trim();
    }
  }
  const r = await api('/api/replay/start', body);
  if (!r.ok) { alert(r.error); return; }
  if ($('r-logbox').hidden && mode === 'run') $('r-logbtn').click();
}

$('r-dry').onclick = () => start('dry');
$('r-check').onclick = () => start('check');
$('r-run').onclick = () => start('run');
$('r-stop').onclick = () => api('/api/replay/stop', {});
$('r-dataset').addEventListener('input', lookup);
$('r-episode').addEventListener('change', () => { store('fui-replay-episode', $('r-episode').value); relabelSpeeds(); render(); });
$('r-speed').addEventListener('change', () => { store('fui-replay-speed', $('r-speed').value); render(); });
$('r-ip').addEventListener('input', () => { store('fui-robot-ip', $('r-ip').value.trim()); render(); });
$('r-match').addEventListener('change', render);

$('r-logbtn').onclick = () => {
  const box = $('r-logbox');
  box.hidden = !box.hidden;
  $('r-logbtn').textContent = box.hidden ? 'Show log ▾' : 'Hide log ▴';
  if (!box.hidden) { const t = $('r-tail'); t.textContent = tail('Replay', 200); t.scrollTop = t.scrollHeight; }
};
document.addEventListener('fui:logs', () => {
  const t = $('r-tail');
  if ($('r-logbox').hidden) return;
  const atBottom = t.scrollTop + t.clientHeight >= t.scrollHeight - 4;
  t.textContent = tail('Replay', 200);
  if (atBottom) t.scrollTop = t.scrollHeight;
});
document.addEventListener('fui:status', render);

loadOptions();
