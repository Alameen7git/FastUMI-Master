// Record (with task list, machine-loss recovery, live camera), Sync, Convert
// (with optional Hub push) and Push.
'use strict';

const $ = id => document.getElementById(id);
let TASKS = {};          // datasets on disk: name -> {raw, synced, lerobot}
let REG = [];            // operator task list
let HF = null;           // {user} or {error}
let NAMING = {today: '', name_format: 'UMI_{date}_T{task_id}_{operator}{num}', last: null, highest: null};

// ── sections (chosen by the step bar via #hash) ─────────────────────────────
function showTab() {
  const name = (location.hash || '#collect').slice(1);
  document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.id === 'tab-' + name));
}
window.addEventListener('hashchange', () => { showTab(); if (FUI.status) updateAll(FUI.status); });
showTab();

function goStep(hash) { location.hash = hash; window.scrollTo(0, 0); }

// The big one-line instruction at the top of each section.
function instruct(id, kind, text, link) {
  const box = $(id);
  box.className = 'instruction' + (kind ? ' ' + kind : '');
  const t = box.querySelector('.txt');
  t.textContent = text;
  if (link) t.append(' ', el('a', {class: 'btn primary', href: link.href}, link.label));
}


async function refreshData() {
  const [t, r] = await Promise.all([api('/api/tasks'), api('/api/registry')]);
  TASKS = t.tasks || {};
  REG = r.tasks || [];
  if (r.today) NAMING = {today: r.today, name_format: r.name_format, last: r.last_number, highest: r.highest_number};
  renderLastNumbers();
  renderTaskSelect();
  updateTaskForm();
  if (editing === null) renderTaskRows();
  if (FUI.status) updateAll(FUI.status);
}

// ── Record: sub-tabs ────────────────────────────────────────────────────────
function showSub(name) {
  document.querySelectorAll('.subtab').forEach(b => b.classList.toggle('active', b.dataset.sub === name));
  $('sub-record').hidden = name !== 'record';
  $('sub-tasks').hidden = name !== 'tasks';
  if (FUI.status) updateAll(FUI.status);
}
document.querySelectorAll('[data-sub]').forEach(b => { b.onclick = () => showSub(b.dataset.sub); });
document.querySelectorAll('[data-goto-sub]').forEach(b => { b.onclick = () => showSub(b.dataset.gotoSub); });

// ── Record: task picker ─────────────────────────────────────────────────────
// Pick from the task list (T7, T4.1, …), or switch with the link under it to
// typing a new name. A typed task is added to the list when recording starts.
let typedMode = stored('fui-task-typed') === '1';
function isTyped() { return typedMode; }
function stored(key) { try { return localStorage.getItem(key); } catch (e) { return null; } }
function store(key, v) { try { localStorage.setItem(key, v); } catch (e) { /* storage blocked */ } }
function taskLabel(t) { return 'T' + t.id; }

// {name, entry} for the chosen task; entry is null for a typed name not in the list.
function selectedTask() {
  const v = $('c-task').value;
  if (isTyped()) {
    const name = $('c-typed').value;
    return name ? {name, entry: REG.find(t => t.name === name) || null} : null;
  }
  const entry = REG.find(t => String(t.id) === v);
  return entry ? {name: entry.name, entry} : null;
}

function datasetName(taskId, operator, num) {
  return NAMING.name_format.replace('{date}', NAMING.today).replace('{task_id}', String(taskId))
    .replace('{operator}', operator).replace('{num}', String(num));
}
// Task numbers are typed by the operator (7, 4.1, …); the T is shown in front.
function normNum(v) {
  const m = String(v || '').trim().match(/^[Tt]?\s*0*(\d+)(?:\.(\d+))?$/);
  return m ? m[1] + (m[2] !== undefined ? '.' + m[2] : '') : null;
}
function numberOwner(num) { return REG.find(t => String(t.id) === num); }
// Why a number typed for a new task can't be used, or '' when it can.
function numProblem(raw) {
  if (!String(raw).trim()) return 'Type the task number.';
  const num = normNum(raw);
  if (num === null) return 'Task number must be a number like 7 or 4.1.';
  const other = numberOwner(num);
  return other ? `T${num} is already used by "${other.name}". Pick another number.` : '';
}
function renderLastNumbers() {
  const {last, highest} = NAMING;
  for (const id of ['t-last-num', 'c-last-num']) {
    const box = $(id);
    box.innerHTML = '';
    if (!last) { box.textContent = 'No task numbers used yet.'; continue; }
    box.append('Last used: ', el('strong', {text: 'T' + last}));
    if (highest && highest !== last) box.append('  ·  highest: ', el('strong', {text: 'T' + highest}));
  }
}
// A typed task not in the list yet uses the number typed under its name.
function taskId(t) { return t.entry ? String(t.entry.id) : normNum($('c-typed-num').value); }
function needsNumber(t) { return isTyped() && !!t && !t.entry; }

// Dataset numbers this operator already used for this task today (mirrors dataset_numbers()).
function datasetNumbers(id, op) {
  const prefix = datasetName(id, op, '').replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const re = new RegExp('^' + prefix + '(\\d+)$');
  return Object.keys(TASKS).map(n => (n.match(re) || [])[1]).filter(Boolean).map(Number).sort((a, b) => a - b);
}

// 'new' or an existing number; reset when the task or operator changes.
let numChoice = null, numKey = null, chosenNum = null;
function opEndsWithDigit() { return /\d$/.test($('c-operator').value); }

function renderTaskSelect() {
  const sel = $('c-task'), keep = sel.value || stored('fui-last-task');
  sel.innerHTML = '';
  if (!REG.length) sel.append(el('option', {value: ''}, 'No tasks in the list yet'));
  for (const t of REG) {
    const label = `${taskLabel(t)}  ${t.name}` + (t.description ? ` — ${t.description}` : '');
    sel.append(el('option', {value: String(t.id)}, label));
  }
  if (keep && REG.some(t => String(t.id) === keep)) sel.value = keep;
  onTaskPicked(false);
}

function onTaskPicked(userChange) {
  const typed = isTyped();
  $('c-task').disabled = typed || !REG.length;
  $('c-task-mode').checked = typed;
  $('c-typed').hidden = !typed;
  $('c-typed-msg').hidden = !typed;
  $('c-typed-num-wrap').hidden = !needsNumber(selectedTask());
  if ($('c-task').value) store('fui-last-task', $('c-task').value);
  renderDatasetName();
}

function renderDatasetName() {
  const t = selectedTask(), op = $('c-operator').value;
  const typedOk = !isTyped() || cTypedValid();
  $('c-typed-num-wrap').hidden = !needsNumber(t);
  const numOk = !needsNumber(t) || !numProblem($('c-typed-num').value);
  const ready = t && op && typedOk && numOk && cOpValid() && !opEndsWithDigit() && NAMING.today;
  $('c-dsname').hidden = !ready;
  if (!ready) { chosenNum = null; return; }
  const id = taskId(t);
  const key = `${id}|${op}|${NAMING.today}`;
  if (key !== numKey) { numKey = key; numChoice = null; }
  const used = datasetNumbers(id, op);
  const latest = used.length ? used[used.length - 1] : null, next = (latest || 0) + 1;
  // A number just recorded into may not be on disk yet (next refresh): keep it.
  if (numChoice !== 'new' && !used.includes(numChoice) && numChoice !== next) {
    numChoice = latest === null ? 'new' : latest;
  }
  chosenNum = numChoice === 'new' ? next : numChoice;

  const box = $('c-dsnum');
  box.innerHTML = '';
  if (latest !== null) {
    const eps = n => { const d = TASKS[datasetName(id, op, n)]; return d ? Math.max(d.raw, d.synced) : 0; };
    const option = (value, label) => {
      const r = el('input', {type: 'radio', name: 'c-dsnum', value: String(value)});
      r.checked = numChoice === value;
      r.onchange = () => { numChoice = value; renderDatasetName(); updateAll(FUI.status); };
      return el('label', {class: 'check'}, r, label);
    };
    box.append(option(latest, `Add to ${op}${latest} (${eps(latest)} episode(s) so far)`),
               option('new', `Start a new dataset: ${op}${next}`));
  }
  const ds = datasetName(id, op, chosenNum);
  $('c-dsname-v').textContent = ds;
  $('c-info').textContent = used.includes(chosenNum)
    ? 'New episodes are added after the ones already in this folder.'
    : 'A new folder.' + (t.entry ? (isTyped() ? ` "${t.name}" is already in your task list as T${id}.` : '')
      : ` "${t.name}" will be added to your task list as T${id}.`);
}

const cTypedValid = attachNameValidation($('c-typed'), $('c-typed-msg'), () => {
  renderDatasetName(); updateAll(FUI.status);
}, 'Task name');
const cOpValid = attachNameValidation($('c-operator'), $('c-operator-msg'), () => {
  store('fui-operator', $('c-operator').value); renderDatasetName(); updateAll(FUI.status);
}, 'Operator name');
if (stored('fui-operator')) setField('c-operator', stored('fui-operator'));
$('c-task').addEventListener('change', () => { onTaskPicked(true); updateAll(FUI.status); });
$('c-task-mode').onchange = () => {
  typedMode = $('c-task-mode').checked;
  store('fui-task-typed', typedMode ? '1' : '0');
  onTaskPicked(true);
  (typedMode ? $('c-typed') : $('c-task')).focus();
  updateAll(FUI.status);
};

function updateCollectForm(s) {
  const t = selectedTask();
  const typed = isTyped();
  let reason = '';
  if (!s) reason = 'Connecting…';
  else if (s.jobs.collection.alive) reason = 'A recording session is already running.';
  else if (!t) reason = typed ? 'Type the task name.'
    : REG.length ? 'Pick a task.' : 'No tasks in the list yet: tick "Type the task name manually", or add one on the Tasks tab.';
  else if (typed && !cTypedValid()) reason = 'Fix the task name first.';
  else if (needsNumber(t) && numProblem($('c-typed-num').value)) reason = numProblem($('c-typed-num').value);
  else if (!$('c-operator').value) reason = 'Type your operator name.';
  else if (!cOpValid()) reason = 'Fix the operator name first.';
  else if (opEndsWithDigit()) reason = "Operator name can't end with a number: n1, n2, … is added for you.";
  else if (s.collect_blocked) reason = s.collect_blocked;
  $('c-start').disabled = !!reason;
  $('c-reason').textContent = reason;
  $('c-reason').classList.toggle('blocked', !!reason);
  if (!s || s.jobs.collection.alive) return;
  if ($('sub-tasks').hidden === false) {
    instruct('c-instr', '', 'Add the tasks you plan to record: type each one\'s number (T7, T4.1, …) and name. You can edit them later.');
  } else if (!$('c-done').hidden) {
    instruct('c-instr', 'good', 'Recording finished. Go on to Synchronize, or record more episodes.');
  } else if (s.collect_blocked) {
    instruct('c-instr', 'bad', 'The machines are not ready yet.', {href: '/', label: '← Go to Machines'});
  } else {
    instruct('c-instr', 'good', 'Pick or type the task, type your name, then press ● START RECORDING.');
  }
}

$('collect-form').addEventListener('submit', async ev => {
  ev.preventDefault();
  $('c-msg').innerHTML = '';
  const t = selectedTask();
  if (!t) return;
  doneHiddenFor = null;
  const res = await api('/api/collect/start', {
    task: t.name, operator: $('c-operator').value, episodes: $('c-episodes').value, dataset_num: chosenNum,
    task_number: needsNumber(t) ? $('c-typed-num').value : undefined,
  });
  if (!res.ok) return showProblem($('c-msg'), res);
  numChoice = chosenNum;   // "Record more" afterwards adds to this dataset
  if (!t.entry) { $('c-typed').value = ''; $('c-typed-num').value = ''; await refreshData(); selectTaskByName(t.name); }
});

// ── Record: keys ────────────────────────────────────────────────────────────
async function sendKey(key) {
  const r = await api('/api/collect/key', {key});
  if (!r.ok) showProblem($('c-msg'), r);
}
document.querySelectorAll('[data-key]').forEach(b => { b.onclick = () => pressKey(b.dataset.key); });

// E (■ Finish): during a take, ask whether to keep it (the recorder's E saves
// it, X throws it away); between takes, just confirm.
function pressKey(key) {
  if (key !== 'e') return sendKey(key);
  const s = FUI.status, job = s && s.jobs.collection, p = (job && job.progress) || {};
  const dlg = $('c-end-dlg');
  if (!job || !job.alive || dlg.open) return;
  const saved = p.saved || 0, total = p.total || job.meta.episodes;
  const recording = p.phase === 'recording';
  const ep = p.episode || saved + 1;
  $('c-end-title').textContent = recording ? `Save episode ${ep} before ending?` : 'End the session?';
  $('c-end-text').textContent = recording
    ? `Episode ${ep} is still recording. ${saved} of ${total} episode(s) are saved so far.`
    : `${saved} of ${total} episode(s) are saved. They are kept.`;
  $('c-end-save').textContent = recording ? `Save episode ${ep} & end` : 'End session';
  $('c-end-discard').hidden = !recording;
  $('c-end-cancel').textContent = recording ? 'Keep recording' : 'Cancel';
  dlg.returnValue = 'cancel';
  dlg.showModal();
  $('c-end-cancel').focus();
}
$('c-end-dlg').addEventListener('close', () => {
  const v = $('c-end-dlg').returnValue;
  if (v === 'save') sendKey('e');
  else if (v === 'discard') sendKey('x');
});
$('c-continue').onclick = () => sendKey('c');
$('c-savefinish').onclick = () => {
  if (confirm('Stop here? All episodes saved so far are kept and the session ends.')) sendKey('e');
};

const KEYMAP = {' ': 'space', r: 'r', R: 'r', e: 'e', E: 'e', c: 'c', C: 'c'};
// 5-key USB keypad (SayoDevice 5K2U) sends the digits 1-5. Unlike Space these
// are one-way, so a double press can't undo itself: 5 / 4 only START a take
// (ignored while recording), 1 / 2 only STOP + save it (ignored otherwise). 3 is unused.
const PAD_START = new Set(['5', '4']), PAD_STOP = new Set(['1', '2']);
function keyTarget(ev) {
  const t = ev.target.tagName;
  return !(t === 'INPUT' || t === 'SELECT' || t === 'TEXTAREA');
}
document.addEventListener('keydown', ev => {
  const s = FUI.status;
  let k = KEYMAP[ev.key];
  if (PAD_START.has(ev.key) || PAD_STOP.has(ev.key)) {
    const phase = ((s && s.jobs.collection.progress) || {}).phase;
    if (PAD_START.has(ev.key) ? phase === 'ready' : phase === 'recording') k = 'space';
    else if (s && s.jobs.collection.alive && keyTarget(ev)) { ev.preventDefault(); return; }
  }
  if (!k || !s || !s.jobs.collection.alive || !keyTarget(ev) || ev.repeat) return;
  if (ev.ctrlKey || ev.altKey || ev.metaKey) return;
  if ((location.hash || '#collect') !== '#collect') return;
  if ($('c-end-dlg').open) return;
  ev.preventDefault();
  pressKey(k);
});
document.addEventListener('keyup', ev => {  // stop Space from also "clicking" a focused button
  if (ev.key === ' ' && keyTarget(ev) && FUI.status && FUI.status.jobs.collection.alive) ev.preventDefault();
});

// ── Record: session panels ──────────────────────────────────────────────────
const PHASE = {starting: 'Starting…', ready: 'Ready', recording: '● Recording', saving: 'Saving…',
               paused: 'Stopped — machine problem', ended: 'Finished'};
const NODE_NAME = {ros: 'ROS', t265: 'T265 tracker', gopro: 'Camera'};
let doneHiddenFor = null;   // session the operator chose "Record more" for

function sessionKey(s) { return `${s.session.task}|${s.jobs.collection.started_at}`; }

$('c-more').onclick = async () => {
  doneHiddenFor = sessionKey(FUI.status);
  await api('/api/session/dismiss', {});
  updateAll(FUI.status);
};
$('c-next').onclick = async () => {
  doneHiddenFor = sessionKey(FUI.status);
  await api('/api/session/dismiss', {});
  goStep('#sync');
};
document.addEventListener('fui:record-more', () => { $('c-more').click(); goStep('#collect'); showSub('record'); });

// The machines during a session: restart one between takes (T265 drift, a
// stuck camera) without leaving the page. Not during a take: the server refuses
// that too, unless the machine already stopped working.
const RESTART_ASK = {
  t265: 'Restart the T265?\n\nPut the UMI in the home holder first: the T265 sets home where it starts.',
  ros: 'Restart ROS?\n\nThe recorder usually cannot reconnect to a new ROS. Better: ■ Finish the session, then ' +
       'restart from Machines. Restart anyway?',
  gopro: 'Restart the camera?',
};
function renderNodeStrip(s, phase) {
  const strip = $('c-nodes');
  strip.innerHTML = '';
  const inTake = phase === 'recording';
  for (const k of ['ros', 't265', 'gopro']) {
    const st = s.jobs[k].state;
    const btn = el('button', {type: 'button', class: 'small', text: '↻ Restart'});
    btn.disabled = st === 'starting' || st === 'stopping' || s.sequence_busy || (inTake && st === 'running');
    btn.title = inTake ? 'Stop or redo the take first' : `Stop and start ${NODE_NAME[k]} again`;
    btn.onclick = async () => {
      if (!confirm(RESTART_ASK[k])) return;
      const r = await api(`/api/node/${k}/restart`, {});
      if (!r.ok) alert(r.error);
    };
    strip.append(el('div', {class: 'node-chip st-' + st}, el('span', {class: 'led'}),
      el('strong', {text: NODE_NAME[k]}), el('span', {class: 'state-word', text: STATE_LABEL[st]}), btn));
  }
}

function renderProblem(s, phase) {
  const pr = s.collect_problem;
  const box = $('c-problem');
  box.hidden = !pr;
  if (!pr) return false;
  const paused = phase === 'paused';
  $('c-problem-title').textContent = paused
    ? 'Recording stopped: a machine stopped working'
    : 'A machine stopped working';
  const names = pr.nodes.map(k => NODE_NAME[k]).join(', ');
  $('c-problem-text').textContent = (pr.alert ? pr.alert.replace(/^⚠\s*/, '') + ' ' : '') +
    (paused ? 'The take in progress was not saved. ' : '') +
    (names ? `Restart ${names} below and wait for green.` : 'Wait for all machines to be green.');
  const list = $('c-problem-nodes');
  list.innerHTML = '';
  for (const k of pr.nodes) {
    const st = pr.states[k];
    const row = el('div', {class: 'problem-node st-' + st},
      el('span', {class: 'led'}), el('strong', {text: NODE_NAME[k]}),
      el('span', {class: 'state-word', text: STATE_LABEL[st]}), el('span', {class: 'grow'}));
    const btn = el('button', {type: 'button', class: 'primary', text: `↻ Restart ${NODE_NAME[k]}`});
    btn.disabled = st === 'starting' || st === 'stopping' || s.sequence_busy;
    btn.onclick = async () => { const r = await api(`/api/node/${k}/restart`, {}); if (!r.ok) alert(r.error); };
    row.append(btn);
    if (k === 'ros') {
      row.append(el('div', {class: 'small muted full', text:
        'If ROS itself stopped, the recorder usually cannot reconnect. Choose Save & finish, then start again from Machines.'}));
    }
    list.append(row);
  }
  $('c-continue').hidden = !paused;
  $('c-continue').disabled = !pr.all_green;
  $('c-continue').textContent = pr.all_green ? '▶ Continue recording' : '▶ Continue (waiting for green)';
  return true;
}

function renderCollection(s) {
  const job = s.jobs.collection, p = job.progress || {};
  const alive = job.alive;
  const phase = alive ? (p.phase || 'starting') : 'ended';
  const finished = !alive && s.session.ended && !!s.session.task && !!job.meta.task &&
    doneHiddenFor !== sessionKey(s);
  $('c-setup').hidden = alive || finished;
  $('c-done').hidden = !finished;
  $('c-live').hidden = !alive;
  if (finished) {
    const sv = s.session.saved || 0;
    $('c-done-title').textContent = job.state === 'failed' ? 'Recording ended with a problem' : 'Recording finished';
    $('c-done-text').textContent = `${sv} episode(s) saved in "${s.session.task}".` +
      (job.state === 'failed' ? ` The recorder stopped: ${job.reason}. Open LOG for details.` : '');
    setState($('c-done'), job.state === 'failed' ? 'failed' : 'done');
  }
  setState($('c-live'), job.state);
  $('c-state').textContent = alive ? 'Running' : STATE_LABEL[job.state];
  $('c-v-task').textContent = job.meta.task || '—';
  $('c-v-ep').textContent = p.episode ? `${p.episode} / ${p.total}` : '—';
  $('c-v-saved').textContent = job.meta.task ? `${p.saved || 0} / ${job.meta.episodes}` : '—';
  $('c-v-phase').textContent = PHASE[phase] || phase;
  $('c-v-phase').className = 'phase-box phase-' + phase;
  $('c-statusline').textContent = alive && phase === 'recording' ? job.status_line : '';

  const problem = alive && renderProblem(s, phase);
  renderNodeStrip(s, phase);
  const recording = phase === 'recording';
  $('c-keypad').hidden = phase === 'paused';
  $('k-space').disabled = !alive || phase === 'saving' || phase === 'starting' || (problem && !recording);
  $('k-space').className = recording ? 'stop' : 'go';
  $('k-space-label').textContent = recording ? '■ Stop take' : '▶ Start take';
  $('k-r').disabled = !recording;
  $('k-e').disabled = !alive;

  $('c-alert').innerHTML = '';
  if (!alive) return;
  let instr;
  if (problem) {
    instr = ['bad', phase === 'paused'
      ? 'Recording stopped because a machine stopped. Restart it below, or Save & finish.'
      : 'A machine stopped. Restart it below before the next take.'];
  } else {
    instr = {
      starting: ['', 'Starting the recorder… wait a moment.'],
      ready: ['good', `Episode ${p.episode || 1}: get ready, then press ▶ START TAKE.`],
      recording: ['bad', 'Recording. Do the task, then press ■ STOP TAKE.'],
      saving: ['', 'Saving… please wait.'],
      ended: ['', 'Finishing…'],
    }[phase] || ['', ''];
  }
  instruct('c-instr', instr[0], instr[1]);
}

// ── Record: live camera ─────────────────────────────────────────────────────
const cam = $('c-cam');
cam.addEventListener('toggle', async () => {
  if (cam.open) {
    $('c-cam-msg').textContent = 'Opening…';
    const r = await api('/api/preview/start', {});
    if (!r.ok) { $('c-cam-msg').textContent = r.error; }
  } else {
    $('c-cam-img').removeAttribute('src');
    api('/api/preview/stop', {});
  }
});
window.addEventListener('pagehide', () => { if (cam.open) navigator.sendBeacon('/api/preview/stop', '{}'); });

function renderCamera(s) {
  const job = s.jobs.preview, img = $('c-cam-img');
  setState(cam, cam.open ? job.state : 'stopped');
  $('c-cam-state').textContent = cam.open ? (job.state === 'running' ? 'live' : STATE_LABEL[job.state]) : 'closed';
  if (!cam.open) return;
  if (job.state === 'running') {
    if (!img.getAttribute('src')) img.src = s.preview_url + '?t=' + Date.now();
    $('c-cam-msg').textContent = '';
  } else {
    img.removeAttribute('src');
    if (job.state === 'failed') $('c-cam-msg').textContent = 'Camera preview stopped: ' + job.reason;
    else if (job.state === 'starting') $('c-cam-msg').textContent = 'Opening…';
    else if (!s.jobs.gopro.alive) $('c-cam-msg').textContent = 'The camera is off. Start it on the Machines page.';
  }
}

function selectTaskByName(name) {
  const t = REG.find(x => x.name === name);
  if (t) { typedMode = false; store('fui-task-typed', '0'); $('c-task').value = String(t.id); onTaskPicked(true); }
}

// ── Tasks: add / edit / remove ──────────────────────────────────────────────
const tValid = attachNameValidation($('t-name'), $('t-name-msg'), () => updateTaskForm());
$('t-num').addEventListener('input', updateTaskForm);
$('c-typed-num').addEventListener('input', () => {
  $('c-typed-num').classList.toggle('invalid', !!$('c-typed-num').value.trim() && normNum($('c-typed-num').value) === null);
  renderDatasetName(); updateAll(FUI.status);
});
function updateTaskForm() {
  const name = $('t-name').value, num = $('t-num').value;
  $('t-num').classList.toggle('invalid', !!num.trim() && normNum(num) === null);
  let reason = '';
  if (numProblem(num)) reason = numProblem(num);
  else if (!name) reason = 'Type a task name.';
  else if (!tValid()) reason = 'Fix the task name first.';
  else if (REG.some(t => t.name === name)) reason = `Task "${name}" is already in the list.`;
  $('t-add').disabled = !!reason;
  const show = reason && !['Type a task name.', 'Type the task number.'].includes(reason);
  $('t-reason').textContent = show ? reason : '';
  $('t-reason').classList.toggle('blocked', !!show);
  const d = TASKS[name];
  $('t-exists').innerHTML = '';
  if (d && !reason) {
    $('t-exists').append(el('div', {class: 'notice ok',
      text: `A dataset folder called "${name}" already exists. It will be listed under this task.`}));
  }
}
$('t-form').addEventListener('submit', async ev => {
  ev.preventDefault();
  $('t-msg').innerHTML = '';
  const r = await api('/api/registry', {
    number: $('t-num').value, name: $('t-name').value, description: $('t-desc').value,
  });
  if (!r.ok) return showProblem($('t-msg'), r);
  $('t-num').value = ''; $('t-name').value = ''; $('t-desc').value = '';
  $('t-name').dispatchEvent(new Event('input'));
  store('fui-last-task', String(r.task.id));
  $('c-task').value = '';
  await refreshData();
  $('t-msg').append(el('div', {class: 'notice ok', text: `Added task ${taskLabel(r.task)} "${r.task.name}". Pick it on the Record tab.`}));
});

let editing = null;   // id of the row being edited
function renderTaskRows() {
  const body = $('t-rows');
  body.innerHTML = '';
  if (!REG.length) {
    body.append(el('tr', {}, el('td', {colspan: 5, class: 'muted', text: 'No tasks yet. Add one above.'})));
    return;
  }
  for (const t of REG) {
    if (editing === t.id) { body.append(editRow(t)); continue; }
    const actions = el('td', {class: 'actions'},
      el('button', {type: 'button', class: 'small', onclick: () => { editing = t.id; renderTaskRows(); }}, 'Edit'),
      el('button', {type: 'button', class: 'small stop', onclick: () => removeTask(t)}, 'Remove'));
    body.append(el('tr', {},
      el('td', {class: 'num', text: taskLabel(t)}),
      el('td', {}, el('code', {text: t.name})),
      el('td', {class: 'small', text: t.description || '—'}),
      datasetsCell(t),
      actions));
  }
}
// Collapsed by default: a task can collect hundreds of datasets. Which rows
// are open survives the 5 s refresh.
const openDatasets = new Set();
function datasetsCell(t) {
  const ds = t.datasets || [];
  if (!ds.length) return el('td', {text: '—'});
  const box = el('details', {class: 'ds-list'},
    el('summary', {text: ds.length === 1 ? '1 dataset' : `${ds.length} datasets`}));
  box.open = openDatasets.has(t.id);
  box.addEventListener('toggle', () => { box.open ? openDatasets.add(t.id) : openDatasets.delete(t.id); });
  const list = el('div', {class: 'ds-items'});
  for (const d of [...ds].reverse()) list.append(el('div', {class: 'small ds', text: d.name}));   // newest first
  box.append(list);
  return el('td', {}, box);
}

function editRow(t) {
  const name = el('input', {type: 'text', value: t.name});
  if (t.name_locked) { name.disabled = true; name.title = 'A dataset folder has this name, so it is fixed.'; }
  const desc = el('input', {type: 'text', value: t.description || ''});
  const msg = el('div', {});
  const save = el('button', {type: 'button', class: 'small go'}, 'Save');
  save.onclick = async () => {
    const r = await api(`/api/registry/${t.id}`, {name: name.value, description: desc.value});
    if (!r.ok) return showProblem(msg, r);
    editing = null; refreshData();
  };
  const cancel = el('button', {type: 'button', class: 'small', onclick: () => { editing = null; renderTaskRows(); }}, 'Cancel');
  return el('tr', {class: 'editing'},
    el('td', {class: 'num', text: taskLabel(t)}),
    el('td', {}, name, t.name_locked ? el('div', {class: 'small muted', text: 'name fixed (a folder has this name)'}) : ''),
    el('td', {}, desc), datasetsCell(t),
    el('td', {class: 'actions'}, save, cancel, msg));
}
async function removeTask(t) {
  if (!confirm(`Remove task ${taskLabel(t)} "${t.name}" from the list?\n\nRecordings on disk are NOT deleted.`)) return;
  const r = await api(`/api/registry/${t.id}/delete`, {});
  if (!r.ok) return showProblem($('t-list-msg'), r);
  refreshData();
}

// ── shared progress rendering (sync, convert, push) ─────────────────────────
function renderBatch(prefix, job) {
  const box = $(prefix + '-job'), p = job.progress || {};
  setState(box, job.state);
  box.querySelector('.state-word').textContent = job.state === 'running' ? 'Working' : STATE_LABEL[job.state];
  const bar = $(prefix + '-bar');
  const known = typeof p.percent === 'number' && !p.pushing;
  bar.classList.toggle('indeterminate', job.state === 'running' && !known);
  bar.classList.toggle('failed', job.state === 'failed');
  bar.firstElementChild.style.width =
    job.state === 'done' ? '100%' : (known ? p.percent + '%' : (job.state === 'running' ? '' : '0'));
  let step = '';
  if (job.state === 'running') {
    step = p.pushing ? 'Uploading to Hugging Face…' : (known ? `${p.percent}%  ·  ${p.step}` : 'working…');
  } else if (job.state === 'done') {
    const what = job.meta.task || (job.meta.dataset || '').split('/').pop();
    step = '✔ ' + (what ? what + ': ' : '') + (p.step || 'finished');
  }
  else if (job.state === 'failed') step = '✖ ' + job.reason;
  if (job.errors.length && $(prefix + '-logbox').hidden) {
    step += `   ▲ ${job.errors.length} problem(s), press SHOW LOG to see them`;
  }
  $(prefix + '-step').textContent = step;
  const errs = $(prefix + '-errors');
  errs.innerHTML = '';
  if (job.errors.length) {
    errs.append(el('div', {class: 'notice error'}, el('p', {text: `${job.errors.length} problem(s):`}),
      el('pre', {class: 'errors', text: job.errors.slice(-30).join('\n')})));
  }
  $(prefix + '-stop').disabled = !job.alive;
  const stage = $(prefix + '-stage');
  if (stage) stage.textContent = job.alive ? (p.stage || '') : '';
  const pushed = $(prefix + '-pushed');
  if (pushed) {
    pushed.innerHTML = '';
    for (const url of p.pushed || []) {
      pushed.append(el('div', {class: 'notice ok'}, 'Uploaded to Hugging Face: ',
        el('a', {href: url, target: '_blank', rel: 'noopener', text: url})));
    }
  }
}

function batchInstruction(id, job, task, reason, words) {
  if (job.state === 'running') {
    instruct(id, '', `${words.busy}… wait until the bar is full.`);
  } else if (job.state === 'failed' && task && job.meta[words.key] === task) {
    instruct(id, 'bad', `${words.name} had a problem: ${job.reason}. Press SHOW LOG for details.`);
  } else if (job.state === 'done' && task && job.meta[words.key] === task) {
    instruct(id, 'good', `${words.name} finished${job.errors.length ? ' (with some problems, see below)' : ''}.`);
  } else if (reason) {
    instruct(id, '', reason);
  } else {
    instruct(id, 'good', `Ready. Press ${words.button}.`);
  }
}

// SHOW LOG / HIDE LOG buttons under each progress bar.
document.querySelectorAll('[data-logbox]').forEach(btn => {
  btn.onclick = () => {
    const box = $(btn.dataset.logbox);
    box.hidden = !box.hidden;
    btn.textContent = box.hidden ? 'Show log ▾' : 'Hide log ▴';
    if (!box.hidden) { const t = box.querySelector('.tail'); t.scrollTop = t.scrollHeight; }
    if (FUI.status) updateAll(FUI.status);
  };
});

function renderTail(id, src) {
  const t = $(id);
  const atBottom = t.scrollTop + t.clientHeight >= t.scrollHeight - 4;
  t.textContent = tail(src, 200);
  if (atBottom) t.scrollTop = t.scrollHeight;
}
document.addEventListener('fui:logs', () => {
  renderTail('s-tail', 'Sync'); renderTail('v-tail', 'Convert'); renderTail('p-tail', 'Push');
});

function setField(id, value) {
  $(id).value = value;
  $(id).dispatchEvent(new Event('input'));
}

// ── Synchronize ─────────────────────────────────────────────────────────────
const sValid = attachNameValidation($('s-task'), $('s-name-msg'), () => updateAll(FUI.status), 'Dataset name');

function updateSyncForm(s) {
  const task = $('s-task').value, info = TASKS[task];
  $('s-info').textContent = info ? `${info.raw} recorded episode(s) waiting, ${info.synced} already synced.` : '';
  let reason = '';
  if (!s) reason = 'Connecting…';
  else if (!s.scripts.sync) reason = 'convert_episodes.py not found in this checkout.';
  else if (s.jobs.collection.alive) reason = 'Wait until recording is finished.';
  else if (s.jobs.sync.alive) reason = 'Sync is running.';
  else if (!sValid()) reason = task ? 'Fix the dataset name first.' : 'Type or pick the dataset you recorded.';
  else if (!info) reason = `No dataset called "${task}".`;
  else if (!info.raw) reason = `"${task}" has nothing left to sync.`;
  $('s-start').disabled = !!reason;
  $('s-reason').textContent = reason;
  $('s-reason').classList.toggle('blocked', !!reason);
  if (!s) return;
  const job = s.jobs.sync;
  $('s-next-box').hidden = !(job.state === 'done' && job.meta.task);
  batchInstruction('s-instr', job, task, reason, {name: 'Sync', busy: 'Syncing', button: 'SYNC NOW', key: 'task'});
}

$('sync-form').addEventListener('submit', async ev => {
  ev.preventDefault();
  $('s-msg').innerHTML = '';
  if (!$('s-keep').checked &&
      !confirm('The raw video files for each synced episode will be DELETED. Continue?')) return;
  const res = await api('/api/sync/start', {
    task: $('s-task').value, parallel: $('s-parallel').value, keep_raw: $('s-keep').checked,
  });
  if (!res.ok) showProblem($('s-msg'), res);
});
$('s-stop').onclick = () => api('/api/sync/stop', {});
$('s-next').onclick = () => {
  goStep('#convert');
};

// ── Hugging Face login status ───────────────────────────────────────────────
async function loadHF() {
  HF = await api('/api/hf');
  const text = HF.user ? `Logged in to Hugging Face as ${HF.user}.`
    : 'Not logged in to Hugging Face. Run "huggingface-cli login" in a terminal once, then reload this page.';
  for (const id of ['v-hf-login', 'p-hf-login']) {
    $(id).textContent = text;
    $(id).classList.toggle('warn-text', !HF.user);
  }
  if (FUI.status) updateAll(FUI.status);
}
// The Hub name is not typed: <logged-in user>/<dataset name>, or where it was pushed before.
function hubName(task) {
  const info = TASKS[task];
  if (info && info.hub && info.hub.repo_id) return info.hub.repo_id;
  return HF && HF.user && task ? `${HF.user}/${task}` : '';
}

// ── Convert (+ optional push) ───────────────────────────────────────────────
const vValid = attachNameValidation($('v-task'), $('v-name-msg'), () => {
  updateAll(FUI.status);
}, 'Dataset name');
$('v-push').addEventListener('change', () => { $('v-push-fields').hidden = !$('v-push').checked; updateAll(FUI.status); });
$('v-space').addEventListener('change', () => updateAll(FUI.status));
// Calibration file for the robot-replay outputs: config/calib_<dataset>.json is
// picked by itself when it exists, until the operator chooses one.
let configPicked = false;
if ($('v-config')) $('v-config').addEventListener('change', () => { configPicked = true; });
function updateConfigPicker(task) {
  if (!$('v-config')) return;
  const replay = $('v-space').value !== 'pose';
  $('v-config-box').hidden = !replay;
  $('v-replay-note').hidden = !replay;
  if (configPicked) return;
  const own = [...$('v-config').options].find(o => o.value === `calib_${task}.json`);
  $('v-config').value = own ? own.value : $('v-config').options[0].value;
}

function copyWords(info) {
  return info.lerobot.length ? ' Already converted: ' + info.lerobot.map(l => l.name).join(', ') + '.' : '';
}

function updateConvertForm(s) {
  const task = $('v-task').value, info = TASKS[task];
  const push = $('v-push').checked, repo = hubName(task);
  $('v-info').textContent = info ? `${info.synced} synced episode(s).` + copyWords(info) : '';
  $('v-repo').textContent = repo || 'type the dataset name first';
  updateConfigPicker(task);
  let reason = '';
  if (!s) reason = 'Connecting…';
  else if (s.jobs.convert.alive) reason = 'Convert is running.';
  else if (s.jobs.sync.alive && s.jobs.sync.meta.task === task) reason = 'Wait until sync is finished.';
  else if (!vValid()) reason = task ? 'Fix the dataset name first.' : 'Type or pick the dataset you synced.';
  else if (!info) reason = `No dataset called "${task}".`;
  else if (!info.synced) reason = `"${task}" is not synced yet. Do Sync first.`;
  else if (push && ['cart-both', 'all'].includes($('v-space').value)) reason = 'Pushing needs an output type that makes one dataset (see Advanced).';
  else if (push && !repo) reason = 'Log in to Hugging Face first (see above).';
  $('v-start').disabled = !!reason;
  $('v-start').textContent = push ? 'Convert & push' : 'Convert';
  $('v-reason').textContent = reason;
  $('v-reason').classList.toggle('blocked', !!reason);
  if (!s) return;
  const job = s.jobs.convert;
  $('v-next-box').hidden = !(job.state === 'done' && job.meta.task);
  batchInstruction('v-instr', job, task, reason,
    {name: 'Convert', busy: job.progress.pushing ? 'Uploading' : 'Converting',
     button: push ? 'CONVERT & PUSH' : 'CONVERT', key: 'task'});
}

async function startConvert(overwrite) {
  $('v-msg').innerHTML = '';
  const push = $('v-push').checked ? {public: $('v-public').checked} : null;
  const res = await api('/api/convert/start', {
    task: $('v-task').value, space: $('v-space').value, fps: $('v-fps').value,
    episodes: $('v-episodes').value, with_video: $('v-video').checked, overwrite, push,
    config: $('v-config') ? $('v-config').value : null,
  });
  if (res.ok) return;
  if (res.exists && !overwrite) {
    if (confirm(res.error)) return startConvert(true);
    return;
  }
  showProblem($('v-msg'), res);
}
$('conv-form').addEventListener('submit', ev => { ev.preventDefault(); startConvert(false); });
$('v-stop').onclick = () => api('/api/convert/stop', {});

// ── Push (a converted dataset) ──────────────────────────────────────────────
const pValid = attachNameValidation($('p-dataset'), $('p-name-msg'), () => updateAll(FUI.status), 'Dataset name');

function updatePushForm(s) {
  const ds = $('p-dataset').value.trim(), info = TASKS[ds];
  const copy = info && (info.lerobot.find(l => l.name === ds + '/lerobot') || info.lerobot.find(l => l.name === ds + '_lerobot'));
  const repo = hubName(ds);
  $('p-info').textContent = copy
    ? `${copy.episodes ?? '?'} episode(s), ${copy.fps ?? '?'} fps · ${copy.path}` : '';
  $('p-repo-row').hidden = !(copy && repo);
  $('p-repo').textContent = repo;
  let reason = '';
  if (!s) reason = 'Connecting…';
  else if (!s.scripts.push) reason = 'No push script found.';
  else if (s.jobs.push.alive) reason = 'An upload is running.';
  else if (!ds) reason = 'Type the dataset name.';
  else if (!pValid()) reason = 'Fix the dataset name first.';
  else if (!info) reason = `No dataset called "${ds}".`;
  else if (!copy) reason = `"${ds}" has not been converted yet. Convert it first.`;
  else if (s.jobs.convert.alive && s.jobs.convert.meta.task === ds) reason = `"${ds}" is being converted. Wait for it to finish.`;
  else if (!repo) reason = 'Log in to Hugging Face first (see above).';
  $('p-start').disabled = !!reason;
  $('p-reason').textContent = reason;
  $('p-reason').classList.toggle('blocked', !!reason);
  if (s) {
    batchInstruction('p-instr', s.jobs.push, ds, reason,
      {name: 'Upload', busy: 'Uploading', button: 'PUSH TO HUB', key: 'task'});
  }
}
$('push-form').addEventListener('submit', async ev => {
  ev.preventDefault();
  $('p-msg').innerHTML = '';
  const res = await api('/api/push/start', {task: $('p-dataset').value.trim(), public: $('p-public').checked});
  if (!res.ok) showProblem($('p-msg'), res);
  refreshData();
});
$('p-stop').onclick = () => api('/api/push/stop', {});

// ── stop the machines once recording is over ────────────────────────────────
async function stopMachines() {
  if (!confirm('Stop ROS, the T265 and the camera?\n\nYou can start them again on the Machines page.')) return;
  const r = await api('/api/nodes/stop_all', {});
  if (!r.ok) alert(r.error);
}
$('m-on-stop').onclick = stopMachines;
$('c-stopnodes').onclick = stopMachines;

function renderMachinesOn(s) {
  const on = ['ros', 't265', 'gopro'].filter(k => s.jobs[k].alive);
  const tab = (location.hash || '#collect').slice(1);
  $('m-on').hidden = !on.length || tab === 'collect' || s.jobs.collection.alive;
  $('m-on-list').textContent = on.map(k => NODE_NAME[k]).join(', ');
  const busy = s.sequence_busy || on.some(k => s.jobs[k].state === 'stopping');
  $('m-on-stop').disabled = busy;
  $('c-stopnodes').hidden = !on.length;
  $('c-stopnodes').disabled = busy;
}

// ── status fan-out ──────────────────────────────────────────────────────────
let refreshedFrom = null;
function updateAll(s) {
  if (!s) return;
  renderCollection(s);
  renderMachinesOn(s);
  renderCamera(s);
  renderBatch('s', s.jobs.sync);
  renderBatch('v', s.jobs.convert);
  renderBatch('p', s.jobs.push);
  updateCollectForm(s); updateSyncForm(s); updateConvertForm(s); updatePushForm(s);
}
document.addEventListener('fui:status', e => {
  const s = e.detail, sess = s.session;
  // After a session, reload the dataset list (once per session).
  if (sess.task && !s.jobs.collection.alive && refreshedFrom !== sessionKey(s)) {
    refreshedFrom = sessionKey(s);
    refreshData();
  }
  updateAll(s);
});

['s-task', 'v-task', 'p-dataset'].forEach(id => prefillRecorded($(id)));
refreshData();
loadHF();
setInterval(refreshData, 5000);

// ── T265 home check ─────────────────────────────────────────────────────────
// Live. Home = where the T265 started. In the home holder it must read ~0;
// anything else is drift. A jump (the T265 snapping its estimate) turns the
// panel red at once: a take can end "at home" only because of such a snap.
const HOME_WORD = {ok: 'At home', warn: 'Near home', bad: 'Off home', jump: 'T265 jumped',
                   stale: 'T265 not sending'};
const cm = mm => (mm / 10).toFixed(1);

function renderHome(r) {
  const off = !r.ok || !r.available;
  $('h-chip').className = 'home-chip ' + (off ? 'off' : r.level);
  if (off) {
    $('h-word').textContent = 'T265 home';
    $('h-hint').textContent = r.reason || r.error || '';
    ['h-dist', 'h-rot', 'h-xyz', 'h-jumps'].forEach(id => { $(id).textContent = '—'; });
    $('h-meta').textContent = '';
    return;
  }
  const jump = r.last_jump;
  $('h-word').textContent = {
    warn: HOME_WORD.warn + ': ' + cm(r.dist_mm) + ' cm, ' + r.rot_deg.toFixed(1) + '°',
    bad: HOME_WORD.bad + ': ' + cm(r.dist_mm) + ' cm, ' + r.rot_deg.toFixed(1) + '°',
    jump: HOME_WORD.jump + ' ' + cm(jump ? jump.mm : 0) + ' cm',
  }[r.level] || HOME_WORD[r.level];
  $('h-hint').textContent = {
    ok: 'Within ' + cm(r.ok_mm) + ' cm and ' + r.ok_deg + '° of where the T265 started.',
    warn: 'Put the UMI in the home holder. If it still reads this in the holder, the T265 has drifted.',
    bad: 'In the holder and still off? The T265 has drifted: restart it with the UMI in the holder.',
    jump: 'The T265 position jumped (no hand moves that fast). If a take is running, redo it (R).',
    stale: 'No pose for ' + r.age_s.toFixed(1) + ' s. Check the T265 on the Machines page.',
  }[r.level];
  $('h-dist').textContent = cm(r.dist_mm) + ' cm';
  $('h-rot').textContent = r.rot_deg.toFixed(1) + '°';
  $('h-xyz').textContent = [r.x_mm, r.y_mm, r.z_mm].map(cm).join(' / ');
  $('h-jumps').textContent = r.jumps + (jump ? ' (last ' + cm(jump.mm) + ' cm, ' + Math.round(jump.ago_s) + ' s ago)' : '');
  $('h-meta').textContent = 'Home = where the T265 started (' + new Date(r.home_at * 1000).toLocaleTimeString() + ')' +
    (r.restarts ? '. The T265 stream restarted ' + r.restarts + '×, home was set again: only right if the UMI was in the holder' : '') +
    (r.home_approx ? '. Rotation home is approximate (the T265 was already running when the check started)' : '') +
    '. Jump = more than ' + cm(r.jump_mm) + ' cm between two poses (5 ms).';
}

async function pollHome() {
  const s = FUI.status;
  const show = !!(s && s.jobs.t265.alive);
  $('h-panel').hidden = !show;
  if (show) renderHome(await api('/api/home'));
  setTimeout(pollHome, 300);
}
pollHome();
