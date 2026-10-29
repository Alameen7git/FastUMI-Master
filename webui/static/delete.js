// Delete page: remove bad episodes from a dataset. The server then converts the
// LeRobot copy again and, if the dataset was on Hugging Face, pushes it again.
'use strict';

const $ = id => document.getElementById(id);
let TASKS = {};
let EPS = null;      // {task, episodes: [n…], deleted: [...], lerobot, pushed_to}
let watching = null; // dataset whose update the progress panel follows
const valid = attachNameValidation($('del-task'), $('del-name-msg'), () => { EPS = null; loadEpisodes(); }, 'Dataset name');

function typed() { return $('del-task').value; }

function instruct(kind, text) {
  $('del-instr').className = 'instruction' + (kind ? ' ' + kind : '');
  $('del-instr').querySelector('.txt').textContent = text;
}

function ranges(nums) {   // [1,2,3,5] -> "1–3, 5"
  const out = [];
  for (let i = 0; i < nums.length; i++) {
    let j = i;
    while (j + 1 < nums.length && nums[j + 1] === nums[j] + 1) j++;
    out.push(j > i ? `${nums[i]}–${nums[j]}` : String(nums[i]));
    i = j;
  }
  return out.join(', ');
}

async function refresh() {
  const r = await api('/api/tasks');
  TASKS = r.tasks || {};
  if (!EPS && typed()) loadEpisodes();   // a name typed in before the list arrived
  else render();
}

async function loadEpisodes() {
  const task = typed();
  if (!task || !TASKS[task] || !valid()) { EPS = null; render(); return; }
  if (EPS && EPS.task === task) { render(); return; }
  const r = await api('/api/episodes?task=' + encodeURIComponent(task));
  if (typed() === task) EPS = r.ok ? r : null;
  render();
}

function afterWords() {
  if (EPS.pushed_to) return `After deleting, the LeRobot copy is converted again and ${EPS.pushed_to} on Hugging Face is updated automatically.`;
  if (EPS.lerobot) return 'After deleting, the LeRobot copy is converted again automatically (it is not on Hugging Face yet).';
  return 'This dataset has not been converted yet: the next Convert leaves these episodes out.';
}

function render() {
  const s = FUI.status, task = typed(), ready = !!(EPS && EPS.task === task);
  let reason = '';
  if (!s) reason = 'Connecting…';
  else if (!task) reason = 'Type the dataset name.';
  else if (!valid()) reason = 'Fix the dataset name first.';
  else if (!TASKS[task]) reason = `No dataset called "${task}".`;
  else if (!ready) reason = 'Loading…';
  else if (!EPS.episodes.length) reason = 'This dataset has no episodes left.';
  else if (!$('del-eps').value.trim()) reason = 'Type the episode numbers to delete.';
  $('del-go').disabled = !!reason;

  $('del-have').textContent = ready
    ? (EPS.episodes.length ? `Episodes in this dataset: ${ranges(EPS.episodes)}` : 'This dataset has no episodes left.') : '';
  $('del-after').textContent = ready ? afterWords() : '';
  const log = $('del-log');
  log.innerHTML = '';
  const deleted = ready ? EPS.deleted : [];
  for (const e of deleted.slice().reverse()) {
    log.append(el('tr', {}, el('td', {text: String(e.episode)}), el('td', {text: e.deleted_at || ''}), el('td', {text: e.reason || '—'})));
  }
  $('del-log-box').hidden = !deleted.length;
  $('del-log-n').textContent = String(deleted.length);

  const u = s ? renderUpdate(s) : {state: 'stopped'};
  if (deleting) instruct('', 'Deleting… wait until the bar is full.');
  else if (u.state === 'running') instruct('', 'Updating the LeRobot copy' + (u.push ? ' and Hugging Face' : '') + '… wait until the bar is full (you can leave this page).');
  else if (u.state === 'failed') instruct('bad', 'The update had a problem: ' + u.reason + '. Press SHOW LOG for details.');
  else if (u.state === 'done' && u.mine) instruct('good', u.push ? 'Done: the LeRobot copy and Hugging Face are up to date.' : 'Done: the LeRobot copy is up to date.');
  else if (reason) instruct('', reason);
  else instruct('good', 'Ready. Press 🗑 DELETE EPISODES.');
}

// Progress: deleting, then the automatic convert (+ push), like the Convert tab.
let deleting = null;   // {task, n} while the delete request is running
let lastDelete = null; // {task, n, update} after it answered
function renderUpdate(s) {
  const job = s.jobs.convert, p = job.progress || {};
  const pending = !!watching && s.updates_pending.includes(watching);
  const mine = !!watching && job.meta.auto && job.meta.task === watching
    && (job.started_at || 0) * 1000 >= (lastDelete ? lastDelete.at - 5000 : 0);
  // The update was asked for but the status (1 s poll) does not show it yet.
  const starting = !!lastDelete && lastDelete.update && lastDelete.task === watching && !pending && !mine;
  let state = 'stopped', stage = '', step = '', percent = null;
  if (deleting) {
    state = 'running'; stage = `Deleting ${deleting.n} episode(s) from "${deleting.task}"`; step = 'deleting…';
  } else if (pending && !(mine && job.alive)) {
    state = 'running'; stage = 'Waiting for Convert / Push to finish another dataset'; step = 'waiting…';
  } else if (starting) {
    state = 'running'; stage = 'Starting the update'; step = 'starting…';
  } else if (mine) {
    state = job.state;
    const known = typeof p.percent === 'number' && !p.pushing;
    if (job.state === 'running' && !job.alive) {
      step = 'finishing…';
    } else if (job.alive) {
      stage = p.pushing ? `Uploading to Hugging Face (${job.meta.push.repo_id})`
        : `Converting "${watching}" again` + (job.meta.push ? ', then uploading to Hugging Face' : '');
      step = p.pushing ? 'Uploading to Hugging Face…' : (known ? `${p.percent}%  ·  ${p.step}` : 'working…');
      if (known) percent = p.percent;
    } else if (job.state === 'done') {
      step = '✔ Deleted, converted again' + (job.meta.push ? ' and updated on Hugging Face' : '') + ` (${p.step || 'done'})`;
    } else if (job.state === 'failed') {
      step = '✖ ' + job.reason;
    } else if (job.state === 'stopped') {
      step = 'Stopped.';
    }
  } else if (lastDelete && lastDelete.task === typed() && !lastDelete.update) {
    state = 'done'; step = `✔ Deleted ${lastDelete.n} episode(s). Nothing to update (not converted yet).`;
  }
  const box = $('del-job');
  setState(box, state);
  box.querySelector('.state-word').textContent = state === 'running' ? 'Working' : STATE_LABEL[state];
  const bar = $('del-bar');
  bar.classList.toggle('indeterminate', state === 'running' && percent === null);
  bar.classList.toggle('failed', state === 'failed');
  bar.firstElementChild.style.width = state === 'done' ? '100%' : (percent !== null ? percent + '%' : (state === 'running' ? '' : '0'));
  $('del-stage').textContent = stage;
  const errs = mine ? job.errors : [];
  if (errs.length && $('del-logbox').hidden) step += `   ▲ ${errs.length} problem(s), press SHOW LOG to see them`;
  $('del-step').textContent = step;
  $('del-errors').innerHTML = '';
  if (errs.length) {
    $('del-errors').append(el('div', {class: 'notice error'}, el('p', {text: `${errs.length} problem(s):`}),
      el('pre', {class: 'errors', text: errs.slice(-30).join('\n')})));
  }
  const pushed = $('del-pushed');
  pushed.innerHTML = '';
  for (const url of (mine && p.pushed) || []) {
    pushed.append(el('div', {class: 'notice ok'}, 'Updated on Hugging Face: ',
      el('a', {href: url, target: '_blank', rel: 'noopener', text: url})));
  }
  return {state, mine, push: mine ? job.meta.push : null, reason: job.reason};
}

$('del-logbtn').onclick = () => {
  const box = $('del-logbox');
  box.hidden = !box.hidden;
  $('del-logbtn').textContent = box.hidden ? 'Show log ▾' : 'Hide log ▴';
  if (!box.hidden) { const t = $('del-tail'); t.textContent = tail('Convert', 200); t.scrollTop = t.scrollHeight; }
  render();
};
document.addEventListener('fui:logs', () => {
  const t = $('del-tail');
  if ($('del-logbox').hidden) return;
  const atBottom = t.scrollTop + t.clientHeight >= t.scrollHeight - 4;
  t.textContent = tail('Convert', 200);
  if (atBottom) t.scrollTop = t.scrollHeight;
});

$('del-eps').addEventListener('input', render);
$('del-form').addEventListener('submit', async ev => {
  ev.preventDefault();
  if ($('del-go').disabled) return;
  $('del-msg').innerHTML = '';
  const task = EPS.task, text = $('del-eps').value;
  if (!confirm(`Delete episode(s) ${text.trim()} from "${task}"?\n\nThe raw recording and the synced file are removed for good.\n` +
               afterWords())) return;
  $('del-go').disabled = true;
  deleting = {task, n: text.trim()};
  watching = null;
  render();
  const r = await api('/api/episodes/delete', {task, episodes: text, reason: $('del-reason').value});
  deleting = null;
  if (!r.ok) { render(); return showProblem($('del-msg'), r); }
  lastDelete = {task, n: r.deleted.length, update: r.update.lerobot, at: Date.now()};
  EPS = Object.assign({}, EPS, {episodes: r.episodes, deleted: r.log});
  $('del-eps').value = ''; $('del-reason').value = '';
  const up = r.update;
  let words = `Deleted episode(s) ${r.deleted.join(', ')} from "${task}".`;
  if (up.repo_id) words += ` Converting again and updating ${up.repo_id} on Hugging Face (see Progress below).`;
  else if (up.lerobot) words += ' Converting the LeRobot copy again (see Progress below).';
  else words += ' The numbers are saved: the next Convert + Push leaves them out and lists them on Hugging Face.';
  const msg = el('div', {class: 'notice ok'}, words);
  if (up.note) msg.append(el('div', {}, up.note));
  if (r.stale_lerobot.length) {
    msg.append(el('div', {}, `Not updated: ${r.stale_lerobot.join(', ')} still has them (convert it again on the Convert tab).`));
  }
  $('del-msg').append(msg);
  if (up.lerobot) watching = task;
  refresh();
});
document.addEventListener('fui:status', render);

// Follow an update that is already running (e.g. after reopening this page).
document.addEventListener('fui:status', e => {
  const job = e.detail.jobs.convert;
  if (!watching && job.alive && job.meta.auto) watching = job.meta.task;
}, {once: true});

prefillRecorded($('del-task'));
refresh();
setInterval(refresh, 10000);
