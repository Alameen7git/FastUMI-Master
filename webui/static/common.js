// Shared by every page: status polling, the debug console, task-name
// validation and the end-of-session prompt.
'use strict';

const FUI = {
  status: null,
  logs: {},            // source -> [{seq,t,src,line}], capped per source
  PER_SOURCE: 2000,
  SOURCES: ['ROS', 'T265', 'GoPro', 'Collection', 'Preview', 'Sync', 'Convert', 'Push', 'Viz', 'UI'],
};

async function api(path, body) {
  const opts = body === undefined ? {} : {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body || {}),
  };
  try {
    const r = await fetch(path, opts);
    return await r.json();
  } catch (e) {
    return {ok: false, error: 'UI server not reachable (' + e.message + ')'};
  }
}

function el(tag, attrs, ...children) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (k === 'class') n.className = v;
    else if (k === 'text') n.textContent = v;
    else if (k.startsWith('on')) n.addEventListener(k.slice(2), v);
    else n.setAttribute(k, v);
  }
  for (const c of children) n.append(c);
  return n;
}

function setState(node, state) {
  node.className = node.className.replace(/\bst-\w+/g, '').trim() + ' st-' + state;
}

// Plain words for the operator.
const STATE_LABEL = {
  stopped: 'Off', starting: 'Starting…', running: 'Ready', stopping: 'Stopping…',
  done: 'Done', failed: 'Problem',
};

// One light for a group of jobs: red beats amber beats green beats off.
function worstState(states) {
  if (states.includes('failed')) return 'failed';
  if (states.some(s => s === 'starting' || s === 'stopping')) return 'starting';
  if (states.every(s => s === 'running')) return 'running';
  if (states.includes('running')) return 'starting';   // only some are up
  return 'stopped';
}

function currentStep() {
  if (location.pathname === '/visualize') return 'viz';
  if (location.pathname === '/delete') return 'delete';
  if (location.pathname === '/collect') return (location.hash || '#collect').slice(1);
  return 'machines';
}

// Step lights: green = on (machines up, recorder or viewer running),
// amber = working (sync / convert / push in progress), red = problem.
// A finished job goes back to off; its page still shows the result.
function stepState(job, batch) {
  if (job.state === 'done') return 'stopped';
  if (batch && job.state === 'running') return 'starting';
  return job.state;
}

function renderSteps(s) {
  const j = s ? s.jobs : null;
  const cur = currentStep();
  document.querySelectorAll('[data-step]').forEach(a => {
    a.classList.toggle('active', a.dataset.step === cur);
    if (!j) return;
    const st = {
      machines: worstState(['ros', 't265', 'gopro'].map(k => j[k].state)),
      collect: stepState(j.collection), sync: stepState(j.sync, true), convert: stepState(j.convert, true),
      push: stepState(j.push, true), viz: stepState(j.viz),
      delete: j.convert.meta.auto ? stepState(j.convert, true) : (s.updates_pending.length ? 'starting' : 'stopped'),
    }[a.dataset.step];
    setState(a, st);
  });
}
window.addEventListener('hashchange', () => renderSteps(FUI.status));

// ── status polling ──────────────────────────────────────────────────────────
async function pollStatus() {
  const s = await api('/api/status');
  if (s && s.jobs) {
    FUI.status = s;
    renderSteps(s);
    renderSessionBanner(s.session);
    // The header (logo + steps) is not needed while recording: hide it.
    document.body.classList.toggle('recording', !!s.jobs.collection.alive);
    document.dispatchEvent(new CustomEvent('fui:status', {detail: s}));
  }
  setTimeout(pollStatus, 1000);
}

// ── debug console ───────────────────────────────────────────────────────────
let logSeq = 0;
const hidden = new Set();
const MAX_DOM = 8000;

function fmtTime(t) {
  const d = new Date(t * 1000);
  return d.toTimeString().slice(0, 8) + '.' + String(d.getMilliseconds()).padStart(3, '0');
}

function isErrLine(line) {
  return /Traceback|Error|ERROR|FAILED|⚠|NOT READY|UNHEALTHY|not started|aborted/.test(line);
}

function initConsole() {
  const filters = document.getElementById('console-filters');
  for (const src of FUI.SOURCES) {
    const cb = el('input', {type: 'checkbox', checked: ''});
    cb.addEventListener('change', () => {
      cb.checked ? hidden.delete(src) : hidden.add(src);
      document.querySelectorAll(`#console-body .ln[data-src="${src}"]`)
        .forEach(n => { n.style.display = cb.checked ? '' : 'none'; });
    });
    filters.append(el('label', {class: 'src-' + src}, cb, src), ' ');
  }
  document.getElementById('console-clear').addEventListener('click', () => {
    document.getElementById('console-body').innerHTML = '';
  });
  let open = false;
  try { open = localStorage.getItem('fui-log-open') === '1'; } catch (e) { /* storage blocked */ }
  setLogOpen(open);
  document.getElementById('log-toggle').addEventListener('click', () => setLogOpen(!logOpen));
  document.getElementById('console-close').addEventListener('click', () => setLogOpen(false));
  document.addEventListener('keydown', ev => { if (ev.key === 'Escape' && logOpen) setLogOpen(false); });
}

// ── log drawer ──────────────────────────────────────────────────────────────
let logOpen = false, unseenErrors = 0;
function setLogOpen(open) {
  logOpen = open;
  document.body.classList.toggle('log-open', open);
  document.getElementById('log-toggle').setAttribute('aria-expanded', String(open));
  try { localStorage.setItem('fui-log-open', open ? '1' : '0'); } catch (e) { /* storage blocked */ }
  if (open) {
    unseenErrors = 0;
    const body = document.getElementById('console-body');
    body.scrollTop = body.scrollHeight;
  }
  renderErrorCount();
}
function renderErrorCount() {
  const c = document.getElementById('log-count');
  c.hidden = logOpen || unseenErrors === 0;
  c.textContent = unseenErrors > 99 ? '99+' : String(unseenErrors);
}

async function pollLogs() {
  const r = await api('/api/logs?since=' + logSeq);
  if (r && r.rows) {
    logSeq = r.last;
    const body = document.getElementById('console-body');
    const auto = document.getElementById('console-autoscroll').checked;
    const frag = document.createDocumentFragment();
    for (const row of r.rows) {
      const buf = FUI.logs[row.src] || (FUI.logs[row.src] = []);
      buf.push(row);
      if (buf.length > FUI.PER_SOURCE) buf.shift();
      const ln = el('div', {class: 'ln' + (isErrLine(row.line) ? ' err' : ''), 'data-src': row.src},
        el('span', {class: 'ts', text: fmtTime(row.t) + ' '}),
        el('span', {class: 'src src-' + row.src, text: '[' + row.src + '] '}),
        row.line);
      if (hidden.has(row.src)) ln.style.display = 'none';
      if (!logOpen && isErrLine(row.line)) unseenErrors++;
      frag.append(ln);
    }
    body.append(frag);
    while (body.childElementCount > MAX_DOM) body.firstElementChild.remove();
    if (auto && r.rows.length) body.scrollTop = body.scrollHeight;
    renderErrorCount();
    if (r.rows.length) document.dispatchEvent(new CustomEvent('fui:logs', {detail: r.rows}));
  }
  setTimeout(pollLogs, 700);
}

// Last n lines of one source, for per-step log tails.
function tail(src, n) {
  return (FUI.logs[src] || []).slice(-n).map(r => r.line).join('\n');
}

// ── task-name validation (mirrors validate_name() in app.py) ────────────────
function nameProblem(raw, what = 'Task name') {
  if (!raw.trim()) return {error: `${what} is required.`, suggestion: null};
  if (/\s/.test(raw)) {
    return {error: `${what} can't contain spaces — use underscores instead.`,
            suggestion: raw.trim().replace(/\s+/g, '_'), typed: raw};
  }
  if (!/^[A-Za-z0-9_.\-]+$/.test(raw) || raw.startsWith('.')) {
    return {error: 'Use only letters, digits, underscores, dashes and dots.',
            suggestion: raw.replace(/[^A-Za-z0-9_.\-]+/g, '_').replace(/^\.+/, ''), typed: raw};
  }
  return null;
}

// Wires an <input> to a message box. Returns () => true when the name is valid.
function attachNameValidation(input, box, onChange, what) {
  function render(notify) {
    const p = nameProblem(input.value, what);
    box.innerHTML = '';
    input.classList.toggle('invalid', !!p && input.value !== '');
    if (p && input.value !== '') {
      const n = el('div', {class: 'notice warn'}, el('p', {text: '⚠️ ' + p.error}));
      if (p.typed !== undefined) {
        n.append(el('p', {}, 'You typed: ', el('code', {text: p.typed})));
      }
      if (p.suggestion) {
        n.append(el('p', {}, 'Did you mean: ', el('code', {text: p.suggestion}), '? ',
          el('button', {type: 'button', onclick: () => {
            input.value = p.suggestion; render(); input.focus();
          }}, 'Use this instead')));
      }
      box.append(n);
    }
    if (onChange && notify !== false) onChange();
  }
  input.addEventListener('input', render);
  render(false);  // caller may not have stored the returned validator yet
  return () => nameProblem(input.value, what) === null;
}

// After a recording, the dataset it made is typed in for the operator (once per
// new recording). A name the operator typed or cleared is left alone.
function prefillRecorded(input) {
  let seen = null, filled = null;
  document.addEventListener('fui:status', e => {
    const name = e.detail.last_recorded;
    if (!name || name === seen) return;
    seen = name;
    if (input.value !== '' && input.value !== filled) return;
    input.value = filled = name;
    input.dispatchEvent(new Event('input'));
  });
}

// Shows a server-side validation failure (same shape) in a box.
function showProblem(box, res) {
  box.innerHTML = '';
  box.append(el('div', {class: 'notice error'}, el('p', {text: res.error || 'Request failed.'})));
}

// ── end-of-session prompt ───────────────────────────────────────────────────
function renderSessionBanner(sess) {
  const b = document.getElementById('session-banner');
  b.classList.toggle('show', !!(sess && sess.ended));
  if (sess && sess.ended) {
    let d = `Task "${sess.task}": ${sess.saved} of ${sess.total} episode(s) saved.`;
    if (sess.exit_code) d += ` data_collection.py exited with code ${sess.exit_code} — see the console.`;
    document.getElementById('session-detail').textContent = d;
  }
}

function initSessionBanner() {
  document.getElementById('session-more').addEventListener('click', async () => {
    await api('/api/session/dismiss', {});
    if (location.pathname !== '/collect') location.href = '/collect#collect';
    else document.dispatchEvent(new CustomEvent('fui:record-more'));
  });
  document.getElementById('session-stop').addEventListener('click', async () => {
    const r = await api('/api/session/stop_nodes', {});
    if (!r.ok) alert(r.error);
  });
}


// ── light / dark theme ──────────────────────────────────────────────────────
function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  const btn = document.getElementById('theme-toggle');
  btn.querySelector('.tl').textContent = theme === 'dark' ? 'Light mode' : 'Dark mode';
  btn.setAttribute('aria-pressed', String(theme === 'dark'));
  try { localStorage.setItem('fui-theme', theme); } catch (e) { /* storage blocked */ }
}
applyTheme(document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light');
document.getElementById('theme-toggle').addEventListener('click', () => {
  applyTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark');
});

initConsole();
initSessionBanner();
pollStatus();
pollLogs();
