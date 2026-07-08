#!/usr/bin/env python3
"""FastUMI web ops UI -- one local page that wraps the whole data-collection
workflow (start/stop the roscore+T265+camera+data_collection stack, run the
episode worker, browse results) so an operator doesn't need a terminal.

This is a supervision layer on top of the existing scripts, not a rewrite of
them: it launches the same commands start_collection.sh/stop_collection.sh
already use as managed background processes instead of terminal windows, and
otherwise only ever *reads* the existing manifest.json/error.log/HDF5 schema
that data_collection.py and episode_worker.py already produce. None of the
claim-lock/atomic-write/finalize logic in those scripts is touched here.

Usage:
    python3 ops_server.py [--port 8000]

Then open http://<this-machine-ip>:<port> from any device on the same network
(no auth -- trusted-network use only).
"""
import argparse
import json
import os
import queue
import shlex
import signal
import subprocess
import sys
import threading
import time
from http.server import ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import psutil

import episode_manifest as em
import visualize_dataset as viz

with open('config/config.json', 'r') as f:
    config = json.load(f)

cfg = config['task_config']
topic_health_cfg = cfg.get('topic_health', {})

REPO_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT   = config['device_settings']['data_dir']
LOG_DIR     = os.path.join(REPO_DIR, 'logs')
ROS_SETUP   = '/opt/ros/noetic/setup.bash'
CONDA_SETUP = '/home/nuc8/miniconda3/etc/profile.d/conda.sh'
CONDA_ENV   = 'FastUMI'

HEALTH_PATH = os.path.join(DATA_ROOT, '.preview', 'topic_health.json')
STALE_SEC = {
    'video_topic': topic_health_cfg.get('video_stale_sec', 1.5),
    'trajectory_topic': topic_health_cfg.get('trajectory_stale_sec', 1.0),
}
# If preview_bridge.py's own snapshot write itself goes stale (it died, or was
# never started), we can no longer vouch for either topic -- don't silently
# report green just because we have no fresher data to contradict it.
HEALTH_FILE_STALE_SEC = max(STALE_SEC.values()) + 3 * topic_health_cfg.get('flush_interval_sec', 0.5)

os.makedirs(LOG_DIR, exist_ok=True)


def _topic_healthy(topic_key):
    """True if preview_bridge.py's snapshot shows a message on topic_key within
    its configured staleness threshold. False (not an exception) if the
    snapshot file is missing, unparseable, or itself too old to trust."""
    try:
        with open(HEALTH_PATH) as f:
            snapshot = json.load(f)
        mtime = os.path.getmtime(HEALTH_PATH)
    except (OSError, json.JSONDecodeError):
        return False
    if time.time() - mtime > HEALTH_FILE_STALE_SEC:
        return False
    last_seen = snapshot.get(topic_key, 0.0)
    return (time.time() - last_seen) <= STALE_SEC[topic_key]


# ── psutil-based liveness check (same cmdline-substring approach as monitor_cpu.py) ──

def _process_alive(match_substrings):
    for proc in psutil.process_iter(['cmdline']):
        try:
            cmd = ' '.join(proc.info['cmdline'] or []).lower()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if any(s in cmd for s in match_substrings):
            return True
    return False


# ── Managed background processes ──────────────────────────────────────────────

class ManagedProcess:
    """Wraps one background process (roscore, a roslaunch, or a python node).

    Status is derived two ways: our own Popen handle (authoritative for
    processes we started -- lets us tell "exited on its own" apart from "we
    stopped it"), cross-checked against a psutil cmdline match so a process
    started outside this UI (e.g. from start_collection.sh) still shows up as
    running rather than "stopped" just because we don't hold its handle.

    If topic_key is given, "running" additionally requires that topic to have
    published recently (per _topic_healthy) -- a process can stay alive while
    its topic silently stops publishing (camera unplugged, T265 lost
    tracking), which is a distinct, worse state than just "not running" and
    is reported as 'stalled' rather than collapsed into either 'running' or
    'error'.
    """

    def __init__(self, name, build_cmd, match_substrings, stdin_pipe=False, topic_key=None):
        self.name = name
        self.build_cmd = build_cmd
        self.match_substrings = match_substrings
        self.stdin_pipe = stdin_pipe
        self.topic_key = topic_key
        self.proc = None
        self.expected_running = False
        self.log_path = os.path.join(LOG_DIR, f'{name}.log')
        self._lock = threading.Lock()

    def start(self, **kwargs):
        with self._lock:
            cmd = self.build_cmd(**kwargs)
            log_f = open(self.log_path, 'a')
            log_f.write(f'\n--- launched {time.strftime("%Y-%m-%d %H:%M:%S")} ---\n{cmd}\n\n')
            log_f.flush()
            self.proc = subprocess.Popen(
                ['bash', '-c', cmd],
                stdout=log_f, stderr=subprocess.STDOUT,
                stdin=subprocess.PIPE if self.stdin_pipe else subprocess.DEVNULL,
                cwd=REPO_DIR, start_new_session=True, bufsize=0,
            )
            self.expected_running = True
            return self.proc

    def write_control_byte(self, data):
        """Only meaningful for stdin_pipe processes (data_collection.py)."""
        with self._lock:
            if self.proc is None or self.proc.stdin is None or self.proc.poll() is not None:
                return False
            try:
                self.proc.stdin.write(data)
                self.proc.stdin.flush()
                return True
            except (BrokenPipeError, OSError):
                return False

    def stop(self, timeout=15):
        """SIGTERM then wait; SIGKILL as a backstop, matching stop_collection.sh."""
        with self._lock:
            self.expected_running = False
            proc = self.proc
        if proc is None or proc.poll() is not None:
            return True
        try:
            proc.terminate()
        except ProcessLookupError:
            return True
        try:
            proc.wait(timeout=timeout)
            return True
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
                proc.wait(timeout=5)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                pass
            return False

    def status(self):
        proc = self.proc
        if proc is not None:
            rc = proc.poll()
            if rc is None:
                return self._maybe_stalled('running')
            if self.expected_running:
                return 'error'  # exited on its own before we asked it to stop
        alive = _process_alive(self.match_substrings)
        return self._maybe_stalled('running') if alive else 'stopped'

    def _maybe_stalled(self, running_status):
        if self.topic_key is not None and not _topic_healthy(self.topic_key):
            return 'stalled'
        return running_status

    def log_tail(self, n=30):
        if not os.path.isfile(self.log_path):
            return ''
        with open(self.log_path, 'rb') as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 65536))
            lines = f.read().decode(errors='replace').splitlines()
        return '\n'.join(lines[-n:])


def _roscore_cmd(**_):
    return f'source {ROS_SETUP} && exec roscore'


def _t265_cmd(**_):
    return f'source {ROS_SETUP} && exec roslaunch realsense2_camera rs_t265.launch'


def _camera_cmd(**_):
    return (f'source {CONDA_SETUP} && conda activate {CONDA_ENV} && source {ROS_SETUP} '
            f'&& cd {shlex.quote(REPO_DIR)} && exec python3 cam_capture_node.py')


def _preview_cmd(**_):
    return (f'source {CONDA_SETUP} && conda activate {CONDA_ENV} && source {ROS_SETUP} '
            f'&& cd {shlex.quote(REPO_DIR)} && exec python3 preview_bridge.py')


def _data_collection_cmd(task, **_):
    # A large num_episodes -- the operator (not a fixed count) decides when the
    # session ends, via Stop Collection -> SIGTERM, the same graceful path
    # stop_collection.sh already uses (sigterm_handler treats it like 'E').
    return (f'source {CONDA_SETUP} && conda activate {CONDA_ENV} && source {ROS_SETUP} '
            f'&& cd {shlex.quote(REPO_DIR)} && exec python3 data_collection.py '
            f'--task {shlex.quote(task)} --num_episodes 100000')


PROCESSES = {
    'roscore':         ManagedProcess('roscore', _roscore_cmd, ['roscore']),
    'rs_t265':         ManagedProcess('rs_t265', _t265_cmd, ['rs_t265.launch', 'realsense2_camera'],
                                       topic_key='trajectory_topic'),
    'camera':          ManagedProcess('camera', _camera_cmd, ['cam_capture_node.py'],
                                       topic_key='video_topic'),
    'preview':         ManagedProcess('preview', _preview_cmd, ['preview_bridge.py']),
    'data_collection': ManagedProcess('data_collection', _data_collection_cmd,
                                       ['data_collection.py'], stdin_pipe=True),
}
START_ORDER = ['roscore', 'rs_t265', 'camera', 'preview', 'data_collection']
STAGE_WAIT_SEC = {'roscore': 2.0, 'rs_t265': 3.0, 'camera': 1.5, 'preview': 1.0, 'data_collection': 0}


# ── Collection session state ──────────────────────────────────────────────────

class CollectionSession:
    """Tracks the operator-facing session on top of the ManagedProcess layer:
    which task is active, whether an episode is believed to be mid-recording
    (self-tracked from button clicks, exactly mirroring the same "trust the
    REC indicator" UX a human at a real keyboard already relies on), and the
    running count of episodes seen this session -- read-only off the same
    raw/episode_*/manifest.json files episode_worker.py already trusts."""

    def __init__(self):
        self.lock = threading.Lock()
        self.task = None
        self.recording = False
        self.starting = False
        self.stopping = False
        self.start_log = []
        self.seen_episodes = []  # [(episode_index, first_seen_ts), ...] this session
        self._seen_set = set()
        self._poll_thread = None
        self._poll_stop = threading.Event()

    def _poll_loop(self):
        while not self._poll_stop.is_set():
            task = self.task
            if task:
                raw_dir = os.path.join(DATA_ROOT, task, 'raw')
                if os.path.isdir(raw_dir):
                    for name in sorted(os.listdir(raw_dir)):
                        if not name.startswith('episode_'):
                            continue
                        manifest_path = os.path.join(raw_dir, name, em.MANIFEST_NAME)
                        if not os.path.isfile(manifest_path):
                            continue
                        try:
                            idx = int(name[len('episode_'):])
                        except ValueError:
                            continue
                        if idx not in self._seen_set:
                            self._seen_set.add(idx)
                            self.seen_episodes.append((idx, time.time()))
                            with self.lock:
                                self.recording = False  # a manifest appearing means a stop+finalize happened
            time.sleep(1.0)

    def start_polling(self, task):
        with self.lock:
            self.task = task
            self.recording = False
            self.seen_episodes = []
            self._seen_set = set()
        self._poll_stop.clear()
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()

    def stop_polling(self):
        self._poll_stop.set()

    def snapshot(self):
        with self.lock:
            return {
                'task': self.task,
                'recording': self.recording,
                'starting': self.starting,
                'stopping': self.stopping,
                'episode_count': len(self.seen_episodes),
                'recent_episodes': list(reversed(self.seen_episodes[-5:])),
            }


session = CollectionSession()


def _run_start_sequence(task):
    session.starting = True
    try:
        session.start_polling(task)
        for name in START_ORDER:
            proc_def = PROCESSES[name]
            if proc_def.status() == 'running':
                continue  # already up (e.g. started earlier outside the UI) -- don't double-launch
            proc_def.start(task=task)
            time.sleep(STAGE_WAIT_SEC[name])
    finally:
        session.starting = False


def _run_stop_sequence():
    session.stopping = True
    try:
        # data_collection.py first -- SIGTERM is caught and treated like 'E',
        # finalizing any in-progress episode before exit. Wait for it before
        # touching anything upstream, exactly like stop_collection.sh does.
        PROCESSES['data_collection'].stop(timeout=15)
        PROCESSES['preview'].stop(timeout=5)
        PROCESSES['camera'].stop(timeout=5)
        PROCESSES['rs_t265'].stop(timeout=5)
        PROCESSES['roscore'].stop(timeout=5)
        session.stop_polling()
    finally:
        session.stopping = False


# ── Episode-worker run state (Processing tab) ─────────────────────────────────

class WorkerRun:
    def __init__(self):
        self.lock = threading.Lock()
        self.proc = None
        self.lines = []
        self.subscribers = []  # list of queue.Queue, one per open SSE connection
        self.done = False
        self.returncode = None

    def is_running(self):
        with self.lock:
            return self.proc is not None and self.proc.poll() is None

    def _reader(self):
        for line in self.proc.stdout:
            line = line.rstrip('\n')
            with self.lock:
                self.lines.append(line)
                subs = list(self.subscribers)
            for q in subs:
                q.put(line)
        self.proc.wait()
        with self.lock:
            self.done = True
            self.returncode = self.proc.returncode
            subs = list(self.subscribers)
        for q in subs:
            q.put(None)  # sentinel: stream ended

    def start(self, task):
        with self.lock:
            if self.proc is not None and self.proc.poll() is None:
                return False
            task_arg = shlex.quote(task) if task else ''
            cmd = (f'source {CONDA_SETUP} && conda activate {CONDA_ENV} '
                   f'&& cd {shlex.quote(REPO_DIR)} && exec python3 episode_worker.py --once {task_arg}')
            self.proc = subprocess.Popen(['bash', '-c', cmd], stdout=subprocess.PIPE,
                                          stderr=subprocess.STDOUT, text=True, bufsize=1,
                                          cwd=REPO_DIR, start_new_session=True)
            self.lines = []
            self.done = False
            self.returncode = None
        threading.Thread(target=self._reader, daemon=True).start()
        return True

    def subscribe(self):
        q = queue.Queue()
        with self.lock:
            for line in self.lines:  # replay what already happened
                q.put(line)
            if self.done:
                q.put(None)
            else:
                self.subscribers.append(q)
        return q

    def unsubscribe(self, q):
        with self.lock:
            if q in self.subscribers:
                self.subscribers.remove(q)


worker_run = WorkerRun()


def _failed_episodes(task):
    """Scan raw/*/manifest.json for status=failed, across one task or all of them."""
    failed = []
    tasks = [task] if task else em.list_tasks(DATA_ROOT)
    for t in tasks:
        raw_dir = os.path.join(DATA_ROOT, t, 'raw')
        if not os.path.isdir(raw_dir):
            continue
        for name in sorted(os.listdir(raw_dir)):
            ep_dir = os.path.join(raw_dir, name)
            manifest_path = os.path.join(ep_dir, em.MANIFEST_NAME)
            if not os.path.isfile(manifest_path):
                continue
            try:
                manifest = em.read_manifest(ep_dir)
            except (json.JSONDecodeError, OSError):
                continue
            if manifest.get('status') != em.STATUS_FAILED:
                continue
            error_log = ''
            error_path = os.path.join(ep_dir, em.ERROR_LOG_NAME)
            if os.path.isfile(error_path):
                with open(error_path) as f:
                    error_log = f.read()[-4000:]
            failed.append({
                'task': t, 'episode_index': manifest.get('episode_index'),
                'error': manifest.get('error', ''), 'error_log': error_log,
            })
    return failed


# ── HTML ───────────────────────────────────────────────────────────────────────
#
# Visual language: an instrument panel, not a generic dashboard template --
# grounded in what this actually is (a control surface standing next to a
# UR7e arm, a GoPro and a T265, read at a glance from arm's length, possibly
# on a tablet). Dark by design (glare control next to hardware, and status
# colors that can't drift with a light/dark OS toggle) rather than by default
# -- a single committed visual world, not a missing second theme. Two type
# voices: a humanist sans for controls/labels, a monospace for anything that
# *is* machine state (counters, timestamps, log lines) -- both drawn from
# Ubuntu's own system font stack, since that's the actual OS this runs on, so
# the one visual "material" choice this page makes is grounded in its own
# deployment rather than an arbitrary webfont. Semantic color (running/
# stalled/error) is kept separate from the teal accent used for primary
# actions, per instrument-panel convention -- state is never encoded by hue
# alone, every dot pairs with an explicit mono status word.

INDEX_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FastUMI Ops</title>
<style>
:root {
  color-scheme: dark;
  --bg: #12161c;
  --panel: #171d25;
  --panel-raised: #1d242e;
  --border: #262e39;
  --border-strong: #3a4552;
  --text: #e7ecf1;
  --text-dim: #8996a6;
  --text-faint: #566173;
  --accent: #3ecfb8;
  --accent-dim: #2c9c8a;
  --accent-soft: rgba(62, 207, 184, 0.14);
  --ok: #4fc47e;
  --warn: #e0a83c;
  --danger: #e2564f;
  --danger-soft: rgba(226, 86, 79, 0.14);
  --idle: #4a5568;
  --radius: 4px;
  --font-sans: "Ubuntu", -apple-system, "Segoe UI", Roboto, sans-serif;
  --font-mono: "Ubuntu Mono", "JetBrains Mono", "SF Mono", Consolas, monospace;
}
* { box-sizing: border-box; }
body {
  margin: 0; min-height: 100vh;
  background: var(--bg);
  background-image: radial-gradient(circle at 12% -10%, rgba(62,207,184,0.06), transparent 42%);
  color: var(--text); font-family: var(--font-sans); font-size: 15px; line-height: 1.45;
  -webkit-font-smoothing: antialiased;
}
button, input, select { font-family: inherit; }
a { color: var(--accent); }

/* Topbar */
.topbar { display: flex; align-items: center; justify-content: space-between;
          padding: 14px 22px; border-bottom: 1px solid var(--border); background: var(--panel); }
.wordmark { font-family: var(--font-mono); font-weight: 700; font-size: 15px; letter-spacing: 0.06em; }
.wordmark span { color: var(--accent); font-weight: 400; }
.tabs { display: flex; gap: 4px; }
.tab { font-family: var(--font-mono); font-size: 12px; letter-spacing: 0.07em; text-transform: uppercase;
       padding: 10px 18px; cursor: pointer; color: var(--text-dim); border-radius: var(--radius);
       border: 1px solid transparent; transition: color .15s, border-color .15s, background .15s; }
.tab:hover { color: var(--text); }
.tab.active { color: var(--accent); border-color: var(--border-strong); background: var(--panel-raised); }

main { max-width: 1080px; margin: 0 auto; padding: 22px; }
.panel { display: none; }
.panel.active { display: grid; gap: 16px; }

/* Instrument card */
.card { background: var(--panel); border: 1px solid var(--border); border-radius: var(--radius);
        padding: 18px 20px; }
.card > .eyebrow { font-family: var(--font-mono); font-size: 11px; letter-spacing: 0.1em;
                    text-transform: uppercase; color: var(--text-faint); margin: 0 0 14px; }

.field { display: flex; flex-direction: column; gap: 6px; }
.field label { font-family: var(--font-mono); font-size: 11px; letter-spacing: 0.08em;
                text-transform: uppercase; color: var(--text-faint); }
input, select {
  background: var(--panel-raised); color: var(--text); border: 1px solid var(--border-strong);
  border-radius: var(--radius); padding: 10px 12px; font-size: 14px; min-height: 42px;
}
input:focus, select:focus, button:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }
.col { display: flex; flex-direction: column; gap: 8px; }

/* Status rail */
.rail { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; }
.lamp { display: flex; flex-direction: column; gap: 8px; padding: 13px 15px;
        background: var(--panel-raised); border: 1px solid var(--border); border-radius: var(--radius);
        cursor: pointer; transition: border-color .15s; user-select: none; }
.lamp:hover { border-color: var(--border-strong); }
.lamp-head { display: flex; align-items: center; gap: 9px; }
.dot { width: 10px; height: 10px; border-radius: 50%; background: var(--idle); flex: none;
       box-shadow: 0 0 0 0 transparent; transition: background .2s, box-shadow .2s; }
.dot.running { background: var(--ok); box-shadow: 0 0 9px 1px rgba(79,196,126,0.55); }
.dot.stalled { background: var(--warn); box-shadow: 0 0 9px 1px rgba(224,168,60,0.5); }
.dot.error   { background: var(--danger); box-shadow: 0 0 9px 1px rgba(226,86,79,0.5); }
.lamp-label { font-size: 14px; font-weight: 500; }
.lamp-state { font-family: var(--font-mono); font-size: 11px; letter-spacing: 0.06em;
              text-transform: uppercase; color: var(--text-faint); }
.log { display: none; white-space: pre-wrap; font-family: var(--font-mono); font-size: 11.5px;
       background: #0b0e12; border: 1px solid var(--border); color: var(--text-dim);
       padding: 10px 12px; border-radius: var(--radius); margin-top: 4px; max-height: 180px;
       overflow-y: auto; grid-column: 1 / -1; }
.log.open { display: block; }

/* Buttons */
button { border: 1px solid var(--border-strong); background: var(--panel-raised); color: var(--text);
         border-radius: var(--radius); padding: 11px 20px; font-size: 13.5px; font-weight: 600;
         cursor: pointer; min-height: 44px; transition: filter .1s, transform .05s; }
button:hover:not(:disabled) { filter: brightness(1.15); }
button:active:not(:disabled) { transform: scale(0.98); }
button:disabled { opacity: 0.35; cursor: not-allowed; }
button.primary { background: var(--accent); border-color: var(--accent); color: #082019; }
button.danger  { background: var(--danger); border-color: var(--danger); color: #200807; }
button.ghost   { background: transparent; }
.controls-row { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }
.divider { height: 1px; background: var(--border); margin: 16px 0; }

/* Episode toggle -- the most-pressed control during a session, its own visual language */
.episode-toggle { width: 100%; min-height: 66px; font-size: 16px; letter-spacing: 0.02em;
                   display: flex; align-items: center; justify-content: center; gap: 10px; }
.episode-toggle:not(.live) { background: var(--accent-soft); border-color: var(--accent-dim); color: var(--accent); }
.episode-toggle.live { background: var(--danger-soft); border-color: var(--danger); color: var(--danger);
                        animation: pulse-border 1.4s ease-in-out infinite; }
.rec-dot { width: 11px; height: 11px; border-radius: 50%; background: var(--danger); display: none; }
.episode-toggle.live .rec-dot { display: inline-block; animation: pulse-dot 1.4s ease-in-out infinite; }
@keyframes pulse-dot { 0%, 100% { opacity: 1; } 50% { opacity: 0.25; } }
@keyframes pulse-border { 0%, 100% { box-shadow: 0 0 0 0 rgba(226,86,79,0.35); }
                           50% { box-shadow: 0 0 0 7px rgba(226,86,79,0); } }
@media (prefers-reduced-motion: reduce) { .episode-toggle.live, .episode-toggle.live .rec-dot { animation: none; } }

/* Session info */
.session-stats { display: flex; gap: 30px; align-items: baseline; flex-wrap: wrap; }
.stat-value { font-family: var(--font-mono); font-size: 34px; font-weight: 700;
              font-variant-numeric: tabular-nums; color: var(--accent); line-height: 1; }
.stat-label { font-family: var(--font-mono); font-size: 11px; letter-spacing: 0.08em;
              text-transform: uppercase; color: var(--text-faint); margin-top: 4px; }
#episodeList { font-family: var(--font-mono); font-size: 12.5px; color: var(--text-dim);
               margin-top: 14px; display: flex; flex-direction: column; gap: 4px; }
#episodeList .ok-mark { color: var(--ok); }
.empty-note { color: var(--text-faint); font-size: 13px; font-family: var(--font-mono); }

/* Preview -- viewfinder framing */
.viewfinder { position: relative; background: #05070a; border-radius: var(--radius); overflow: hidden;
              max-width: 480px; aspect-ratio: 16/9; display: flex; align-items: center; justify-content: center; }
.viewfinder img { width: 100%; height: 100%; object-fit: cover; display: block; }
.viewfinder .placeholder { font-family: var(--font-mono); font-size: 12px; color: var(--text-faint);
                            letter-spacing: 0.08em; }
.viewfinder .bracket { position: absolute; width: 18px; height: 18px; border: 2px solid var(--accent); opacity: 0.75; }
.viewfinder .bracket.tl { top: 8px; left: 8px; border-right: none; border-bottom: none; }
.viewfinder .bracket.tr { top: 8px; right: 8px; border-left: none; border-bottom: none; }
.viewfinder .bracket.bl { bottom: 8px; left: 8px; border-right: none; border-top: none; }
.viewfinder .bracket.br { bottom: 8px; right: 8px; border-left: none; border-top: none; }

/* Processing */
.queue-row { display: flex; align-items: center; gap: 26px; flex-wrap: wrap; }
#workerLog { white-space: pre-wrap; font-family: var(--font-mono); font-size: 12.5px; color: #a8d8cd;
             background: #0b0e12; border: 1px solid var(--border); padding: 12px 14px; border-radius: var(--radius);
             height: 320px; overflow-y: auto; }
#workerLog.streaming::after { content: "\\2588"; animation: blink 1s step-start infinite; color: var(--accent); }
@keyframes blink { 50% { opacity: 0; } }
.failed-ep { background: var(--danger-soft); border: 1px solid var(--border); border-left: 3px solid var(--danger);
             border-radius: var(--radius); padding: 11px 14px; margin-bottom: 8px; font-size: 13px; }
.failed-ep b { font-family: var(--font-mono); }
.failed-ep pre { white-space: pre-wrap; font-family: var(--font-mono); font-size: 11px; color: var(--text-dim);
                 margin: 8px 0 0; max-height: 160px; overflow-y: auto; }

/* Visualization */
.viz-row { display: flex; gap: 16px; flex-wrap: wrap; align-items: flex-end; }
.viz-main { display: flex; gap: 20px; flex-wrap: wrap; align-items: flex-start; }
video { max-width: 640px; width: 100%; background: #05070a; border-radius: var(--radius); border: 1px solid var(--border); }
canvas { background: var(--panel-raised); border: 1px solid var(--border); border-radius: var(--radius);
         cursor: crosshair; max-width: 100%; }
#meta { font-family: var(--font-mono); font-size: 12px; color: var(--text-dim); margin-top: 8px; }
#vizStatus { margin-top: 12px; font-size: 13px; color: var(--danger); min-height: 18px; }
</style>
</head>
<body>
  <header class="topbar">
    <div class="wordmark">FASTUMI <span>/ OPS</span></div>
    <nav class="tabs">
      <div class="tab active" data-tab="collect">Collection</div>
      <div class="tab" data-tab="process">Processing</div>
      <div class="tab" data-tab="viz">Visualization</div>
    </nav>
  </header>

  <main>
    <!-- Collection -->
    <div class="panel active" id="panel-collect">
      <div class="card">
        <p class="eyebrow">Session</p>
        <div class="field" style="max-width:320px">
          <label for="taskInput">Task name</label>
          <input id="taskInput" list="taskList" placeholder="e.g. pick_and_place">
          <datalist id="taskList"></datalist>
        </div>
      </div>

      <div class="card">
        <p class="eyebrow">System Status</p>
        <div class="rail" id="chips"></div>
      </div>

      <div class="card">
        <p class="eyebrow">Controls</p>
        <div class="controls-row">
          <button class="primary" id="btnStart">Start Collection</button>
          <button class="danger" id="btnStop">Stop Collection</button>
        </div>
        <div class="divider"></div>
        <button class="episode-toggle" id="btnToggleEp" disabled>
          <span class="rec-dot"></span><span id="toggleEpLabel">Start Episode</span>
        </button>
        <div class="controls-row" style="margin-top:10px">
          <button class="ghost" id="btnRedo" disabled>Redo Last Episode</button>
        </div>
      </div>

      <div class="card">
        <p class="eyebrow">Session Info</p>
        <div class="session-stats">
          <div><div class="stat-value" id="epCount">0</div><div class="stat-label">Episodes Recorded</div></div>
        </div>
        <div id="episodeList"></div>
      </div>

      <div class="card">
        <p class="eyebrow">Live Preview</p>
        <div class="viewfinder">
          <img id="preview" src="" alt="" style="display:none">
          <div class="placeholder" id="previewPlaceholder">NO SIGNAL</div>
          <span class="bracket tl"></span><span class="bracket tr"></span>
          <span class="bracket bl"></span><span class="bracket br"></span>
        </div>
      </div>
    </div>

    <!-- Processing -->
    <div class="panel" id="panel-process">
      <div class="card">
        <p class="eyebrow">Queue</p>
        <div class="queue-row">
          <div class="field">
            <label for="procTaskSelect">Scope</label>
            <select id="procTaskSelect"><option value="">All tasks</option></select>
          </div>
          <div><div class="stat-value" id="queueDepth" style="font-size:28px">-</div><div class="stat-label">Pending</div></div>
          <button class="primary" id="btnRunWorker">Run Worker</button>
        </div>
      </div>

      <div class="card">
        <p class="eyebrow">Live Log</p>
        <div id="workerLog"></div>
      </div>

      <div class="card">
        <p class="eyebrow">Failed Episodes</p>
        <div id="failedList"><div class="empty-note">None</div></div>
      </div>
    </div>

    <!-- Visualization -->
    <div class="panel" id="panel-viz">
      <div class="card">
        <div class="viz-row">
          <div class="field"><label for="vizTaskSelect">Task</label><select id="vizTaskSelect"></select></div>
          <div class="field"><label for="vizEpisodeSelect">Episode</label><select id="vizEpisodeSelect"></select></div>
        </div>
        <div class="viz-main" style="margin-top:18px">
          <div class="col">
            <video id="vizPlayer" controls></video>
            <div id="meta"></div>
          </div>
          <canvas id="vizPlot" width="640" height="320"></canvas>
        </div>
        <div id="vizStatus"></div>
      </div>
    </div>
  </main>

<script>
// -- Tabs --------------------------------------------------------------------
document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById('panel-' + tab.dataset.tab).classList.add('active');
  });
});

async function fetchJSON(url, opts) {
  const res = await fetch(url, opts);
  const body = await res.json();
  if (!res.ok) throw new Error(body.error || ('request failed: ' + url));
  return body;
}

// -- Collection tab ------------------------------------------------------------
const CHIP_NAMES = [['roscore','Roscore'],['rs_t265','T265'],['camera','Camera'],
                     ['data_collection','Data Collection'],['preview','Preview']];
const chipsEl = document.getElementById('chips');
CHIP_NAMES.forEach(([key, label]) => {
  const lamp = document.createElement('div');
  lamp.className = 'lamp';
  lamp.innerHTML = `<div class="lamp-head"><span class="dot" id="dot-${key}"></span>
    <span class="lamp-label">${label}</span></div><span class="lamp-state" id="state-${key}">STOPPED</span>`;
  lamp.addEventListener('click', () => toggleLog(key));
  const log = document.createElement('div');
  log.className = 'log';
  log.id = 'log-' + key;
  chipsEl.appendChild(lamp);
  chipsEl.appendChild(log);
});

function toggleLog(key) {
  document.getElementById('log-' + key).classList.toggle('open');
}

async function refreshTaskList() {
  try {
    const data = await fetchJSON('/api/tasks');
    document.getElementById('taskList').innerHTML =
      data.tasks.map(t => `<option value="${t}">`).join('');
    const procSel = document.getElementById('procTaskSelect');
    const current = procSel.value;
    procSel.innerHTML = '<option value="">All tasks</option>' +
      data.tasks.map(t => `<option value="${t}">${t}</option>`).join('');
    procSel.value = current;
  } catch (e) { /* best effort */ }
}

const STATE_LABELS = { running: 'RUNNING', stalled: 'STALLED', error: 'ERROR', stopped: 'STOPPED' };

async function refreshCollectionStatus() {
  try {
    const data = await fetchJSON('/api/collection/status');
    for (const [key] of CHIP_NAMES) {
      const status = data.processes[key] || 'stopped';
      const dot = document.getElementById('dot-' + key);
      dot.className = 'dot ' + (status === 'stopped' ? '' : status);
      const stateEl = document.getElementById('state-' + key);
      stateEl.textContent = STATE_LABELS[status] || status.toUpperCase();
      stateEl.title = status === 'stalled' ? 'Process alive but its topic has stopped publishing' : '';
      document.getElementById('log-' + key).textContent = data.logs[key] || '(no log output yet)';
    }
    document.getElementById('epCount').textContent = data.session.episode_count;
    document.getElementById('episodeList').innerHTML = data.session.recent_episodes.length
      ? data.session.recent_episodes.map(([idx]) => `<div><span class="ok-mark">✓</span> episode ${idx} saved</div>`).join('')
      : '<div class="empty-note">No episodes yet this session</div>';

    const toggleBtn = document.getElementById('btnToggleEp');
    toggleBtn.classList.toggle('live', data.session.recording);
    document.getElementById('toggleEpLabel').textContent = data.session.recording ? 'Stop Episode' : 'Start Episode';
    const dcRunning = data.processes.data_collection === 'running';
    toggleBtn.disabled = !dcRunning || data.session.starting;
    document.getElementById('btnRedo').disabled = !dcRunning || data.session.recording;
    document.getElementById('btnStart').disabled = data.session.starting || data.session.stopping;
    document.getElementById('btnStop').disabled = data.session.starting || data.session.stopping;
    if (data.session.task && !document.getElementById('taskInput').value) {
      document.getElementById('taskInput').value = data.session.task;
    }
  } catch (e) { /* server may be mid-restart -- ignore and retry next tick */ }
}

document.getElementById('btnStart').addEventListener('click', async () => {
  const task = document.getElementById('taskInput').value.trim();
  if (!task) { alert('Enter a task name first.'); return; }
  await fetchJSON('/api/collection/start', {method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({task})});
});
document.getElementById('btnStop').addEventListener('click', async () => {
  await fetchJSON('/api/collection/stop', {method: 'POST'});
});
document.getElementById('btnToggleEp').addEventListener('click', async () => {
  await fetchJSON('/api/collection/key', {method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({key: 'space'})});
});
document.getElementById('btnRedo').addEventListener('click', async () => {
  if (!confirm('Redo the last saved episode? This deletes it.')) return;
  await fetchJSON('/api/collection/key', {method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({key: 'r'})});
});

const previewImg = document.getElementById('preview');
const previewPlaceholder = document.getElementById('previewPlaceholder');
previewImg.addEventListener('load', () => { previewImg.style.display = 'block'; previewPlaceholder.style.display = 'none'; });
previewImg.addEventListener('error', () => { previewImg.style.display = 'none'; previewPlaceholder.style.display = 'flex'; });
function refreshPreview() {
  previewImg.src = '/preview.jpg?t=' + Date.now();
}

setInterval(refreshCollectionStatus, 1000);
setInterval(refreshPreview, 500);
setInterval(refreshTaskList, 5000);
refreshTaskList();
refreshCollectionStatus();

// -- Processing tab -------------------------------------------------------------
async function refreshQueue() {
  const task = document.getElementById('procTaskSelect').value;
  try {
    const data = await fetchJSON('/api/processing/queue?task=' + encodeURIComponent(task));
    document.getElementById('queueDepth').textContent = data.pending;
  } catch (e) {}
  try {
    const failed = await fetchJSON('/api/processing/failed?task=' + encodeURIComponent(task));
    document.getElementById('failedList').innerHTML = failed.failed.length === 0
      ? '<div class="empty-note">None</div>'
      : failed.failed.map(f => `<div class="failed-ep"><b>${f.task}</b> · episode ${f.episode_index} — ${f.error}
          <pre>${(f.error_log || '').replace(/</g,'&lt;')}</pre></div>`).join('');
  } catch (e) {}
}
document.getElementById('procTaskSelect').addEventListener('change', refreshQueue);
setInterval(refreshQueue, 3000);
refreshQueue();

let workerES = null;
document.getElementById('btnRunWorker').addEventListener('click', async () => {
  const task = document.getElementById('procTaskSelect').value;
  const logEl = document.getElementById('workerLog');
  logEl.textContent = '';
  logEl.classList.add('streaming');
  document.getElementById('btnRunWorker').disabled = true;
  try {
    await fetchJSON('/api/processing/run', {method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({task})});
  } catch (e) {
    document.getElementById('btnRunWorker').disabled = false;
    logEl.classList.remove('streaming');
    alert(e.message);
    return;
  }
  if (workerES) workerES.close();
  workerES = new EventSource('/api/processing/stream');
  workerES.onmessage = (ev) => {
    if (ev.data === '__DONE__') {
      workerES.close();
      logEl.classList.remove('streaming');
      document.getElementById('btnRunWorker').disabled = false;
      refreshQueue();
      return;
    }
    logEl.textContent += ev.data + '\\n';
    logEl.scrollTop = logEl.scrollHeight;
  };
});

// -- Visualization tab (mirrors the standalone visualize_dataset.py page) ----
const vizTaskSelect = document.getElementById('vizTaskSelect');
const vizEpisodeSelect = document.getElementById('vizEpisodeSelect');
const vizPlayer = document.getElementById('vizPlayer');
const vizCanvas = document.getElementById('vizPlot');
const vizCtx = vizCanvas.getContext('2d');
const vizStatusEl = document.getElementById('vizStatus');
const metaEl = document.getElementById('meta');
const COLORS = ['#3ecfb8','#e0a83c','#4fc47e','#e2564f','#8ab4f8','#c58af9','#f28fb1'];
let vizQpos = null, vizLabels = [], vizRaf = null;

function vizShowError(msg) { vizStatusEl.textContent = msg; }
function vizClearError() { vizStatusEl.textContent = ''; }

async function loadVizDataset() {
  try {
    const data = await fetchJSON('/api/dataset');
    vizLabels = data.qpos_labels;
    const tasks = Object.keys(data.tasks);
    if (tasks.length === 0) { vizShowError('No processed episodes found yet.'); return; }
    vizTaskSelect.innerHTML = tasks.map(t => `<option value="${t}">${t}</option>`).join('');
    vizTaskSelect.dataset.episodes = JSON.stringify(data.tasks);
    populateVizEpisodes();
  } catch (e) { vizShowError('Failed to load dataset: ' + e.message); }
}

function populateVizEpisodes() {
  const allEpisodes = JSON.parse(vizTaskSelect.dataset.episodes || '{}');
  const eps = allEpisodes[vizTaskSelect.value] || [];
  vizEpisodeSelect.innerHTML = eps.map(e => `<option value="${e}">${e}</option>`).join('');
  loadVizEpisode();
}

async function loadVizEpisode() {
  vizClearError();
  const task = vizTaskSelect.value, episode = vizEpisodeSelect.value;
  if (task === undefined || episode === '' || episode === undefined) return;
  try {
    const data = await fetchJSON(`/api/episode?task=${encodeURIComponent(task)}&episode=${episode}`);
    vizQpos = data.qpos;
    metaEl.textContent = `cameras: ${data.cam_names.join(', ')}   frames: ${data.n_frames}   fps: ${data.fps.toFixed(1)}`;
    vizPlayer.src = `/video/${encodeURIComponent(task)}/episode_${episode}.mp4`;
    vizPlayer.load();
    drawVizPlot(0);
  } catch (e) { vizShowError('Failed to load episode: ' + e.message); vizQpos = null; }
}

function drawVizPlot(cursorFrac) {
  const W = vizCanvas.width, H = vizCanvas.height, pad = 30;
  vizCtx.clearRect(0, 0, W, H);
  vizCtx.strokeStyle = 'rgba(137,150,166,0.15)';
  vizCtx.lineWidth = 1;
  for (let gy = 0; gy <= 4; gy++) {
    const y = pad + (gy / 4) * (H - 2 * pad);
    vizCtx.beginPath(); vizCtx.moveTo(pad, y); vizCtx.lineTo(W - pad, y); vizCtx.stroke();
  }
  if (!vizQpos || vizQpos.length === 0) return;
  const n = vizQpos.length, dims = vizQpos[0].length;
  for (let d = 0; d < dims; d++) {
    const col = vizQpos.map(row => row[d]);
    const min = Math.min(...col), max = Math.max(...col);
    const range = (max - min) || 1;
    vizCtx.strokeStyle = COLORS[d % COLORS.length];
    vizCtx.lineWidth = 1.5;
    vizCtx.beginPath();
    for (let i = 0; i < n; i++) {
      const x = pad + (n > 1 ? (i / (n - 1)) * (W - 2 * pad) : 0);
      const norm = (col[i] - min) / range;
      const y = H - pad - norm * (H - 2 * pad);
      if (i === 0) vizCtx.moveTo(x, y); else vizCtx.lineTo(x, y);
    }
    vizCtx.stroke();
  }
  const legendLabels = vizLabels && vizLabels.length === dims ? vizLabels : Array.from({length: dims}, (_, i) => 'dim ' + i);
  vizCtx.font = '11px "Ubuntu Mono", monospace';
  for (let d = 0; d < dims; d++) {
    const lx = pad + (d % 4) * 90, ly = 14 + Math.floor(d / 4) * 14;
    vizCtx.fillStyle = COLORS[d % COLORS.length];
    vizCtx.fillRect(lx, ly - 8, 10, 10);
    vizCtx.fillStyle = '#8996a6';
    vizCtx.fillText(legendLabels[d], lx + 14, ly);
  }
  const cx = pad + cursorFrac * (W - 2 * pad);
  vizCtx.strokeStyle = '#e7ecf1';
  vizCtx.lineWidth = 1;
  vizCtx.beginPath();
  vizCtx.moveTo(cx, pad);
  vizCtx.lineTo(cx, H - pad);
  vizCtx.stroke();
}

function vizCurrentFrac() {
  if (!vizPlayer.duration || !isFinite(vizPlayer.duration)) return 0;
  return Math.min(1, Math.max(0, vizPlayer.currentTime / vizPlayer.duration));
}
function vizRenderLoop() {
  drawVizPlot(vizCurrentFrac());
  if (!vizPlayer.paused && !vizPlayer.ended) vizRaf = requestAnimationFrame(vizRenderLoop);
}
vizPlayer.addEventListener('play', () => { cancelAnimationFrame(vizRaf); vizRenderLoop(); });
vizPlayer.addEventListener('seeked', () => drawVizPlot(vizCurrentFrac()));
vizPlayer.addEventListener('pause', () => drawVizPlot(vizCurrentFrac()));
vizPlayer.addEventListener('loadedmetadata', () => drawVizPlot(0));
vizPlayer.addEventListener('error', () => vizShowError('Video failed to load/encode for this episode.'));
vizCanvas.addEventListener('click', (e) => {
  if (!vizPlayer.duration) return;
  const rect = vizCanvas.getBoundingClientRect();
  const pad = 30 * (vizCanvas.width / rect.width);
  const xPix = (e.clientX - rect.left) * (vizCanvas.width / rect.width);
  const frac = Math.min(1, Math.max(0, (xPix - pad) / (vizCanvas.width - 2 * pad)));
  vizPlayer.currentTime = frac * vizPlayer.duration;
});
vizTaskSelect.addEventListener('change', populateVizEpisodes);
vizEpisodeSelect.addEventListener('change', loadVizEpisode);
loadVizDataset();
</script>
</body>
</html>
""".encode('utf-8')


# ── HTTP handler ───────────────────────────────────────────────────────────────

class OpsHandler(viz.Handler):
    """Extends visualize_dataset.py's Handler (same http.server foundation) with
    the Collection/Processing routes. /api/dataset, /api/episode, /video/..., and
    _json/_error/log_message are all inherited untouched from viz.Handler."""

    def _read_json_body(self):
        length = int(self.headers.get('Content-Length', 0))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length) or b'{}')

    def _serve_preview(self):
        path = os.path.join(DATA_ROOT, '.preview', 'latest.jpg')
        if not os.path.isfile(path):
            return self._error(404, 'no preview frame yet')
        with open(path, 'rb') as f:
            body = f.read()
        self.send_response(200)
        self.send_header('Content-Type', 'image/jpeg')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_worker_stream(self):
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        q = worker_run.subscribe()
        try:
            while True:
                item = q.get()
                if item is None:
                    self.wfile.write(b'data: __DONE__\n\n')
                    self.wfile.flush()
                    break
                data = item.replace('\r', '').replace('\n', ' ')
                self.wfile.write(f'data: {data}\n\n'.encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            worker_run.unsubscribe(q)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        try:
            if path in ('/', '/index.html'):
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(INDEX_HTML)))
                self.end_headers()
                self.wfile.write(INDEX_HTML)

            elif path == '/preview.jpg':
                self._serve_preview()

            elif path == '/api/tasks':
                self._json({'tasks': em.list_tasks(DATA_ROOT)})

            elif path == '/api/collection/status':
                self._json({
                    'processes': {name: p.status() for name, p in PROCESSES.items()},
                    'logs': {name: p.log_tail() for name, p in PROCESSES.items()},
                    'session': session.snapshot(),
                })

            elif path == '/api/processing/queue':
                task = qs.get('task', [''])[0] or None
                self._json({'pending': len(em.list_pending(DATA_ROOT, task=task))})

            elif path == '/api/processing/failed':
                task = qs.get('task', [''])[0] or None
                self._json({'failed': _failed_episodes(task)})

            elif path == '/api/processing/stream':
                self._serve_worker_stream()

            elif path == '/api/dataset' or path == '/api/episode' or path.startswith('/video/'):
                super().do_GET()

            else:
                self._error(404, f'not found: {path}')
        except Exception as e:
            try:
                self._error(500, f'internal error: {e}')
            except Exception:
                pass

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            if path == '/api/collection/start':
                body = self._read_json_body()
                task = (body.get('task') or '').strip()
                if not task:
                    return self._error(400, 'task is required')
                if session.starting or session.stopping:
                    return self._error(409, 'a start/stop is already in progress')
                threading.Thread(target=_run_start_sequence, args=(task,), daemon=True).start()
                self._json({'ok': True})

            elif path == '/api/collection/stop':
                if session.starting or session.stopping:
                    return self._error(409, 'a start/stop is already in progress')
                threading.Thread(target=_run_stop_sequence, daemon=True).start()
                self._json({'ok': True})

            elif path == '/api/collection/key':
                body = self._read_json_body()
                key = body.get('key')
                byte_map = {'space': b' ', 'r': b'r', 'e': b'e'}
                if key not in byte_map:
                    return self._error(400, f'invalid key: {key}')
                if PROCESSES['data_collection'].status() != 'running':
                    return self._error(409, 'data_collection is not running')
                ok = PROCESSES['data_collection'].write_control_byte(byte_map[key])
                if not ok:
                    return self._error(500, 'failed to write control byte -- process may have exited')
                if key == 'space':
                    with session.lock:
                        session.recording = not session.recording
                self._json({'ok': True})

            elif path == '/api/processing/run':
                body = self._read_json_body()
                task = (body.get('task') or '').strip() or None
                if not worker_run.start(task):
                    return self._error(409, 'a worker run is already in progress')
                self._json({'ok': True})

            else:
                self._error(404, f'not found: {path}')
        except Exception as e:
            try:
                self._error(500, f'internal error: {e}')
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()

    server = ThreadingHTTPServer(('0.0.0.0', args.port), OpsHandler)
    print(f'FastUMI ops UI running at http://0.0.0.0:{args.port}  (Ctrl+C to stop)')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nStopped.')
        for p in PROCESSES.values():
            if p.status() == 'running':
                print(f'  leaving {p.name} running (stop it from the UI or stop_collection.sh)')


if __name__ == '__main__':
    main()
