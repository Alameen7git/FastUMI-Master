#!/usr/bin/env python3
"""Local web dashboard for sanity-checking a FastUMI dataset -- browse every
task/episode, play back the camera video, and see the synced qpos/action
(odometry) trace next to it, similar to Hugging Face's LeRobot dataset viewer.

Usage:
    python3 visualize_dataset.py [--port 8000] [--no-browser]
    python3 visualize_dataset.py --prepare <task>    # encode that task's videos now, then exit

Then open http://localhost:<port> (opened automatically unless --no-browser).
Episode videos are encoded once (via ffmpeg, 960x540 per camera) and cached under
<data_dir>/.viz_cache/, so repeat views of the same episode are instant. The web
UI runs --prepare after each Sync so episodes are ready before they are opened.
Reading the gzip-compressed 1080p frames is the slow part (~54 ms/frame on one
core), so frames are read in blocks by a pool of worker processes and piped
straight into ffmpeg.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs, unquote

import h5py
import numpy as np

with open('config/config.json', 'r') as f:
    config = json.load(f)

DATA_DIR     = config['device_settings']['data_dir']
CACHE_DIR    = os.path.join(DATA_DIR, '.viz_cache')
CAM_FPS      = config['task_config'].get('camera', {}).get('fps', 30)
STRIDE       = config.get('sync', {}).get('frame_stride', 1)
PLAYBACK_FPS = max(1, CAM_FPS / STRIDE)

QPOS_LABELS = ['X', 'Y', 'Z', 'qx', 'qy', 'qz', 'qw']

PREVIEW_HEIGHT = 540    # viewer video height (width follows the aspect ratio)
READ_BLOCK = 8          # frames per worker read (= the HDF5 chunk depth)
READ_WORKERS = max(1, min(12, (os.cpu_count() or 2) - 2))
_pool = None            # worker pool, created in main()
_encode_lock = threading.Lock()   # one encode at a time: each already uses every core


def scan_dataset():
    """{task_name: [episode_index, ...]} for every task dir under DATA_DIR that
    has at least one processed episode_<n>.hdf5 file."""
    result = {}
    if not os.path.isdir(DATA_DIR):
        return result
    for task in sorted(os.listdir(DATA_DIR)):
        task_dir = os.path.join(DATA_DIR, task)
        if not os.path.isdir(task_dir) or task.startswith('.'):
            continue
        indices = []
        for name in os.listdir(task_dir):
            if name.startswith('episode_') and name.endswith('.hdf5'):
                try:
                    indices.append(int(name[len('episode_'):-len('.hdf5')]))
                except ValueError:
                    continue
        if indices:
            result[task] = sorted(indices)
    return result


def load_episode_arrays(task, episode):
    hdf5_path = os.path.join(DATA_DIR, task, f'episode_{episode}.hdf5')
    if not os.path.isfile(hdf5_path):
        raise FileNotFoundError(f'no such episode file: {hdf5_path}')
    with h5py.File(hdf5_path, 'r') as f:
        if 'observations/images' not in f or 'observations/qpos' not in f or 'action' not in f:
            raise ValueError(f'malformed episode file: {hdf5_path} (missing expected datasets)')
        cam_names = list(f['observations/images'].keys())
        if not cam_names:
            raise ValueError(f'malformed episode file: {hdf5_path} (no camera datasets)')
        n_frames = f[f'observations/images/{cam_names[0]}'].shape[0]
        for cam in cam_names:
            if f[f'observations/images/{cam}'].shape[0] != n_frames:
                raise ValueError(f'malformed episode file: {hdf5_path} (camera "{cam}" frame count mismatch)')
        qpos = f['observations/qpos'][:]
        action = f['action'][:]
        if qpos.shape[0] != n_frames or action.shape[0] != n_frames:
            raise ValueError(f'malformed episode file: {hdf5_path} (qpos/action row count mismatch)')
    return cam_names, qpos, action, n_frames, hdf5_path


def _read_block(args):
    """Worker: frames [a, b) of every camera, side by side."""
    path, cams, a, b = args
    with h5py.File(path, 'r') as f:
        blocks = [f[f'observations/images/{cam}'][a:b] for cam in cams]
    return np.concatenate(blocks, axis=2) if len(blocks) > 1 else blocks[0]


def render_video(task, episode):
    """Encode (and cache) this episode's camera frames to an H.264 mp4 so the
    browser <video> tag can play/seek it natively."""
    cache_path = os.path.join(CACHE_DIR, task, f'episode_{episode}.mp4')
    hdf5_path = os.path.join(DATA_DIR, task, f'episode_{episode}.hdf5')
    if not os.path.isfile(hdf5_path):
        raise FileNotFoundError(f'no such episode file: {hdf5_path}')

    def fresh():
        return os.path.isfile(cache_path) and os.path.getmtime(cache_path) >= os.path.getmtime(hdf5_path)
    if fresh():
        return cache_path
    with _encode_lock:
        if fresh():   # encoded while this request waited (the browser asks several times)
            return cache_path
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with h5py.File(hdf5_path, 'r') as f:
            cams = list(f['observations/images'].keys())
            n, h, w, _ = f[f'observations/images/{cams[0]}'].shape
        width = w * len(cams)
        tmp_out = '{}.{}.tmp.mp4'.format(cache_path, os.getpid())   # --prepare may run at the same time
        cmd = [
            'ffmpeg', '-y', '-loglevel', 'error',
            '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{width}x{h}', '-r', str(PLAYBACK_FPS), '-i', '-',
            '-vf', f'scale=-2:{min(PREVIEW_HEIGHT, h)}',
            '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23',
            '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-f', 'mp4', tmp_out,
        ]
        t0 = time.time()
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        jobs = [(hdf5_path, cams, a, min(n, a + READ_BLOCK)) for a in range(0, n, READ_BLOCK)]
        try:
            try:
                # In order, a window of blocks at a time: memory stays ~2 blocks per worker.
                window = READ_WORKERS * 2
                for start in range(0, len(jobs), window):
                    part = jobs[start:start + window]
                    for block in (_pool.map(_read_block, part) if _pool else map(_read_block, part)):
                        proc.stdin.write(np.ascontiguousarray(block).tobytes())
                proc.stdin.close()
            except BrokenPipeError:
                pass
            err = proc.stderr.read()
            if proc.wait() != 0:
                raise subprocess.CalledProcessError(proc.returncode, cmd, output=err)
            os.replace(tmp_out, cache_path)
        finally:   # stopped or failed: no half-written file left behind
            if proc.poll() is None:
                proc.kill()
            if os.path.exists(tmp_out):
                os.remove(tmp_out)
        print(f'encoded {task}/episode_{episode}: {n} frames in {time.time() - t0:.1f} s', flush=True)
    return cache_path


def prepare(task):
    """Encode every episode of a task that is not cached yet (run after Sync)."""
    episodes = scan_dataset().get(task, [])
    if not episodes:
        print(f'No synced episodes in {task}.')
        return 0
    folder = os.path.join(CACHE_DIR, task)
    if os.path.isdir(folder):   # left by a run that was killed
        for name in os.listdir(folder):
            path = os.path.join(folder, name)
            if name.endswith('.tmp.mp4') and time.time() - os.path.getmtime(path) > 600:
                os.remove(path)
    print(f'Preparing {len(episodes)} episode video(s) for {task}', flush=True)
    failed = 0
    for i, ep in enumerate(episodes, 1):
        try:
            render_video(task, ep)
            print(f'[{i}/{len(episodes)}] episode_{ep} ready', flush=True)
        except Exception as e:   # one bad file must not stop the rest
            failed += 1
            print(f'[{i}/{len(episodes)}] episode_{ep} FAILED: {e}', flush=True)
    print(f'Done: {len(episodes) - failed}/{len(episodes)} episode video(s) ready.')
    return 1 if failed else 0


INDEX_HTML = b"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>FastUMI Dataset Viewer</title>
<style>
  body { font-family: -apple-system, Arial, sans-serif; margin: 0; padding: 20px;
         background: #1e1e1e; color: #e8e8e8; }
  h1 { font-size: 18px; font-weight: 600; margin: 0 0 16px; }
  .controls { display: flex; gap: 16px; align-items: center; margin-bottom: 16px; flex-wrap: wrap; }
  .controls label { font-size: 13px; color: #aaa; }
  select { font-size: 14px; padding: 4px 8px; background: #2a2a2a; color: #e8e8e8;
           border: 1px solid #444; border-radius: 4px; }
  .main { display: flex; gap: 20px; flex-wrap: wrap; align-items: flex-start; }
  video { max-width: 640px; width: 100%; background: #000; border-radius: 6px; }
  canvas { background: #262626; border-radius: 6px; cursor: crosshair; max-width: 100%; }
  .keys-hint { font-size: 12px; color: #999; }
  .keys-hint kbd { background: #333; border: 1px solid #555; border-radius: 3px; padding: 0 5px;
                   font: 11px monospace; color: #e8e8e8; }
  .vwrap { position: relative; }
  #toast { position: absolute; top: 12px; left: 50%; transform: translateX(-50%); pointer-events: none;
           background: rgba(0,0,0,.72); color: #fff; font-size: 15px; font-weight: 600;
           padding: 6px 14px; border-radius: 6px; opacity: 0; transition: opacity .25s; }
  #toast.show { opacity: 1; }
  #status { margin-top: 12px; font-size: 13px; color: #f28b82; min-height: 18px; }
  #meta { font-size: 12px; color: #999; margin-top: 6px; }
  .col { display: flex; flex-direction: column; }
</style>
</head>
<body>
  <h1>FastUMI Dataset Viewer</h1>
  <div class="controls">
    <label>Task
      <div><select id="taskSelect"></select></div>
    </label>
    <label>Episode
      <div><select id="episodeSelect"></select></div>
    </label>
    <div class="keys-hint"><kbd>Space</kbd> play / pause &nbsp; <kbd>&rarr;</kbd> +10 s &nbsp; <kbd>&larr;</kbd> &minus;5 s
      &nbsp; <kbd>&darr;</kbd> next episode &nbsp; <kbd>&uarr;</kbd> previous episode</div>
  </div>
  <div class="main">
    <div class="col">
      <div class="vwrap"><video id="player" controls></video><div id="toast"></div></div>
      <div id="meta"></div>
    </div>
    <canvas id="plot" width="720" height="500"></canvas>
  </div>
  <div id="status"></div>

<script>
const taskSelect = document.getElementById('taskSelect');
const episodeSelect = document.getElementById('episodeSelect');
const player = document.getElementById('player');
const canvas = document.getElementById('plot');
const ctx = canvas.getContext('2d');
const statusEl = document.getElementById('status');
const metaEl = document.getElementById('meta');

const COLORS = ['#e6194b','#3cb44b','#4363d8','#f58231','#911eb4','#46b8b8','#f032e6'];
let qpos = null;
let labels = [];
let nFrames = 0;
let fps = 10;
let rafHandle = null;

function showError(msg) { statusEl.textContent = msg; }
function clearError() { statusEl.textContent = ''; }

async function fetchJSON(url) {
  const res = await fetch(url);
  const body = await res.json();
  if (!res.ok) throw new Error(body.error || ('request failed: ' + url));
  return body;
}

async function loadDataset() {
  try {
    const data = await fetchJSON('/api/dataset');
    labels = data.qpos_labels;
    const tasks = Object.keys(data.tasks);
    if (tasks.length === 0) {
      showError('No processed episodes found under the dataset directory.');
      return;
    }
    taskSelect.innerHTML = tasks.map(t => `<option value="${t}">${t}</option>`).join('');
    taskSelect.dataset.episodes = JSON.stringify(data.tasks);
    const wanted = new URLSearchParams(location.search).get('task');  // e.g. ?task=PnP_block11
    if (wanted && tasks.includes(wanted)) taskSelect.value = wanted;
    populateEpisodes();
  } catch (e) {
    showError('Failed to load dataset: ' + e.message);
  }
}

function populateEpisodes() {
  const allEpisodes = JSON.parse(taskSelect.dataset.episodes);
  const eps = allEpisodes[taskSelect.value] || [];
  episodeSelect.innerHTML = eps.map(e => `<option value="${e}">${e}</option>`).join('');
  loadEpisode();
}

// Encode the next episode in the background while this one is watched
// (a HEAD request makes the server encode it; the server does one at a time).
function prefetchNext(task) {
  const next = episodeSelect.options[episodeSelect.selectedIndex + 1];
  if (!next) return;
  const url = `/video/${encodeURIComponent(task)}/episode_${next.value}.mp4`;
  player.addEventListener('canplay', () => fetch(url, {method: 'HEAD'}).catch(() => {}), {once: true});
}

async function loadEpisode() {
  clearError();
  const task = taskSelect.value;
  const episode = episodeSelect.value;
  if (task === undefined || episode === '' || episode === undefined) return;

  try {
    const data = await fetchJSON(`/api/episode?task=${encodeURIComponent(task)}&episode=${episode}`);
    qpos = data.qpos;
    nFrames = data.n_frames;
    fps = data.fps;
    metaEl.textContent = `cameras: ${data.cam_names.join(', ')}   frames: ${nFrames}   fps: ${fps.toFixed(1)}`;

    player.src = `/video/${encodeURIComponent(task)}/episode_${episode}.mp4`;
    player.load();
    prefetchNext(task);
    drawPlot(0);
  } catch (e) {
    showError('Failed to load episode: ' + e.message);
    qpos = null;
  }
}

// T265 pose from the odometry topic: position in metres (odom frame, from where
// the camera started), orientation as a unit quaternion (no unit).
const PANELS = [
  {title: 'Position', unit: 'm', dims: [0, 1, 2], colors: ['#e6194b', '#3cb44b', '#4f8cff']},
  {title: 'Orientation, quaternion', unit: 'no unit', dims: [3, 4, 5, 6],
   colors: ['#f58231', '#c77dff', '#46b8b8', '#f032e6']},
];
const PAD_L = 64, PAD_R = 16;

function niceStep(range, target) {   // 1, 2 or 5 x 10^k giving about `target` ticks
  const raw = range / target, mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const f = raw / mag;
  return (f < 1.5 ? 1 : f < 3.5 ? 2 : f < 7.5 ? 5 : 10) * mag;
}
function fmt(v, step) {
  const dec = Math.max(0, -Math.floor(Math.log10(step) + 1e-9));
  return (Math.abs(v) < step / 1e6 ? 0 : v).toFixed(dec);
}

function drawPanel(p, top, h, W, n) {
  const col = d => qpos.map(row => row[d]);
  let min = Infinity, max = -Infinity;
  for (const d of p.dims) for (const v of col(d)) { if (v < min) min = v; if (v > max) max = v; }
  if (max - min < 1e-6) { min -= 0.5; max += 0.5; }
  const margin = (max - min) * 0.05; min -= margin; max += margin;
  const y = v => top + h - (v - min) / (max - min) * h;
  const x = i => PAD_L + (n > 1 ? i / (n - 1) : 0) * (W - PAD_L - PAD_R);

  // grid + y ticks
  const step = niceStep(max - min, 5);
  ctx.font = '11px sans-serif'; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
  for (let v = Math.ceil(min / step) * step; v <= max; v += step) {
    ctx.strokeStyle = '#3a3a3a'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(PAD_L, y(v)); ctx.lineTo(W - PAD_R, y(v)); ctx.stroke();
    ctx.fillStyle = '#aaa'; ctx.fillText(fmt(v, step), PAD_L - 6, y(v));
  }
  // axes
  ctx.strokeStyle = '#888';
  ctx.beginPath(); ctx.moveTo(PAD_L, top); ctx.lineTo(PAD_L, top + h); ctx.lineTo(W - PAD_R, top + h); ctx.stroke();
  // y-axis title
  ctx.save(); ctx.translate(14, top + h / 2); ctx.rotate(-Math.PI / 2);
  ctx.textAlign = 'center'; ctx.fillStyle = '#ccc'; ctx.font = '12px sans-serif';
  ctx.fillText(p.unit === 'no unit' ? 'value' : `${p.title.toLowerCase()} (${p.unit})`, 0, 0);
  ctx.restore();

  // lines
  p.dims.forEach((d, k) => {
    const c = col(d);
    ctx.strokeStyle = p.colors[k]; ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < n; i++) { if (i === 0) ctx.moveTo(x(i), y(c[i])); else ctx.lineTo(x(i), y(c[i])); }
    ctx.stroke();
  });

  // title + legend above the panel
  ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic';
  ctx.font = 'bold 12px sans-serif'; ctx.fillStyle = '#e8e8e8';
  const title = `${p.title} (${p.unit})`;
  ctx.fillText(title, PAD_L, top - 10);
  let lx = PAD_L + ctx.measureText(title).width + 18;
  ctx.font = '11px sans-serif';
  p.dims.forEach((d, k) => {
    const name = labels && labels[d] ? labels[d] : 'dim ' + d;
    ctx.fillStyle = p.colors[k]; ctx.fillRect(lx, top - 19, 10, 10);
    ctx.fillStyle = '#ccc'; ctx.fillText(name, lx + 14, top - 10);
    lx += ctx.measureText(name).width + 30;
  });
}

function drawPlot(cursorFrac) {
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  if (!qpos || qpos.length === 0) return;
  const n = qpos.length, dims = qpos[0].length;
  const panels = dims === 7 ? PANELS
    : [{title: 'Values', unit: 'no unit', dims: [...Array(dims).keys()], colors: PANELS[0].colors.concat(PANELS[1].colors)}];
  const top0 = 30, bottom = H - 44, gap = 44;
  const h = (bottom - top0 - gap * (panels.length - 1)) / panels.length;
  panels.forEach((p, k) => drawPanel(p, top0 + k * (h + gap), h, W, n));

  // time axis (seconds) under the last panel
  const dur = n > 1 ? (n - 1) / fps : 0;
  const xs = t => PAD_L + (dur ? t / dur : 0) * (W - PAD_L - PAD_R);
  const step = niceStep(dur || 1, 8);
  ctx.fillStyle = '#aaa'; ctx.font = '11px sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
  for (let t = 0; t <= dur + 1e-9; t += step) {
    ctx.strokeStyle = '#888'; ctx.beginPath(); ctx.moveTo(xs(t), bottom); ctx.lineTo(xs(t), bottom + 4); ctx.stroke();
    ctx.fillText(fmt(t, step), xs(t), bottom + 6);
  }
  ctx.fillStyle = '#ccc'; ctx.font = '12px sans-serif';
  ctx.fillText('time (s)', PAD_L + (W - PAD_L - PAD_R) / 2, bottom + 24);

  // cursor (video position) across both panels
  const cx = PAD_L + cursorFrac * (W - PAD_L - PAD_R);
  ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(cx, top0); ctx.lineTo(cx, bottom); ctx.stroke();
}

function currentFrac() {
  if (!player.duration || !isFinite(player.duration)) return 0;
  return Math.min(1, Math.max(0, player.currentTime / player.duration));
}

function renderLoop() {
  drawPlot(currentFrac());
  if (!player.paused && !player.ended) {
    rafHandle = requestAnimationFrame(renderLoop);
  }
}

player.addEventListener('play', () => {
  cancelAnimationFrame(rafHandle);
  renderLoop();
});
player.addEventListener('seeked', () => drawPlot(currentFrac()));
player.addEventListener('pause', () => drawPlot(currentFrac()));
player.addEventListener('loadedmetadata', () => drawPlot(0));
player.addEventListener('error', () => showError('Video failed to load/encode for this episode.'));

canvas.addEventListener('click', (e) => {
  if (!player.duration) return;
  const rect = canvas.getBoundingClientRect();
  const xPix = (e.clientX - rect.left) * (canvas.width / rect.width);
  const frac = Math.min(1, Math.max(0, (xPix - PAD_L) / (canvas.width - PAD_L - PAD_R)));
  player.currentTime = frac * player.duration;
});

taskSelect.addEventListener('change', populateEpisodes);
episodeSelect.addEventListener('change', loadEpisode);

// --- hot keys ---
// Space play/pause, Right +10 s, Left -5 s, Down next episode, Up previous.
// Also sent by the web UI's View page (postMessage) when this viewer is embedded
// there, so they work without clicking into the frame first.
const toastEl = document.getElementById('toast');
let toastTimer = null;
function toast(text) {
  toastEl.textContent = text;
  toastEl.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toastEl.classList.remove('show'), 900);
}
function seekBy(sec) {
  if (!player.duration || !isFinite(player.duration)) return;
  player.currentTime = Math.min(player.duration, Math.max(0, player.currentTime + sec));
  toast((sec > 0 ? '+' : '\u2212') + Math.abs(sec) + ' s');
}
let resumeAfterSwitch = false;
function stepEpisode(delta) {
  const i = episodeSelect.selectedIndex + delta;
  if (i < 0 || i >= episodeSelect.options.length) {
    toast(delta > 0 ? 'Last episode' : 'First episode');
    return;
  }
  resumeAfterSwitch = !player.paused && !player.ended;   // keep playing across episodes
  episodeSelect.selectedIndex = i;
  toast('Episode ' + episodeSelect.value);
  loadEpisode();
}
player.addEventListener('loadeddata', () => {
  if (resumeAfterSwitch) { resumeAfterSwitch = false; player.play().catch(() => {}); }
});
const HOTKEYS = {
  ' ': () => { if (player.paused || player.ended) player.play().catch(() => {}); else player.pause(); },
  'ArrowRight': () => seekBy(10),
  'ArrowLeft': () => seekBy(-5),
  'ArrowDown': () => stepEpisode(1),
  'ArrowUp': () => stepEpisode(-1),
};
function hotkey(key) {
  const fn = HOTKEYS[key];
  if (fn) fn();
  return !!fn;
}
// Capture phase: runs before the <video> and <select> built-in keys (which would
// otherwise also seek, or change the episode twice).
document.addEventListener('keydown', ev => {
  if (ev.ctrlKey || ev.metaKey || ev.altKey || ev.repeat && ev.key === ' ') return;
  if (ev.target.matches && ev.target.matches('input, textarea')) return;
  if (hotkey(ev.key)) { ev.preventDefault(); ev.stopPropagation(); }
}, true);
window.addEventListener('message', ev => {
  if (ev.data && ev.data.type === 'fastumi-viewer-key') hotkey(ev.data.key);
});

loadDataset();
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _json(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status, message):
        self._json({'error': message}, status=status)

    def _serve_video(self, method):
        """Handle GET/HEAD for /video/<task>/episode_<n>.mp4 with proper Range
        support. Chrome/Firefox <video> elements issue a Range request to start
        buffering and require a real 206 Partial Content response -- a server
        that always answers 200 with the full body causes the element to fail
        loading (this was the actual bug, not the ffmpeg encode itself)."""
        path = urlparse(self.path).path
        rest = path[len('/video/'):]
        if '/' not in rest or not rest.endswith('.mp4') or not rest.split('/', 1)[1].startswith('episode_'):
            return self._error(400, 'expected /video/<task>/episode_<n>.mp4')
        task, fname = rest.split('/', 1)
        task = unquote(task)  # self.path is raw/percent-encoded; task names can contain spaces
        try:
            episode = int(fname[len('episode_'):-len('.mp4')])
        except ValueError:
            return self._error(400, f'invalid video path: {path}')

        try:
            video_path = render_video(task, episode)
        except FileNotFoundError as e:
            return self._error(404, str(e))
        except subprocess.CalledProcessError as e:
            detail = e.output.decode(errors='replace') if e.output else str(e)
            return self._error(500, f'ffmpeg failed to encode episode: {detail}')

        file_size = os.path.getsize(video_path)
        range_header = self.headers.get('Range')

        if range_header:
            try:
                units, _, range_spec = range_header.partition('=')
                start_s, _, end_s = range_spec.partition('-')
                start = int(start_s) if start_s else 0
                end = int(end_s) if end_s else file_size - 1
                end = min(end, file_size - 1)
                if units != 'bytes' or start > end or start >= file_size:
                    raise ValueError('unsatisfiable range')
            except ValueError:
                self.send_response(416)
                self.send_header('Content-Range', f'bytes */{file_size}')
                self.end_headers()
                return

            length = end - start + 1
            self.send_response(206)
            self.send_header('Content-Type', 'video/mp4')
            self.send_header('Accept-Ranges', 'bytes')
            self.send_header('Content-Range', f'bytes {start}-{end}/{file_size}')
            self.send_header('Content-Length', str(length))
            self.end_headers()
            if method == 'GET':
                with open(video_path, 'rb') as f:
                    f.seek(start)
                    self.wfile.write(f.read(length))
        else:
            self.send_response(200)
            self.send_header('Content-Type', 'video/mp4')
            self.send_header('Accept-Ranges', 'bytes')
            self.send_header('Content-Length', str(file_size))
            self.end_headers()
            if method == 'GET':
                with open(video_path, 'rb') as f:
                    self.wfile.write(f.read())

    def do_HEAD(self):
        try:
            if urlparse(self.path).path.startswith('/video/'):
                self._serve_video('HEAD')
            else:
                self._error(404, f'not found: {self.path}')
        except Exception as e:
            self._error(500, f'internal error: {e}')

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

            elif path == '/api/dataset':
                self._json({'tasks': scan_dataset(), 'fps': PLAYBACK_FPS, 'qpos_labels': QPOS_LABELS})

            elif path == '/api/episode':
                task = qs.get('task', [None])[0]
                episode = qs.get('episode', [None])[0]
                if task is None or episode is None:
                    return self._error(400, 'task and episode query params are required')
                try:
                    episode = int(episode)
                except ValueError:
                    return self._error(400, f'invalid episode index: {episode}')
                try:
                    cam_names, qpos, action, n_frames, _ = load_episode_arrays(task, episode)
                except FileNotFoundError as e:
                    return self._error(404, str(e))
                except ValueError as e:
                    return self._error(422, str(e))
                self._json({
                    'task': task, 'episode': episode, 'cam_names': cam_names,
                    'n_frames': n_frames, 'fps': PLAYBACK_FPS,
                    'qpos': qpos.tolist(), 'action': action.tolist(),
                })

            elif path.startswith('/video/'):
                self._serve_video('GET')

            else:
                self._error(404, f'not found: {path}')
        except Exception as e:
            self._error(500, f'internal error: {e}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--no-browser', action='store_true', help='Do not auto-open a browser tab.')
    parser.add_argument('--prepare', metavar='TASK', help='encode the videos of this task now and exit')
    args = parser.parse_args()

    if shutil.which('ffmpeg') is None:
        print('Error: ffmpeg not found on PATH -- required to encode episode videos.', file=sys.stderr)
        sys.exit(1)

    os.makedirs(CACHE_DIR, exist_ok=True)
    global _pool
    # HDF5 is not fork-safe once a file has been opened: workers come from a clean
    # forkserver process. Warm them up now so the first episode does not pay for it.
    _pool = ProcessPoolExecutor(READ_WORKERS, mp_context=multiprocessing.get_context('forkserver'))
    list(_pool.map(abs, range(READ_WORKERS)))
    if args.prepare:
        try:
            sys.exit(prepare(args.prepare))
        except KeyboardInterrupt:   # paused by the web UI (a recording started)
            print('Stopped.')
            sys.exit(130)
    server = ThreadingHTTPServer(('0.0.0.0', args.port), Handler)
    url = f'http://localhost:{args.port}'
    print(f'FastUMI dataset viewer running at {url}  (Ctrl+C to stop)')

    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nStopped.')


if __name__ == '__main__':
    main()
