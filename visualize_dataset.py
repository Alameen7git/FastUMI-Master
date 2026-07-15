#!/usr/bin/env python3
"""Local web dashboard for sanity-checking a FastUMI dataset -- browse every
task/episode, play back the camera video, and see the synced qpos/action
(odometry) trace next to it, similar to Hugging Face's LeRobot dataset viewer.

Usage:
    python3 visualize_dataset.py [--port 8000] [--no-browser]

Then open http://localhost:<port> (opened automatically unless --no-browser).
Episode videos are encoded once (via ffmpeg) and cached under
<data_dir>/.viz_cache/, so repeat views of the same episode are instant.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs, unquote

import cv2
import h5py
import numpy as np

with open('config/config.json', 'r') as f:
    config = json.load(f)

DATA_DIR     = config['device_settings']['data_dir']
CACHE_DIR    = os.path.join(DATA_DIR, '.viz_cache')
CAM_FPS      = config['task_config'].get('camera', {}).get('fps', 30)
STRIDE       = config.get('sync', {}).get('frame_stride', 1)
PLAYBACK_FPS = max(1, CAM_FPS / STRIDE)

QPOS_LABELS = ['Pos X', 'Pos Y', 'Pos Z', 'Q_X', 'Q_Y', 'Q_Z', 'Q_W']


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


def render_video(task, episode):
    """Encode (and cache) this episode's camera frames to an H.264 mp4 so the
    browser <video> tag can play/seek it natively."""
    cache_path = os.path.join(CACHE_DIR, task, f'episode_{episode}.mp4')
    hdf5_path = os.path.join(DATA_DIR, task, f'episode_{episode}.hdf5')
    if not os.path.isfile(hdf5_path):
        raise FileNotFoundError(f'no such episode file: {hdf5_path}')
    if os.path.isfile(cache_path) and os.path.getmtime(cache_path) >= os.path.getmtime(hdf5_path):
        return cache_path

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with h5py.File(hdf5_path, 'r') as f:
        cam_names = list(f['observations/images'].keys())
        images = [f[f'observations/images/{cam}'][:] for cam in cam_names]

    n_frames = images[0].shape[0]
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(n_frames):
            frame = np.concatenate([img[i] for img in images], axis=1) if len(images) > 1 else images[0][i]
            cv2.imwrite(os.path.join(tmp, f'{i:06d}.jpg'), frame)

        tmp_out = cache_path + '.tmp.mp4'
        cmd = [
            'ffmpeg', '-y', '-loglevel', 'error',
            '-framerate', str(PLAYBACK_FPS),
            '-i', os.path.join(tmp, '%06d.jpg'),
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
            tmp_out,
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        os.replace(tmp_out, cache_path)

    return cache_path


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
  canvas { background: #262626; border-radius: 6px; cursor: crosshair; max-width: 100%; display: block; }
  canvas + canvas { margin-top: 8px; }
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
  </div>
  <div class="main">
    <div class="col">
      <video id="player" controls></video>
      <div id="meta"></div>
    </div>
    <div class="col">
      <canvas id="plotPos" width="640" height="200"></canvas>
      <canvas id="plotQuat" width="640" height="200"></canvas>
    </div>
  </div>
  <div id="status"></div>

<script>
const taskSelect = document.getElementById('taskSelect');
const episodeSelect = document.getElementById('episodeSelect');
const player = document.getElementById('player');
const plotPos = document.getElementById('plotPos');
const plotPosCtx = plotPos.getContext('2d');
const plotQuat = document.getElementById('plotQuat');
const plotQuatCtx = plotQuat.getContext('2d');
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
    drawPlot(0);
  } catch (e) {
    showError('Failed to load episode: ' + e.message);
    qpos = null;
  }
}

const PAD_L = 55, PAD_R = 12, PAD_T = 14;

// Split qpos columns into the position group (meters) and quaternion group
// (unitless) by label prefix, so each gets its own auto-scaled Y-axis instead
// of sharing one axis where position (meters) and quaternion (-1..1) drown
// each other out.
function splitDims() {
  const posIdx = [], quatIdx = [];
  const dims = qpos && qpos.length ? qpos[0].length : 0;
  for (let d = 0; d < dims; d++) {
    const label = labels[d] || '';
    if (label.startsWith('Pos')) posIdx.push(d);
    else quatIdx.push(d);
  }
  return { posIdx, quatIdx };
}

// Pick a "nice" rounded tick step (1/2/5 * 10^n) so gridline labels read as
// sensible numbers rather than raw min/max fractions.
function niceTicks(min, max, count) {
  if (min === max) { min -= 1; max += 1; }
  const range = max - min;
  const rawStep = range / count;
  const mag = Math.pow(10, Math.floor(Math.log10(rawStep)));
  const norm = rawStep / mag;
  let step;
  if (norm < 1.5) step = 1 * mag;
  else if (norm < 3) step = 2 * mag;
  else if (norm < 7) step = 5 * mag;
  else step = 10 * mag;
  const niceMin = Math.floor(min / step) * step;
  const niceMax = Math.ceil(max / step) * step;
  const count2 = Math.round((niceMax - niceMin) / step);
  const ticks = [];
  // Compute each tick as niceMin + i*step (not accumulated addition) to avoid
  // floating-point drift producing labels like "-1.4e-17" instead of "0".
  for (let i = 0; i <= count2; i++) ticks.push(niceMin + i * step);
  return ticks;
}

function formatNum(v) {
  // Snap floating-point noise (e.g. -1.4e-17 from tick-step arithmetic) to a
  // clean value before deciding how to display it.
  v = Math.round(v * 1e6) / 1e6;
  const av = Math.abs(v);
  if (av !== 0 && (av < 0.001 || av >= 1000)) return v.toExponential(1);
  return (Math.round(v * 1000) / 1000).toString();
}

function drawSubplot(canvas, ctx, dimIndices, unitSuffix, showTimeLabels, cursorFrac) {
  const W = canvas.width, H = canvas.height;
  const padB = showTimeLabels ? 28 : 10;
  ctx.clearRect(0, 0, W, H);
  if (!qpos || qpos.length === 0 || dimIndices.length === 0) return;

  const n = qpos.length;
  const durationSec = fps > 0 ? (n - 1) / fps : 0;
  const plotW = W - PAD_L - PAD_R;
  const plotH = H - PAD_T - padB;

  // real min/max of this episode's actual data for these dims -- no fixed range
  let dataMin = Infinity, dataMax = -Infinity;
  dimIndices.forEach(d => {
    for (let i = 0; i < n; i++) {
      const v = qpos[i][d];
      if (v < dataMin) dataMin = v;
      if (v > dataMax) dataMax = v;
    }
  });
  if (!isFinite(dataMin) || !isFinite(dataMax)) return;

  const ticks = niceTicks(dataMin, dataMax, 5);
  const axisMin = ticks[0], axisMax = ticks[ticks.length - 1];
  const yRange = (axisMax - axisMin) || 1;
  const yOf = v => PAD_T + plotH - ((v - axisMin) / yRange) * plotH;

  // horizontal gridlines + numeric Y labels
  ctx.font = '10px sans-serif';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  ticks.forEach(t => {
    const y = yOf(t);
    ctx.strokeStyle = '#3a3a3a';
    ctx.beginPath();
    ctx.moveTo(PAD_L, y);
    ctx.lineTo(PAD_L + plotW, y);
    ctx.stroke();
    ctx.fillStyle = '#999';
    ctx.fillText(formatNum(t), PAD_L - 6, y);
  });

  if (unitSuffix) {
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillStyle = '#777';
    ctx.fillText(unitSuffix, 2, 2);
  }

  // vertical gridlines at fixed time fractions; numeric seconds only on the
  // bottom subplot so the two stacked charts don't duplicate the same labels
  const xTickCount = 6;
  for (let k = 0; k <= xTickCount; k++) {
    const frac = k / xTickCount;
    const x = PAD_L + frac * plotW;
    ctx.strokeStyle = '#2e2e2e';
    ctx.beginPath();
    ctx.moveTo(x, PAD_T);
    ctx.lineTo(x, PAD_T + plotH);
    ctx.stroke();
    if (showTimeLabels) {
      ctx.fillStyle = '#999';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      ctx.fillText((frac * durationSec).toFixed(1) + 's', x, PAD_T + plotH + 4);
    }
  }
  if (showTimeLabels) {
    ctx.textAlign = 'right';
    ctx.textBaseline = 'bottom';
    ctx.fillStyle = '#777';
    ctx.fillText('time (s)', PAD_L + plotW, H - 2);
  }

  ctx.strokeStyle = '#555';
  ctx.lineWidth = 1;
  ctx.strokeRect(PAD_L, PAD_T, plotW, plotH);

  // data lines, true-scaled relative to each other within this subplot
  dimIndices.forEach(d => {
    ctx.strokeStyle = COLORS[d % COLORS.length];
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const x = PAD_L + (n > 1 ? (i / (n - 1)) * plotW : 0);
      const y = yOf(qpos[i][d]);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
  });

  // legend
  ctx.font = '11px sans-serif';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'alphabetic';
  dimIndices.forEach((d, idx) => {
    const lx = PAD_L + idx * 90;
    const ly = PAD_T + 10;
    ctx.fillStyle = COLORS[d % COLORS.length];
    ctx.fillRect(lx, ly - 8, 10, 10);
    ctx.fillStyle = '#ccc';
    ctx.fillText(labels[d] || ('dim ' + d), lx + 14, ly);
  });

  // playback cursor
  const cx = PAD_L + cursorFrac * plotW;
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(cx, PAD_T);
  ctx.lineTo(cx, PAD_T + plotH);
  ctx.stroke();
}

function drawPlot(cursorFrac) {
  const { posIdx, quatIdx } = splitDims();
  drawSubplot(plotPos, plotPosCtx, posIdx, 'm', false, cursorFrac);
  drawSubplot(plotQuat, plotQuatCtx, quatIdx, '', true, cursorFrac);
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

function attachSeekHandler(canvas) {
  canvas.addEventListener('click', (e) => {
    if (!player.duration) return;
    const rect = canvas.getBoundingClientRect();
    const scale = canvas.width / rect.width;
    const padL = PAD_L * scale, padR = PAD_R * scale;
    const xPix = (e.clientX - rect.left) * scale;
    const frac = Math.min(1, Math.max(0, (xPix - padL) / (canvas.width - padL - padR)));
    player.currentTime = frac * player.duration;
  });
}
attachSeekHandler(plotPos);
attachSeekHandler(plotQuat);

taskSelect.addEventListener('change', populateEpisodes);
episodeSelect.addEventListener('change', loadEpisode);

// Keyboard shortcuts: Left/Right step through the episode dropdown (via the
// same loadEpisode() the dropdown's own change handler uses), Space toggles
// play/pause. Suppressed while an input/select has focus so they don't
// interfere with normal dropdown selection or text entry.
function isTypingTarget(el) {
  if (!el) return false;
  const tag = el.tagName;
  return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || el.isContentEditable;
}

function stepEpisode(delta) {
  const opts = episodeSelect.options;
  if (opts.length === 0) return;
  // Stop at the first/last episode rather than wrapping around, so repeated
  // presses can't accidentally jump from the last episode back to the first.
  const idx = Math.min(opts.length - 1, Math.max(0, episodeSelect.selectedIndex + delta));
  if (idx === episodeSelect.selectedIndex) return;
  episodeSelect.selectedIndex = idx;
  loadEpisode();
}

document.addEventListener('keydown', (e) => {
  if (isTypingTarget(document.activeElement)) return;
  if (e.key === 'ArrowRight') {
    e.preventDefault();
    stepEpisode(1);
  } else if (e.key === 'ArrowLeft') {
    e.preventDefault();
    stepEpisode(-1);
  } else if (e.code === 'Space' || e.key === ' ') {
    e.preventDefault();
    if (player.paused) player.play(); else player.pause();
  }
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
    args = parser.parse_args()

    if shutil.which('ffmpeg') is None:
        print('Error: ffmpeg not found on PATH -- required to encode episode videos.', file=sys.stderr)
        sys.exit(1)

    os.makedirs(CACHE_DIR, exist_ok=True)
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
