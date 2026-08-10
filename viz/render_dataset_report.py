#!/usr/bin/env python3
"""Generate one self-contained HTML report covering every episode in a task's
dataset directory: real camera footage next to the simulated UR7e replay, plus
synced joint-angle/TCP-position trajectory charts -- with keyboard navigation
across episodes (built on the same per-episode pipeline as render_episode_report.py).

Hotkeys (in the generated report):
    Right arrow  -- next episode
    Left arrow   -- previous episode
    Space        -- pause/unpause the video

Everything (fonts, every episode's video, every episode's trajectory data) is
embedded in the output HTML -- just open it in a browser, no server needed.
Large datasets make for a large file (each episode adds roughly its video's
size); that's the tradeoff for "no server, just double-click it."

Usage:
    python3 render_dataset_report.py <path/to/task_dir>
    python3 render_dataset_report.py <path/to/task_dir> --out dataset_report.html
    python3 render_dataset_report.py <path/to/task_dir> --episodes 1-10
    python3 render_dataset_report.py <path/to/task_dir> --episodes 1,3,7
"""
import argparse
import glob
import json
import os
import re

import mujoco

import visualize_episode_mujoco as vem
from render_episode_report import (
    FONT_FILES, FONTS_DIR, JOINT_NAMES, _b64_file, compute_report_data,
)

HTML_TEMPLATE = r"""<!doctype html>
<meta charset="utf-8">
<title>__TITLE__</title>
<style>
@font-face { font-family: 'Plex Sans Cond'; font-weight: 600; src: url(data:font/woff2;base64,__SANS_COND_600__) format('woff2'); font-display: swap; }
@font-face { font-family: 'Plex Sans'; font-weight: 400; src: url(data:font/woff2;base64,__SANS_400__) format('woff2'); font-display: swap; }
@font-face { font-family: 'Plex Sans'; font-weight: 600; src: url(data:font/woff2;base64,__SANS_600__) format('woff2'); font-display: swap; }
@font-face { font-family: 'Plex Mono'; font-weight: 400; src: url(data:font/woff2;base64,__MONO_400__) format('woff2'); font-display: swap; }
@font-face { font-family: 'Plex Mono'; font-weight: 500; src: url(data:font/woff2;base64,__MONO_500__) format('woff2'); font-display: swap; }

:root {
  --bg: #f4f6f8; --panel: #ffffff; --border: #d8dde3; --text: #1a1f26; --text-muted: #5b6472;
  --accent: #3d7ba0; --accent-soft: #e4eef4; --good: #2f8f5f; --good-soft: #e3f3ea; --warn: #b5791f; --warn-soft: #f7ecd9;
  --shadow: 0 1px 2px rgba(20, 23, 28, 0.06), 0 4px 16px rgba(20, 23, 28, 0.05);
  color-scheme: light dark;
}
@media (prefers-color-scheme: dark) {
  :root { --bg: #14171c; --panel: #1c2129; --border: #2a313c; --text: #e8ecf1; --text-muted: #8b95a3;
    --accent: #7dadcc; --accent-soft: #22303a; --good: #5fbf8f; --good-soft: #1c2e26; --warn: #e0a64d; --warn-soft: #2e2618;
    --shadow: 0 1px 2px rgba(0,0,0,0.4), 0 8px 24px rgba(0,0,0,0.3); }
}
:root[data-theme="dark"] { --bg: #14171c; --panel: #1c2129; --border: #2a313c; --text: #e8ecf1; --text-muted: #8b95a3;
  --accent: #7dadcc; --accent-soft: #22303a; --good: #5fbf8f; --good-soft: #1c2e26; --warn: #e0a64d; --warn-soft: #2e2618;
  --shadow: 0 1px 2px rgba(0,0,0,0.4), 0 8px 24px rgba(0,0,0,0.3); }
:root[data-theme="light"] { --bg: #f4f6f8; --panel: #ffffff; --border: #d8dde3; --text: #1a1f26; --text-muted: #5b6472;
  --accent: #3d7ba0; --accent-soft: #e4eef4; --good: #2f8f5f; --good-soft: #e3f3ea; --warn: #b5791f; --warn-soft: #f7ecd9;
  --shadow: 0 1px 2px rgba(20,23,28,0.06), 0 4px 16px rgba(20,23,28,0.05); }

* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text); font-family: 'Plex Sans', system-ui, sans-serif; font-size: 15px; line-height: 1.5; }
.wrap { max-width: 1080px; margin: 0 auto; padding: 28px 24px 64px; }
.eyebrow { font-family: 'Plex Mono', monospace; font-size: 12px; letter-spacing: 0.08em; text-transform: uppercase; color: var(--accent); margin: 0 0 6px; }

.nav-row { display: flex; align-items: baseline; justify-content: space-between; gap: 16px; flex-wrap: wrap; margin-bottom: 4px; }
h1 { font-family: 'Plex Sans Cond', 'Plex Sans', sans-serif; font-weight: 600; font-size: 26px; letter-spacing: -0.01em; margin: 0; text-wrap: balance; }
.nav-controls { display: flex; align-items: center; gap: 8px; font-family: 'Plex Mono', monospace; font-size: 13px; color: var(--text-muted); }
.nav-controls select {
  font-family: 'Plex Mono', monospace; font-size: 13px; background: var(--panel); color: var(--text);
  border: 1px solid var(--border); border-radius: 6px; padding: 4px 8px;
}
.nav-btn {
  background: var(--panel); border: 1px solid var(--border); color: var(--text); border-radius: 6px;
  width: 28px; height: 28px; font-family: 'Plex Mono', monospace; font-size: 14px; cursor: pointer;
  display: inline-flex; align-items: center; justify-content: center;
}
.nav-btn:hover { background: var(--accent-soft); }
.nav-btn:disabled { opacity: 0.35; cursor: default; }
.nav-btn:focus-visible, .nav-controls select:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }

.subtitle { color: var(--text-muted); font-size: 14px; margin: 0 0 6px; font-family: 'Plex Mono', monospace; }
.hotkeys { color: var(--text-muted); font-size: 12px; margin: 0 0 24px; font-family: 'Plex Mono', monospace; }
.hotkeys kbd {
  font-family: 'Plex Mono', monospace; font-size: 11px; border: 1px solid var(--border); border-radius: 4px;
  padding: 1px 5px; background: var(--panel); box-shadow: var(--shadow);
}

.stat-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; margin-bottom: 28px; }
.stat { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 12px 14px; box-shadow: var(--shadow); }
.stat .label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--text-muted); margin-bottom: 5px; }
.stat .value { font-family: 'Plex Mono', monospace; font-weight: 500; font-size: 18px; font-variant-numeric: tabular-nums; }
.pill { display: inline-flex; align-items: center; gap: 5px; font-family: 'Plex Mono', monospace; font-size: 11px; font-weight: 500; padding: 2px 8px; border-radius: 999px; margin-top: 4px; }
.pill.good { background: var(--good-soft); color: var(--good); }
.pill.warn { background: var(--warn-soft); color: var(--warn); }
.pill::before { content: ''; width: 6px; height: 6px; border-radius: 50%; background: currentColor; }

.panel { background: var(--panel); border: 1px solid var(--border); border-radius: 14px; box-shadow: var(--shadow); padding: 20px; margin-bottom: 24px; }
.panel h2 { font-family: 'Plex Sans Cond', sans-serif; font-weight: 600; font-size: 17px; margin: 0 0 4px; }
.panel .desc { color: var(--text-muted); font-size: 13px; margin: 0 0 16px; }

video { width: 100%; display: block; border-radius: 8px; background: #000; }

.chart-wrap { overflow-x: auto; }
svg.chart { display: block; width: 100%; height: auto; }
.axis-label { fill: var(--text-muted); font-family: 'Plex Mono', monospace; font-size: 10.5px; }
.gridline { stroke: var(--border); stroke-width: 1; }
.playhead { stroke: var(--text); stroke-width: 1; stroke-dasharray: 3 3; opacity: 0.7; }
.endpoint { r: 3; }

.legend { display: flex; flex-wrap: wrap; gap: 6px 14px; margin-top: 14px; font-family: 'Plex Mono', monospace; font-size: 12px; }
.legend button { display: inline-flex; align-items: center; gap: 6px; background: none; border: none; color: var(--text);
  font-family: inherit; font-size: inherit; cursor: pointer; padding: 3px 6px; border-radius: 6px; opacity: 1; transition: opacity 0.15s; }
.legend button.off { opacity: 0.32; }
.legend button:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }
.legend .swatch { width: 10px; height: 10px; border-radius: 2px; flex: none; }

footer { color: var(--text-muted); font-family: 'Plex Mono', monospace; font-size: 12px; margin-top: 32px; text-align: center; }
</style>

<div class="wrap">
  <p class="eyebrow">FastUMI &middot; UR7e dataset report</p>
  <div class="nav-row">
    <h1 id="ep-title">__TASK__</h1>
    <div class="nav-controls">
      <button class="nav-btn" id="btn-prev" aria-label="Previous episode">&#8592;</button>
      <select id="ep-select"></select>
      <button class="nav-btn" id="btn-next" aria-label="Next episode">&#8594;</button>
    </div>
  </div>
  <p class="subtitle" id="ep-subtitle"></p>
  <p class="hotkeys"><kbd>&#8592;</kbd> prev &nbsp; <kbd>&#8594;</kbd> next &nbsp; <kbd>space</kbd> play/pause</p>

  <div class="stat-row">
    <div class="stat"><div class="label">Frames</div><div class="value" id="stat-frames"></div></div>
    <div class="stat"><div class="label">Duration</div><div class="value" id="stat-duration"></div></div>
    <div class="stat">
      <div class="label">FPS</div><div class="value" id="stat-fps"></div>
      <span class="pill" id="stat-fps-pill"></span>
    </div>
    <div class="stat"><div class="label">IK branch</div><div class="value" id="stat-branch"></div></div>
  </div>

  <div class="panel">
    <h2>Real footage vs. simulated replay</h2>
    <p class="desc">Left: recorded camera feed. Right: UR7e driven by the IK-solved joint trajectory.</p>
    <video id="player" controls></video>
  </div>

  <div class="panel">
    <h2>Joint angles over time</h2>
    <p class="desc">All 6 joints, degrees. Click a legend entry to isolate it. Dashed line tracks the video's current frame.</p>
    <div class="chart-wrap"><svg class="chart" id="chart-joints" viewBox="0 0 960 320" preserveAspectRatio="xMidYMid meet"></svg></div>
    <div class="legend" id="legend-joints"></div>
  </div>

  <div class="panel">
    <h2>TCP position over time</h2>
    <p class="desc">Simulated end-effector position in base_link frame, meters.</p>
    <div class="chart-wrap"><svg class="chart" id="chart-tcp" viewBox="0 0 960 260" preserveAspectRatio="xMidYMid meet"></svg></div>
    <div class="legend" id="legend-tcp"></div>
  </div>

  <footer id="footer-text"></footer>
</div>

<script>
const EPISODES = __EPISODES_JSON__;

const COLORS = ['#e6753a', '#3d9e6b', '#3d7ba0', '#b5459a', '#8a7a2e', '#7a5ce0'];
const TCP_COLORS = ['#e6753a', '#3d9e6b', '#3d7ba0'];
const TCP_LABELS = ['x', 'y', 'z'];

const select = document.getElementById('ep-select');
const player = document.getElementById('player');
const btnPrev = document.getElementById('btn-prev');
const btnNext = document.getElementById('btn-next');

select.innerHTML = EPISODES.map((e, i) => `<option value="${i}">${e.episode}</option>`).join('');

let current = 0;
let hiddenSeries = new Set();

function buildChart(svgId, legendId, series) {
  const svg = document.getElementById(svgId);
  const W = 960, H = svg.viewBox.baseVal.height;
  const padL = 44, padR = 14, padT = 14, padB = 26;
  const plotW = W - padL - padR, plotH = H - padT - padB;

  const allVals = series.flatMap(s => s.values);
  let vMin = Math.min(...allVals), vMax = Math.max(...allVals);
  if (vMin === vMax) { vMin -= 1; vMax += 1; }
  const pad = (vMax - vMin) * 0.08;
  vMin -= pad; vMax += pad;

  const tMax = series.times[series.times.length - 1] || 1;
  const xOf = t => padL + (t / tMax) * plotW;
  const yOf = v => padT + plotH - ((v - vMin) / (vMax - vMin)) * plotH;

  let svgContent = '';
  const yTicks = 5;
  for (let i = 0; i <= yTicks; i++) {
    const v = vMin + (vMax - vMin) * i / yTicks;
    const y = yOf(v);
    svgContent += `<line class="gridline" x1="${padL}" y1="${y}" x2="${padL + plotW}" y2="${y}"/>`;
    svgContent += `<text class="axis-label" x="${padL - 6}" y="${y}" text-anchor="end" dominant-baseline="middle">${v.toFixed(Math.abs(v) < 5 ? 2 : 0)}</text>`;
  }
  const xTicks = 6;
  for (let i = 0; i <= xTicks; i++) {
    const t = tMax * i / xTicks;
    const x = xOf(t);
    svgContent += `<text class="axis-label" x="${x}" y="${padT + plotH + 16}" text-anchor="middle">${t.toFixed(1)}s</text>`;
  }
  svgContent += `<rect x="${padL}" y="${padT}" width="${plotW}" height="${plotH}" fill="none" stroke="var(--border)"/>`;

  series.list.forEach(s => {
    const hidden = hiddenSeries.has(s.key);
    const pts = s.values.map((v, i) => `${xOf(series.times[i])},${yOf(v)}`).join(' ');
    svgContent += `<polyline points="${pts}" fill="none" stroke="${s.color}" stroke-width="1.8" data-series="${s.key}" style="display:${hidden ? 'none' : ''}"/>`;
    const lastV = s.values[s.values.length - 1];
    svgContent += `<circle class="endpoint" cx="${xOf(tMax)}" cy="${yOf(lastV)}" fill="${s.color}" data-series="${s.key}" style="display:${hidden ? 'none' : ''}"/>`;
  });

  svgContent += `<line class="playhead" id="${svgId}-playhead" x1="${padL}" y1="${padT}" x2="${padL}" y2="${padT + plotH}"/>`;
  svg.innerHTML = svgContent;

  const legend = document.getElementById(legendId);
  legend.innerHTML = series.list.map(s => `
    <button data-series="${s.key}" class="${hiddenSeries.has(s.key) ? 'off' : ''}"><span class="swatch" style="background:${s.color}"></span>${s.label}</button>`).join('');
  legend.querySelectorAll('button').forEach(btn => {
    btn.addEventListener('click', () => {
      const key = btn.dataset.series;
      if (hiddenSeries.has(key)) hiddenSeries.delete(key); else hiddenSeries.add(key);
      btn.classList.toggle('off');
      svg.querySelectorAll(`[data-series="${key}"]`).forEach(el => {
        el.style.display = hiddenSeries.has(key) ? 'none' : '';
      });
    });
  });

  return xOf;
}

let jointXOf, tcpXOf, jointTMax;

function loadEpisode(i, opts) {
  opts = opts || {};
  i = Math.max(0, Math.min(EPISODES.length - 1, i));
  current = i;
  const ep = EPISODES[i];

  select.value = String(i);
  btnPrev.disabled = i === 0;
  btnNext.disabled = i === EPISODES.length - 1;

  document.getElementById('ep-title').textContent = `${ep.task} — ${ep.episode}`;
  document.getElementById('ep-subtitle').textContent = `episode ${i + 1} of ${EPISODES.length}`;
  document.getElementById('stat-frames').textContent = ep.n_frames;
  document.getElementById('stat-duration').textContent = ep.duration_s.toFixed(2) + ' s';
  document.getElementById('stat-fps').textContent = ep.fps.toFixed(2);
  const fpsPill = document.getElementById('stat-fps-pill');
  fpsPill.textContent = ep.fps_measured ? 'measured' : 'fallback';
  fpsPill.className = 'pill ' + (ep.fps_measured ? 'good' : 'warn');
  document.getElementById('stat-branch').textContent = '#' + ep.preferred_branch;
  document.getElementById('footer-text').textContent = `generated by render_dataset_report.py · ${EPISODES.length} episode(s)`;

  const wasPlaying = !player.paused && !opts.forcePause;
  player.pause();
  player.src = 'data:video/mp4;base64,' + ep.video_b64;
  player.load();
  if (wasPlaying) {
    player.oncanplay = () => { player.play(); player.oncanplay = null; };
  }

  const times = ep.joint_deg.map((_, k) => k / ep.fps);
  jointTMax = times[times.length - 1] || 1;

  const jointList = ep.joint_names.map((name, j) => ({
    key: name, label: name.replace('_', ' '), color: COLORS[j % COLORS.length],
    values: ep.joint_deg.map(row => row[j]),
  }));
  jointXOf = buildChart('chart-joints', 'legend-joints', { list: jointList, times });

  const tcpList = TCP_LABELS.map((label, j) => ({
    key: label, label: label, color: TCP_COLORS[j],
    values: ep.tcp_pos_m.map(row => row[j]),
  }));
  tcpXOf = buildChart('chart-tcp', 'legend-tcp', { list: tcpList, times });

  syncPlayhead();
}

function syncPlayhead() {
  const t = Math.min(player.currentTime || 0, jointTMax);
  document.getElementById('chart-joints-playhead').setAttribute('x1', jointXOf(t));
  document.getElementById('chart-joints-playhead').setAttribute('x2', jointXOf(t));
  document.getElementById('chart-tcp-playhead').setAttribute('x1', tcpXOf(t));
  document.getElementById('chart-tcp-playhead').setAttribute('x2', tcpXOf(t));
}

player.addEventListener('timeupdate', syncPlayhead);
player.addEventListener('seeking', syncPlayhead);

select.addEventListener('change', () => loadEpisode(parseInt(select.value, 10)));
btnPrev.addEventListener('click', () => loadEpisode(current - 1));
btnNext.addEventListener('click', () => loadEpisode(current + 1));

function isTypingTarget(el) {
  if (!el) return false;
  const tag = el.tagName;
  return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || el.isContentEditable;
}

document.addEventListener('keydown', (e) => {
  if (isTypingTarget(document.activeElement)) return;
  if (e.key === 'ArrowRight') {
    e.preventDefault();
    loadEpisode(current + 1);
  } else if (e.key === 'ArrowLeft') {
    e.preventDefault();
    loadEpisode(current - 1);
  } else if (e.code === 'Space' || e.key === ' ') {
    e.preventDefault();
    if (player.paused) player.play(); else player.pause();
  }
});

loadEpisode(0);
</script>
"""


def _episode_sort_key(path):
    m = re.search(r'episode_(\d+)\.hdf5$', os.path.basename(path))
    return int(m.group(1)) if m else path


def find_episodes(task_dir, selection):
    all_paths = sorted(glob.glob(os.path.join(task_dir, 'episode_*.hdf5')), key=_episode_sort_key)
    if selection is None:
        return all_paths

    wanted = set()
    for part in selection.split(','):
        part = part.strip()
        if '-' in part:
            lo, hi = part.split('-')
            wanted.update(range(int(lo), int(hi) + 1))
        else:
            wanted.add(int(part))

    def idx(path):
        m = re.search(r'episode_(\d+)\.hdf5$', os.path.basename(path))
        return int(m.group(1)) if m else None

    return [p for p in all_paths if idx(p) in wanted]


def build_dataset_html(episodes_data, task_name):
    fonts_b64 = {key: _b64_file(os.path.join(FONTS_DIR, fname)) for key, fname in FONT_FILES.items()}

    html = HTML_TEMPLATE
    html = html.replace('__TITLE__', f'{task_name} — dataset report')
    html = html.replace('__TASK__', task_name)
    for key, b64 in fonts_b64.items():
        html = html.replace(f'__{key.upper()}__', b64)
    html = html.replace('__EPISODES_JSON__', json.dumps(episodes_data))
    return html


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('task_dir', help='directory containing episode_N.hdf5 files')
    parser.add_argument('--out', default=None, help='output .html path (default: <task_dir>/dataset_report.html)')
    parser.add_argument('--episodes', default=None,
                         help='which episodes to include, e.g. "1-10" or "1,3,7" (default: all)')
    parser.add_argument('--step', type=int, default=2, help='render every Nth frame in each sim video (default 2)')
    parser.add_argument('--azimuth', type=float, default=340)
    parser.add_argument('--elevation', type=float, default=-30)
    parser.add_argument('--distance', type=float, default=1.9)
    args = parser.parse_args()

    task_dir = os.path.abspath(args.task_dir)
    task_name = os.path.basename(task_dir)
    out_path = args.out or os.path.join(task_dir, 'dataset_report.html')

    episode_paths = find_episodes(task_dir, args.episodes)
    if not episode_paths:
        print(f'No episode_N.hdf5 files found in {task_dir}' + (f' matching --episodes {args.episodes}' if args.episodes else ''))
        return

    print(f'Found {len(episode_paths)} episode(s). Rendering each...')
    episodes_data = []
    for n, path in enumerate(episode_paths, 1):
        print(f'[{n}/{len(episode_paths)}] {os.path.basename(path)}')
        tmp_video = os.path.join(os.path.dirname(out_path), f'.tmp_{os.path.basename(path)}.mp4')
        vem.render(path, tmp_video, args.step, args.azimuth, args.elevation, args.distance)
        data = compute_report_data(path)
        data['video_b64'] = _b64_file(tmp_video)
        os.remove(tmp_video)
        episodes_data.append(data)

    print('Building combined report...')
    html = build_dataset_html(episodes_data, task_name)

    with open(out_path, 'w') as f:
        f.write(html)

    print(f'Saved -> {out_path}  ({os.path.getsize(out_path) / 1e6:.1f} MB)')
    print('Open it directly in a browser -- no server needed. '
          'Left/Right arrows switch episodes, Space pauses/unpauses.')


if __name__ == '__main__':
    main()
