#!/usr/bin/env python3
"""Generate a self-contained HTML report for one recorded episode: the real
camera footage next to the simulated UR7e replay (via visualize_episode_mujoco.py),
plus synced joint-angle and TCP-position trajectory charts.

Everything (fonts, video, data) is embedded in the output HTML -- just open it
in a browser (double-click, or `file://` it), no server needed.

Usage:
    python3 render_episode_report.py <path/to/episode_N.hdf5>
    python3 render_episode_report.py <path/to/episode_N.hdf5> --out report.html
    python3 render_episode_report.py <path/to/episode_N.hdf5> --step 1 --keep-video
"""
import argparse
import base64
import json
import os
import sys

import mujoco
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in [_ROOT] + [os.path.join(_ROOT, _d) for _d in ('conversion', 'viz', 'replay_pipeline', 'replay', 'lib')]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import data_processing_to_joint as dpj
import visualize_episode_mujoco as vem

REPO_DIR = _ROOT
FONTS_DIR = os.path.join(REPO_DIR, 'assets', 'fonts')

FONT_FILES = {
    'sans_cond_600': 'plex_sans_cond_600.woff2',
    'sans_400': 'plex_sans_400.woff2',
    'sans_600': 'plex_sans_600.woff2',
    'mono_400': 'plex_mono_400.woff2',
    'mono_500': 'plex_mono_500.woff2',
}

JOINT_NAMES = ['shoulder_pan', 'shoulder_lift', 'elbow', 'wrist_1', 'wrist_2', 'wrist_3']


def _b64_file(path):
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('ascii')


def compute_report_data(episode_path):
    """Joint-angle and TCP-position trajectory, via the same transform+IK chain
    visualize_episode_mujoco.py uses for the video -- so the charts and the
    video are always showing the same solve, not two different computations."""
    config = dpj.config
    joint_traj, images, raw_t265_pos = vem.compute_joint_trajectory(episode_path, config)
    fps, measured = vem._episode_fps(episode_path, joint_traj.shape[0])

    model = mujoco.MjModel.from_xml_path(vem.SCENE_XML)
    data = mujoco.MjData(model)
    tcp_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'tcp')
    model.site_pos[tcp_site] = [0, 0, config['distances']['flange_to_tcp']]

    N = joint_traj.shape[0]
    tcp_pos = np.zeros((N, 3))
    for i in range(N):
        data.qpos[:6] = joint_traj[i]
        mujoco.mj_forward(model, data)
        tcp_pos[i] = data.site_xpos[tcp_site]

    task_name = os.path.basename(os.path.dirname(os.path.abspath(episode_path)))
    episode_name = os.path.basename(episode_path)

    return {
        'task': task_name,
        'episode': episode_name,
        'n_frames': N,
        'fps': fps,
        'fps_measured': bool(measured),
        'duration_s': N / fps,
        'joint_names': JOINT_NAMES,
        'joint_deg': np.degrees(joint_traj).round(3).tolist(),
        'tcp_pos_m': tcp_pos.round(5).tolist(),
        'preferred_branch': int(dpj._PREFERRED_BRANCH),
    }


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
  --accent: #3d7ba0; --good: #2f8f5f; --good-soft: #e3f3ea; --warn: #b5791f; --warn-soft: #f7ecd9;
  --shadow: 0 1px 2px rgba(20, 23, 28, 0.06), 0 4px 16px rgba(20, 23, 28, 0.05);
  color-scheme: light dark;
}
@media (prefers-color-scheme: dark) {
  :root { --bg: #14171c; --panel: #1c2129; --border: #2a313c; --text: #e8ecf1; --text-muted: #8b95a3;
    --accent: #7dadcc; --good: #5fbf8f; --good-soft: #1c2e26; --warn: #e0a64d; --warn-soft: #2e2618;
    --shadow: 0 1px 2px rgba(0,0,0,0.4), 0 8px 24px rgba(0,0,0,0.3); }
}
:root[data-theme="dark"] { --bg: #14171c; --panel: #1c2129; --border: #2a313c; --text: #e8ecf1; --text-muted: #8b95a3;
  --accent: #7dadcc; --good: #5fbf8f; --good-soft: #1c2e26; --warn: #e0a64d; --warn-soft: #2e2618;
  --shadow: 0 1px 2px rgba(0,0,0,0.4), 0 8px 24px rgba(0,0,0,0.3); }
:root[data-theme="light"] { --bg: #f4f6f8; --panel: #ffffff; --border: #d8dde3; --text: #1a1f26; --text-muted: #5b6472;
  --accent: #3d7ba0; --good: #2f8f5f; --good-soft: #e3f3ea; --warn: #b5791f; --warn-soft: #f7ecd9;
  --shadow: 0 1px 2px rgba(20,23,28,0.06), 0 4px 16px rgba(20,23,28,0.05); }

* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text); font-family: 'Plex Sans', system-ui, sans-serif; font-size: 15px; line-height: 1.5; }
.wrap { max-width: 1080px; margin: 0 auto; padding: 28px 24px 64px; }
.eyebrow { font-family: 'Plex Mono', monospace; font-size: 12px; letter-spacing: 0.08em; text-transform: uppercase; color: var(--accent); margin: 0 0 6px; }
h1 { font-family: 'Plex Sans Cond', 'Plex Sans', sans-serif; font-weight: 600; font-size: 26px; letter-spacing: -0.01em; margin: 0 0 4px; text-wrap: balance; word-break: break-word; }
.subtitle { color: var(--text-muted); font-size: 14px; margin: 0 0 24px; font-family: 'Plex Mono', monospace; }
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
  <p class="eyebrow">FastUMI &middot; UR7e episode report</p>
  <h1>__TASK__ — __EPISODE__</h1>
  <p class="subtitle">Real T265 demonstration vs. simulated UR7e trajectory</p>

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
    <video id="player" controls>
      <source src="data:video/mp4;base64,__VIDEO__" type="video/mp4">
    </video>
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

  <footer>generated by render_episode_report.py</footer>
</div>

<script>
const DATA = __TRAJ_JSON__;

const COLORS = ['#e6753a', '#3d9e6b', '#3d7ba0', '#b5459a', '#8a7a2e', '#7a5ce0'];
const TCP_COLORS = ['#e6753a', '#3d9e6b', '#3d7ba0'];
const TCP_LABELS = ['x', 'y', 'z'];

document.getElementById('stat-frames').textContent = DATA.n_frames;
document.getElementById('stat-duration').textContent = DATA.duration_s.toFixed(2) + ' s';
document.getElementById('stat-fps').textContent = DATA.fps.toFixed(2);
const fpsPill = document.getElementById('stat-fps-pill');
fpsPill.textContent = DATA.fps_measured ? 'measured' : 'fallback';
fpsPill.className = 'pill ' + (DATA.fps_measured ? 'good' : 'warn');
document.getElementById('stat-branch').textContent = '#' + DATA.preferred_branch;

const times = DATA.joint_deg.map((_, i) => i / DATA.fps);

function buildChart(svgId, legendId, series) {
  const svg = document.getElementById(svgId);
  const W = 960, H = svg.viewBox.baseVal.height;
  const padL = 44, padR = 14, padT = 14, padB = 26;
  const plotW = W - padL - padR, plotH = H - padT - padB;

  let allVals = series.flatMap(s => s.values);
  let vMin = Math.min(...allVals), vMax = Math.max(...allVals);
  if (vMin === vMax) { vMin -= 1; vMax += 1; }
  const pad = (vMax - vMin) * 0.08;
  vMin -= pad; vMax += pad;

  const tMax = times[times.length - 1] || 1;
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

  series.forEach(s => {
    const pts = s.values.map((v, i) => `${xOf(times[i])},${yOf(v)}`).join(' ');
    svgContent += `<polyline points="${pts}" fill="none" stroke="${s.color}" stroke-width="1.8" data-series="${s.key}"/>`;
    const lastV = s.values[s.values.length - 1];
    svgContent += `<circle class="endpoint" cx="${xOf(tMax)}" cy="${yOf(lastV)}" fill="${s.color}" data-series="${s.key}"/>`;
  });

  svgContent += `<line class="playhead" id="${svgId}-playhead" x1="${padL}" y1="${padT}" x2="${padL}" y2="${padT + plotH}"/>`;
  svg.innerHTML = svgContent;

  const legend = document.getElementById(legendId);
  legend.innerHTML = series.map(s => `
    <button data-series="${s.key}"><span class="swatch" style="background:${s.color}"></span>${s.label}</button>`).join('');
  legend.querySelectorAll('button').forEach(btn => {
    btn.addEventListener('click', () => {
      const key = btn.dataset.series;
      const on = !btn.classList.contains('off');
      btn.classList.toggle('off', on);
      svg.querySelectorAll(`[data-series="${key}"]`).forEach(el => { el.style.display = on ? 'none' : ''; });
    });
  });

  return { xOf, tMax };
}

const jointSeries = DATA.joint_names.map((name, j) => ({
  key: name, label: name.replace('_', ' '), color: COLORS[j % COLORS.length],
  values: DATA.joint_deg.map(row => row[j]),
}));
const jointChart = buildChart('chart-joints', 'legend-joints', jointSeries);

const tcpSeries = TCP_LABELS.map((label, j) => ({
  key: label, label: label, color: TCP_COLORS[j],
  values: DATA.tcp_pos_m.map(row => row[j]),
}));
const tcpChart = buildChart('chart-tcp', 'legend-tcp', tcpSeries);

const video = document.getElementById('player');
function syncPlayhead() {
  const t = Math.min(video.currentTime, jointChart.tMax);
  [[jointChart, 'chart-joints'], [tcpChart, 'chart-tcp']].forEach(([c, svgId]) => {
    const ph = document.getElementById(svgId + '-playhead');
    const x = c.xOf(t);
    ph.setAttribute('x1', x);
    ph.setAttribute('x2', x);
  });
}
video.addEventListener('timeupdate', syncPlayhead);
video.addEventListener('seeking', syncPlayhead);
syncPlayhead();
</script>
"""


def build_html(traj_data, video_path):
    fonts_b64 = {key: _b64_file(os.path.join(FONTS_DIR, fname)) for key, fname in FONT_FILES.items()}
    video_b64 = _b64_file(video_path)

    title = f"{traj_data['task']} — {traj_data['episode']}"
    html = HTML_TEMPLATE
    html = html.replace('__TITLE__', title)
    html = html.replace('__TASK__', traj_data['task'])
    html = html.replace('__EPISODE__', traj_data['episode'])
    for key, b64 in fonts_b64.items():
        html = html.replace(f'__{key.upper()}__', b64)
    html = html.replace('__VIDEO__', video_b64)
    html = html.replace('__TRAJ_JSON__', json.dumps(traj_data))
    return html


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('episode', help='path to episode_N.hdf5')
    parser.add_argument('--out', default=None, help='output .html path (default: <episode>_report.html)')
    parser.add_argument('--step', type=int, default=2, help='render every Nth frame in the sim video (default 2)')
    parser.add_argument('--azimuth', type=float, default=340)
    parser.add_argument('--elevation', type=float, default=-30)
    parser.add_argument('--distance', type=float, default=1.9)
    parser.add_argument('--keep-video', action='store_true',
                         help='keep the intermediate side-by-side .mp4 next to the report (default: delete after embedding)')
    args = parser.parse_args()

    out_path = args.out or os.path.splitext(args.episode)[0] + '_report.html'
    video_path = os.path.splitext(out_path)[0] + '.mp4'

    print('Rendering side-by-side video...')
    vem.render(args.episode, video_path, args.step, args.azimuth, args.elevation, args.distance)

    print('Computing trajectory data...')
    traj_data = compute_report_data(args.episode)

    print('Embedding into report...')
    html = build_html(traj_data, video_path)

    with open(out_path, 'w') as f:
        f.write(html)

    if not args.keep_video:
        os.remove(video_path)

    print(f'Saved -> {out_path}  ({os.path.getsize(out_path) / 1e6:.1f} MB)')
    print('Open it directly in a browser -- everything is embedded, no server needed.')


if __name__ == '__main__':
    main()
