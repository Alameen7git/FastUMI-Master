#!/usr/bin/env python3
"""
Check the real frame rate of recorded episodes.

video.mp4 always says 60 fps (data_collection.py writes it with a fixed rate),
so this measures what actually arrived, from each frame's ROS timestamp in
raw/episode_N/timestamps.csv, and the T265 rate from trajectory.csv.

    python3 viz/check_fps.py                    # newest dataset in data_dir
    python3 viz/check_fps.py UMI_2026-11-05_T1_N1
    python3 viz/check_fps.py UMI_2026-11-05_T1_N1 --episode 3
"""
import argparse
import csv
import json
import os
import re
import statistics
import subprocess
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TARGET_FPS = 60.0
T265_HZ = 200.0


def data_dir():
    with open(os.path.join(REPO, 'config', 'config.json')) as f:
        return json.load(f)['device_settings']['data_dir']


def read_stamps(path):
    with open(path) as f:
        rows = csv.reader(f)
        next(rows, None)                       # header
        return [float(r[-1] if 'timestamps' in path else r[0]) for r in rows if r]


def mp4_frames(path):
    try:
        out = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v', '-count_packets',
                              '-show_entries', 'stream=nb_read_packets', '-of', 'csv=p=0', path],
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=60)
        return int(out.stdout.strip())
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def rate(stamps, nominal):
    """fps, median gap (ms), worst gap (ms), estimated missing frames."""
    if len(stamps) < 2:
        return None
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    span = stamps[-1] - stamps[0]
    # Missing = frames a steady stream would have had in this time, minus the
    # ones that arrived. Counting long gaps instead would also count jitter
    # (a late frame followed by an early one), which loses nothing.
    expected = int(round(span * nominal)) + 1
    missing = max(0, expected - len(stamps))
    return {'fps': (len(stamps) - 1) / span if span > 0 else 0.0, 'secs': span,
            'median_ms': statistics.median(gaps) * 1000, 'worst_ms': max(gaps) * 1000,
            'missing': missing, 'expected': max(expected, len(stamps))}


def check_episode(ep_dir):
    n = int(re.search(r'(\d+)$', ep_dir).group(1))
    cam = rate(read_stamps(os.path.join(ep_dir, 'timestamps.csv')), TARGET_FPS)
    traj_path = os.path.join(ep_dir, 'trajectory.csv')
    traj = rate(read_stamps(traj_path), T265_HZ) if os.path.exists(traj_path) else None
    frames = mp4_frames(os.path.join(ep_dir, 'video.mp4'))
    stamps = len(read_stamps(os.path.join(ep_dir, 'timestamps.csv')))
    if cam is None:
        print('  episode {:>3}: too few frames'.format(n))
        return False
    good = cam['fps'] >= TARGET_FPS * 0.97 and cam['missing'] <= max(1, cam['expected'] // 500)
    print('  episode {:>3}: {:6.2f} fps  {:5.1f}s  {:>5} frames  median {:5.2f} ms  worst {:6.1f} ms  '
          'missing {:>3} ({:.1f}%)  T265 {}  {}'.format(
              n, cam['fps'], cam['secs'], stamps, cam['median_ms'], cam['worst_ms'], cam['missing'],
              100.0 * cam['missing'] / cam['expected'],
              '{:5.1f} Hz'.format(traj['fps']) if traj else '  n/a  ',
              'OK' if good else 'LOW'))
    if frames is not None and frames != stamps:
        print('               video.mp4 has {} frames but {} timestamps'.format(frames, stamps))
    return good


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument('dataset', nargs='?', help='dataset folder name or path (default: newest)')
    ap.add_argument('--episode', type=int, help='only this episode')
    args = ap.parse_args()

    root = data_dir()
    if args.dataset:
        ds = args.dataset if os.path.isdir(args.dataset) else os.path.join(root, args.dataset)
    else:
        cands = [os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d, 'raw'))]
        if not cands:
            sys.exit('No datasets with a raw/ folder in {}'.format(root))
        ds = max(cands, key=lambda d: os.path.getmtime(os.path.join(d, 'raw')))
    raw = os.path.join(ds, 'raw')
    if not os.path.isdir(raw):
        sys.exit('No raw/ folder in {}'.format(ds))
    eps = sorted((d for d in os.listdir(raw) if re.match(r'episode_\d+$', d)), key=lambda d: int(d.split('_')[1]))
    if args.episode is not None:
        eps = [d for d in eps if d == 'episode_{}'.format(args.episode)]
    if not eps:
        sys.exit('No episodes found in {}'.format(raw))

    print('{}  (target {:.0f} fps; "missing" = frames a steady 60 fps would have had)'.format(os.path.basename(ds), TARGET_FPS))
    results = [check_episode(os.path.join(raw, d)) for d in eps]
    bad = results.count(False)
    print('\n{} of {} episode(s) at {:.0f} fps.'.format(len(results) - bad, len(results), TARGET_FPS)
          if not bad else '\n{} of {} episode(s) below {:.0f} fps or dropping frames.'.format(bad, len(results), TARGET_FPS))


if __name__ == '__main__':
    main()
