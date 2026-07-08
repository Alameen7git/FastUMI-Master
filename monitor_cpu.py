#!/usr/bin/env python3
"""Standalone background CPU/RSS monitor for FastUMI pipeline validation runs.

Polls every --interval seconds for a fixed set of pipeline processes, matched
by cmdline substring rather than PID since cam_capture_node.py, data_collection.py,
and episode_worker.py all start and stop over the course of a test. A process
absent from a given tick simply has no row for that tick, so the log itself
encodes when each stage was alive -- that's what the post-run summary keys off
of to split "live capture" from "worker processing".

Usage:
    python3 monitor_cpu.py [--interval 2.0] [--out logs/cpu_monitor_<ts>.log]

Stop with Ctrl+C or SIGTERM -- both flush and close the log cleanly.
"""
import argparse
import csv
import signal
import time

import psutil

MATCHERS = {
    'roscore':          lambda cmd: 'roscore' in cmd,
    'rs_t265':          lambda cmd: 'rs_t265' in cmd or 'realsense2_camera' in cmd,
    'cam_capture_node': lambda cmd: 'cam_capture_node.py' in cmd,
    'data_collection':  lambda cmd: 'data_collection.py' in cmd,
    'episode_worker':   lambda cmd: 'episode_worker.py' in cmd,
}

_stop = False


def _handle_stop(signum, frame):
    global _stop
    _stop = True


def match_label(cmdline_str):
    for label, fn in MATCHERS.items():
        if fn(cmdline_str):
            return label
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--interval', type=float, default=2.0)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    out_path = args.out or time.strftime('logs/cpu_monitor_%Y%m%d_%H%M%S.log')

    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    tracked = {}  # pid -> (psutil.Process, label)

    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp_iso', 'epoch', 'label', 'pid', 'cpu_percent', 'rss_mb'])
        f.flush()
        print(f'Logging to {out_path} every {args.interval}s (Ctrl+C to stop)')

        while not _stop:
            now = time.time()
            now_iso = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now))

            seen_pids = set()
            newly_primed = set()
            for proc in psutil.process_iter(['pid', 'cmdline']):
                try:
                    cmdline = proc.info['cmdline'] or []
                    cmd_str = ' '.join(cmdline).lower()
                    if not cmd_str:
                        continue
                    label = match_label(cmd_str)
                    if label is None:
                        continue
                    pid = proc.info['pid']
                    seen_pids.add(pid)
                    if pid not in tracked:
                        p = psutil.Process(pid)
                        p.cpu_percent(None)  # prime -- first read is always 0.0
                        tracked[pid] = (p, label)
                        newly_primed.add(pid)
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue

            # Drop processes that exited since the last tick.
            for pid in list(tracked):
                if pid not in seen_pids:
                    del tracked[pid]

            for pid, (p, label) in list(tracked.items()):
                if pid in newly_primed:
                    continue  # no valid delta yet -- skip this tick only
                try:
                    cpu = p.cpu_percent(None)
                    rss_mb = p.memory_info().rss / (1024 * 1024)
                    writer.writerow([now_iso, f'{now:.3f}', label, pid, f'{cpu:.1f}', f'{rss_mb:.1f}'])
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    del tracked[pid]
            f.flush()

            time.sleep(args.interval)

    print(f'Stopped. Log saved to {out_path}')


if __name__ == '__main__':
    main()
