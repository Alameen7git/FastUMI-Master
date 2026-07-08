#!/usr/bin/env python3
"""Summarize a monitor_cpu.py log: per-process avg/peak CPU% and RSS, split into
the "live capture" window (while data_collection.py was running) and the
"worker processing" window (while episode_worker.py was running), plus simple
red flags for sustained near-100% CPU or steadily growing memory.

Usage:
    python3 parse_cpu_log.py logs/cpu_monitor_<ts>.log
"""
import csv
import sys
from collections import defaultdict

SUSTAINED_CPU_THRESHOLD = 90.0   # %
SUSTAINED_CPU_FRACTION  = 0.3    # flag if >=30% of samples are above threshold
MEM_GROWTH_FLAG_RATIO   = 1.2    # flag if last quartile avg RSS > 1.2x first quartile avg


def load_rows(path):
    rows = []
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            rows.append({
                'epoch': float(r['epoch']),
                'label': r['label'],
                'pid': r['pid'],
                'cpu': float(r['cpu_percent']),
                'rss_mb': float(r['rss_mb']),
            })
    return rows


def window_for(rows, label):
    epochs = [r['epoch'] for r in rows if r['label'] == label]
    if not epochs:
        return None
    return min(epochs), max(epochs)


def in_window(epoch, window):
    return window is not None and window[0] <= epoch <= window[1]


def summarize(rows, window, window_name):
    by_label = defaultdict(list)
    for r in rows:
        if in_window(r['epoch'], window):
            by_label[r['label']].append(r)

    print(f'\n=== {window_name} '
          f'({"%.1fs" % (window[1] - window[0]) if window else "n/a"}) ===')
    if not window:
        print('  (no samples -- process never observed running)')
        return

    for label, samples in sorted(by_label.items()):
        cpus = [s['cpu'] for s in samples]
        mems = [s['rss_mb'] for s in samples]
        avg_cpu, peak_cpu = sum(cpus) / len(cpus), max(cpus)
        avg_mem, peak_mem = sum(mems) / len(mems), max(mems)
        print(f'  {label:18s}  n={len(samples):4d}  '
              f'CPU avg={avg_cpu:6.1f}%  peak={peak_cpu:6.1f}%   '
              f'RSS avg={avg_mem:7.1f}MB  peak={peak_mem:7.1f}MB')

        sustained = sum(1 for c in cpus if c >= SUSTAINED_CPU_THRESHOLD) / len(cpus)
        if sustained >= SUSTAINED_CPU_FRACTION:
            print(f'    FLAG: {sustained*100:.0f}% of samples >= {SUSTAINED_CPU_THRESHOLD:.0f}% CPU '
                  f'-- looks pegged, not just a brief spike')

        if len(mems) >= 8:
            q = len(mems) // 4
            first_q = sum(mems[:q]) / q
            last_q = sum(mems[-q:]) / q
            if first_q > 0 and last_q / first_q >= MEM_GROWTH_FLAG_RATIO:
                print(f'    FLAG: RSS grew {first_q:.1f}MB -> {last_q:.1f}MB '
                      f'({last_q/first_q:.2f}x) over the window -- possible leak, not flat usage')


def main():
    if len(sys.argv) != 2:
        print(f'Usage: {sys.argv[0]} <cpu_monitor_log>')
        sys.exit(1)

    rows = load_rows(sys.argv[1])
    if not rows:
        print('Log is empty -- nothing to summarize.')
        sys.exit(1)

    capture_window = window_for(rows, 'data_collection')
    worker_window = window_for(rows, 'episode_worker')

    summarize(rows, capture_window, 'LIVE CAPTURE window (data_collection.py alive)')
    summarize(rows, worker_window, 'WORKER PROCESSING window (episode_worker.py alive)')

    if worker_window:
        total = worker_window[1] - worker_window[0]
        print(f'\nWorker total wall-clock (first seen -> last seen): {total:.1f}s')
        print('Compare against worker.log per-episode timings for the expected baseline.')


if __name__ == '__main__':
    main()
