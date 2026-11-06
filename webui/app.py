#!/usr/bin/env python3
"""
FastUMI operator UI — steps 1–7 of the collection workflow.

    ./bin/fastumi-ui start      (preferred: handles browser + clean shutdown)
    python3 webui/app.py        (bare server on UI_PORT)

All commands are defined in webui/settings.py.
"""
import argparse
import glob
import json
import logging
import math
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import settings as S  # noqa: E402
from jobs import FAILED, RUNNING, STARTING, Job, LogHub, Registry  # noqa: E402

from flask import Flask, jsonify, render_template, request, send_from_directory  # noqa: E402

hub = LogHub(per_source=2000)
registry = Registry()


# ── readiness / health checks ─────────────────────────────────────────────────
def port_open(port, host='127.0.0.1'):
    s = socket.socket()
    s.settimeout(0.5)
    try:
        s.connect((host, port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def ros_check(job):
    if port_open(S.ROS_MASTER_PORT):
        return True, 'master listening on :{}'.format(S.ROS_MASTER_PORT)
    return False, 'nothing listening on :{}'.format(S.ROS_MASTER_PORT)


def topic_check(topic):
    """Green only once a real message arrives — a running roslaunch whose
    device is unplugged would otherwise look healthy."""
    def check(job):
        if not port_open(S.ROS_MASTER_PORT):
            return False, 'ROS master is not reachable'
        cmd = S.ROS_PREFIX + 'timeout {t} rostopic echo -n 1 --noarr {topic}/header'.format(
            t=S.TOPIC_CHECK_TIMEOUT_S, topic=topic)
        try:
            r = subprocess.run(['bash', '-c', cmd], stdout=subprocess.DEVNULL,
                               stderr=subprocess.PIPE, timeout=S.TOPIC_CHECK_TIMEOUT_S + 3)
        except subprocess.TimeoutExpired:
            return False, 'no message on {}'.format(topic)
        if r.returncode == 0:
            return True, 'publishing on {}'.format(topic)
        return False, 'no message on {} within {}s'.format(topic, S.TOPIC_CHECK_TIMEOUT_S)
    return check


def camera_check(topic):
    """Green only for a live picture: the Elgato capture card keeps sending
    frames with no GoPro attached, so a message alone proves nothing."""
    def check(job):
        if not port_open(S.ROS_MASTER_PORT):
            return False, 'ROS master is not reachable'
        cmd = S.ROS_PREFIX + 'timeout {t} {py} -u {script} --topic {topic} --timeout {w}'.format(
            t=S.TOPIC_CHECK_TIMEOUT_S + 6, py=shlex.quote(S.FASTUMI_PY),
            script=shlex.quote(os.path.join(S.REPO_ROOT, S.CAMERA_CHECK_SCRIPT)),
            topic=shlex.quote(topic), w=S.TOPIC_CHECK_TIMEOUT_S)
        try:
            r = subprocess.run(['bash', '-c', cmd], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                               text=True, timeout=S.TOPIC_CHECK_TIMEOUT_S + 9)
        except subprocess.TimeoutExpired:
            return False, 'no picture on {}'.format(topic)
        lines = r.stdout.strip().splitlines()
        return r.returncode == 0, (lines[-1] if lines else 'camera check failed (exit {})'.format(r.returncode))
    return check


def http_check(url):
    def check(job):
        try:
            with urllib.request.urlopen(url, timeout=1.5) as r:
                return r.status == 200, 'serving {}'.format(url)
        except Exception as e:
            return False, '{} not answering ({})'.format(url, e)
    return check


# ── capture card (Elgato) ─────────────────────────────────────────────────────
def _read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return ''


def find_capture_card():
    """The capture card to open: {dev, name, serial, product, speed} or None.
    Looks through /sys for the first video capture node (index 0) of a USB
    device from S.CAPTURE_VENDORS, so any card model or serial works."""
    if S.CAPTURE_DEVICE:
        if not os.path.exists(S.CAPTURE_DEVICE):
            return None
        return {'dev': S.CAPTURE_DEVICE, 'name': S.CAPTURE_DEVICE, 'serial': '', 'product': '', 'speed': ''}
    nodes = glob.glob('/sys/class/video4linux/video*')
    for node in sorted(nodes, key=lambda n: int(re.sub(r'\D', '', os.path.basename(n)) or 0)):
        if _read(os.path.join(node, 'index')) != '0':     # index 1 is the metadata node
            continue
        usb = os.path.realpath(os.path.join(node, 'device'))
        while usb != '/' and not os.path.exists(os.path.join(usb, 'idVendor')):
            usb = os.path.dirname(usb)
        vendor = _read(os.path.join(usb, 'idVendor'))
        if vendor not in S.CAPTURE_VENDORS:
            continue
        return {'dev': '/dev/' + os.path.basename(node),
                'name': _read(os.path.join(usb, 'product')) or _read(os.path.join(node, 'name')),
                'serial': _read(os.path.join(usb, 'serial')),
                'product': '{}:{}'.format(vendor, _read(os.path.join(usb, 'idProduct'))),
                'speed': _read(os.path.join(usb, 'speed'))}
    return None


def card_words(c):
    usb = {'480': 'USB 2', '5000': 'USB 3', '10000': 'USB 3.1', '20000': 'USB 3.2'}.get(c['speed'], '')
    return '{} {}{}at {}'.format(c['name'], '(serial {}) '.format(c['serial']) if c['serial'] else '',
                                 usb + ' ' if usb else '', c['dev'])


def grace_check(seconds):
    def check(job):
        up = time.time() - (job.started_at or time.time())
        return up >= seconds, 'alive for {:.0f}s'.format(up)
    return check


# ── session bookkeeping for data collection ───────────────────────────────────
session = {'ended': False, 'task': None, 'saved': 0, 'total': 0, 'exit_code': None,
           'completed_tasks': []}
session_lock = threading.Lock()

EP_RE = re.compile(r'───\s*Episode\s+(\d+)/(\d+)')
SAVED_RE = re.compile(r'Done\.\s*\((\d+)/(\d+)')
ALLDONE_RE = re.compile(r'All done\.\s*(\d+)/(\d+) episodes saved')


def collect_line(job, line):
    p = job.progress
    m = EP_RE.search(line)
    if m:
        p.update(episode=int(m.group(1)), total=int(m.group(2)), phase='ready')
        job.mark_ready('script is waiting for Space')
    elif 'FastUMI  |  Robot:' in line:
        job.mark_ready('script started')
    elif 'Saving episode' in line or 'Stopped. Saving' in line:
        p['phase'] = 'saving'
    elif '⚠' in line:
        p['phase'] = 'paused'
        p['alert'] = line.strip()
    elif 'take discarded' in line:
        p['phase'] = 'ready'
    m = SAVED_RE.search(line) or ALLDONE_RE.search(line)
    if m:
        p['saved'] = int(m.group(1))
        p.setdefault('total', int(m.group(2)))
    if 'Session ended' in line or 'All done.' in line:
        p['phase'] = 'ended'


def collect_status(job, seg):
    if seg.startswith('● REC'):
        job.progress['phase'] = 'recording'
        job.progress.pop('alert', None)


def collect_exit(job):
    with session_lock:
        session.update(ended=True, task=job.meta.get('task'),
                       saved=job.progress.get('saved', 0),
                       total=job.meta.get('episodes'), exit_code=job.exit_code)
        if job.meta.get('task') and job.meta['task'] not in session['completed_tasks']:
            session['completed_tasks'].append(job.meta['task'])
    hub.add('UI', '[Collection] session ended — {} episode(s) saved for task "{}"'.format(
        session['saved'], session['task']))
    if job.meta.get('task') and session['saved']:
        remember_recorded(job.meta['task'])
    prep_hold[0] = 0.0
    background(start_next_prep)   # videos paused for the recording


# The dataset recorded last: pre-filled on the Sync, Convert, Push, View and
# Delete pages. Kept in RUN_DIR so it survives a UI restart.
LAST_RECORDED_FILE = os.path.join(S.RUN_DIR, 'last_recorded.json')


def remember_recorded(dataset):
    try:
        with open(LAST_RECORDED_FILE + '.tmp', 'w') as f:
            json.dump({'dataset': dataset, 'at': time.time()}, f)
        os.replace(LAST_RECORDED_FILE + '.tmp', LAST_RECORDED_FILE)
    except OSError as e:
        hub.add('UI', '[Collection] could not remember the dataset name: {}'.format(e))


def last_recorded():
    try:
        with open(LAST_RECORDED_FILE) as f:
            name = json.load(f).get('dataset')
    except (OSError, ValueError, AttributeError):
        return None
    return name if name and os.path.isdir(os.path.join(S.DATA_DIR, name)) else None


# ── progress parsers for sync / convert ───────────────────────────────────────
SYNC_FOUND_RE = re.compile(r'Found (\d+) raw episode')
SYNC_EP_RE = re.compile(r'Episode (\d+): (DONE|SKIPPED|FAILED)(.*)')
ERROR_HINTS = ('Traceback', 'Error', 'ERROR', 'No raw episodes')


def _is_error(line):
    return any(h in line for h in ERROR_HINTS)


def sync_line(job, line):
    p = job.progress
    m = SYNC_FOUND_RE.search(line)
    if m:
        p.update(total=int(m.group(1)), done=0, skipped=0, failed=0)
    m = SYNC_EP_RE.search(line)
    if m:
        k = {'DONE': 'done', 'SKIPPED': 'skipped', 'FAILED': 'failed'}[m.group(2)]
        p[k] = p.get(k, 0) + 1
        if k == 'failed':
            job.errors.append(line.strip())
    elif _is_error(line):
        job.errors.append(line.strip())
    if p.get('total'):
        n = p.get('done', 0) + p.get('skipped', 0) + p.get('failed', 0)
        p['percent'] = round(100.0 * n / p['total'])
        p['step'] = '{}/{} episodes'.format(n, p['total'])


PUSHED_RE = re.compile(r'Pushed to:\s*(https://\S+)')


def push_line(job, line):
    p = job.progress
    if '>>> PUSH' in line:
        p['stage'] = 'Uploading to Hugging Face Hub'
        p['pushing'] = True
    m = PUSHED_RE.search(line)
    if m:
        p.setdefault('pushed', []).append(m.group(1))
        p['pushing'] = False
    if job.key == 'push' and _is_error(line):
        job.errors.append(line.strip())


CONV_TOTAL_RE = re.compile(r'Converting (\d+) episode')
CONV_EP_RE = re.compile(r'^\[(\d+)/(\d+)\] episode_')
S1_TOTAL_RE = re.compile(r'(?:^|\s)episodes:\s*(\d+)')
S1_EP_RE = re.compile(r'\bepisode_\d+: T=')
S2_EP_RE = re.compile(r'\bsaved \S+\.npz')
STAGE2_RUNS = {'joint': 1, 'cart-urbase': 1, 'cart-baselink': 1, 'cart-both': 2, 'all': 3}


def convert_line(job, line):
    p = job.progress
    if job.meta.get('space') == 'pose':
        m = CONV_TOTAL_RE.search(line)
        if m:
            p.update(stage='Writing the LeRobot dataset', total=int(m.group(1)), done=0)
        m = CONV_EP_RE.search(line)
        if m:
            p['done'] = int(m.group(1))
        if _is_error(line):
            job.errors.append(line.strip())
        push_line(job, line)
        if p.get('total') and not (p.get('pushing') or p.get('pushed')):
            p['percent'] = round(100.0 * p['done'] / p['total'])
            p['step'] = '{}/{} episodes'.format(p['done'], p['total'])
        return
    if '>>> STAGE 1/2' in line:
        p.update(stage='Stage 1/2 — pose → joints (FastUMI env)', s1=0, s2=0, s2_runs=0)
    elif '>>> STAGE 2' in line:
        p['s2_runs'] = p.get('s2_runs', 0) + 1
        p['stage'] = 'Stage 2 — writing LeRobot dataset (base env), pass {}/{}'.format(
            p['s2_runs'], STAGE2_RUNS.get(job.meta.get('space'), 1))
    m = S1_TOTAL_RE.search(line)
    if m:
        p['total'] = int(m.group(1))
    if S1_EP_RE.search(line):
        p['s1'] = p.get('s1', 0) + 1
        if 'VIOLATIONS' in line:   # unreachable frames: check the calibration before replaying it
            job.errors.append(line.strip())
    if S2_EP_RE.search(line):
        p['s2'] = p.get('s2', 0) + 1
    if _is_error(line):
        job.errors.append(line.strip())
    push_line(job, line)
    if p.get('pushing') or p.get('pushed'):
        return
    total = p.get('total')
    if total:
        runs = STAGE2_RUNS.get(job.meta.get('space'), 1)
        units = total * (1 + runs)
        p['percent'] = min(100, round(100.0 * (p.get('s1', 0) + p.get('s2', 0)) / units))
        p['step'] = 'stage 1: {}/{}  ·  stage 2: {}/{}'.format(
            p.get('s1', 0), total, p.get('s2', 0), total * runs)


# replay/replay_lerobot.py (Robot page): dry run, --check, --execute.
REPLAY_EP_RE = re.compile(r' episode (\d+): (\d+) frames, ([\d.]+)s, (\d+) waypoint')
REPLAY_DUR_RE = re.compile(r'(?:EFFECTIVE playback duration|estimated playback duration): ([\d.]+)s')
REPLAY_PEAK_RE = re.compile(r'effective peak \|velocity\|: ([\d.]+) deg/s')
REPLAY_JOINTS_RE = re.compile(r'^\s*(Current robot joints|Episode start|first waypoint) \(deg\): (\[.*\])')
REPLAY_DIFF_RE = re.compile(r'Largest joint difference: ([\d.]+) deg')
REPLAY_MODE_RE = re.compile(r'Robot mode: (.+?)\s+Safety: (.+)$')
REPLAY_APPROACH_RE = re.compile(r'max joint delta ([\d.]+) deg\) at speed=([\d.]+) rad/s')
REPLAY_STREAM_RE = re.compile(r'Streaming (\d+) servoJ waypoints at ([\d.]+) Hz')
REPLAY_PLOT_RE = re.compile(r'plot -> (.+\.png)$')
REPLAY_ERROR_HINTS = ERROR_HINTS + ('ABORT', 'Refusing', 'Aborted', 'Could not', 'not found', 'No robot IP')


def replay_line(job, line):
    p = job.progress
    m = REPLAY_EP_RE.search(line)
    if m:
        p.update(frames=int(m.group(2)), secs=float(m.group(3)), outside=int(m.group(4)))
    m = REPLAY_DUR_RE.search(line)
    if m:
        p['duration'] = float(m.group(1))
    m = REPLAY_PEAK_RE.search(line)
    if m:
        p['peak_vel'] = float(m.group(1))
    m = REPLAY_JOINTS_RE.search(line)
    if m:
        key = {'Current robot joints': 'current', 'Episode start': 'start', 'first waypoint': 'start'}[m.group(1)]
        p[key] = json.loads(m.group(2))
    m = REPLAY_DIFF_RE.search(line)
    if m:
        p['diff'] = float(m.group(1))
    m = REPLAY_MODE_RE.search(line)
    if m:
        p.update(robot_mode=m.group(1).strip(), safety=m.group(2).strip())
    m = REPLAY_APPROACH_RE.search(line)
    if m:
        p.update(phase='approach', phase_t=time.time(),
                 phase_s=math.radians(float(m.group(1))) / max(float(m.group(2)), 1e-6))
    elif 'Reached trajectory start' in line:
        p.update(phase='reached', phase_t=time.time())
    m = REPLAY_STREAM_RE.search(line)
    if m:
        p.update(phase='playing', phase_t=time.time(), phase_s=int(m.group(1)) / float(m.group(2)))
    if 'Playback complete' in line:
        p['phase'] = 'complete'
    elif 'Interrupted' in line:
        p['phase'] = 'interrupted'
    m = REPLAY_PLOT_RE.search(line)
    if m:
        p['plot'] = os.path.basename(m.group(1).strip())
    if any(h in line for h in REPLAY_ERROR_HINTS):
        job.errors.append(line.strip())


def replay_exit(job):
    """Remember a clean dry run / robot check: Run needs both, for the same episode."""
    if job.state == 'done' and job.meta.get('mode') in ('dry', 'check'):
        replay_passed[job.meta['mode']] = job.meta['key']


PREP_TOTAL_RE = re.compile(r'Preparing (\d+) episode video')
PREP_EP_RE = re.compile(r'^\[(\d+)/(\d+)\] episode_')


def vizprep_line(job, line):
    p = job.progress
    m = PREP_TOTAL_RE.search(line)
    if m:
        p.update(stage='Preparing viewer videos', total=int(m.group(1)), done=0, percent=0)
    m = PREP_EP_RE.search(line)
    if m:
        p.update(done=int(m.group(1)), total=int(m.group(2)))
        p['percent'] = round(100.0 * p['done'] / p['total'])
        p['step'] = '{}/{} videos'.format(p['done'], p['total'])
    if 'FAILED' in line:
        job.errors.append(line.strip())


# ── viewer videos made ahead of time (after Sync) ─────────────────────────────
prep_queue = []
prep_lock = threading.Lock()
prep_hold = [0.0]   # no new start before this time (a recording is starting)


def prepare_videos(task):
    """Queue a dataset for visualize_dataset.py --prepare."""
    if not (S.VIZ_PREPARE_AFTER_SYNC and S.VIZ_SCRIPT and task):
        return
    with prep_lock:
        if task not in prep_queue:
            prep_queue.append(task)
    start_next_prep()


def start_next_prep():
    """Run the next queued dataset, unless recording (the recorder gets the CPU)."""
    with prep_lock:
        job = JOBS['vizprep']
        if not prep_queue or job.alive() or JOBS['collection'].alive() or time.time() < prep_hold[0]:
            return
        task = prep_queue.pop(0)
        cmd = S.FASTUMI_PREFIX + 'exec nice -n 10 python3 -u {} --prepare {}'.format(S.VIZ_SCRIPT, shlex.quote(task))
        good, why = job.start(cmd, meta={'task': task})
        if not good:
            prep_queue.insert(0, task)


def pause_prep_for_recording():
    prep_hold[0] = time.time() + 15
    job = JOBS['vizprep']
    if job.alive():
        with prep_lock:
            task = job.meta.get('task')
            if task and task not in prep_queue:
                prep_queue.insert(0, task)   # finished videos stay cached; the rest are done later
        hub.add('UI', '[Viz] pausing the viewer videos while recording')
        background(job.stop)


def node_exit(job):
    """A machine process died. If a recording session is running, remember
    it so the Record page can offer 'restart this node' or 'save & finish'."""
    if JOBS['collection'].alive() and not job._stop_requested:
        with session_lock:
            lost = session.setdefault('lost_nodes', [])
            if job.key not in lost:
                lost.append(job.key)
        hub.add('UI', '[Collection] WARNING: {} stopped during the recording session ({})'.format(
            job.tag, job.reason))


# ── tracked jobs ──────────────────────────────────────────────────────────────
JOBS = {
    'ros': Job('ros', 'ROS', hub, registry, ready_check=ros_check, health_check=ros_check,
               ready_timeout=S.READY_TIMEOUT_S['ros'], on_exit=node_exit),
    't265': Job('t265', 'T265', hub, registry, ready_check=topic_check(S.T265_TOPIC),
                health_check=topic_check(S.T265_TOPIC), ready_timeout=S.READY_TIMEOUT_S['t265'],
                on_exit=node_exit),
    'gopro': Job('gopro', 'GoPro', hub, registry, ready_check=camera_check(S.GOPRO_TOPIC),
                 health_check=camera_check(S.GOPRO_TOPIC), ready_timeout=S.READY_TIMEOUT_S['gopro'],
                 on_exit=node_exit),
    'collection': Job('collection', 'Collection', hub, registry, use_pty=True,
                      ready_check=grace_check(4), ready_timeout=60,
                      stop_key=S.COLLECT_KEYS['e'], stop_grace=S.COLLECT_STOP_GRACE_S,
                      on_line=collect_line, on_status=collect_status, on_exit=collect_exit),
    'sync': Job('sync', 'Sync', hub, registry, kind='batch', on_line=sync_line,
                on_exit=lambda j: j.state == 'done' and prepare_videos(j.meta.get('task'))),
    'vizprep': Job('vizprep', 'Viz', hub, registry, kind='batch', on_line=lambda j, l: vizprep_line(j, l),
                   on_exit=lambda j: start_next_prep()),
    'convert': Job('convert', 'Convert', hub, registry, kind='batch', on_line=convert_line,
                   on_exit=lambda j: remember_push(j)),
    'viz': Job('viz', 'Viz', hub, registry,
               ready_check=http_check('http://127.0.0.1:{}/api/dataset'.format(S.VIZ_PORT)),
               health_check=http_check('http://127.0.0.1:{}/api/dataset'.format(S.VIZ_PORT)),
               ready_timeout=30),
    'preview': Job('preview', 'Preview', hub, registry,
                   ready_check=http_check('http://127.0.0.1:{}/snapshot.jpg'.format(S.PREVIEW_PORT)),
                   ready_timeout=20),
    'push': Job('push', 'Push', hub, registry, kind='batch', on_line=push_line,
                on_exit=lambda j: remember_push(j)),
    # pty: replay_ur7e.py reads its typed CONFIRM_FAST from the terminal
    'replay': Job('replay', 'Replay', hub, registry, kind='batch', use_pty=True,
                  on_line=replay_line, on_exit=replay_exit),
}
replay_passed = {}   # mode -> key of the last clean dry run / robot check
NODE_ORDER = ['ros', 't265', 'gopro']           # start order; stop is reversed
NODE_CMDS = {
    'ros': S.ROS_PREFIX + 'exec ' + S.ROS_CMD,
    't265': S.ROS_PREFIX + 'exec ' + S.T265_CMD,
    'gopro': S.ROS_PREFIX + 'exec ' + S.GOPRO_CMD,   # {device}, {pixel_format} filled in by start_gopro()
}


def is_green(key):
    return JOBS[key].state == RUNNING and JOBS[key].alive()


def monitor():
    """Readiness polling for starting jobs, periodic health for running ones."""
    tick = 0
    while True:
        for job in JOBS.values():
            if job.state == STARTING:
                threading.Thread(target=job.poll_ready, daemon=True).start()
            elif job.key in ('t265', 'gopro', 'preview') and JOBS['collection'].alive():
                # data_collection.py runs its own feed watchdog; don't add a
                # second subscriber to the 1080p image topic while recording.
                continue
            elif tick % 3 == 0:
                threading.Thread(target=job.poll_health, daemon=True).start()
        tick += 1
        time.sleep(2)


# ── node control ──────────────────────────────────────────────────────────────
sequence_lock = threading.Lock()


def node_precondition(key):
    if key == 'ros':
        if not JOBS['ros'].alive() and port_open(S.ROS_MASTER_PORT):
            return ('A ROS master is already running on :{} outside this UI '
                    '(old start_collection.sh terminal?). Stop it first, e.g. ./stop_collection.sh'
                    ).format(S.ROS_MASTER_PORT)
    elif key in ('t265', 'gopro'):
        if not is_green('ros'):
            return 'ROS must be green before starting {}'.format(JOBS[key].tag)
    return None


def start_node(key):
    why = node_precondition(key)
    if why:
        hub.add('UI', '[{}] not started: {}'.format(JOBS[key].tag, why))
        return False, why
    if key == 'gopro':
        return start_gopro()
    return JOBS[key].start(NODE_CMDS[key])


# The camera follows the capture card: card_watch() restarts it when the card
# is swapped or re-plugged, and starts it when a card arrives after Start was
# pressed with none plugged in. Only while the camera is wanted (started and
# not stopped by the operator), and never while recording.
card = {'now': None, 'used': None, 'wanted': False}


def start_gopro():
    c = find_capture_card()
    card['wanted'] = True
    card['now'] = c
    if c is None:
        card['used'] = None
        why = 'no capture card found: plug in the Elgato card (the camera starts by itself once it is plugged in)'
        hub.add('UI', '[GoPro] not started: ' + why)
        return False, why
    card['used'] = (c['serial'], c['dev'])
    fmt = pixel_format(c)
    hub.add('UI', '[GoPro] capture card: {}, sending {}'.format(card_words(c), fmt.upper()))
    return JOBS['gopro'].start(NODE_CMDS['gopro'].format(device=shlex.quote(c['dev']), pixel_format=fmt))


def pixel_format(c):
    """YUYV on USB 3 (no JPEG decoding, full 60 fps), MJPEG on USB 2 (see settings)."""
    if S.PIXEL_FORMAT:
        return S.PIXEL_FORMAT
    try:
        usb3 = int(c['speed']) >= 5000
    except (TypeError, ValueError):
        usb3 = False
    return S.PIXEL_FORMAT_USB3 if usb3 else S.PIXEL_FORMAT_USB2


def card_watch():
    last = None
    changed_at = 0.0
    while True:
        c = find_capture_card()
        card['now'] = c
        now = (c['serial'], c['dev']) if c else None
        if now != last:
            hub.add('UI', '[GoPro] capture card {}'.format(
                'plugged in: ' + card_words(c) if c else 'unplugged'))
            last, changed_at = now, time.time()
        job = JOBS['gopro']
        if (card['wanted'] and not sequence_lock.locked() and not JOBS['collection'].alive()
                and now != card['used']):
            if now is None:
                if job.alive():
                    hub.add('UI', '[GoPro] stopping the camera; it starts again when a card is plugged in')
                    job.stop()
                card['used'] = None
            elif time.time() - changed_at >= S.CARD_SETTLE_S and is_green('ros'):
                hub.add('UI', '[GoPro] {} the camera on the new capture card'.format(
                    'restarting' if job.alive() else 'starting'))
                if job.alive():
                    job.stop()
                    time.sleep(0.5)
                if not start_gopro()[0]:
                    card['used'] = now     # don't retry in a loop; the node shows why it failed
        time.sleep(S.CARD_POLL_S)


def wait_state(key, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = JOBS[key].state
        if st == RUNNING:
            return True
        if st == FAILED or not JOBS[key].alive():
            return False
        time.sleep(0.3)
    return False


def start_all():
    with sequence_lock:
        hub.add('UI', '=== Start All: ROS → T265 → GoPro ===')
        for i, key in enumerate(NODE_ORDER):
            if is_green(key):
                continue
            ok, why = True, ''
            if JOBS[key].state != STARTING:
                if JOBS[key].alive():      # alive but red: restart it
                    JOBS[key].stop()
                ok, why = start_node(key)
            if not ok or not wait_state(key, S.READY_TIMEOUT_S[key] + 5):
                hub.add('UI', '=== Start All aborted at {}: {} ==='.format(
                    JOBS[key].tag, why or JOBS[key].reason or 'not ready'))
                return
            if i < len(NODE_ORDER) - 1:
                time.sleep(2)
        hub.add('UI', '=== Start All: all nodes green ===')


def stop_nodes(include_collection=True):
    with sequence_lock:
        order = (['collection'] if include_collection else []) + ['preview'] + list(reversed(NODE_ORDER))
        hub.add('UI', '=== Stop: {} ==='.format(' → '.join(JOBS[k].tag for k in order)))
        card['wanted'] = False
        for key in order:
            if JOBS[key].alive():
                JOBS[key].stop()
        hub.add('UI', '=== Stop complete ===')


def restart_all():
    stop_nodes()
    time.sleep(1)
    start_all()


def background(fn, *a):
    threading.Thread(target=fn, args=a, daemon=True).start()


_shutdown_once = threading.Lock()
_shutdown_done = threading.Event()


def shutdown_everything():
    """The single shutdown path: Stop All + background jobs. Used by the
    Stop All button's big brother (/api/shutdown), SIGTERM/SIGHUP and
    bin/fastumi-ui's trap."""
    if not _shutdown_once.acquire(False):
        _shutdown_done.wait()     # another caller is already stopping everything
        return
    hub.add('UI', '=== Shutting down FastUMI UI ===')
    side = [threading.Thread(target=JOBS[k].stop) for k in ('sync', 'convert', 'viz', 'push', 'vizprep')]
    for t in side:
        t.start()
    stop_nodes()
    for t in side:
        t.join()
    try:
        os.remove(os.path.join(S.RUN_DIR, 'server.pid'))
    except OSError:
        pass
    _shutdown_done.set()


# ── datasets on disk ──────────────────────────────────────────────────────────
NAME_OK_RE = re.compile(r'^[A-Za-z0-9_.\-]+$')


def validate_name(name, what='Task name'):
    """Returns None if fine, else {'error', 'suggestion'} — mirrored in common.js."""
    raw = name or ''
    if not raw.strip():
        return {'error': '{} is required.'.format(what), 'suggestion': None}
    if re.search(r'\s', raw):
        return {'error': "{} can't contain spaces — use underscores instead.".format(what),
                'suggestion': re.sub(r'\s+', '_', raw.strip()), 'typed': raw}
    if not NAME_OK_RE.match(raw) or raw.startswith('.'):
        return {'error': 'Use only letters, digits, underscores, dashes and dots.',
                'suggestion': re.sub(r'[^A-Za-z0-9_.\-]+', '_', raw).lstrip('.'), 'typed': raw}
    return None


def dataset_name(task_id, operator, num):
    """UMI_<date>_T<task number>_<operator><num> (settings.DATASET_NAME_FORMAT)."""
    return S.DATASET_NAME_FORMAT.format(date=time.strftime(S.DATASET_DATE_FORMAT),
                                        task_id=task_id, operator=operator, num=num)


def dataset_numbers(task_id, operator):
    """Numbers already used today by this operator for this task."""
    pattern = re.compile('^' + re.escape(dataset_name(task_id, operator, '')) + r'(\d+)$')
    names = os.listdir(S.DATA_DIR) if os.path.isdir(S.DATA_DIR) else []
    return sorted(int(m.group(1)) for m in map(pattern.match, names)
                  if m and os.path.isdir(os.path.join(S.DATA_DIR, m.group(0))))


def lerobot_info(path, name):
    entry = {'name': name, 'path': path}
    try:
        with open(os.path.join(path, 'meta', 'info.json')) as f:
            info = json.load(f)
        entry.update(episodes=info.get('total_episodes'), frames=info.get('total_frames'),
                     fps=info.get('fps'),
                     video=any(v.get('dtype') == 'video' for v in info.get('features', {}).values()))
    except (OSError, ValueError, AttributeError):
        pass
    return entry


def is_lerobot(path):
    return os.path.isfile(os.path.join(path, 'meta', 'info.json'))


def lerobot_dir(task):
    """Where Convert writes a dataset's LeRobot copy: <dataset>/lerobot."""
    return os.path.join(S.DATA_DIR, task, S.LEROBOT_SUBDIR)


def current_copy(task):
    """The dataset's LeRobot copy: <dataset>/lerobot, or an old-layout <dataset>_lerobot."""
    for path in (lerobot_dir(task), os.path.join(S.DATA_DIR, task + '_lerobot')):
        if is_lerobot(path):
            return path
    return None


def scan_tasks():
    out = {}
    if not os.path.isdir(S.DATA_DIR):
        return out
    for name in sorted(os.listdir(S.DATA_DIR)):
        d = os.path.join(S.DATA_DIR, name)
        if name.startswith('.') or not os.path.isdir(d):
            continue
        if is_lerobot(d):
            continue  # an old-layout <dataset>_lerobot folder, attached to its dataset below
        # raw/ = the original recordings, lerobot*/ = converted copies
        lerobot = [lerobot_info(os.path.join(d, sub), name + '/' + sub) for sub in sorted(os.listdir(d))
                   if sub.startswith(S.LEROBOT_SUBDIR) and is_lerobot(os.path.join(d, sub))]
        out[name] = {
            'raw': len(glob.glob(os.path.join(d, 'raw', 'episode_*'))),
            'synced': len(glob.glob(os.path.join(d, 'episode_*.hdf5'))),
            'lerobot': lerobot,
            'hub': read_export(name).get('hub'),
        }
    for name in sorted(os.listdir(S.DATA_DIR)):
        d = os.path.join(S.DATA_DIR, name)
        base = name.split('_lerobot')[0]
        if base != name and base in out and is_lerobot(d):
            out[base]['lerobot'].append(lerobot_info(d, name))
    return out


# ── what was converted / pushed (<dataset>/lerobot_export.json) ──────────────
export_lock = threading.Lock()


def read_export(task):
    try:
        with open(os.path.join(S.DATA_DIR, task, S.EXPORT_FILE)) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def write_export(task, **parts):
    with export_lock:
        rec = read_export(task)
        rec.update(parts)
        path = os.path.join(S.DATA_DIR, task, S.EXPORT_FILE)
        with open(path + '.tmp', 'w') as f:
            json.dump(rec, f, indent=2)
        os.replace(path + '.tmp', path)


# ── Flask ─────────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder='templates', static_folder='static')


def bad(msg, code=400, **extra):
    d = {'ok': False, 'error': msg}
    d.update(extra)
    return jsonify(d), code


def ok(**extra):
    d = {'ok': True}
    d.update(extra)
    return jsonify(d)


@app.route('/')
def page_nodes():
    return render_template('nodes.html', page='nodes', S=S, cmds=NODE_CMDS)


@app.route('/collect')
def page_collect():
    return render_template('collect.html', page='collect', S=S, configs=list_configs())


@app.route('/visualize')
def page_visualize():
    return render_template('visualize.html', page='visualize', S=S)


@app.route('/delete')
def page_delete():
    return render_template('delete.html', page='delete', S=S)


@app.route('/replay')
def page_replay():
    return render_template('replay.html', page='replay', S=S, configs=list_configs())


ALERT_NODE = (('GoPro', 'gopro'), ('video', 'gopro'), ('T265', 't265'), ('trajectory', 't265'))


def collect_problem(sess):
    """While recording: which machines need attention, and why."""
    job = JOBS['collection']
    if not job.alive():
        return None
    alert = job.progress.get('alert') if job.progress.get('phase') == 'paused' else None
    nodes = [k for k in NODE_ORDER if not is_green(k)]
    if alert:
        nodes += [k for word, k in ALERT_NODE if word in alert and k not in nodes]
    nodes = [k for k in NODE_ORDER if k in nodes]
    if not nodes and not alert:
        return None
    return {'nodes': nodes, 'alert': alert,
            'states': {k: JOBS[k].state for k in NODE_ORDER},
            'all_green': all(is_green(k) for k in NODE_ORDER)}


@app.route('/api/status')
def api_status():
    deps_missing = [JOBS[k].tag for k in NODE_ORDER if not is_green(k)]
    with session_lock:
        sess = dict(session)
    return jsonify({
        'jobs': {k: j.status() for k, j in JOBS.items()},
        'session': sess,
        'collect_problem': collect_problem(sess),
        'collect_blocked': ('Waiting for {} to be green.'.format(', '.join(deps_missing))
                            if deps_missing else None),
        'sequence_busy': sequence_lock.locked(),
        'capture_card': {'card': card['now'], 'waiting': card['wanted'] and card['now'] is None
                         and not JOBS['gopro'].alive()},
        'updates_pending': sorted(pending_updates),
        'last_recorded': None if JOBS['collection'].alive() else last_recorded(),
        'scripts': {'sync': S.SYNC_SCRIPT, 'convert': S.CONVERT_SCRIPT, 'replay': S.REPLAY_SCRIPT,
                    'viz': S.VIZ_SCRIPT, 'push': S.PUSH_SCRIPT},
        'now': time.time(),
        'replay_passed': replay_passed,
        'viz_url': 'http://{}:{}/'.format(request.host.split(':')[0], S.VIZ_PORT),
        'preview_url': 'http://{}:{}/stream.mjpg'.format(request.host.split(':')[0], S.PREVIEW_PORT),
    })


@app.route('/api/logs')
def api_logs():
    last, rows = hub.since(int(request.args.get('since', 0)))
    return jsonify({'last': last, 'rows': rows})


@app.route('/api/tasks')
def api_tasks():
    return jsonify({'data_dir': S.DATA_DIR, 'tasks': scan_tasks()})


# ── deleting episodes (Delete page) ───────────────────────────────────────────
# Each dataset keeps a log of what was deleted in <dataset>/deleted_episodes.json.
# convert_to_lerobot.py copies it into the LeRobot dataset (meta/fastumi_episodes.json)
# and push_to_hub.py prints it on the Hub dataset card; data_collection.py never
# reuses a deleted number. After a delete the LeRobot copy is converted again
# and, if the dataset was pushed before, pushed again (update_after_delete).
EP_FILE_RE = re.compile(r'episode_(\d+)(?:\.hdf5)?$')


def episode_numbers(task, raw=True):
    d = os.path.join(S.DATA_DIR, task)
    nums = set()
    for sub in (d, os.path.join(d, 'raw')) if raw else (d,):
        if os.path.isdir(sub):
            nums.update(int(m.group(1)) for m in map(EP_FILE_RE.match, os.listdir(sub)) if m)
    return sorted(nums)


def read_deleted(task):
    try:
        with open(os.path.join(S.DATA_DIR, task, S.DELETED_EPISODES_FILE)) as f:
            return json.load(f).get('deleted', [])
    except (OSError, ValueError):
        return []


def parse_episode_list(text):
    """'3, 7, 10-12' -> [3, 7, 10, 11, 12]; raises ValueError with a message."""
    out = set()
    for part in re.split(r'[,\s]+', str(text).strip()):
        if not part:
            continue
        m = re.fullmatch(r'(\d+)(?:-(\d+))?', part)
        if not m:
            raise ValueError('"{}" is not an episode number. Use numbers like 3, 7, 10-12.'.format(part))
        lo, hi = int(m.group(1)), int(m.group(2) or m.group(1))
        if hi < lo or hi - lo > 1000:
            raise ValueError('"{}" is not a valid range.'.format(part))
        out.update(range(lo, hi + 1))
    if not out:
        raise ValueError('Type the episode numbers to delete, e.g. 3, 7.')
    return sorted(out)


def auto_update_running(task):
    job = JOBS['convert']
    return job.alive() and job.meta.get('auto') and job.meta.get('task') == task


def busy_with(task):
    """Why this dataset's episodes can't be deleted right now, or None."""
    checks = (('collection', 'Recording'), ('sync', 'Sync'), ('convert', 'Convert'), ('push', 'Push'))
    for key, label in checks:
        job = JOBS[key]
        if job.alive() and job.meta.get('task') == task and not (key == 'convert' and job.meta.get('auto')):
            return '{} is running on "{}". Wait for it to finish.'.format(label, task)
    return None


def plan_update(task):
    """(command, job meta, note) to rewrite <dataset>/lerobot after a delete and
    push it again if it was pushed before; (None, None, why) when there is nothing to do."""
    rec = read_export(task)
    copy = current_copy(task)
    pushed = pushed_repo(task)
    if copy is None and not pushed:
        return None, None, None   # never converted: the next Convert has the deletions
    synced = episode_numbers(task, raw=False)
    if not synced:
        return None, None, 'No synced episodes are left, so the LeRobot copy was not rewritten.'
    conv = rec.get('convert') or {}
    episodes = conv.get('episodes') or ''
    if episodes:   # it was converted with "only these episodes": keep that choice
        episodes = ','.join(n for n in episodes.split(',') if n and int(n) in synced)
        if not episodes:
            return None, None, 'None of the episodes chosen for the LeRobot copy are left.'
    with_video = conv.get('with_video')
    if with_video is None:
        with_video = lerobot_info(copy, '').get('video', True) if copy else True
    push, note = None, None
    if pushed:
        push, note = hub_target(task)
        if note:
            note += ' Hugging Face was not updated.'
    cmd, outputs = convert_command(task, 'pose', conv.get('fps'), episodes, with_video, push)
    return cmd, {'task': task, 'space': 'pose', 'outputs': outputs, 'push': push, 'auto': True}, note


pending_updates = set()
pending_lock = threading.Lock()


def update_after_delete(task):
    """Start the update now, or when Convert and Push are free (another dataset may be using them)."""
    with pending_lock:
        if task in pending_updates:
            return
        pending_updates.add(task)

    def go():
        try:
            while True:
                if not (JOBS['convert'].alive() or JOBS['push'].alive()):
                    cmd, meta, note = plan_update(task)   # planned now: sees every delete so far
                    if note:
                        hub.add('UI', '[Delete] {}: {}'.format(task, note))
                    if cmd is None:
                        return
                    hub.add('UI', '[Delete] {}: rewriting the LeRobot copy{}'.format(
                        task, ' and pushing it to ' + meta['push']['repo_id'] if meta['push'] else ''))
                    if JOBS['convert'].start(cmd, meta=meta)[0]:
                        return
                time.sleep(2)
        finally:
            with pending_lock:
                pending_updates.discard(task)
    background(go)


@app.route('/api/episodes')
def api_episodes():
    task = request.args.get('task', '')
    if validate_name(task, 'Dataset name') or task not in scan_tasks():
        return bad('No dataset named "{}"'.format(task), 404)
    return ok(task=task, episodes=episode_numbers(task), deleted=read_deleted(task),
              lerobot=current_copy(task) is not None, pushed_to=pushed_repo(task),
              updating=bool(auto_update_running(task)) or task in pending_updates)


@app.route('/api/episodes/delete', methods=['POST'])
def api_episodes_delete():
    body = request.get_json(force=True, silent=True) or {}
    task = body.get('task', '')
    problem = validate_name(task, 'Dataset name')
    if problem:
        return bad(problem['error'], problem=problem)
    if task not in scan_tasks():
        return bad('No dataset named "{}".'.format(task))
    try:
        wanted = parse_episode_list(body.get('episodes', ''))
    except ValueError as e:
        return bad(str(e))
    have = episode_numbers(task)
    missing = [n for n in wanted if n not in have]
    if missing:
        return bad('"{}" has no episode {}. Its episodes are: {}.'.format(
            task, ', '.join(map(str, missing)), ', '.join(map(str, have)) or 'none'))
    why = busy_with(task)
    if why:
        return bad(why)
    if auto_update_running(task):
        # The update from the last delete: stop it, it restarts below with this delete too.
        hub.add('UI', '[Delete] {}: restarting the LeRobot / Hugging Face update'.format(task))
        JOBS['convert'].stop()
    reason = str(body.get('reason', '')).strip()[:300]
    d = os.path.join(S.DATA_DIR, task)
    log = read_deleted(task)
    now = time.strftime('%Y-%m-%d %H:%M:%S')
    for n in wanted:
        removed = []
        raw = os.path.join(d, 'raw', 'episode_{}'.format(n))
        synced = os.path.join(d, 'episode_{}.hdf5'.format(n))
        if os.path.isdir(raw):
            shutil.rmtree(raw)
            removed.append('raw')
        if os.path.isfile(synced):
            os.remove(synced)
            removed.append('synced')
        cached = os.path.join(S.DATA_DIR, '.viz_cache', task, 'episode_{}.mp4'.format(n))
        if os.path.isfile(cached):
            os.remove(cached)
        log.append({'episode': n, 'deleted_at': now, 'reason': reason, 'removed': removed})
        hub.add('UI', '[Delete] {} episode {} ({})'.format(task, n, ', '.join(removed)))
    tmp = os.path.join(d, S.DELETED_EPISODES_FILE + '.tmp')
    with open(tmp, 'w') as f:
        json.dump({'dataset': task, 'deleted': log}, f, indent=2)
    os.replace(tmp, os.path.join(d, S.DELETED_EPISODES_FILE))
    _, meta, note = plan_update(task)
    if meta:
        update_after_delete(task)
    # Copies the update does not rewrite (robot-replay output types).
    stale = [e['name'] for e in scan_tasks().get(task, {}).get('lerobot', [])
             if not meta or e['path'] != lerobot_dir(task)]
    return ok(deleted=wanted, episodes=episode_numbers(task), log=log, stale_lerobot=stale,
              update={'lerobot': bool(meta), 'repo_id': meta['push']['repo_id'] if meta and meta['push'] else None,
                      'note': note})


@app.route('/api/validate_name')
def api_validate_name():
    return jsonify({'problem': validate_name(request.args.get('name', ''))})


@app.route('/api/node/<key>/<action>', methods=['POST'])
def api_node(key, action):
    if key not in NODE_ORDER:
        return bad('unknown node {}'.format(key), 404)
    if sequence_lock.locked():
        return bad('A Start/Stop All sequence is in progress.')
    job = JOBS[key]
    if action == 'start':
        good, why = start_node(key)
        return ok() if good else bad(why)
    if action == 'stop':
        if key == 'gopro':
            card['wanted'] = False
        background(job.stop)
        return ok()
    if action == 'restart':
        def go():
            job.stop()
            time.sleep(0.5)
            if start_node(key)[0] and wait_state(key, S.READY_TIMEOUT_S[key] + 5):
                with session_lock:
                    if key in session.get('lost_nodes', []):
                        session['lost_nodes'].remove(key)
        background(go)
        return ok()
    return bad('unknown action {}'.format(action), 404)


@app.route('/api/nodes/<action>', methods=['POST'])
def api_nodes(action):
    if sequence_lock.locked():
        return bad('A Start/Stop All sequence is already in progress.')
    fn = {'start_all': start_all, 'stop_all': stop_nodes, 'restart_all': restart_all}.get(action)
    if fn is None:
        return bad('unknown action', 404)
    background(fn)
    return ok()


@app.route('/api/collect/start', methods=['POST'])
def api_collect_start():
    body = request.get_json(force=True, silent=True) or {}
    task = body.get('task', '')
    operator = body.get('operator', '')
    for value, what in ((task, 'Task name'), (operator, 'Operator name')):
        problem = validate_name(value, what)
        if problem:
            return bad(problem['error'], problem=problem)
    if operator[-1].isdigit():
        return bad("Operator name can't end with a number: the dataset number (n1, n2, …) is added for you.")
    try:
        episodes = int(body.get('episodes', S.COLLECT_DEFAULT_EPISODES))
        assert episodes > 0
    except (ValueError, TypeError, AssertionError):
        return bad('Num episodes must be a positive whole number.')
    task_id, err = ensure_task(task, episodes, body.get('task_number'))
    if err:
        return err
    used = dataset_numbers(task_id, operator)
    nxt = (max(used) if used else 0) + 1
    try:
        num = int(body.get('dataset_num') or nxt)
    except (TypeError, ValueError):
        num = 0
    if num not in used and num != nxt:
        return bad('Dataset number must be one of {} or the next one, {}.'.format(used, nxt) if used
                   else 'The first dataset number is 1.')
    missing = [JOBS[k].tag for k in NODE_ORDER if not is_green(k)]
    if missing:
        return bad('Start Recording needs {} green first.'.format(', '.join(missing)))
    dataset = dataset_name(task_id, operator, num)
    cmd = S.FASTUMI_PREFIX + 'exec python3 -u {} --task {} --num_episodes {}'.format(
        S.COLLECT_SCRIPT, shlex.quote(dataset), episodes)
    with session_lock:
        session.update(ended=False, task=dataset, saved=0, total=episodes, exit_code=None, lost_nodes=[])
    pause_prep_for_recording()
    good, why = JOBS['collection'].start(cmd, meta={'task': dataset, 'episodes': episodes,
                                                    'base_task': task, 'operator': operator})
    if not good:
        prep_hold[0] = 0.0
        background(start_next_prep)
    if good:
        remember_dataset(task_id, dataset)
    return ok(dataset=dataset) if good else bad(why)


@app.route('/api/collect/key', methods=['POST'])
def api_collect_key():
    key = (request.get_json(force=True, silent=True) or {}).get('key', '')
    ch = S.COLLECT_KEYS.get(key)
    if ch is None:
        return bad('unknown key')
    if not JOBS['collection'].send_key(ch):
        return bad('Data collection is not running.')
    hub.add('UI', '[Collection] key sent: {}'.format('SPACE' if ch == ' ' else ch.upper()))
    return ok()


@app.route('/api/collect/stop', methods=['POST'])
def api_collect_stop():
    background(JOBS['collection'].stop)
    return ok()


@app.route('/api/session/<action>', methods=['POST'])
def api_session(action):
    with session_lock:
        session['ended'] = False
    if action == 'stop_nodes':
        if sequence_lock.locked():
            return bad('A Start/Stop All sequence is already in progress.')
        background(stop_nodes)
    return ok()


def _task_from_body(body, need_key):
    task = body.get('task', '')
    problem = validate_name(task)
    if problem:
        return None, bad(problem['error'], problem=problem)
    info = scan_tasks().get(task)
    if info is None:
        return None, bad('No dataset named "{}" under {}'.format(task, S.DATA_DIR))
    if not info[need_key]:
        what = {'raw': 'raw episodes (raw/episode_N/) to synchronize',
                'synced': 'synced episodes (episode_N.hdf5) — run Sync first'}[need_key]
        return None, bad('"{}" has no {}.'.format(task, what))
    return task, None


@app.route('/api/sync/<action>', methods=['POST'])
def api_sync(action):
    job = JOBS['sync']
    if action == 'stop':
        background(job.stop)
        return ok()
    if S.SYNC_SCRIPT is None:
        return bad('convert_episodes.py not found in this checkout.')
    if JOBS['collection'].alive():
        return bad('Wait for the data collection session to finish before synchronizing.')
    body = request.get_json(force=True, silent=True) or {}
    task, err = _task_from_body(body, 'raw')
    if err:
        return err
    try:
        parallel = max(1, int(body.get('parallel', S.SYNC_DEFAULT_PARALLEL)))
    except (TypeError, ValueError):
        return bad('Parallel workers must be a whole number.')
    cmd = S.FASTUMI_PREFIX + 'exec python3 -u {} --task {} --parallel {} --fps {}{}'.format(
        S.SYNC_SCRIPT, shlex.quote(task), parallel, S.DATASET_FPS,
        ' --keep-raw' if body.get('keep_raw', True) else '')
    good, why = job.start(cmd, meta={'task': task})
    return ok() if good else bad(why)


def convert_outputs(task, space):
    """Output folders for a dataset and output type."""
    if space == 'pose':
        return [lerobot_dir(task)]
    # make_replay_dataset.sh picks its own folders (no --out): <data_dir>/<dataset>_lerobot…
    base = os.path.join(S.DATA_DIR.rstrip('/'), task) + '_lerobot'
    return {'joint': [base], 'cart-urbase': [base], 'cart-baselink': [base],
            'cart-both': [base + '_urbase', base + '_baselink'],
            'all': [base, base + '_cart_urbase', base + '_cart_baselink']}[space]


def list_configs():
    """Calibration / config files Convert can use (config/*.json), the default first."""
    names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(S.CONFIG_DIR, '*.json')))
    return sorted(names, key=lambda n: n != S.DEFAULT_CONFIG)


def convert_command(task, space='pose', fps=None, episodes='', with_video=True, push=None, config=None):
    """Shell command for Convert (+ push when push = {repo_id, public})."""
    outputs = convert_outputs(task, space)
    if space == 'pose':
        args = [S.LEROBOT_PY, '-u', S.CONVERT_SCRIPT, '--src', os.path.join(S.DATA_DIR, task),
                '--out', outputs[0], '--overwrite',
                '--width', str(S.CONVERT_IMAGE_SIZE[0]), '--height', str(S.CONVERT_IMAGE_SIZE[1])]
        if push:
            args += ['--repo-id', push['repo_id']]
        if not with_video:
            args.append('--no-video')
        if fps:
            args += ['--fps', str(fps)]
    else:
        args = ['./' + S.REPLAY_SCRIPT, '--task', task, '--space', space, '--fps', str(fps or S.DATASET_FPS)]
        if with_video:
            args.append('--with-video')
    if episodes:
        args += ['--episodes', episodes]
    desc = task_description(task)
    if desc:
        args += ['--task-name', desc]
    cmd = ' '.join(shlex.quote(a) for a in args)
    env = ''
    if space != 'pose' and config:
        env = 'export CONFIG={}; '.format(shlex.quote(os.path.join(S.CONFIG_DIR, config)))
    if push:
        return env + cmd + " && echo '>>> PUSH to Hugging Face Hub' && " + push_cmd(
            push['repo_id'], outputs[0], push['public']), outputs
    return env + 'exec ' + cmd, outputs


@app.route('/api/convert/<action>', methods=['POST'])
def api_convert(action):
    job = JOBS['convert']
    if action == 'stop':
        background(job.stop)
        return ok()
    body = request.get_json(force=True, silent=True) or {}
    task, err = _task_from_body(body, 'synced')
    if err:
        return err
    space = body.get('space', 'pose')
    if space not in S.CONVERT_SPACES:
        return bad('Output type must be one of ' + ', '.join(S.CONVERT_SPACES))
    try:
        # Empty = the rate Sync wrote into the episodes (DATASET_FPS).
        fps = int(body['fps']) if body.get('fps') else None
        assert fps is None or fps > 0
    except (TypeError, ValueError, AssertionError):
        return bad('FPS must be a positive whole number.')
    episodes = (body.get('episodes') or '').replace(' ', '')
    if episodes and not re.match(r'^\d+(,\d+)*$', episodes):
        return bad('Episodes must be a comma-separated list like 1,2,5.')
    outputs = convert_outputs(task, space)
    push = None
    if body.get('push'):
        if len(outputs) != 1:
            return bad('Pushing is only available for output types that make one dataset.')
        if S.PUSH_SCRIPT is None:
            return bad('No push script found.')
        push, why = hub_target(task, (body.get('push') or {}).get('public'))
        if why:
            return bad(why)
    existing = [o for o in outputs if os.path.exists(o)]
    if existing and not body.get('overwrite'):
        return bad('This dataset was converted before. Converting again deletes and rewrites:\n  ' +
                   '\n  '.join(existing) + '\nConfirm to overwrite.', exists=existing)
    with_video = bool(body.get('with_video', True))
    config = body.get('config') or S.DEFAULT_CONFIG
    if config not in list_configs():
        return bad('No calibration file config/{}.'.format(config))
    cmd, outputs = convert_command(task, space, fps, episodes, with_video, push, config)
    good, why = job.start(cmd, meta={'task': task, 'space': space, 'outputs': outputs, 'push': push,
                                     'config': None if space == 'pose' else config})
    if not good:
        return bad(why)
    if space == 'pose':
        write_export(task, convert={'fps': fps, 'episodes': episodes, 'with_video': with_video})
    return ok()


@app.route('/api/viz/<action>', methods=['POST'])
def api_viz(action):
    job = JOBS['viz']
    if action == 'stop':
        background(job.stop)
        return ok()
    if S.VIZ_SCRIPT is None:
        return bad('visualize_dataset.py not found in this checkout.')
    body = request.get_json(force=True, silent=True) or {}
    if body.get('task') is not None:   # operator-typed dataset name
        _, err = _task_from_body(body, 'synced')
        if err:
            return err
    if job.alive():
        return ok(already=True)
    if port_open(S.VIZ_PORT):
        return bad('Port {} is already in use by another process (a dashboard started '
                   'outside this UI?). Stop it or change VIZ_PORT in webui/settings.py.'.format(S.VIZ_PORT))
    cmd = S.FASTUMI_PREFIX + 'exec python3 -u {} --port {} --no-browser'.format(S.VIZ_SCRIPT, S.VIZ_PORT)
    good, why = job.start(cmd)
    return ok() if good else bad(why)


# ── Robot page: replay a LeRobot joint dataset on the UR7e ───────────────────
# Three steps, each a run of replay/replay_lerobot.py: a dry run (no robot), a
# read-only robot check (which robot answered, where it is), then the real run.
# The real run is refused until a dry run and a check of the same episode (and
# the same robot IP) finished cleanly, and needs the operator to type RUN (and
# CONFIRM_FAST above REPLAY_FAST_SPEED). Stop sends SIGINT: the script stops the
# arm (stopJ / servoStop). The e-stop is still the real stop.
HUB_REPO_RE = re.compile(r'^[\w.-]+/[\w.-]+$')
IP_RE = re.compile(r'^\d{1,3}(\.\d{1,3}){3}$')
JOINT_NAMES = ('UR7_joint_0', 'Joint1')
hub_info_cache = {}


def _joint_info(info):
    names = ((info.get('features') or {}).get('action') or {}).get('names') or []
    return {'episodes': info.get('total_episodes'), 'frames': info.get('total_frames'), 'fps': info.get('fps'),
            'joint': bool(names) and names[0] in JOINT_NAMES, 'action': names}


def local_joint_datasets():
    """LeRobot folders in data_dir whose action is joint angles (Convert's robot-replay output)."""
    out = []
    for info_path in sorted(glob.glob(os.path.join(S.DATA_DIR, '*', '*', 'meta', 'info.json')) +
                            glob.glob(os.path.join(S.DATA_DIR, '*', 'meta', 'info.json'))):
        try:
            with open(info_path) as f:
                info = _joint_info(json.load(f))
        except (OSError, ValueError):
            continue
        if info['joint']:
            path = os.path.dirname(os.path.dirname(info_path))
            out.append(dict(info, path=path, label=os.path.relpath(path, S.DATA_DIR)))
    return out


def replay_dataset(text):
    """(dataset argument, error) for a typed local folder / data_dir-relative folder / Hub repo id."""
    text = (text or '').strip()
    if not text:
        return None, 'Type a dataset: a Hugging Face repo id (user/name) or a folder.'
    for cand in (text, os.path.join(S.DATA_DIR, text)):
        if os.path.isfile(os.path.join(cand, 'meta', 'info.json')):
            return os.path.realpath(cand), None
    if HUB_REPO_RE.match(text):
        return text, None
    return None, 'Not a LeRobot folder or a Hugging Face repo id (user/name): {}'.format(text)


def hub_info(repo):
    if repo not in hub_info_cache:
        code = ('import json,sys;from huggingface_hub import hf_hub_download;'
                "print(json.dumps(json.load(open(hf_hub_download(sys.argv[1],'meta/info.json',repo_type='dataset')))))")
        try:
            out = subprocess.run([S.LEROBOT_PY, '-c', code, repo], stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, text=True, timeout=60)
        except subprocess.TimeoutExpired:
            return None, 'Hugging Face did not answer in 60 s.'
        if out.returncode != 0:
            last = (out.stderr.strip().splitlines() or ['unknown error'])[-1]
            return None, 'Could not read {} from Hugging Face: {}'.format(repo, last[:300])
        hub_info_cache[repo] = _joint_info(json.loads(out.stdout))
    return hub_info_cache[repo], None


def default_robot_ip():
    try:
        with open(os.path.join(S.CONFIG_DIR, S.DEFAULT_CONFIG)) as f:
            ip = json.load(f).get('ur7e_hardware', {}).get('robot_ip') or ''
    except (OSError, ValueError):
        ip = ''
    return ip if IP_RE.match(ip) else ''


@app.route('/api/replay/options')
def api_replay_options():
    return jsonify({'datasets': local_joint_datasets(), 'robot_ip': default_robot_ip(),
                    'speeds': S.REPLAY_SPEEDS, 'fast': S.REPLAY_FAST_SPEED,
                    'script': S.REPLAY_LEROBOT_SCRIPT})


@app.route('/api/replay/info')
def api_replay_info():
    ds, why = replay_dataset(request.args.get('dataset'))
    if why:
        return bad(why)
    if os.path.isdir(ds):
        try:
            with open(os.path.join(ds, 'meta', 'info.json')) as f:
                info = _joint_info(json.load(f))
        except (OSError, ValueError) as e:
            return bad('Could not read meta/info.json: {}'.format(e))
    else:
        info, why = hub_info(ds)
        if why:
            return bad(why)
    if not info['joint']:
        return bad('This is not a joint-angle dataset (action: {}). Convert it with Advanced → Output type → '
                   '"Robot replay: joint angles".'.format(', '.join(info['action'][:7]) or 'unknown'))
    return ok(dataset=ds, **info)


@app.route('/api/replay/<action>', methods=['POST'])
def api_replay(action):
    job = JOBS['replay']
    if action == 'stop':
        background(job.stop)
        return ok()
    if action != 'start':
        return bad('Unknown action.', 404)
    if S.REPLAY_LEROBOT_SCRIPT is None:
        return bad('replay/replay_lerobot.py is not in this checkout (branch ur7e-dev7.0).')
    body = request.get_json(force=True, silent=True) or {}
    mode = body.get('mode')
    if mode not in ('dry', 'check', 'run'):
        return bad('mode must be dry, check or run.')
    ds, why = replay_dataset(body.get('dataset'))
    if why:
        return bad(why)
    try:
        episode = int(body.get('episode'))
        speed = float(body.get('speed'))
        assert episode >= 0 and 0 < speed <= 1
    except (TypeError, ValueError, AssertionError):
        return bad('Episode must be a whole number from 0, and speed between 0 and 1.')
    ip = (body.get('robot_ip') or '').strip()
    if mode != 'dry' and not IP_RE.match(ip):
        return bad('Type the robot IP (from the pendant), e.g. 192.168.1.102.')
    if mode == 'run':
        if replay_passed.get('dry') != [ds, episode, speed] or replay_passed.get('check') != [ds, episode, ip]:
            return bad('Do the dry run and the robot check for this episode (and this speed and robot IP) first.')
        if body.get('confirm') != 'RUN':
            return bad('Not started: RUN was not typed.')
        if speed > S.REPLAY_FAST_SPEED and body.get('confirm_fast') != 'CONFIRM_FAST':
            return bad('Not started: CONFIRM_FAST was not typed.')
    if job.alive():
        return bad('A replay is already running. Stop it first.')
    args = [S.FASTUMI_PY, '-u', S.REPLAY_LEROBOT_SCRIPT, ds, '--episode', str(episode), '--speed', str(speed),
            '--replay-dir', S.REPLAY_OUT_DIR]
    if mode != 'dry':
        args += ['--robot-ip', ip]
    args += {'dry': [], 'check': ['--check'], 'run': ['--execute', '--yes']}[mode]
    key = {'dry': [ds, episode, speed], 'check': [ds, episode, ip], 'run': [ds, episode, speed, ip]}[mode]
    replay_passed.pop(mode, None)
    good, why = job.start('exec ' + ' '.join(shlex.quote(a) for a in args),
                          meta={'mode': mode, 'dataset': ds, 'episode': episode, 'speed': speed, 'ip': ip, 'key': key})
    if not good:
        return bad(why)
    if mode == 'run' and speed > S.REPLAY_FAST_SPEED:
        job.send_key('CONFIRM_FAST\n')   # the operator typed it on the page; replay_ur7e asks for it on the terminal
    return ok()


@app.route('/api/replay/plot/<name>')
def api_replay_plot(name):
    if not name.endswith('.png'):
        return bad('Not a plot.', 404)
    return send_from_directory(S.REPLAY_OUT_DIR, name)


# ── task list (operator-defined, numbered) ────────────────────────────────────
registry_lock = threading.Lock()


def load_registry():
    try:
        with open(S.TASKS_FILE) as f:
            reg = json.load(f)
        reg.setdefault('next', 1)
        reg.setdefault('tasks', [])
        return reg
    except (OSError, ValueError):
        return {'next': 1, 'tasks': []}


def save_registry(reg):
    os.makedirs(os.path.dirname(S.TASKS_FILE), exist_ok=True)
    tmp = S.TASKS_FILE + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(reg, f, indent=2)
    os.replace(tmp, S.TASKS_FILE)


def task_datasets(t):
    """Dataset folders recorded for a task: UMI_<date>_<task>_<operator> ones,
    plus a folder named exactly like the task (recorded before the naming rule)."""
    return [t['name']] + [d for d in t.get('datasets', []) if d != t['name']]


def registry_view():
    disk = scan_tasks()
    out = []
    for t in load_registry()['tasks']:
        found = [(n, disk[n]) for n in task_datasets(t) if n in disk]
        out.append(dict(t, raw=sum(d['raw'] for _, d in found),
                        synced=sum(d['synced'] for _, d in found),
                        recorded=sum(max(d['raw'], d['synced']) for _, d in found),
                        datasets=[{'name': n, 'raw': d['raw'], 'synced': d['synced']} for n, d in found],
                        on_disk=bool(found), name_locked=t['name'] in disk))
    return out


TASK_NUM_RE = re.compile(r'^[Tt]?\s*0*(\d+)(?:\.(\d+))?$')


def parse_task_number(value):
    """'4', 'T4', '4.1' -> '4', '4', '4.1'; raises ValueError. Typed by the operator."""
    m = TASK_NUM_RE.match(str(value if value is not None else '').strip())
    if not m:
        raise ValueError('Task number must be a number like 7 or 4.1 (the T is added for you).')
    return m.group(1) + ('.' + m.group(2) if m.group(2) is not None else '')


def number_key(num):
    whole, _, dec = str(num).partition('.')
    return (int(whole), int(dec) if dec else -1)


def task_by_number(reg, num):
    return next((t for t in reg['tasks'] if str(t['id']) == str(num)), None)


def last_numbers(reg):
    """(most recently added task number, highest task number), or Nones."""
    tasks = reg['tasks']
    if not tasks:
        return None, None
    last = max(tasks, key=lambda t: t.get('created', 0))['id']
    return str(last), str(max((t['id'] for t in tasks), key=number_key))


def new_task_number(reg, value):
    """(number, None) or (None, error response) for a number typed for a new task."""
    try:
        num = parse_task_number(value)
    except ValueError as e:
        return None, bad(str(e))
    other = task_by_number(reg, num)
    if other:
        return None, bad('T{} is already used by "{}". Pick another number.'.format(num, other['name']))
    return num, None


def ensure_task(task, episodes, number=None):
    """(task number, None); a task typed on the Record page is added to the list
    with the number the operator typed. (None, error response) if that number is bad."""
    with registry_lock:
        reg = load_registry()
        cur = next((t for t in reg['tasks'] if t['name'] == task), None)
        if cur is None:
            num, err = new_task_number(reg, number)
            if err:
                return None, err
            cur = {'name': task, 'episodes': episodes, 'description': '', 'id': num, 'created': time.time()}
            reg['tasks'].append(cur)
            save_registry(reg)
            hub.add('UI', '[Tasks] added T{} {} (typed on the Record page)'.format(cur['id'], task))
        return str(cur['id']), None


def remember_dataset(task_id, dataset):
    """Called when recording starts: file the dataset under its task."""
    with registry_lock:
        reg = load_registry()
        cur = task_by_number(reg, task_id)
        if cur is not None and dataset not in cur.setdefault('datasets', []):
            cur['datasets'].append(dataset)
            save_registry(reg)


def _task_fields(body, reg, current=None):
    name = (body.get('name') if body.get('name') is not None else (current or {}).get('name', ''))
    problem = validate_name(name)
    if problem:
        return None, bad(problem['error'], problem=problem)
    if any(t['name'] == name and t is not current for t in reg['tasks']):
        return None, bad('There is already a task called "{}".'.format(name))
    try:
        episodes = int(body.get('episodes', (current or {}).get('episodes', S.COLLECT_DEFAULT_EPISODES)))
        assert episodes > 0
    except (TypeError, ValueError, AssertionError):
        return None, bad('Episodes must be a positive whole number.')
    desc = str(body.get('description', (current or {}).get('description', ''))).strip()[:300]
    return {'name': name, 'episodes': episodes, 'description': desc}, None


@app.route('/api/registry')
def api_registry():
    return jsonify({'tasks': registry_view(), 'file': S.TASKS_FILE,
                    'today': time.strftime(S.DATASET_DATE_FORMAT), 'name_format': S.DATASET_NAME_FORMAT,
                    'last_number': last_numbers(load_registry())[0],
                    'highest_number': last_numbers(load_registry())[1]})


@app.route('/api/registry', methods=['POST'])
def api_registry_add():
    body = request.get_json(force=True, silent=True) or {}
    with registry_lock:
        reg = load_registry()
        fields, err = _task_fields(body, reg)
        if err:
            return err
        num, err = new_task_number(reg, body.get('number'))
        if err:
            return err
        task = dict(fields, id=num, created=time.time())
        reg['tasks'].append(task)
        save_registry(reg)
    hub.add('UI', '[Tasks] added T{} {}'.format(task['id'], task['name']))
    return ok(task=task)


@app.route('/api/registry/<tid>', methods=['POST'])
def api_registry_update(tid):
    body = request.get_json(force=True, silent=True) or {}
    with registry_lock:
        reg = load_registry()
        cur = task_by_number(reg, tid)
        if cur is None:
            return bad('No task T{}.'.format(tid), 404)
        fields, err = _task_fields(body, reg, cur)
        if err:
            return err
        if fields['name'] != cur['name']:
            # New folders are named by task number, so a rename is safe; only a
            # folder named exactly like the task (recorded before that rule) would be lost.
            if cur['name'] in scan_tasks():
                return bad('A dataset folder is named "{}", so this task\'s name can\'t change. '
                           'You can still edit the description.'.format(cur['name']))
            if JOBS['collection'].alive() and JOBS['collection'].meta.get('base_task') == cur['name']:
                return bad('This task is being recorded right now.')
        cur.update(fields)
        save_registry(reg)
    hub.add('UI', '[Tasks] edited T{} {}'.format(tid, cur['name']))
    return ok(task=cur)


@app.route('/api/registry/<tid>/delete', methods=['POST'])
def api_registry_delete(tid):
    with registry_lock:
        reg = load_registry()
        cur = task_by_number(reg, tid)
        if cur is None:
            return bad('No task T{}.'.format(tid), 404)
        if JOBS['collection'].alive() and JOBS['collection'].meta.get('base_task') == cur['name']:
            return bad('This task is being recorded right now.')
        reg['tasks'].remove(cur)
        save_registry(reg)
    hub.add('UI', '[Tasks] removed T{} {} from the list (recordings on disk are kept)'.format(tid, cur['name']))
    return ok()


def task_description(dataset):
    """The operator's description of the task a dataset was recorded for."""
    for t in load_registry()['tasks']:
        if dataset in task_datasets(t) and t.get('description'):
            return t['description']
    return None


# ── live camera preview ───────────────────────────────────────────────────────
@app.route('/api/preview/<action>', methods=['POST'])
def api_preview(action):
    job = JOBS['preview']
    if action == 'stop':
        background(job.stop)
        return ok()
    if job.alive():
        return ok(already=True)
    if not is_green('gopro'):
        return bad('The camera is not ready. Start it on the Machines page first.')
    if port_open(S.PREVIEW_PORT):
        return bad('Port {} is in use by another program (change PREVIEW_PORT in webui/settings.py).'
                   .format(S.PREVIEW_PORT))
    cmd = S.FASTUMI_PREFIX + 'exec python3 -u {} --topic {} --port {} --width {} --fps {}'.format(
        S.PREVIEW_SCRIPT, shlex.quote(S.PREVIEW_TOPIC), S.PREVIEW_PORT, S.PREVIEW_WIDTH, S.PREVIEW_FPS)
    good, why = job.start(cmd)
    return ok() if good else bad(why)


# ── Hugging Face Hub ──────────────────────────────────────────────────────────
REPO_ID_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.\-]*/[A-Za-z0-9][A-Za-z0-9_.\-]*$')
_hf_cache = {'t': 0, 'data': None}


def hf_whoami(refresh=False):
    """Who is logged in to Hugging Face (base env), cached for a minute."""
    if _hf_cache['data'] is None or time.time() - _hf_cache['t'] > 60 or refresh:
        code = ('import json\nfrom huggingface_hub import whoami\n'
                'try:\n    print(json.dumps({"user": whoami()["name"]}))\n'
                'except Exception as e:\n    print(json.dumps({"error": str(e).splitlines()[0][:200]}))')
        try:
            r = subprocess.run([S.LEROBOT_PY, '-c', code], capture_output=True, text=True, timeout=15)
            data = json.loads(r.stdout.strip().splitlines()[-1])
        except Exception as e:
            data = {'error': 'could not check login ({})'.format(e)}
        _hf_cache.update(t=time.time(), data=data)
    return _hf_cache['data']


@app.route('/api/hf')
def api_hf():
    return jsonify(hf_whoami(bool(request.args.get('refresh'))))


def hub_target(task, public=None):
    """({repo_id, public}, None) for a dataset, or (None, why). The Hub name is
    <logged-in user>/<dataset name>, or the repo it was pushed to before."""
    rec = read_export(task).get('hub') or {}
    repo_id = rec.get('repo_id')
    if not repo_id:
        user = hf_whoami().get('user')
        if not user:
            return None, ('Not logged in to Hugging Face. Run "huggingface-cli login" in a terminal once, '
                          'then try again.')
        repo_id = '{}/{}'.format(user, task)
    why = check_repo_id(repo_id)
    if why:
        return None, why
    return {'repo_id': repo_id, 'public': bool(rec.get('public')) if public is None else bool(public)}, None


def push_cmd(repo_id, root, public):
    return '{} -u {} --repo-id {} --root {}{}'.format(
        shlex.quote(S.LEROBOT_PY), shlex.quote(S.PUSH_SCRIPT), shlex.quote(repo_id),
        shlex.quote(root), ' --public' if public else '')


_pushed_cache = {}


def pushed_repo(task):
    """The Hub repo this dataset was pushed to, or None. Datasets pushed before
    this UI kept a record are found by looking for <user>/<dataset> on the Hub."""
    rec = read_export(task).get('hub')
    if rec and rec.get('repo_id'):
        return rec['repo_id']
    target, why = hub_target(task)
    if why:
        return None
    hit = _pushed_cache.get(target['repo_id'])
    if hit and time.time() - hit[0] < 120:
        return hit[1]
    code = ('import sys\nfrom huggingface_hub import HfApi\n'
            'try:\n    HfApi().dataset_info(sys.argv[1])\nexcept Exception:\n    sys.exit(1)')
    try:
        found = subprocess.run([S.LEROBOT_PY, '-c', code, target['repo_id']], capture_output=True,
                               timeout=20).returncode == 0
    except Exception:
        found = False   # offline: treated as not pushed
    _pushed_cache[target['repo_id']] = (time.time(), target['repo_id'] if found else None)
    return target['repo_id'] if found else None


def check_repo_id(repo_id):
    if not REPO_ID_RE.match(repo_id or '') or len(repo_id.split('/')[1]) > 96:
        return 'Hub name "{}" is not valid (letters, digits, _ . - only, at most 96).'.format(repo_id)
    return None


def remember_push(job):
    """After a successful upload, remember where the dataset went (Delete re-pushes there)."""
    pushed, target, task = job.progress.get('pushed'), job.meta.get('push'), job.meta.get('task')
    if job.state == 'done' and pushed and target and task and os.path.isdir(os.path.join(S.DATA_DIR, task)):
        write_export(task, hub=dict(target, url=pushed[-1], pushed_at=time.strftime('%Y-%m-%d %H:%M:%S')))


@app.route('/api/push/<action>', methods=['POST'])
def api_push(action):
    job = JOBS['push']
    if action == 'stop':
        background(job.stop)
        return ok()
    if S.PUSH_SCRIPT is None:
        return bad('No push script found.')
    body = request.get_json(force=True, silent=True) or {}
    task, err = _task_from_body(body, 'synced')
    if err:
        return err
    path = current_copy(task)
    if path is None:
        return bad('"{}" has not been converted yet. Convert it first.'.format(task))
    if JOBS['convert'].alive() and JOBS['convert'].meta.get('task') == task:
        return bad('"{}" is being converted right now. Wait for it to finish.'.format(task))
    push, why = hub_target(task, body.get('public'))
    if why:
        return bad(why)
    good, why = job.start(push_cmd(push['repo_id'], path, push['public']),
                          meta={'task': task, 'dataset': path, 'push': push})
    return ok() if good else bad(why)


@app.route('/api/shutdown', methods=['POST'])
def api_shutdown():
    def go():
        shutdown_everything()
        os._exit(0)
    background(go)
    return ok()


# ── process entry ─────────────────────────────────────────────────────────────
def kill_stale():
    """Kill process groups a previous UI run left behind (crash / kill -9)."""
    path = os.path.join(S.RUN_DIR, 'pgids.json')
    try:
        with open(path) as f:
            pgids = json.load(f)
    except (OSError, ValueError):
        return
    for key, pgid in pgids.items():
        try:
            if os.getpgid(pgid) != pgid:
                continue  # pid reused by something that isn't our group leader
        except OSError:
            continue
        print('  killing leftover {} (process group {})'.format(key, pgid))
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(pgid, sig)
            except OSError:
                break
            time.sleep(2)
    os.remove(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kill-stale', action='store_true', help='clean up after a crashed run and exit')
    args = ap.parse_args()
    if args.kill_stale:
        kill_stale()
        return

    kill_stale()
    os.makedirs(S.RUN_DIR, exist_ok=True)
    with open(os.path.join(S.RUN_DIR, 'server.pid'), 'w') as f:
        f.write(str(os.getpid()))

    def on_signal(signum, frame):
        def go():
            shutdown_everything()
            os._exit(0)
        threading.Thread(target=go, daemon=True).start()

    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(sig, on_signal)

    logging.getLogger('werkzeug').setLevel(logging.WARNING)  # hide 1 Hz polling requests
    threading.Thread(target=monitor, daemon=True).start()
    threading.Thread(target=card_watch, daemon=True).start()
    hub.add('UI', 'FastUMI UI up on http://{}:{}  (repo {}, data_dir {})'.format(
        S.UI_HOST, S.UI_PORT, S.REPO_ROOT, S.DATA_DIR))
    for name, path in (('sync', S.SYNC_SCRIPT), ('convert', S.CONVERT_SCRIPT), ('visualize', S.VIZ_SCRIPT)):
        if path is None:
            hub.add('UI', 'WARNING: no script found for {} step on this branch'.format(name))
    app.run(host=S.UI_HOST, port=S.UI_PORT, threaded=True, use_reloader=False)


if __name__ == '__main__':
    main()
