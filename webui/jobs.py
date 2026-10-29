"""
One job-tracking pattern for every long-running process the UI starts:
the four nodes, sync, convert and the visualization dashboard.

A Job owns one process group (started with start_new_session=True so that
roslaunch's children, ProcessPoolExecutor workers, image_view windows etc. are
all stopped together), tails its output into the shared LogHub, and exposes a
small status dict the browser polls.
"""
import codecs
import collections
import errno
import json
import os
import pty
import re
import signal
import subprocess
import termios
import threading
import time

from settings import REPO_ROOT, RUN_DIR

ANSI_RE = re.compile(r'\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07]*\x07|\x1b[()][A-Za-z0-9]')

# Job states. running/done render green, failed red, starting/stopping amber,
# stopped grey.
STOPPED, STARTING, RUNNING, STOPPING, DONE, FAILED = (
    'stopped', 'starting', 'running', 'stopping', 'done', 'failed')


class LogHub(object):
    """Interleaved, timestamped, source-tagged log with per-source scrollback."""

    def __init__(self, per_source=2000):
        self._lock = threading.Lock()
        self._seq = 0
        self._per_source = per_source
        self._by_source = {}

    def add(self, source, text):
        with self._lock:
            buf = self._by_source.setdefault(
                source, collections.deque(maxlen=self._per_source))
            for line in text.splitlines() or ['']:
                self._seq += 1
                buf.append((self._seq, time.time(), source, line))

    def since(self, seq, limit=4000):
        with self._lock:
            rows = [r for buf in self._by_source.values() for r in buf if r[0] > seq]
            last = self._seq
        rows.sort(key=lambda r: r[0])
        rows = rows[-limit:]
        return last, [{'seq': s, 't': t, 'src': src, 'line': line} for s, t, src, line in rows]


class Job(object):
    """
    kind='service'  long-lived (nodes, dashboard): green once ready_check passes.
    kind='batch'    runs to completion (sync, convert): green while running and
                    after a clean exit, red on a nonzero exit.
    """

    def __init__(self, key, tag, hub, registry, kind='service', use_pty=False,
                 ready_check=None, ready_timeout=20, health_check=None,
                 stop_key=None, stop_grace=8, on_line=None, on_status=None, on_exit=None):
        self.key, self.tag, self.hub, self.registry = key, tag, hub, registry
        self.kind, self.use_pty = kind, use_pty
        self.ready_check, self.ready_timeout = ready_check, ready_timeout
        self.health_check = health_check
        self.stop_key, self.stop_grace = stop_key, stop_grace
        self.on_line, self.on_status, self.on_exit = on_line, on_status, on_exit

        self.lock = threading.RLock()
        self.proc = None
        self.master_fd = None
        self.state = STOPPED
        self.reason = ''
        self.exit_code = None
        self.started_at = None
        self.cmd = ''
        self.meta = {}
        self.status_line = ''
        self.progress = {}
        self.errors = collections.deque(maxlen=200)
        self._stop_requested = False
        self._health_failures = 0

    # ── lifecycle ────────────────────────────────────────────────────────────
    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def start(self, cmd, meta=None):
        with self.lock:
            if self.alive():
                return False, '{} is already running'.format(self.tag)
            self.cmd, self.meta = cmd, dict(meta or {})
            self.reason, self.exit_code, self.status_line = '', None, ''
            self.progress, self._stop_requested, self._health_failures = {}, False, 0
            self.errors.clear()
            env = dict(os.environ, PYTHONUNBUFFERED='1')
            argv = ['bash', '-c', cmd]
            self.hub.add('UI', '[{}] $ {}'.format(self.tag, cmd))
            try:
                if self.use_pty:
                    master, slave = pty.openpty()
                    attrs = termios.tcgetattr(slave)
                    attrs[1] &= ~termios.ONLCR          # plain \n, so \r means "rewrite line"
                    termios.tcsetattr(slave, termios.TCSANOW, attrs)
                    self.proc = subprocess.Popen(
                        argv, cwd=REPO_ROOT, env=env, stdin=slave, stdout=slave,
                        stderr=slave, start_new_session=True, close_fds=True)
                    os.close(slave)
                    self.master_fd = master
                    out_fd = master
                else:
                    self.proc = subprocess.Popen(
                        argv, cwd=REPO_ROOT, env=env, stdin=subprocess.DEVNULL,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        start_new_session=True)
                    self.master_fd = None
                    out_fd = self.proc.stdout.fileno()
            except OSError as e:
                self._set(FAILED, 'could not start: {}'.format(e))
                return False, self.reason
            self.started_at = time.time()
            self._set(STARTING if self.kind == 'service' else RUNNING, '')
            self.registry.remember(self)
            proc = self.proc
            threading.Thread(target=self._reader, args=(out_fd,), daemon=True).start()
            threading.Thread(target=self._waiter, args=(proc,), daemon=True).start()
            return True, ''

    def stop(self, wait=True):
        """Graceful stop: optional key (data_collection's 'E'), then SIGINT,
        SIGTERM, SIGKILL to the whole process group."""
        with self.lock:
            if not self.alive():
                return
            self._stop_requested = True
            self._set(STOPPING, '')
            proc = self.proc
        t = threading.Thread(target=self._escalate, args=(proc,), daemon=True)
        t.start()
        if wait:
            t.join()

    def _escalate(self, proc):
        steps = []
        if self.stop_key is not None:
            steps.append(('key', self.stop_grace))
        steps += [(signal.SIGINT, 8 if self.stop_key is None else 5),
                  (signal.SIGTERM, 4), (signal.SIGKILL, 2)]
        for sig, grace in steps:
            if proc.poll() is not None:
                break
            if sig == 'key':
                self.send_key(self.stop_key)
            else:
                self._signal_group(proc, sig)
            deadline = time.time() + grace
            while time.time() < deadline and proc.poll() is None:
                time.sleep(0.1)
        # The leader may exit before its children (roslaunch); sweep the group.
        self._signal_group(proc, signal.SIGKILL)

    @staticmethod
    def _signal_group(proc, sig):
        try:
            os.killpg(proc.pid, sig)
        except OSError:
            pass

    def send_key(self, ch):
        if self.master_fd is None or not self.alive():
            return False
        try:
            os.write(self.master_fd, ch.encode())
            return True
        except OSError:
            return False

    # ── io threads ───────────────────────────────────────────────────────────
    def _reader(self, fd):
        buf = ''
        decoder = codecs.getincrementaldecoder('utf-8')('replace')
        while True:
            try:
                chunk = os.read(fd, 4096)
            except OSError as e:
                if e.errno == errno.EINTR:
                    continue
                break           # EIO: pty slave closed
            if not chunk:
                break
            buf += ANSI_RE.sub('', decoder.decode(chunk))
            buf = self._consume(buf)
        if buf.strip():
            self._line(buf)

    def _consume(self, buf):
        while True:
            i_n, i_r = buf.find('\n'), buf.find('\r')
            if i_n < 0 and i_r < 0:
                return buf
            if i_r >= 0 and (i_n < 0 or i_r < i_n):
                if i_r + 1 < len(buf) and buf[i_r + 1] == '\n':
                    self._line(buf[:i_r])
                    buf = buf[i_r + 2:]
                elif i_r + 1 == len(buf):
                    return buf  # wait to see whether \n follows
                else:
                    # carriage-return progress line (REC timer, tqdm): keep the
                    # latest as a live status instead of flooding the log
                    seg = buf[:i_r].strip()
                    if seg:
                        self.status_line = seg
                        if self.on_status:
                            self.on_status(self, seg)
                    buf = buf[i_r + 1:]
            else:
                self._line(buf[:i_n])
                buf = buf[i_n + 1:]

    def _line(self, line):
        line = line.rstrip()
        if not line.strip():
            return
        self.hub.add(self.tag, line)
        if self.on_line:
            try:
                self.on_line(self, line)
            except Exception as e:  # never let a parser kill the reader
                self.hub.add('UI', '[{}] progress parser error: {}'.format(self.tag, e))

    def _waiter(self, proc):
        code = proc.wait()
        time.sleep(0.2)  # let the reader flush the last lines
        with self.lock:
            if proc is not self.proc:
                return
            self.exit_code = code
            if self.master_fd is not None:
                try:
                    os.close(self.master_fd)
                except OSError:
                    pass
                self.master_fd = None
            if self._stop_requested:
                self._set(STOPPED, 'stopped by operator')
            elif code == 0:
                self._set(DONE if self.kind == 'batch' else STOPPED, 'exited normally (code 0)')
            else:
                why = 'exited with code {}'.format(code)
                if code < 0:
                    why = 'killed by signal {}'.format(-code)
                self._set(FAILED, why)
            self.registry.forget(self)
        self.hub.add('UI', '[{}] {}'.format(self.tag, self.reason))
        if self.on_exit:
            self.on_exit(self)

    # ── readiness / health, driven by the app's monitor thread ──────────────
    def poll_ready(self):
        with self.lock:
            if self.state != STARTING or not self.alive():
                return
            started = self.started_at
        ok, why = self.ready_check(self) if self.ready_check else (True, '')
        with self.lock:
            if self.state != STARTING:
                return
            if ok:
                self._set(RUNNING, why or 'ready')
                self.hub.add('UI', '[{}] ready{}'.format(self.tag, ' — ' + why if why else ''))
            elif time.time() - started > self.ready_timeout:
                self._set(FAILED, 'readiness check failed after {}s: {}'.format(self.ready_timeout, why))
                self.hub.add('UI', '[{}] NOT READY: {}'.format(self.tag, self.reason))

    def poll_health(self):
        if self.health_check is None:
            return
        with self.lock:
            if self.state not in (RUNNING, FAILED) or not self.alive() or self._stop_requested:
                return
        ok, why = self.health_check(self)
        with self.lock:
            if ok:
                self._health_failures = 0
                if self.state == FAILED:
                    self._set(RUNNING, 'recovered')
                    self.hub.add('UI', '[{}] recovered'.format(self.tag))
            else:
                self._health_failures += 1
                if self._health_failures == 2 and self.state == RUNNING:
                    self._set(FAILED, 'process alive but health check failing: {}'.format(why))
                    self.hub.add('UI', '[{}] UNHEALTHY: {}'.format(self.tag, why))

    def mark_ready(self, why=''):
        with self.lock:
            if self.state == STARTING:
                self._set(RUNNING, why or 'ready')

    def _set(self, state, reason):
        self.state, self.reason = state, reason

    def status(self):
        with self.lock:
            return {
                'key': self.key, 'tag': self.tag, 'state': self.state,
                'reason': self.reason, 'alive': self.alive(),
                'pid': self.proc.pid if self.alive() else None,
                'exit_code': self.exit_code, 'started_at': self.started_at,
                'cmd': self.cmd, 'meta': self.meta, 'status_line': self.status_line,
                'progress': self.progress, 'errors': list(self.errors),
            }


class Registry(object):
    """Persists the process-group ids of everything we started, so
    `bin/fastumi-ui stop` can still clean up if the Flask process died."""

    def __init__(self):
        self._lock = threading.Lock()
        self._path = os.path.join(RUN_DIR, 'pgids.json')
        os.makedirs(RUN_DIR, exist_ok=True)

    def _rw(self, fn):
        with self._lock:
            try:
                with open(self._path) as f:
                    data = json.load(f)
            except (OSError, ValueError):
                data = {}
            fn(data)
            tmp = self._path + '.tmp'
            with open(tmp, 'w') as f:
                json.dump(data, f)
            os.replace(tmp, self._path)

    def remember(self, job):
        self._rw(lambda d: d.__setitem__(job.key, job.proc.pid))

    def forget(self, job):
        self._rw(lambda d: d.pop(job.key, None))

