#!/usr/bin/env python3
"""
T265 home check for the web UI: subscribes to the T265 odometry and serves how
far the UMI is from the T265's start pose (its origin). Run in the ROS env.

    python3 webui/home_monitor.py --topic /camera/odom/sample --port 8082

    /pose   JSON: distance (mm) and rotation (deg) from home, x/y/z, age,
            T265 jumps (503 until the first message; also the readiness check)

A jump is a step between two consecutive poses larger than --jump-mm: at
200 Hz no hand moves that far, so it is the T265 snapping its estimate.

The T265 sets its origin where it starts, so position home is always (0,0,0).
Rotation home is the orientation of the first message: it is only exact when
this script starts together with the T265 (the UI restarts it whenever the
T265 node starts). A gap in the stream means the T265 restarted, so home is
captured again.
"""
import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import rospy
from nav_msgs.msg import Odometry

RESTART_GAP_S = 2.0   # no message for this long, then data again = T265 restarted

state = {'home_q': None, 'home_at': None, 'home_approx': False, 'restarts': 0,
         'pos': None, 'q': None, 'cov': None, 't': 0.0, 'stamp': 0.0, 'n': 0, 'jumps': []}
lock = threading.Lock()


def rot_deg(q1, q2):
    """Angle between two xyzw quaternions, degrees."""
    d = abs(float(np.dot(q1, q2)))
    return float(np.degrees(2 * np.arccos(min(1.0, d))))


def make_callback(home_approx, jump_mm):
    def on_odom(msg):
        p, o = msg.pose.pose.position, msg.pose.pose.orientation
        q = np.array([o.x, o.y, o.z, o.w], dtype=np.float64)
        q /= np.linalg.norm(q) or 1.0
        now, stamp = time.time(), msg.header.stamp.to_sec()
        with lock:
            if state['home_q'] is None:
                state.update(home_q=q, home_at=now, home_approx=home_approx)
            elif now - state['t'] > RESTART_GAP_S:
                state.update(home_q=q, home_at=now, home_approx=False, restarts=state['restarts'] + 1)
            elif state['pos'] is not None and 0 < stamp - state['stamp'] <= 0.02:   # consecutive poses only
                step = float(np.linalg.norm(np.subtract((p.x, p.y, p.z), state['pos']))) * 1000
                if step > jump_mm:
                    state['jumps'] = (state['jumps'] + [(now, round(step, 1))])[-50:]
            state.update(pos=(p.x, p.y, p.z), q=q, cov=msg.pose.covariance[0], t=now, stamp=stamp,
                         n=state['n'] + 1)
    return on_odom


def snapshot():
    with lock:
        if state['pos'] is None:
            return None
        pos = np.array(state['pos'])
        return {
            'x_mm': round(pos[0] * 1000, 1), 'y_mm': round(pos[1] * 1000, 1), 'z_mm': round(pos[2] * 1000, 1),
            'dist_mm': round(float(np.linalg.norm(pos)) * 1000, 1),
            'rot_deg': round(rot_deg(state['q'], state['home_q']), 2),
            'home_q': [round(v, 5) for v in state['home_q']],
            'home_at': state['home_at'], 'home_approx': state['home_approx'],
            'restarts': state['restarts'], 'cov': state['cov'],
            'age_s': round(time.time() - state['t'], 3), 'messages': state['n'],
            'jumps': len(state['jumps']),
            'last_jump': ({'ago_s': round(time.time() - state['jumps'][-1][0], 1), 'mm': state['jumps'][-1][1]}
                          if state['jumps'] else None),
        }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if not self.path.startswith('/pose'):
            self.send_error(404)
            return
        snap = snapshot()
        if snap is None:
            self.send_error(503, 'no T265 pose yet')
            return
        body = json.dumps(snap).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--topic', default='/camera/odom/sample')
    ap.add_argument('--port', type=int, default=8082)
    ap.add_argument('--t265-age', type=float, default=0.0,
                    help='seconds the T265 had been running when this started (home rotation is approximate if large)')
    ap.add_argument('--jump-mm', type=float, default=30.0, help='step between two poses that counts as a jump')
    args = ap.parse_args()

    rospy.init_node('fastumi_ui_home', anonymous=True, disable_signals=True)
    # The UI starts this once the T265 is green, normally 5-20 s after the node.
    # queue_size large enough that no 200 Hz pose is dropped (a dropped pose
    # would make the next step look like a jump).
    rospy.Subscriber(args.topic, Odometry, make_callback(args.t265_age > 60, args.jump_mm), queue_size=200)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    server.daemon_threads = True
    print('T265 home check of {} on http://localhost:{}/pose'.format(args.topic, args.port), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        rospy.signal_shutdown('home check closed')


if __name__ == '__main__':
    main()
