#!/usr/bin/env python3
"""
Live check of the gripper ArUco tags on the capture card -- same detector,
dictionary and tag ids as the conversion (config/config.json).

    conda activate FastUMI
    python viz/check_aruco_live.py                 # /dev/elgato, 1920x1080
    python viz/check_aruco_live.py --device /dev/video0

The camera must be free: stop the camera / recording in the web UI first.

Window shows: which tags are found (green box = found), the distance between
the two tag centres in px (what becomes the gripper value), tag size in px, and
the both-tags detection rate over the last ~5 s and since the last reset.
Keys:  r = reset counters   q = quit (prints a summary)

Aim for both tags >= 90% during a practice run of the task, not just at rest.
"""
import argparse, collections, json, os, sys, threading, time
import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--device', default='/dev/elgato')
ap.add_argument('--config', default=os.path.join(REPO, 'config', 'config.json'))
args = ap.parse_args()

dcfg = json.load(open(args.config))['data_process_config']
IDS = (dcfg['marker_id_0'], dcfg['marker_id_1'])
DMIN, DMAX = dcfg['distances']['marker_min'], dcfg['distances']['marker_max']
AD = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dcfg['aruco_dict']))
P = cv2.aruco.DetectorParameters_create() if hasattr(cv2.aruco, 'DetectorParameters_create') else cv2.aruco.DetectorParameters()

cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))   # same as recording on USB 3
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080); cap.set(cv2.CAP_PROP_FPS, 60)
if not cap.isOpened():
    sys.exit(f'Cannot open {args.device} -- is the web UI camera / recording still using it?')
latest = {'img': None}


def grab():
    while True:
        ok, f = cap.read()
        if ok:
            latest['img'] = f


threading.Thread(target=grab, daemon=True).start()
win = collections.deque(maxlen=60)
tot = {'n': 0, 'both': 0, 'id0': 0, 'id1': 0}
sizes = collections.deque(maxlen=60)
W = 'ArUco check  (r = reset, q = quit)'
cv2.namedWindow(W, cv2.WINDOW_NORMAL); cv2.resizeWindow(W, 1280, 720)
try:
    cv2.setWindowProperty(W, cv2.WND_PROP_TOPMOST, 1)   # don't open behind the editor/terminal
except Exception:
    pass
print(f'Checking ids {IDS} ({dcfg["aruco_dict"]}) on {args.device}. Press q in the window to quit.')

while True:
    img = latest['img']
    if img is None:
        time.sleep(0.02)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue
    latest['img'] = None
    corners, ids, _ = cv2.aruco.detectMarkers(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), AD, parameters=P)
    found = {int(j): c[0] for c, j in zip(corners, ids.flatten()) if j in IDS} if ids is not None else {}
    both = len(found) == 2
    tot['n'] += 1; tot['both'] += both; tot['id0'] += IDS[0] in found; tot['id1'] += IDS[1] in found
    win.append(both)
    frame = img.copy()
    for j, c in found.items():
        cv2.polylines(frame, [c.astype(int)], True, (0, 255, 0), 6)
        cv2.putText(frame, f'id{j}', tuple(c[0].astype(int) - [0, 12]), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 4)
        sizes.append(float(np.sqrt(cv2.contourArea(c.astype(np.float32)))))
    if both:
        a, b = (found[i].mean(0) for i in IDS)
        d = float(np.linalg.norm(a - b))
        cv2.line(frame, tuple(a.astype(int)), tuple(b.astype(int)), (255, 0, 0), 4)
        g = np.clip((d - DMIN) / (DMAX - DMIN), 0, 1)
        dist = f'distance {d:.0f} px -> gripper {g:.2f} (calibration {DMIN:.0f}-{DMAX:.0f})'
    else:
        dist = 'distance -- (need BOTH tags)'
    n = max(tot['n'], 1)
    lines = [f'tags found now: {sorted(found) or "none"}',
             dist,
             f'both tags: last 5 s {np.mean(win):.0%}   since reset {tot["both"] / n:.0%}  (id{IDS[0]} {tot["id0"] / n:.0%}, id{IDS[1]} {tot["id1"] / n:.0%})',
             f'tag size {np.median(sizes):.0f} px' if sizes else 'tag size --']
    for k, s in enumerate(lines):
        y = 60 + 62 * k
        cv2.putText(frame, s, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 9, cv2.LINE_AA)
        cv2.putText(frame, s, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 3, cv2.LINE_AA)
    cv2.imshow(W, frame)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('r'):
        tot = {'n': 0, 'both': 0, 'id0': 0, 'id1': 0}; win.clear()
    if k == ord('q'):
        break

n = max(tot['n'], 1)
print(f'frames {tot["n"]}: both tags {tot["both"] / n:.0%}, id{IDS[0]} {tot["id0"] / n:.0%}, id{IDS[1]} {tot["id1"] / n:.0%}'
      + (f', tag size {np.median(sizes):.0f} px' if sizes else ''))
cv2.destroyAllWindows()
