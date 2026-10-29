#!/usr/bin/env python3
"""
Is the camera showing a real, moving picture? Used by the web UI as the
camera node's readiness and health check (ROS / FastUMI env).

    python3 webui/camera_check.py --topic /usb_cam/image_raw

The Elgato capture card keeps publishing frames even when no GoPro is plugged
into it (or the GoPro is off), so "a message arrived" is not enough. A live
GoPro picture measured on this rig: grey-level spread (std) ~61 and a mean
change of ~3 levels between frames 0.3 s apart (sensor noise alone). A blank
or frozen "no signal" picture has neither.

Exit codes: 0 live picture · 1 no frames · 2 blank picture · 3 frozen picture.
The last stdout line is a short reason for the UI.
"""
import argparse
import sys
import threading
import time

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image

from camera_preview import decode_small

BLANK_STD = 4.0      # grey-level spread below this = one flat colour (black / blue screen)
FROZEN_DIFF = 0.3    # mean change between samples below this = the same picture over and over


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--topic', default='/usb_cam/image_raw')
    ap.add_argument('--timeout', type=float, default=4.0, help='seconds to wait for the first frame')
    ap.add_argument('--samples', type=int, default=3)
    ap.add_argument('--gap', type=float, default=0.3, help='seconds between samples')
    args = ap.parse_args()

    frames = []
    last = [0.0]
    enough = threading.Event()

    def on_image(msg):
        now = time.time()
        if enough.is_set() or now - last[0] < args.gap:
            return
        last[0] = now
        try:
            img = decode_small(msg, 160)
        except ValueError:
            return
        if img is None:
            print('unsupported image encoding {}'.format(msg.encoding))
            enough.set()
            return
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        frames.append(img.astype(np.int16))
        if len(frames) >= args.samples:
            enough.set()

    rospy.init_node('fastumi_ui_camcheck', anonymous=True, disable_signals=True)
    rospy.Subscriber(args.topic, Image, on_image, queue_size=1, buff_size=2 ** 26)
    enough.wait(args.timeout + args.gap * args.samples)
    rospy.signal_shutdown('check done')

    if not frames:
        print('no picture arrived on {}'.format(args.topic))
        return 1
    spread = max(float(f.std()) for f in frames)
    if spread < BLANK_STD:
        print('camera sends a blank picture (is the GoPro on and plugged into the capture card?)')
        return 2
    if len(frames) >= 2:
        change = max(float(np.abs(f - frames[0]).mean()) for f in frames[1:])
        if change < FROZEN_DIFF:
            print('camera picture is frozen (is the GoPro on and plugged into the capture card?)')
            return 3
    print('live picture on {}'.format(args.topic))
    return 0


if __name__ == '__main__':
    sys.exit(main())
