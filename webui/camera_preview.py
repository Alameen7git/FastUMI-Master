#!/usr/bin/env python3
"""
Live camera preview for the web UI: subscribes to the camera topic, shrinks
frames and serves them as MJPEG. Run in the ROS / FastUMI env.

    python3 webui/camera_preview.py --topic /usb_cam/image_raw --port 8081

    /stream.mjpg   multipart MJPEG for an <img> tag
    /snapshot.jpg  latest frame (also the readiness check)
"""
import argparse
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image

latest = {'jpg': None, 't': 0.0}
cond = threading.Condition()


def decode_small(msg, width):
    """sensor_msgs/Image -> small BGR (or grey) array about `width` px wide,
    or None for an encoding we can't read."""
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    enc = msg.encoding.lower()
    # Drop pixels first (cheap stride view), then do colour work on the small image.
    k = max(1, msg.width // width)
    if enc in ('rgb8', 'bgr8'):
        img = buf.reshape(msg.height, msg.step // 3, 3)[::k, :msg.width:k]
        return np.ascontiguousarray(img[:, :, ::-1] if enc == 'rgb8' else img)
    if enc in ('mono8', '8uc1'):
        return np.ascontiguousarray(buf.reshape(msg.height, msg.step)[::k, :msg.width:k])
    if enc in ('yuyv', 'yuv422_yuy2'):
        return cv2.cvtColor(buf.reshape(msg.height, msg.width, 2), cv2.COLOR_YUV2BGR_YUYV)[::k, ::k]
    if enc in ('uyvy', 'yuv422'):
        return cv2.cvtColor(buf.reshape(msg.height, msg.width, 2), cv2.COLOR_YUV2BGR_UYVY)[::k, ::k]
    return None


def make_callback(width, fps):
    period = 1.0 / fps
    last = [0.0]

    def on_image(msg):
        now = time.time()
        if now - last[0] < period:
            return                      # throttle: skip decoding most frames
        last[0] = now
        try:
            img = decode_small(msg, width)
        except ValueError as e:
            rospy.logwarn_throttle(10, 'preview: bad frame (%s)', e)
            return
        if img is None:
            rospy.logwarn_throttle(10, 'preview: unsupported encoding %s', msg.encoding)
            return
        h = int(img.shape[0] * width / img.shape[1])
        small = cv2.resize(img, (width, h), interpolation=cv2.INTER_AREA)
        ok, jpg = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            with cond:
                latest['jpg'], latest['t'] = jpg.tobytes(), now
                cond.notify_all()
    return on_image


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.startswith('/snapshot.jpg'):
            jpg = latest['jpg']
            if jpg is None:
                self.send_error(503, 'no frame yet')
                return
            self.send_response(200)
            self.send_header('Content-Type', 'image/jpeg')
            self.send_header('Content-Length', str(len(jpg)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(jpg)
        elif self.path.startswith('/stream.mjpg'):
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            seen = 0.0
            try:
                while not rospy.is_shutdown():
                    with cond:
                        cond.wait_for(lambda: latest['t'] > seen, timeout=5)
                        jpg, seen = latest['jpg'], latest['t']
                    if jpg is None:
                        continue
                    self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '
                                     + str(len(jpg)).encode() + b'\r\n\r\n' + jpg + b'\r\n')
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_error(404)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--topic', default='/usb_cam/image_raw')
    ap.add_argument('--port', type=int, default=8081)
    ap.add_argument('--width', type=int, default=640)
    ap.add_argument('--fps', type=float, default=10)
    args = ap.parse_args()

    rospy.init_node('fastumi_ui_preview', anonymous=True, disable_signals=True)
    rospy.Subscriber(args.topic, Image, make_callback(args.width, args.fps),
                     queue_size=1, buff_size=2 ** 26)
    server = ThreadingHTTPServer(('0.0.0.0', args.port), Handler)
    server.daemon_threads = True
    print('camera preview of {} on http://localhost:{}/stream.mjpg'.format(args.topic, args.port), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        rospy.signal_shutdown('preview closed')


if __name__ == '__main__':
    main()
