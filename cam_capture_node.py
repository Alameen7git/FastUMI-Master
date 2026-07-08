#!/usr/bin/env python3
"""Stage 1 camera node — publishes raw MJPEG frames as sensor_msgs/CompressedImage.

Replaces the usb_cam_node + cv_bridge decode path. usb_cam_node decodes MJPEG to
a raw sensor_msgs/Image internally in C++ before publishing, and the old
data_collection.py then re-encoded that raw frame to mp4 with cv2.VideoWriter —
two full codec passes per frame for no benefit, since the sync/HDF5 step only
ever needed 1-in-3 frames.

This node opens the V4L2 device directly via OpenCV with CAP_PROP_CONVERT_RGB=0,
which makes the V4L2 backend hand back the exact JPEG bytes it read from the
driver instead of decoding them — zero decode work in the capture hot path.

There's no catkin package here on purpose: like data_collection.py, this is a
plain rospy client script, run directly with `python3 cam_capture_node.py`
(after sourcing ROS + activating the conda env), same as any other node in
this repo. No catkin workspace or build step required.
"""
import json
import time

import cv2
import rospy
from sensor_msgs.msg import CompressedImage

with open('config/config.json', 'r') as f:
    config = json.load(f)

cfg = config['task_config']
cam_cfg = cfg.get('camera', {})

VIDEO_DEVICE = cam_cfg.get('video_device', '/dev/elgato')
WIDTH        = cfg['cam_width']
HEIGHT       = cfg['cam_height']
FPS          = cam_cfg.get('fps', 60)
TOPIC        = cfg['ros']['video_topic']


def open_capture():
    cap = cv2.VideoCapture(VIDEO_DEVICE, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f'Could not open {VIDEO_DEVICE} via V4L2')

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    # The key line: with CONVERT_RGB=0 the V4L2 backend returns the raw MJPEG
    # bitstream it captured instead of decoding it to BGR first.
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # don't let the driver queue stale frames

    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = ''.join(chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4))
    if fourcc_str.upper() != 'MJPG':
        rospy.logwarn(f'Device reports fourcc {fourcc_str!r}, not MJPG — '
                       f'frames may not actually be JPEG-encoded.')
    if cap.get(cv2.CAP_PROP_CONVERT_RGB) != 0:
        rospy.logwarn('Driver ignored CAP_PROP_CONVERT_RGB=0 — frames will arrive '
                       'pre-decoded, defeating the purpose of this node. Check your '
                       'OpenCV/V4L2 backend version.')
    return cap


def main():
    rospy.init_node('cam_capture_node', anonymous=False)
    # Small queue_size: frames are already tiny (JPEG) and callbacks on the
    # subscriber side are trivial, so nothing should ever back up here.
    pub = rospy.Publisher(TOPIC, CompressedImage, queue_size=2)
    cap = open_capture()
    rospy.loginfo(f'cam_capture_node publishing raw MJPEG on {TOPIC} ({WIDTH}x{HEIGHT}@{FPS}fps)')

    frame_count = 0
    last_report = time.time()
    try:
        while not rospy.is_shutdown():
            ret, buf = cap.read()
            if not ret:
                rospy.logwarn_throttle(5, 'Frame grab failed, retrying...')
                continue

            msg = CompressedImage()
            msg.header.stamp = rospy.Time.now()
            msg.format = 'jpeg'
            msg.data = buf.tobytes()
            pub.publish(msg)

            frame_count += 1
            now = time.time()
            if now - last_report >= 5.0:
                rospy.loginfo(f'{frame_count / (now - last_report):.1f} fps, '
                               f'~{len(msg.data) / 1024:.0f} KB/frame')
                frame_count = 0
                last_report = now
    finally:
        cap.release()


if __name__ == '__main__':
    main()
