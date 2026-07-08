#!/usr/bin/env python3
"""Camera preview + topic-health bridge for the web ops UI.

Two read-only jobs, neither touching cam_capture_node.py/data_collection.py:
  1. Subscribes to the camera's compressed-image topic and writes the newest
     frame to a shared JPEG file at a throttled rate (default 2fps), via
     tmp-file + os.rename so ops_server.py's /preview.jpg endpoint never
     serves a half-written file.
  2. Subscribes to that same topic AND the T265 odometry topic purely to
     track a last-message timestamp for each -- a process can stay alive
     while its topic silently stops publishing (camera unplugged, T265 lost
     tracking), and that's exactly what ops_server.py's status indicators
     need to catch. The timestamps are flushed periodically (not per-message
     -- the odom topic runs at ~200Hz, no need to touch disk that often) to a
     small JSON file, same tmp+rename pattern.

Run directly, same as cam_capture_node.py:
    python3 preview_bridge.py
"""
import json
import os
import time

import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CompressedImage

with open('config/config.json', 'r') as f:
    config = json.load(f)

cfg = config['task_config']
VIDEO_TOPIC      = cfg['ros']['video_topic']
TRAJECTORY_TOPIC = cfg['ros']['trajectory_topic']
FLUSH_INTERVAL   = cfg.get('topic_health', {}).get('flush_interval_sec', 0.5)
PREVIEW_FPS      = config.get('preview', {}).get('fps', 2)
MIN_INTERVAL     = 1.0 / PREVIEW_FPS

PREVIEW_DIR   = os.path.join(config['device_settings']['data_dir'], '.preview')
PREVIEW_PATH  = os.path.join(PREVIEW_DIR, 'latest.jpg')
PREVIEW_TMP   = PREVIEW_PATH + '.tmp'
HEALTH_PATH   = os.path.join(PREVIEW_DIR, 'topic_health.json')
HEALTH_TMP    = HEALTH_PATH + '.tmp'

os.makedirs(PREVIEW_DIR, exist_ok=True)

_last_write = 0.0
_last_seen = {'video_topic': 0.0, 'trajectory_topic': 0.0}


def video_callback(msg):
    global _last_write
    _last_seen['video_topic'] = time.time()
    now = time.time()
    if now - _last_write < MIN_INTERVAL:
        return
    _last_write = now
    with open(PREVIEW_TMP, 'wb') as f:
        f.write(msg.data)
    os.rename(PREVIEW_TMP, PREVIEW_PATH)  # atomic -- readers never see a partial JPEG


def trajectory_callback(msg):
    _last_seen['trajectory_topic'] = time.time()


def flush_health(_event):
    with open(HEALTH_TMP, 'w') as f:
        json.dump(_last_seen, f)
    os.rename(HEALTH_TMP, HEALTH_PATH)  # atomic -- readers never see a partial write


def main():
    rospy.init_node('preview_bridge', anonymous=False)
    rospy.Subscriber(VIDEO_TOPIC, CompressedImage, video_callback, queue_size=1)
    rospy.Subscriber(TRAJECTORY_TOPIC, Odometry, trajectory_callback, queue_size=1)
    rospy.Timer(rospy.Duration(FLUSH_INTERVAL), flush_health)
    rospy.loginfo(f'preview_bridge: preview from {VIDEO_TOPIC} at ~{PREVIEW_FPS}fps, '
                  f'health-tracking {VIDEO_TOPIC} + {TRAJECTORY_TOPIC}')
    rospy.spin()


if __name__ == '__main__':
    main()
