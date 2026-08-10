#!/usr/bin/env python3
"""Publish a converted FastUMI episode's joint trajectory as JointState messages for RViz playback.

Publishes only sensor_msgs/JointState on /joint_states — no communication with the
real UR7e control box. Pair with robot_state_publisher + RViz (see visualize_trajectory.launch).

Usage:
    python3 publish_test_output_trajectory.py --input test_output.hdf5 --rate 20
"""

import argparse

import h5py
import rospy
from sensor_msgs.msg import JointState

JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="test_output.hdf5")
    parser.add_argument("--rate", type=float, default=20.0)
    parser.add_argument("--loop", action="store_true")
    args = parser.parse_args(rospy.myargv()[1:])

    with h5py.File(args.input, "r") as f:
        qpos = f["observations/qpos"][:]
    joint_traj = qpos[:, :6]
    T = joint_traj.shape[0]

    rospy.init_node("fastumi_trajectory_publisher")
    rospy.loginfo(f"Loaded {T} frames from {args.input}, publishing at {args.rate} Hz"
                  f"{' (looping)' if args.loop else ''}")

    pub = rospy.Publisher("/joint_states", JointState, queue_size=10)
    rate = rospy.Rate(args.rate)

    msg = JointState()
    msg.name = JOINT_NAMES

    i = 0
    while not rospy.is_shutdown():
        msg.header.stamp = rospy.Time.now()
        msg.position = joint_traj[i].tolist()
        pub.publish(msg)

        if i < T - 1:
            i += 1
        elif args.loop:
            i = 0
            rospy.loginfo("Reached end of trajectory, looping.")
        elif i == T - 1:
            rospy.loginfo_once("Reached end of trajectory, holding last frame.")

        rate.sleep()


if __name__ == "__main__":
    main()
