#!/usr/bin/env python3
"""Render an episode in MuJoCo using a chosen IK seed branch, to compare how
different branches place the arm. Overrides data_processing_to_joint's
_PREFERRED_BRANCH (which frame-0 seed_joint_angles reads) then reuses the
existing visualize_episode_mujoco.render().

Usage:
    python3 render_branch.py <episode.hdf5> --branch 4 --out out.mp4 --step 2
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_processing_to_joint as dpj
import visualize_episode_mujoco as vem

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('episode')
    p.add_argument('--branch', type=int, required=True)
    p.add_argument('--out', default=None)
    p.add_argument('--step', type=int, default=2)
    p.add_argument('--azimuth', type=float, default=340)
    p.add_argument('--elevation', type=float, default=-30)
    p.add_argument('--distance', type=float, default=1.9)
    a = p.parse_args()

    dpj._PREFERRED_BRANCH = a.branch  # override the frame-0 seed branch
    print(f'Rendering with _PREFERRED_BRANCH = {dpj._PREFERRED_BRANCH}')

    out = a.out or os.path.splitext(a.episode)[0] + f'_branch{a.branch}.mp4'
    vem.render(a.episode, out, a.step, a.azimuth, a.elevation, a.distance)
    print('Saved', out)
