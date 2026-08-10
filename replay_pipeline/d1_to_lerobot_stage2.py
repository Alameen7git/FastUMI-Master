#!/usr/bin/env python3
"""
STAGE 2 of the FastUMI-raw -> LeRobot(D2-schema) converter.  RUN WITH THE
BASE / LEROBOT CONDA ENV (needs lerobot 0.6.0 + h5py + cv2):

    /home/nuc8/miniconda3/bin/python3 d1_to_lerobot_stage2.py \
        --intermediate-dir outputs/d1_lerobot_intermediate \
        --out-root /home/nuc8/FastUMI/dataset/Pick_and_place_the_bottle_lerobot \
        --repo-id embodied-ai/pick_and_place_the_bottle \
        --task "pick and place the bottle" --fps 20

Reads stage 1's per-episode .npz (joint-space numerics) plus the original HDF5
images and writes a LeRobot v3.0 dataset whose schema mirrors D2
(WS04_T000.0_Jul29_T1): joint-angle `action` + the same lead/foll_cmd/foll
position + follower velocity columns, and one camera `observation.images.cam_high`
(the FastUMI `front` view, resized to D2's 640x480).

Why the joint columns are all filled from the same IK solution: D1 is a
*handheld* demo -- there is no leader arm, no follower controller, no encoders.
So `action` (the command your replay executes), `observation.state`, the lead
position, the follower commanded position and the follower achieved position
are all set to the one IK-derived joint trajectory; the follower velocity is
stage 1's finite-difference (gripper velocity 0, exactly as D2 records it).
These duplicated columns are reconstructed, not independently measured -- but
they make the dataset drop-in for a D2 replay that reads any of them.

cam_low and cam_right_wrist are intentionally ABSENT: D1 only ever had one
camera, and fabricating two more would be misleading.  If your replay hard-
requires all three video keys, add --blank-missing-cams to emit black frames
for them (off by default).
"""
import argparse
import glob
import json
import os
import shutil
import sys

import cv2
import h5py
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.configs.video import RGBEncoderConfig

# lerobot 0.6.0 defaults to libsvtav1 (AV1); D2's videos are H.264 and this
# env's torchcodec can't decode the AV1 output, so force H.264 to match D2 and
# keep the dataset loadable via LeRobotDataset[i].  g=2 mirrors D2's GOP size.
H264_ENCODER = RGBEncoderConfig(vcodec='h264', pix_fmt='yuv420p', g=2)

# D2's feature names, mirrored verbatim so the schema matches.
LEAD_NAMES = ['UR7_joint_0', 'UR7_joint_1', 'UR7_joint_2', 'UR7_joint_3',
              'UR7_joint_4', 'UR7_joint_5', 'UR7_gripper_openness']
FOLL_NAMES = ['Joint1', 'Joint2', 'Joint3', 'Joint4', 'Joint5', 'Joint6', 'gripper']

# Cartesian schemas (chosen with --space). The pose is the gripper TCP; the two
# frames/rotations match what each controller expects.
CART_URBASE_NAMES = ['x', 'y', 'z', 'Rx', 'Ry', 'Rz', 'gripper']          # UR base, rotation vector
CART_BASELINK_NAMES = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 'gripper']  # base_link, quaternion

# space -> (npz key holding the per-frame action, feature names)
SPACE = {
    'joint':          ('action',        LEAD_NAMES),
    'cart-urbase':    ('cart_urbase',   CART_URBASE_NAMES),
    'cart-baselink':  ('cart_baselink', CART_BASELINK_NAMES),
}

CAM_W, CAM_H = 640, 480
CAM_HIGH = 'observation.images.cam_high'
CAM_LOW = 'observation.images.cam_low'
CAM_WRIST = 'observation.images.cam_right_wrist'


def build_features(space, with_video, blank_missing):
    if space == 'joint':
        def joints(names):
            return {'dtype': 'float32', 'shape': (7,), 'names': names}
        feats = {
            'observation.state.lead_joint_states.position': joints(LEAD_NAMES),
            'action': joints(LEAD_NAMES),
            'observation.state.foll_cmd_joint_states.position': joints(LEAD_NAMES),
            'observation.state.foll_joint_states.position': joints(FOLL_NAMES),
            'observation.state': joints(FOLL_NAMES),
            'observation.state.foll_joint_states.velocity': joints(FOLL_NAMES),
        }
    else:
        names = SPACE[space][1]
        vec = {'dtype': 'float32', 'shape': (len(names),), 'names': names}
        feats = {'action': vec, 'observation.state': dict(vec)}
    if with_video:
        feats[CAM_HIGH] = {'dtype': 'video', 'shape': (CAM_H, CAM_W, 3)}
        if blank_missing:
            feats[CAM_LOW] = {'dtype': 'video', 'shape': (CAM_H, CAM_W, 3)}
            feats[CAM_WRIST] = {'dtype': 'video', 'shape': (CAM_H, CAM_W, 3)}
    return feats


def load_images_rgb(src_hdf5, T):
    """Read the FastUMI `front` frames, BGR(stored)->RGB, resized to 640x480."""
    with h5py.File(src_hdf5, 'r') as f:
        raw = f['observations/images/front'][:]
    out = np.empty((T, CAM_H, CAM_W, 3), dtype=np.uint8)
    n = min(T, raw.shape[0])
    for t in range(n):
        img = cv2.resize(raw[t], (CAM_W, CAM_H), interpolation=cv2.INTER_AREA)
        out[t] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for t in range(n, T):                       # pad if lengths ever disagree
        out[t] = out[n - 1] if n else 0
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--intermediate-dir', required=True, help="stage 1's --out-dir")
    ap.add_argument('--out-root', required=True, help='dataset root to create (must not already exist)')
    ap.add_argument('--repo-id', required=True, help='e.g. embodied-ai/pick_and_place_the_bottle')
    ap.add_argument('--task', required=True, help='natural-language task string stored per frame')
    ap.add_argument('--fps', type=int, default=None,
                    help='dataset fps (default: rounded from stage-1 measured fps). '
                         'Set this to whatever your D2 replay assumes so timing matches.')
    ap.add_argument('--space', default='joint', choices=list(SPACE),
                    help="'joint' (default, D2 schema): action = 6 joints + gripper. "
                         "'cart-urbase': TCP pose [x,y,z,Rx,Ry,Rz,gripper] in the UR base frame "
                         "(rotation vector, ur_rtde native). 'cart-baselink': TCP pose "
                         "[x,y,z,qx,qy,qz,qw,gripper] in ROS base_link (quaternion).")
    ap.add_argument('--no-video', action='store_true', help='joints-only dataset (no cameras)')
    ap.add_argument('--blank-missing-cams', action='store_true',
                    help='also emit black cam_low / cam_right_wrist so all 3 D2 video keys exist')
    ap.add_argument('--overwrite', action='store_true', help='delete out-root if it exists')
    args = ap.parse_args()

    man_path = os.path.join(args.intermediate_dir, 'manifest.json')
    if not os.path.isfile(man_path):
        sys.exit(f'no manifest.json in {args.intermediate_dir} -- run stage 1 first')
    manifest = json.load(open(man_path))
    npz_files = sorted(glob.glob(os.path.join(args.intermediate_dir, 'episode_*.npz')),
                       key=lambda p: int(os.path.splitext(os.path.basename(p))[0].split('_')[1]))
    if not npz_files:
        sys.exit(f'no episode_*.npz in {args.intermediate_dir}')

    fps = args.fps or int(round(float(np.mean([e['fps'] for e in manifest['episodes']]))))
    with_video = not args.no_video

    if os.path.exists(args.out_root):
        if args.overwrite:
            shutil.rmtree(args.out_root)
        else:
            sys.exit(f'{args.out_root} exists (use --overwrite)')

    features = build_features(args.space, with_video, args.blank_missing_cams)
    npz_key = SPACE[args.space][0]
    print(f'creating LeRobot dataset  repo_id={args.repo_id}  space={args.space}  fps={fps}  video={with_video}\n'
          f'features: {list(features)}\n')

    ds = LeRobotDataset.create(repo_id=args.repo_id, fps=fps, features=features,
                               root=args.out_root, robot_type='ur', use_videos=with_video,
                               rgb_encoder=H264_ENCODER if with_video else None)

    black = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)
    n_viol = 0
    for npz_path in npz_files:
        d = np.load(npz_path, allow_pickle=True)
        act = d[npz_key]                 # (T,7) joints, or (T,7)/(T,8) cartesian
        velocity = d['velocity']         # (T,7) -- joint-space only
        T = act.shape[0]
        if bool(d['any_violation']):
            n_viol += 1
        imgs = load_images_rgb(str(d['src_hdf5']), T) if with_video else None

        for t in range(T):
            a = act[t].astype(np.float32)
            if args.space == 'joint':
                frame = {
                    'observation.state.lead_joint_states.position': a,
                    'action': a,
                    'observation.state.foll_cmd_joint_states.position': a,
                    'observation.state.foll_joint_states.position': a,
                    'observation.state': a,
                    'observation.state.foll_joint_states.velocity': velocity[t].astype(np.float32),
                    'task': args.task,
                }
            else:
                frame = {'action': a, 'observation.state': a, 'task': args.task}
            if with_video:
                frame[CAM_HIGH] = imgs[t]
                if args.blank_missing_cams:
                    frame[CAM_LOW] = black
                    frame[CAM_WRIST] = black
            ds.add_frame(frame)
        ds.save_episode()
        print(f'  saved {os.path.basename(npz_path)}  ({T} frames)')

    print(f'\nDONE -> {args.out_root}')
    print(f'  meta/info.json, data/, ' + ('videos/, ' if with_video else '') + 'meta/stats.json written by lerobot')
    if n_viol:
        print(f'  NOTE: {n_viol} episode(s) had validation flags in stage 1 -- '
              f'review with validate_joint_trajectory.py before hardware replay.')


if __name__ == '__main__':
    main()
