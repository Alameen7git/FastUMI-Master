#!/usr/bin/env python3
"""
One-command UR7e replay: dataset + episode + speed in, safely-paced motion
out. Wraps validate_joint_trajectory.py (convert + safety-check) and
replay_ur7e.py (retime + speed-scale + execute) so you don't have to run
them by hand and pass the intermediate file path yourself.

This does NOT reimplement safety -- it calls straight into those two
scripts' own functions, so "no jerky moments / nothing too fast" comes from
mechanisms that already exist and were already tested there:
  - validate_joint_trajectory.py: IK-reachability + FK cross-check + URDF
    joint-limit checks. An unresolved reachability/position violation
    blocks replay outright (pass --force only if you've reviewed the report
    yourself and accept it).
  - replay_ur7e.py: fits a cubic spline through the recorded waypoints and
    retimes playback (uniformly slows it down) so velocity/acceleration
    respect the URDF/config limits -- this is what actually removes jerk,
    not just resampling. --speed then applies an ADDITIONAL, user-facing
    slowdown on top (default 0.1 = 10% speed), with a non-bypassable typed
    "CONFIRM_FAST" gate above 0.25.

Usage (dry run -- no hardware, just the report + plots):
    python3 replay.py --task Pick_and_place_the_bottle --episode 10
    python3 replay.py --task Pick_and_place_the_bottle --episode 10 --speed 0.2

Usage (real hardware, once you've reviewed the dry run):
    python3 replay.py --task Pick_and_place_the_bottle --episode 10 --speed 0.1 \\
        --execute --robot-ip 192.168.1.102

You can also point straight at an episode file instead of --task/--episode:
    python3 replay.py --episode-path /path/to/episode_10.hdf5 --speed 0.1
"""
import argparse
import json
import os
import sys

import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in [_ROOT] + [os.path.join(_ROOT, _d) for _d in ('conversion','viz','replay_pipeline','replay','lib')]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
import validate_joint_trajectory as vjt
import replay_ur7e as rur


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--task', default=None, help='task subdir under config data_dir, e.g. Pick_and_place_the_bottle')
    parser.add_argument('--episode', type=int, default=None, help='episode index within --task')
    parser.add_argument('--episode-path', default=None, help='direct path to episode_N.hdf5 (alternative to --task/--episode)')
    parser.add_argument('--speed', type=float, default=0.1,
                         help='fraction of full (already jerk-safe-retimed) playback speed (default 0.1 = 10%%). '
                              'Values above 0.25 require typing "CONFIRM_FAST" before real motion.')
    parser.add_argument('--execute', action='store_true', help='connect to the real robot and move it (default: dry run)')
    parser.add_argument('--robot-ip', default=None, help='overrides config.json ur7e_hardware.robot_ip')
    parser.add_argument('--yes', action='store_true', help='skip the typed "RUN" confirmation prompt (never skips CONFIRM_FAST)')
    parser.add_argument('--force', action='store_true', help='proceed even if validation found unresolved reachability/position violations')
    parser.add_argument('--config', default='config/config.json')
    parser.add_argument('--validated-dir', default='outputs/validated_trajectories')
    parser.add_argument('--replay-dir', default='outputs/replay')
    args = parser.parse_args()

    if args.episode_path:
        episode_path = args.episode_path
    elif args.task and args.episode is not None:
        with open(args.config) as f:
            data_dir = json.load(f)['device_settings']['data_dir']
        episode_path = os.path.join(data_dir, args.task, f'episode_{args.episode}.hdf5')
    else:
        print('Specify --episode-path <file>, or both --task <name> and --episode <N>.')
        sys.exit(1)

    if not os.path.isfile(episode_path):
        print(f'Episode not found: {episode_path}')
        sys.exit(1)

    with open(args.config) as f:
        config = json.load(f)

    print(f"=== step 1/2: convert + validate {os.path.basename(episode_path)} ===")
    os.makedirs(args.validated_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(episode_path))[0]
    validated_prefix = os.path.join(args.validated_dir, stem)
    vjt.validate(episode_path, config, validated_prefix)
    print("(raw velocity/acceleration flags above are expected for handheld demo data and are what step 2's\n"
          " retiming fixes; only IK-reachability or joint-position-limit violations actually block replay.)\n")

    print(f"=== step 2/2: {'execute on hardware' if args.execute else 'dry run'} at speed x{args.speed:.2f} ===")
    replay_args = argparse.Namespace(
        validated_hdf5=validated_prefix + '.hdf5',
        config=args.config,
        execute=args.execute,
        robot_ip=args.robot_ip,
        yes=args.yes,
        force=args.force,
        speed_scale=args.speed,
        output_dir=args.replay_dir,
    )
    rur.run(replay_args)


if __name__ == '__main__':
    main()
