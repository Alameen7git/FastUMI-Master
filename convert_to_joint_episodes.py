#!/usr/bin/env python3
"""Convert SPECIFIC episodes of a dataset to joint poses.

Reuses the EXACT same conversion (normalize_ik_and_save_hdf5) that
data_processing_to_joint.py runs on a whole folder -- same config.json, same
base_position/base_orientation, same IK. The only difference is this lets you
pick individual episode numbers instead of processing the entire input_dir,
so you don't have to touch config.json.

Usage:
    python3 convert_to_joint_episodes.py <dataset_dir> <episode_numbers...>

Example (episodes 1 and 2 of the RAM-stick dataset):
    python3 convert_to_joint_episodes.py /home/nuc8/FastUMI/dataset/Insert_the_RAM_stick 1 2

Output goes to <dataset_dir>/output_joint/episode_<N>.hdf5
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_processing_to_joint import normalize_ik_and_save_hdf5


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dataset_dir', help='folder containing episode_N.hdf5 files')
    parser.add_argument('episodes', type=int, nargs='+', help='episode numbers to convert, e.g. 1 2')
    parser.add_argument('--out', default=None,
                        help='output dir (default: <dataset_dir>/output_joint)')
    args = parser.parse_args()

    out_dir = args.out or os.path.join(args.dataset_dir, 'output_joint')
    os.makedirs(out_dir, exist_ok=True)

    for n in args.episodes:
        in_file = os.path.join(args.dataset_dir, f'episode_{n}.hdf5')
        out_file = os.path.join(out_dir, f'episode_{n}.hdf5')
        if not os.path.exists(in_file):
            print(f'SKIP: {in_file} does not exist')
            continue
        print(f'Converting {in_file} ...')
        normalize_ik_and_save_hdf5((in_file, out_file))

    print(f'\nDone. Converted files are in: {out_dir}')


if __name__ == '__main__':
    main()
