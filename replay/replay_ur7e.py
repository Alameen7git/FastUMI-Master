#!/usr/bin/env python3
"""
Replay a UR7e joint trajectory produced by validate_joint_trajectory.py on
the physical arm via ur_rtde, defaulting to a dry run (no hardware, no
ur_rtde import at all) until explicitly told otherwise.

Why retiming, not just resampling: the recorded joint trajectory's
frame-to-frame velocity/acceleration (see validate_joint_trajectory.py's
report) is derived from raw handheld-demo motion at ~20 Hz and is typically
far jerkier than the arm can track once you naively resample it up to the
control rate -- resampling alone preserves that jerk, it just changes the
sample count. Instead this fits a cubic spline through the recorded
waypoints, measures the spline's OWN peak velocity/acceleration, and -- if
that exceeds the URDF/config limits -- uniformly slows down playback
(stretches total duration) until it doesn't. That's an exact, one-shot
computation: stretching time by `scale` divides velocity by `scale` and
acceleration by `scale^2`, since it's a pure reparameterization of the same
path. The safety check that actually matters happens on the trajectory as
it will be sent, not on the raw recording.

Speed safety: --speed-scale (default 0.1 = 10%) stretches playback time by
1/speed_scale on top of the retiming above -- every waypoint is still sent
in the same order, just spaced further apart in wall-clock time. Since
velocity is delta-position/delta-time, that divides the achieved joint
velocity by speed_scale and acceleration by speed_scale^2, same as the
retiming math, applied a second time as a user-controlled margin. A first
physical run should never be at full (already-safety-retimed) speed, so
this can't be skipped -- it just defaults conservative. Above 0.25 it
requires typing "CONFIRM_FAST" in addition to the usual "RUN" prompt.

Usage (dry run, default -- no hardware, no ur_rtde needed):
    python3 replay_ur7e.py outputs/validated_trajectories/episode_10.hdf5

Usage (real hardware, first try -- slow and deliberate):
    python3 replay_ur7e.py outputs/validated_trajectories/episode_10.hdf5 \\
        --execute --robot-ip 192.168.1.102

Usage (real hardware, faster than 25% speed -- extra confirmation required):
    python3 replay_ur7e.py outputs/validated_trajectories/episode_10.hdf5 \\
        --execute --robot-ip 192.168.1.102 --speed-scale 0.5
"""
import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET

import h5py
import numpy as np
from scipy.interpolate import CubicSpline

JOINT_NAMES = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
               'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
RETIME_SAFETY_MARGIN = 1.1  # extra slack on top of the exact scale needed to respect limits
FAST_SPEED_CONFIRM_THRESHOLD = 0.25  # --speed-scale above this needs a typed "CONFIRM_FAST"


def load_urdf_joint_limits(urdf_path):
    tree = ET.parse(urdf_path)
    limits = {}
    for j in tree.findall('.//joint'):
        name = j.get('name')
        if name in JOINT_NAMES:
            lim = j.find('limit')
            limits[name] = (float(lim.get('lower')), float(lim.get('upper')), float(lim.get('velocity')))
    return (np.array([limits[n][0] for n in JOINT_NAMES]),
            np.array([limits[n][1] for n in JOINT_NAMES]),
            np.array([limits[n][2] for n in JOINT_NAMES]))


def build_retimed_trajectory(joint_traj, dt_s, control_period, vel_limit, accel_limit):
    """Cubic-spline the recorded waypoints, retime to respect vel/accel
    limits, then sample at the fixed hardware control period.
    Returns (q, qd, qdd) each (N, 6), and the scale factor applied."""
    t_orig = np.concatenate([[0.0], np.cumsum(dt_s)])
    spl = CubicSpline(t_orig, joint_traj, axis=0)
    spl_d1 = spl.derivative(1)
    spl_d2 = spl.derivative(2)

    probe_t = np.linspace(t_orig[0], t_orig[-1], max(2000, 20 * len(t_orig)))
    peak_vel = np.abs(spl_d1(probe_t)).max(axis=0)
    peak_accel = np.abs(spl_d2(probe_t)).max(axis=0)

    vel_ratio = (peak_vel / vel_limit).max()
    accel_ratio = (peak_accel / accel_limit).max()
    scale = max(1.0, vel_ratio, np.sqrt(accel_ratio)) * RETIME_SAFETY_MARGIN

    total_duration = t_orig[-1] * scale
    n_steps = int(np.ceil(total_duration / control_period)) + 1
    tick_times = np.arange(n_steps) * control_period
    s_eval = np.clip(tick_times / scale, t_orig[0], t_orig[-1])

    q = spl(s_eval)
    qd = spl_d1(s_eval) / scale
    qdd = spl_d2(s_eval) / (scale ** 2)
    return q, qd, qdd, scale, t_orig[-1], total_duration


def dry_run_report(episode_path, q, qd, qdd, control_period, orig_duration, scale, vel_limit, accel_limit,
                    speed_scale, effective_control_period, out_prefix):
    qd_eff = qd * speed_scale
    qdd_eff = qdd * (speed_scale ** 2)
    max_vel_eff = np.abs(qd_eff).max()
    max_accel_eff = np.abs(qdd_eff).max()
    effective_duration = q.shape[0] * effective_control_period

    print(f"--- dry run: {os.path.basename(episode_path)} ---")
    print(f"  recorded duration: {orig_duration:.2f}s  ->  safety-retimed duration: {q.shape[0] * control_period:.2f}s "
          f"(retime x{scale:.2f}, includes {RETIME_SAFETY_MARGIN:.0%} margin)")
    print(f"  speed scale: x{speed_scale:.2f}  ->  EFFECTIVE playback duration: {effective_duration:.2f}s "
          f"({effective_control_period * 1000:.1f} ms/waypoint, {q.shape[0]} waypoints)")
    print(f"  effective peak |velocity|: {np.degrees(max_vel_eff):.1f} deg/s   (URDF limit: {np.degrees(vel_limit.min()):.0f} deg/s)")
    print(f"  effective peak |acceleration|: {np.degrees(max_accel_eff):.1f} deg/s^2   (config limit: {np.degrees(accel_limit.min()):.0f} deg/s^2)")
    print(f"  first waypoint (deg): {np.degrees(q[0]).round(1).tolist()}")
    print(f"  last  waypoint (deg): {np.degrees(q[-1]).round(1).tolist()}")
    print("  DRY RUN -- no hardware connection attempted, ur_rtde was not imported.")
    print("  Pass --execute --robot-ip <ip> to run on the physical arm once you've reviewed this "
          f"(defaults to --speed-scale {speed_scale:.2f}).")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t = np.arange(q.shape[0]) * effective_control_period
    fig, ax = plt.subplots(figsize=(12, 4))
    for j, name in enumerate(JOINT_NAMES):
        ax.plot(t, np.degrees(q[:, j]), label=name)
    ax.set_xlabel('time (s)')
    ax.set_ylabel('joint angle (deg)')
    ax.set_title(f'playback trajectory at speed-scale x{speed_scale:.2f} '
                 f'(x{scale / speed_scale:.2f} slower than recorded overall)')
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    path = out_prefix + '_retimed.png'
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  plot -> {path}")


def execute_on_hardware(q, robot_ip, effective_control_period, lookahead_time, gain,
                         startup_speed, startup_accel, speed_scale, pos_lo, pos_hi):
    if np.any(q < pos_lo[None, :]) or np.any(q > pos_hi[None, :]):
        print("ABORT: retimed trajectory has a waypoint outside the URDF joint position limits.")
        sys.exit(1)

    try:
        import rtde_control
        import rtde_receive
    except ImportError:
        print("ur_rtde is not installed in this environment. Install it with:")
        print("    pip install ur_rtde")
        sys.exit(1)

    print(f"Connecting to UR7e at {robot_ip} ...")
    rtde_c = rtde_control.RTDEControlInterface(robot_ip)
    rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)
    if not rtde_c.isConnected():
        print(f"Could not establish an RTDE control connection to {robot_ip}.")
        sys.exit(1)

    try:
        current_q = np.array(rtde_r.getActualQ())
        first_move_dist_deg = np.degrees(np.abs(q[0] - current_q)).max()
        # moveJ takes absolute speed/acceleration caps (not a time-indexed
        # target like servoJ), so the speed-scale safeguard applies here by
        # scaling those caps directly rather than stretching a duration.
        startup_speed_scaled = startup_speed * speed_scale
        startup_accel_scaled = startup_accel * speed_scale
        print(f"Current robot joints (deg): {np.degrees(current_q).round(1).tolist()}")
        print(f"Moving to trajectory start (deg): {np.degrees(q[0]).round(1).tolist()} "
              f"(max joint delta {first_move_dist_deg:.1f} deg) at speed={startup_speed_scaled:.3f} rad/s "
              f"(x{speed_scale:.2f} of {startup_speed} rad/s), accel={startup_accel_scaled:.3f} rad/s^2 ...")
        rtde_c.moveJ(q[0].tolist(), startup_speed_scaled, startup_accel_scaled)

        print(f"Streaming {q.shape[0]} servoJ waypoints at {1.0 / effective_control_period:.1f} Hz "
              f"(speed-scale x{speed_scale:.2f}) ...")
        for i in range(q.shape[0]):
            t_start = rtde_c.initPeriod()
            rtde_c.servoJ(q[i].tolist(), 0.0, 0.0, effective_control_period, lookahead_time, gain)
            rtde_c.waitPeriod(t_start)
        print("Playback complete.")
    except KeyboardInterrupt:
        print("Interrupted -- stopping robot.")
    finally:
        rtde_c.servoStop()
        rtde_c.stopScript()


def run(args):
    """Core replay logic, factored out of main() so other entry points (e.g.
    replay.py's one-command dataset/episode/speed wrapper) can call it
    directly with a constructed Namespace instead of duplicating this or
    shelling out to this script. `args` needs: validated_hdf5, config,
    execute, robot_ip, yes, force, speed_scale, output_dir."""
    if args.speed_scale <= 0:
        print("ERROR: --speed-scale must be > 0.")
        sys.exit(1)

    with open(args.config) as f:
        config = json.load(f)
    dcfg = config['data_process_config']
    hw = config.get('ur7e_hardware', {})
    pos_lo, pos_hi, vel_limit = load_urdf_joint_limits(dcfg['urdf_path'])
    accel_limit = np.full(6, hw.get('joint_accel_limit_rad_s2', np.pi))
    control_period = hw.get('servo_control_period_s', 0.002)

    with h5py.File(args.validated_hdf5, 'r') as f:
        joint_traj = f['joint_trajectory'][:]
        dt_s = f['dt_s'][:]
        reachable = f['diagnostics/reachable'][:]
        over_pos_limit = f['violations/over_position_limit'][:]

    hard_violation = bool((~reachable).any() or over_pos_limit.any())
    if hard_violation and not args.force:
        print("This validated trajectory has unresolved IK-reachability or joint-position-limit "
              "violations (see validate_joint_trajectory.py's report). Refusing to proceed.")
        print("Re-check the recording/calibration, or pass --force if you've reviewed and accept it.")
        sys.exit(1)

    q, qd, qdd, scale, orig_duration, playback_duration = build_retimed_trajectory(
        joint_traj, dt_s, control_period, vel_limit, accel_limit)

    # User-controlled speed safeguard, layered on top of the safety retiming
    # above: stretching the servoJ streaming period by 1/speed_scale divides
    # the achieved velocity by speed_scale and acceleration by speed_scale^2
    # between every waypoint (same reparameterization math as the retime
    # itself), so it protects against a bad IK jump even where the retimed
    # trajectory was already within limits.
    effective_control_period = control_period / args.speed_scale
    effective_duration = q.shape[0] * effective_control_period

    os.makedirs(args.output_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.validated_hdf5))[0]
    out_prefix = os.path.join(args.output_dir, stem)

    if not args.execute:
        dry_run_report(args.validated_hdf5, q, qd, qdd, control_period, orig_duration, scale,
                        vel_limit, accel_limit, args.speed_scale, effective_control_period, out_prefix)
        return

    robot_ip = args.robot_ip or hw.get('robot_ip')
    if not robot_ip or robot_ip == 'TODO_FILL_IN_ROBOT_IP':
        print("No robot IP set. Pass --robot-ip <ip> or fill in ur7e_hardware.robot_ip in config.json.")
        sys.exit(1)

    print(f"About to move the PHYSICAL UR7e at {robot_ip} through {q.shape[0]} waypoints.")
    print(f"  speed scale: x{args.speed_scale:.2f}  ->  estimated playback duration: {effective_duration:.1f}s "
          f"(recorded duration was {orig_duration:.1f}s)")
    print("  Make sure the workspace is clear and you have the e-stop / teach pendant within reach.")

    # Hard gate: fast playback needs an explicit, non-skippable typed
    # confirmation on top of (not instead of) the usual RUN prompt below --
    # --yes only waives the routine prompt, not this one.
    if args.speed_scale > FAST_SPEED_CONFIRM_THRESHOLD:
        print(f"\n--speed-scale {args.speed_scale:.2f} is above the {FAST_SPEED_CONFIRM_THRESHOLD:.2f} "
              f"conservative threshold.")
        reply = input('Type "CONFIRM_FAST" to proceed at this speed: ')
        if reply.strip() != 'CONFIRM_FAST':
            print('Aborted.')
            sys.exit(1)

    if not args.yes:
        reply = input('Type "RUN" to proceed: ')
        if reply.strip() != 'RUN':
            print('Aborted.')
            sys.exit(1)

    execute_on_hardware(q, robot_ip, effective_control_period,
                         hw.get('servo_lookahead_time_s', 0.1), hw.get('servo_gain', 300),
                         hw.get('startup_move_speed_rad_s', 0.2), hw.get('startup_move_accel_rad_s2', 0.3),
                         args.speed_scale, pos_lo, pos_hi)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('validated_hdf5', help='output of validate_joint_trajectory.py')
    parser.add_argument('--config', default='config/config.json')
    parser.add_argument('--execute', action='store_true', help='connect to the real robot and move it (default: dry run)')
    parser.add_argument('--robot-ip', default=None, help='overrides config.json ur7e_hardware.robot_ip')
    parser.add_argument('--yes', action='store_true', help='skip the typed confirmation prompt before --execute moves the robot')
    parser.add_argument('--force', action='store_true', help='proceed to --execute even if the validated file has unresolved reachability/position violations')
    parser.add_argument('--speed-scale', type=float, default=0.1,
                         help='fraction of full (safety-retimed) playback speed to actually run (default 0.1 = 10%%). '
                              f'Values above {FAST_SPEED_CONFIRM_THRESHOLD:.2f} require typing "CONFIRM_FAST" in addition to "RUN".')
    parser.add_argument('--output-dir', default='outputs/replay')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
