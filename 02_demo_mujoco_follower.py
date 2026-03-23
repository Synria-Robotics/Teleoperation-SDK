#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Synria Robotics Co., Ltd.
# Developer: Edward Zhou, Synria
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# Author: Synria Robotics Team
# Website: https://synriarobotics.ai

"""Leader-to-MuJoCo follower teleoperation demo.

Core modules:
    1. Input decoding: parse leader-arm buttons, trigger, and gripper state.
    2. Shared state: keep the latest hardware snapshot in a thread-safe object.
    3. MuJoCo bridge: load the follower XML and resolve actuator handles.
"""

import math
import time
import argparse
import threading
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

try:
    import mujoco
    import mujoco.viewer
except ImportError:
    print("[ERROR] MuJoCo Python bindings not found.")
    print("        Install with:  pip install mujoco")
    sys.exit(1)

import alicia_d_sdk
from alicia_d_sdk.utils import precise_sleep

# ───────────────────────── constants ──────────────────────────
TRIGGER_THRESHOLD = 900          # gripper < this ⇒ trigger pressed
GRIPPER_SDK_MAX   = 1000.0       # SDK gripper range: 0 (closed) – 1000 (open)
GRIPPER_MJ_MAX    = 0.025        # MuJoCo Gripper slide joint range: 0 – 0.025 m

# Default MuJoCo XML path (from user-provided XML)

DEFAULT_XML = str(
    PROJECT_ROOT
    / "assets" / "mujoco" / "Alicia_D_v5_6" / "gripper_50mm" / "alicia_d_follower.xml"
)


def _try_disable_torque(robot):
    """Disable torque using the first compatible SDK method available.

    Args:
        robot: SDK robot instance created by ``alicia_d_sdk.create_robot``.
    """
    for method_name, args in [
        ("torque_control", ("off",)),
        ("disable_torque", ()),
        ("torque_enable", (False,)),
    ]:
        fn = getattr(robot, method_name, None)
        if fn is not None:
            fn(*args)
            return


def _decode_handle_inputs(raw_status, run_status_text, gripper_value=None):
    """Decode leader handle inputs into normalized teleoperation flags.

    Args:
        raw_status: Raw status bitmask from the SDK parser.
        run_status_text: Human-readable run status string from the SDK state.
        gripper_value: Optional SDK gripper value used to infer trigger state.

    Returns:
        dict: Mapping with keys ``trigger``, ``button1``, and ``button2``.
    """
    left_button = False
    right_button = False
    trigger = gripper_value is not None and gripper_value < TRIGGER_THRESHOLD

    if isinstance(run_status_text, str):
        if run_status_text == "sync":
            left_button = True
        elif run_status_text == "locked":
            right_button = True
        elif run_status_text == "sync_locked":
            left_button = True
            right_button = True

    if isinstance(raw_status, int):
        left_button = left_button or bool(raw_status & 0x10)
        right_button = right_button or bool(raw_status & 0x01)

    return {"trigger": trigger, "button1": left_button, "button2": right_button}


# ─────────────────── gripper value conversion ─────────────────
def gripper_sdk_to_mujoco(sdk_val: float) -> float:
    """Convert an SDK gripper value into a MuJoCo gripper target.

    Args:
        sdk_val: Leader-side gripper value in the range ``0`` to ``1000``.

    Returns:
        float: MuJoCo slide-joint target in meters.

    SDK : 1000 = fully open  →  MuJoCo 0      (fingers at body origins, apart)
    SDK : 0    = fully closed →  MuJoCo 0.025  (fingers moved toward centre)

    Verified via kinematic analysis:
      qpos (0, 0)           → gap ≈ 51 mm → OPEN
      qpos (0.025, -0.025)  → gap ≈  1 mm → CLOSED
    """
    clamped = max(0.0, min(GRIPPER_SDK_MAX, sdk_val))
    return (1.0 - clamped / GRIPPER_SDK_MAX) * GRIPPER_MJ_MAX


def load_mujoco_bundle(xml_path: str):
    """Load the MuJoCo model and resolve all required actuator IDs.

    Args:
        xml_path: Path to the MuJoCo XML model file.

    Returns:
        tuple: ``(model, data, act_ids, grip_l_act_id, grip_r_act_id)``.

    Raises:
        RuntimeError: If any expected arm or gripper actuator is missing.
    """
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    act_names = [f"pos{i+1}" for i in range(6)]
    act_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in act_names]
    grip_l_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "pos_grip_l")
    grip_r_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "pos_grip_r")

    for aid, name in zip(act_ids, act_names):
        if aid < 0:
            raise RuntimeError(f"Actuator '{name}' not found in MuJoCo model")
    if grip_l_act_id < 0 or grip_r_act_id < 0:
        raise RuntimeError("Gripper actuators (pos_grip_l / pos_grip_r) not found in MuJoCo model")

    return model, data, act_ids, grip_l_act_id, grip_r_act_id


# ──────────────────── shared state container ──────────────────
class LeaderState:
    """Thread-safe container holding the latest leader-arm snapshot.

    This object is written by the background collector thread and read by the
    MuJoCo viewer loop.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.joint_angles = [0.0] * 6          # radians
        self.gripper_value = GRIPPER_SDK_MAX    # SDK units
        self.trigger = False
        self.button1 = False                    # left  – deadman switch
        self.button2 = False                    # right – reserved
        self.run_status_text = "unknown"
        self.run_status_raw = None
        self.fps = 0.0

    def update(self, angles, gripper, trigger, btn1, btn2, status_text, status_raw, fps):
        """Store a new leader snapshot atomically.

        Args:
            angles: Latest leader joint angles in radians.
            gripper: Latest SDK gripper value.
            trigger: Whether the analog trigger is considered pressed.
            btn1: Left button state.
            btn2: Right button state.
            status_text: Decoded SDK run status text.
            status_raw: Raw SDK run status bitmask.
            fps: Measured collector loop frequency.
        """
        with self.lock:
            self.joint_angles = list(angles)
            self.gripper_value = gripper if gripper is not None else GRIPPER_SDK_MAX
            self.trigger = trigger
            self.button1 = btn1
            self.button2 = btn2
            self.run_status_text = status_text
            self.run_status_raw = status_raw
            self.fps = fps

    def snapshot(self):
        """Return a consistent copy of the latest leader state.

        Returns:
            dict: Snapshot used by the MuJoCo control loop and HUD renderer.
        """
        with self.lock:
            return {
                "joint_angles": list(self.joint_angles),
                "gripper_value": self.gripper_value,
                "trigger": self.trigger,
                "button1": self.button1,
                "button2": self.button2,
                "run_status_text": self.run_status_text,
                "run_status_raw": self.run_status_raw,
                "fps": self.fps,
            }


# ──────────────── leader polling thread ───────────────────────
def leader_collector(robot, state: LeaderState, stop_event: threading.Event, fps: float, debug: bool):
    """Continuously poll the leader arm and publish the latest state.

    Args:
        robot: SDK robot instance used to query hardware state.
        state: Shared state container updated by this thread.
        stop_event: Stop signal for terminating the polling loop.
        fps: Target polling frequency in Hz.
        debug: Whether to print collector exceptions.
    """
    interval = 1.0 / fps
    while not stop_event.is_set():
        t0 = time.perf_counter()
        try:
            raw = robot.get_robot_state("joint_gripper")
            if raw is None:
                dt = time.perf_counter() - t0
                if dt < interval:
                    precise_sleep(interval - dt)
                continue

            angles = list(raw.angles)[:6]
            if len(angles) < 6:
                angles.extend([0.0] * (6 - len(angles)))

            run_text = getattr(raw, "run_status_text", "unknown")
            run_raw  = getattr(robot.servo_driver.data_parser, "_run_status", None)
            grip_val = getattr(raw, "gripper", None)

            decoded = _decode_handle_inputs(run_raw, run_text, gripper_value=grip_val)

            dt_loop = time.perf_counter() - t0
            actual_fps = 1.0 / dt_loop if dt_loop > 0 else 0.0

            state.update(
                angles=angles,
                gripper=grip_val,
                trigger=decoded["trigger"],
                btn1=decoded["button1"],
                btn2=decoded["button2"],
                status_text=run_text,
                status_raw=run_raw,
                fps=actual_fps,
            )

        except Exception as e:
            if debug:
                print(f"[collector] {e}")

        dt = time.perf_counter() - t0
        if dt < interval:
            precise_sleep(interval - dt)


# ─────────────────── HUD overlay helpers ──────────────────────
def _draw_hud(viewer, snap, enabled, mj_gripper):
    """Render the teleoperation HUD in the MuJoCo viewer.

    Args:
        viewer: Active MuJoCo viewer instance.
        snap: Snapshot dictionary returned by ``LeaderState.snapshot``.
        enabled: Whether teleoperation is currently enabled.
        mj_gripper: Current MuJoCo gripper target in meters.
    """
    # Top-left block
    lines_tl = []
    lines_tl.append(("Leader FPS", f"{snap['fps']:.1f}"))
    lines_tl.append(("Left Btn (deadman)", "PRESSED ✓" if snap["button1"] else "released ✗"))
    lines_tl.append(("Right Btn", "PRESSED" if snap["button2"] else "released"))
    lines_tl.append(("Trigger", "ON" if snap["trigger"] else "OFF"))
    lines_tl.append(("Gripper SDK", f"{snap['gripper_value']:.0f}"))
    lines_tl.append(("Gripper MuJoCo", f"{mj_gripper:.4f} m"))
    lines_tl.append(("Status", snap["run_status_text"]))
    raw_hex = f"0x{snap['run_status_raw']:02X}" if isinstance(snap["run_status_raw"], int) else "--"
    lines_tl.append(("Raw status", raw_hex))

    # Connection state – prominent
    if enabled:
        lines_tl.append((">>> TELEOP", "ACTIVE <<<"))
    else:
        lines_tl.append((">>> TELEOP", "DISABLED <<<"))

    # Joint angles
    for i, a in enumerate(snap["joint_angles"]):
        lines_tl.append((f"J{i+1}", f"{math.degrees(a):+8.2f}°"))

    # Build overlay string (MuJoCo overlay uses \n separated pairs)
    overlay_left = "\n".join(k for k, _ in lines_tl)
    overlay_right = "\n".join(v for _, v in lines_tl)

    # Use custom overlay on top-left corner
    viewer.user_scn.ngeom = 0  # not strictly needed, but keeps it clean
    mujoco.mjr_overlay(
        mujoco.mjtFont.mjFONT_NORMAL,
        mujoco.mjtGridPos.mjGRID_TOPLEFT,
        viewer.viewport,
        overlay_left,
        overlay_right,
        viewer.ctx,
    )


# ──────────────────────── main ────────────────────────────────
def main(args):
    """Run the leader-to-follower teleoperation demo.

    Args:
        args: Parsed command-line arguments.

    Workflow:
        1. Connect to the real leader arm.
        2. Start the polling thread.
        3. Load the follower MuJoCo XML model.
        4. Map leader state onto MuJoCo actuators.
        5. Freeze the follower when the deadman switch is released.
    """
    # ── 1. Connect to leader arm ──────────────────────────────
    robot = alicia_d_sdk.create_robot(
        port=args.port,
        variant=args.variant,
        gripper_type=args.gripper_type,
        debug_mode=args.debug,
    )

    print("=" * 66)
    print("  Leader → MuJoCo Follower Teleoperation")
    print("=" * 66)
    print(f"\n[OK] Leader arm connected on {args.port}")
    print(f"[OK] Variant : {args.variant}")

    if args.disable_torque:
        print("[WARN] Disabling torque – support the arm by hand!")
        _try_disable_torque(robot)
        print("[OK] Torque disabled")
    else:
        print("[INFO] Torque kept enabled for stable Leader input monitoring")

    # ── 2. Prepare MuJoCo XML path ────────────────────────────
    xml_path = args.xml
    print(f"\n[INFO] MuJoCo XML: {xml_path}")

    # ── 3. Start leader polling thread ────────────────────────
    state = LeaderState()
    stop_event = threading.Event()

    worker = threading.Thread(
        target=leader_collector,
        args=(robot, state, stop_event, args.fps, args.debug),
        daemon=True,
    )
    worker.start()
    print(f"[OK] Leader polling thread started @ {args.fps:.0f} Hz")

    # ── 4. Frozen position (used when left button is NOT held) ──
    frozen_joints   = [0.0] * 6
    frozen_gripper  = 0.0   # 0 = open (fingers apart at body origins)
    was_enabled     = False

    # ── 5. Launch MuJoCo viewer ───────────────────────────────
    print("\n" + "-" * 66)
    print("  Controls:")
    print("    Hold LEFT BUTTON  → enable teleop (deadman switch)")
    print("    Release LEFT BTN  → follower freezes (disconnected)")
    print("    Trigger (analog)  → gripper open / close")
    print("    Press R           → reload MuJoCo XML and reopen viewer")
    print("    Close window      → quit")
    print("-" * 66 + "\n")

    reload_requested = True
    window_closed = False

    while reload_requested:
        reload_requested = False

        print(f"[INFO] Loading MuJoCo model: {xml_path}")
        try:
            model, data, act_ids, grip_l_act_id, grip_r_act_id = load_mujoco_bundle(xml_path)
        except Exception as exc:
            print(f"[ERROR] Failed to load MuJoCo XML: {exc}")
            break

        print("[OK] MuJoCo model loaded – arm + gripper actuators resolved")

        key_state = {"reload": False}

        def key_callback(keycode):
            """Handle MuJoCo viewer keyboard shortcuts.

            Args:
                keycode: Integer keycode reported by the viewer.
            """
            try:
                key = chr(keycode).lower()
            except ValueError:
                return
            if key == "r":
                key_state["reload"] = True

        # Physics runs at model timestep; viewer renders at ~60 fps.
        # Multiple substeps per render frame keep the simulation real-time.
        render_interval = 1.0 / 60.0                        # ~60 fps target
        n_substeps = max(1, round(render_interval / model.opt.timestep))

        with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
            print(f"[OK] MuJoCo viewer launched – {n_substeps} substeps/frame")
            print("[OK] Waiting for LEFT BUTTON …")

            while viewer.is_running():
                step_start = time.perf_counter()

                if key_state["reload"]:
                    reload_requested = True
                    print("[INFO] Reload requested – reopening viewer from XML")
                    viewer.close()
                    break

                snap = state.snapshot()
                enabled = snap["button1"]

                if enabled:
                    target_joints = snap["joint_angles"]
                    target_gripper = gripper_sdk_to_mujoco(snap["gripper_value"])

                    for idx, aid in enumerate(act_ids):
                        data.ctrl[aid] = target_joints[idx]

                    data.ctrl[grip_l_act_id] = target_gripper
                    data.ctrl[grip_r_act_id] = -target_gripper

                    frozen_joints = list(target_joints)
                    frozen_gripper = target_gripper

                    if not was_enabled:
                        print("[TELEOP] ▶ Connected – follower tracking leader")
                    was_enabled = True

                else:
                    for idx, aid in enumerate(act_ids):
                        data.ctrl[aid] = frozen_joints[idx]

                    data.ctrl[grip_l_act_id] = frozen_gripper
                    data.ctrl[grip_r_act_id] = -frozen_gripper

                    if was_enabled:
                        print("[TELEOP] ■ Disconnected – follower frozen")
                    was_enabled = False

                for _ in range(n_substeps):
                    mujoco.mj_step(model, data)
                viewer.sync()

                elapsed = time.perf_counter() - step_start
                if elapsed < render_interval:
                    precise_sleep(render_interval - elapsed)

            if not reload_requested:
                window_closed = True

    # ── cleanup ───────────────────────────────────────────────
    print("\n[INFO] Viewer closed – shutting down …")
    stop_event.set()
    worker.join(timeout=2.0)
    robot.disconnect()
    print("[OK] Leader disconnected. Bye!")


# ──────────────────── CLI ─────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Leader → MuJoCo Follower Teleoperation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--port", type=str, default="/dev/ttyACM0",
                        help="Serial port of the leader arm")
    parser.add_argument("--gripper_type", type=str, default="50mm",
                        help="Gripper type (deprecated, use --variant)")
    parser.add_argument("--variant", type=str, default="leader",
                        choices=["gripper_50mm", "gripper_100mm", "leader_ur", "leader", "vertical_50mm"],
                        help="Leader arm variant")
    parser.add_argument("--fps", type=float, default=50.0,
                        help="Leader arm polling rate (Hz)")
    parser.add_argument("--xml", type=str, default=DEFAULT_XML,
                        help="Path to follower MuJoCo XML model")
    parser.add_argument("--disable_torque", action="store_true",
                        help="Disable leader torque on start")
    parser.add_argument("--debug", action="store_true",
                        help="Print debug messages")

    args = parser.parse_args()
    main(args)
