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
    import cv2
except ImportError:
    cv2 = None

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
WRIST_CAMERA_NAME = "wrist_realsense"
WRIST_CAMERA_WINDOW = "Wrist Camera (Intel RealSense-style RGB)"
DEFAULT_WRIST_WIDTH = 848
DEFAULT_WRIST_HEIGHT = 480
DEFAULT_WRIST_FPS = 30.0
WRIST_WINDOW_X = 40
WRIST_WINDOW_Y = 40
WRIST_DEBUG_FRAME = PROJECT_ROOT / "logs" / "wrist_camera_debug.png"

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


def create_wrist_camera_renderer(model, width: int, height: int, enabled: bool):
    """Create the off-screen wrist camera renderer when available.

    Args:
        model: MuJoCo model instance containing the wrist camera.
        width: Output image width in pixels.
        height: Output image height in pixels.
        enabled: Whether wrist-camera preview is requested.

    Returns:
        mujoco.Renderer | None: Renderer for the wrist view, or ``None`` if the
        preview is disabled or unavailable.
    """
    if not enabled:
        return None

    if cv2 is None:
        print("[WARN] OpenCV not found – wrist camera preview disabled")
        return None

    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, WRIST_CAMERA_NAME)
    if camera_id < 0:
        print(f"[WARN] MuJoCo camera '{WRIST_CAMERA_NAME}' not found – preview disabled")
        return None

    buffer_width = model.vis.global_.offwidth
    buffer_height = model.vis.global_.offheight
    if width > buffer_width or height > buffer_height:
        print(
            "[WARN] Wrist camera preview exceeds MuJoCo offscreen framebuffer "
            f"({width}x{height} requested, {buffer_width}x{buffer_height} available)"
        )
        return None

    try:
        renderer = mujoco.Renderer(model, height=height, width=width)
    except Exception as exc:
        print(f"[WARN] Failed to initialize wrist camera renderer: {exc}")
        return None

    return renderer


def open_wrist_camera_window(width: int, height: int):
    """Create or reopen the OpenCV window used for the wrist RGB preview."""
    if cv2 is None:
        return

    cv2.startWindowThread()
    cv2.namedWindow(WRIST_CAMERA_WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow(WRIST_CAMERA_WINDOW, height, width)  # swapped: camera rotated 90° CW
    cv2.moveWindow(WRIST_CAMERA_WINDOW, WRIST_WINDOW_X, WRIST_WINDOW_Y)

    topmost_prop = getattr(cv2, "WND_PROP_TOPMOST", None)
    if topmost_prop is not None:
        try:
            cv2.setWindowProperty(WRIST_CAMERA_WINDOW, topmost_prop, 1)
        except cv2.error:
            pass


def show_wrist_camera_placeholder(width: int, height: int, message: str):
    """Render a visible placeholder frame so the preview window is easy to find."""
    if cv2 is None:
        return

    # Swap dimensions to match 90° CW rotation of the camera
    frame = np.zeros((width, height, 3), dtype=np.uint8)
    frame[:] = (28, 28, 28)
    cv2.rectangle(frame, (0, 0), (height - 1, width - 1), (0, 210, 255), 6)
    cv2.putText(
        frame,
        "Wrist Camera Preview",
        (24, 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        message,
        (24, 96),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 210, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"If hidden, look near the top-left corner at ({WRIST_WINDOW_X}, {WRIST_WINDOW_Y})",
        (24, 134),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )

    cv2.imshow(WRIST_CAMERA_WINDOW, frame)
    cv2.waitKey(1)


def save_wrist_camera_debug_frame(frame_bgr):
    """Save the latest wrist frame to disk for debugging window issues."""
    WRIST_DEBUG_FRAME.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(WRIST_DEBUG_FRAME), frame_bgr)


def close_wrist_camera_window():
    """Close the wrist-camera preview window if it is open."""
    if cv2 is None:
        return

    try:
        cv2.destroyWindow(WRIST_CAMERA_WINDOW)
    except cv2.error:
        pass


def render_wrist_camera_frame(renderer, data, width: int, height: int, enabled: bool):
    """Render and show one wrist-camera frame.

    Args:
        renderer: Off-screen MuJoCo renderer created for the wrist camera.
        data: MuJoCo runtime data.
        width: Preview width in pixels.
        height: Preview height in pixels.
        enabled: Whether teleoperation is currently active.

    Returns:
        tuple[bool, bool]: Window-visible flag and whether this frame looks non-empty.
    """
    if renderer is None or cv2 is None:
        return False, False

    renderer.update_scene(data, camera=WRIST_CAMERA_NAME)
    frame_rgb = renderer.render()
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    # Rotate 90° clockwise – physical camera is mounted sideways
    frame_bgr = cv2.rotate(frame_bgr, cv2.ROTATE_90_CLOCKWISE)
    # Flip both axes so that gripper left/right matches the main viewer's left/right.
    # (Wrist cam moves with the gripper, so without this flip the scene drifts
    #  in the opposite direction to the gripper movement.)
    frame_bgr = cv2.flip(frame_bgr, -1)
    rot_h, rot_w = frame_bgr.shape[:2]

    frame_is_nonempty = bool(np.max(frame_bgr) > 0)

    status_text = "TELEOP ACTIVE" if enabled else "TELEOP HOLD"
    cv2.putText(
        frame_bgr,
        f"Intel RealSense-like RGB  {width}x{height}",
        (14, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame_bgr,
        status_text,
        (14, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (80, 220, 120) if enabled else (0, 210, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.rectangle(frame_bgr, (0, 0), (rot_w - 1, rot_h - 1), (0, 210, 255), 4)

    cv2.imshow(WRIST_CAMERA_WINDOW, frame_bgr)
    cv2.waitKey(1)

    try:
        return cv2.getWindowProperty(WRIST_CAMERA_WINDOW, cv2.WND_PROP_VISIBLE) >= 1, frame_is_nonempty
    except cv2.error:
        return False, frame_is_nonempty


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
    print("    Press V           → toggle wrist camera preview")
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

        key_state = {"reload": False, "toggle_wrist": False}
        wrist_renderer = create_wrist_camera_renderer(
            model,
            width=args.wrist_width,
            height=args.wrist_height,
            enabled=args.show_wrist_camera,
        )
        wrist_window_visible = wrist_renderer is not None
        wrist_window_closed_notice = False
        wrist_render_interval = 1.0 / max(args.wrist_fps, 1.0)
        last_wrist_render_time = 0.0
        wrist_first_frame_saved = False
        wrist_nonempty_frame_seen = False

        if wrist_window_visible:
            open_wrist_camera_window(args.wrist_width, args.wrist_height)
            show_wrist_camera_placeholder(
                args.wrist_width,
                args.wrist_height,
                "Initializing MuJoCo wrist camera...",
            )
            print(
                f"[OK] Wrist camera preview opened – "
                f"{args.wrist_width}x{args.wrist_height} @ {args.wrist_fps:.0f} Hz"
            )
            print(f"[INFO] Wrist preview window moved to ({WRIST_WINDOW_X}, {WRIST_WINDOW_Y})")

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
            elif key == "v":
                key_state["toggle_wrist"] = True

        # Physics runs at model timestep; viewer renders at ~60 fps.
        # n_substeps is chosen so that n_substeps * dt ≈ 1/60 s.
        # render_interval is then snapped to the exact multiple of dt so that
        # simulated time == wall-clock time (no silent slow-down from rounding).
        n_substeps = max(1, round(1.0 / (60.0 * model.opt.timestep)))
        render_interval = n_substeps * model.opt.timestep   # exact wall-clock target

        with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
            print(f"[OK] MuJoCo viewer launched – {n_substeps} substeps/frame")
            print("[OK] Waiting for LEFT BUTTON …")

            while viewer.is_running():
                step_start = time.perf_counter()

                if key_state["reload"]:
                    reload_requested = True
                    print("[INFO] Reload requested – reopening viewer from XML")
                    close_wrist_camera_window()
                    if wrist_renderer is not None:
                        close_fn = getattr(wrist_renderer, "close", None)
                        if callable(close_fn):
                            close_fn()
                    viewer.close()
                    break

                if key_state["toggle_wrist"]:
                    key_state["toggle_wrist"] = False
                    if wrist_renderer is None:
                        print("[WARN] Wrist camera preview is unavailable")
                    else:
                        wrist_window_visible = not wrist_window_visible
                        wrist_window_closed_notice = False
                        if wrist_window_visible:
                            open_wrist_camera_window(args.wrist_width, args.wrist_height)
                            show_wrist_camera_placeholder(
                                args.wrist_width,
                                args.wrist_height,
                                "Wrist camera preview resumed",
                            )
                            print("[INFO] Wrist camera preview enabled")
                        else:
                            close_wrist_camera_window()
                            print("[INFO] Wrist camera preview hidden")

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

                if wrist_renderer is not None and wrist_window_visible:
                    now = time.perf_counter()
                    if now - last_wrist_render_time >= wrist_render_interval:
                        wrist_window_visible, frame_is_nonempty = render_wrist_camera_frame(
                            wrist_renderer,
                            data,
                            width=args.wrist_width,
                            height=args.wrist_height,
                            enabled=enabled,
                        )
                        wrist_nonempty_frame_seen = wrist_nonempty_frame_seen or frame_is_nonempty
                        if not wrist_first_frame_saved:
                            wrist_renderer.update_scene(data, camera=WRIST_CAMERA_NAME)
                            debug_frame_rgb = wrist_renderer.render()
                            debug_frame_bgr = cv2.cvtColor(debug_frame_rgb, cv2.COLOR_RGB2BGR)
                            save_wrist_camera_debug_frame(debug_frame_bgr)
                            wrist_first_frame_saved = True
                            print(f"[INFO] Wrist debug frame saved to {WRIST_DEBUG_FRAME}")
                        if not wrist_nonempty_frame_seen and wrist_first_frame_saved:
                            print("[WARN] Wrist camera is rendering, but the image is still fully black")
                            wrist_nonempty_frame_seen = True
                        last_wrist_render_time = now
                        if not wrist_window_visible and not wrist_window_closed_notice:
                            print("[INFO] Wrist camera window closed – press V to reopen")
                            wrist_window_closed_notice = True

                elapsed = time.perf_counter() - step_start
                if elapsed < render_interval:
                    precise_sleep(render_interval - elapsed)

            if not reload_requested:
                window_closed = True

        close_wrist_camera_window()
        if wrist_renderer is not None:
            close_fn = getattr(wrist_renderer, "close", None)
            if callable(close_fn):
                close_fn()

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
    parser.add_argument("--no_wrist_camera", dest="show_wrist_camera", action="store_false",
                        help="Disable the wrist RGB preview window")
    parser.add_argument("--wrist_width", type=int, default=DEFAULT_WRIST_WIDTH,
                        help="Wrist camera preview width in pixels")
    parser.add_argument("--wrist_height", type=int, default=DEFAULT_WRIST_HEIGHT,
                        help="Wrist camera preview height in pixels")
    parser.add_argument("--wrist_fps", type=float, default=DEFAULT_WRIST_FPS,
                        help="Wrist camera preview refresh rate (Hz)")

    args = parser.parse_args()
    if args.wrist_width <= 0 or args.wrist_height <= 0:
        parser.error("--wrist_width and --wrist_height must be positive")
    if args.wrist_fps <= 0:
        parser.error("--wrist_fps must be positive")
    main(args)
