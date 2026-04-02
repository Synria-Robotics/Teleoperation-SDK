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

"""Leader-to-MuJoCo follower teleoperation demo with dual camera previews.

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
FRONT_CAMERA_NAME = "front_overview"
FRONT_CAMERA_WINDOW = "Front Camera (Arm Overview)"
DEFAULT_FRONT_WIDTH = 640
DEFAULT_FRONT_HEIGHT = 480
DEFAULT_FRONT_FPS = 20.0
FRONT_WINDOW_X = 560
FRONT_WINDOW_Y = 40
FRONT_DEBUG_FRAME = PROJECT_ROOT / "logs" / "front_camera_debug.png"

# Default MuJoCo XML path (from user-provided XML)

DEFAULT_XML = str(
    PROJECT_ROOT
    / "assets" / "mujoco" / "Alicia_D_v5_6" / "gripper_50mm" / "alicia_d_follower_dual_camera.xml"
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


def create_camera_renderer(model, camera_name: str, width: int, height: int, enabled: bool, label: str):
    """Create an off-screen MuJoCo renderer for a named camera preview."""
    if not enabled:
        return None

    if cv2 is None:
        print(f"[WARN] OpenCV not found – {label} disabled")
        return None

    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if camera_id < 0:
        print(f"[WARN] MuJoCo camera '{camera_name}' not found – {label} disabled")
        return None

    buffer_width = model.vis.global_.offwidth
    buffer_height = model.vis.global_.offheight
    if width > buffer_width or height > buffer_height:
        print(
            f"[WARN] {label} exceeds MuJoCo offscreen framebuffer "
            f"({width}x{height} requested, {buffer_width}x{buffer_height} available)"
        )
        return None

    try:
        return mujoco.Renderer(model, height=height, width=width)
    except Exception as exc:
        print(f"[WARN] Failed to initialize {label}: {exc}")
        return None


def open_camera_window(window_name: str, width: int, height: int, window_x: int, window_y: int, rotate_90_cw: bool):
    """Create or reopen an OpenCV preview window."""
    if cv2 is None:
        return

    cv2.startWindowThread()
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
    display_width = height if rotate_90_cw else width
    display_height = width if rotate_90_cw else height
    cv2.resizeWindow(window_name, display_width, display_height)
    cv2.moveWindow(window_name, window_x, window_y)

    topmost_prop = getattr(cv2, "WND_PROP_TOPMOST", None)
    if topmost_prop is not None:
        try:
            cv2.setWindowProperty(window_name, topmost_prop, 1)
        except cv2.error:
            pass


def show_camera_placeholder(
    window_name: str,
    title: str,
    width: int,
    height: int,
    message: str,
    window_x: int,
    window_y: int,
    rotate_90_cw: bool,
):
    """Render a placeholder frame so each preview window is easy to find."""
    if cv2 is None:
        return

    frame_shape = (width, height, 3) if rotate_90_cw else (height, width, 3)
    frame = np.zeros(frame_shape, dtype=np.uint8)
    frame[:] = (28, 28, 28)
    frame_h, frame_w = frame.shape[:2]

    cv2.rectangle(frame, (0, 0), (frame_w - 1, frame_h - 1), (0, 210, 255), 6)
    cv2.putText(
        frame,
        title,
        (24, 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
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
        f"If hidden, look near ({window_x}, {window_y})",
        (24, 134),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )

    cv2.imshow(window_name, frame)
    cv2.waitKey(1)


def save_camera_debug_frame(debug_path: Path, frame_bgr):
    """Save the latest preview frame to disk for debugging window issues."""
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(debug_path), frame_bgr)


def close_camera_window(window_name: str):
    """Close an OpenCV preview window if it is open."""
    if cv2 is None:
        return

    try:
        cv2.destroyWindow(window_name)
    except cv2.error:
        pass


def render_camera_frame(
    renderer,
    data,
    camera_name: str,
    window_name: str,
    width: int,
    height: int,
    enabled: bool,
    overlay_title: str,
    rotate_90_cw: bool = False,
    flip_code=None,
):
    """Render and show one preview frame for a named MuJoCo camera."""
    if renderer is None or cv2 is None:
        return False, False, None

    renderer.update_scene(data, camera=camera_name)
    frame_rgb = renderer.render()
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    if rotate_90_cw:
        frame_bgr = cv2.rotate(frame_bgr, cv2.ROTATE_90_CLOCKWISE)
    if flip_code is not None:
        frame_bgr = cv2.flip(frame_bgr, flip_code)

    frame_h, frame_w = frame_bgr.shape[:2]
    frame_is_nonempty = bool(np.max(frame_bgr) > 0)

    status_text = "TELEOP ACTIVE" if enabled else "TELEOP HOLD"
    cv2.putText(
        frame_bgr,
        f"{overlay_title}  {width}x{height}",
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
    cv2.rectangle(frame_bgr, (0, 0), (frame_w - 1, frame_h - 1), (0, 210, 255), 4)

    cv2.imshow(window_name, frame_bgr)
    cv2.waitKey(1)

    try:
        window_visible = cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error:
        window_visible = False

    return window_visible, frame_is_nonempty, frame_bgr


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
    print("  Leader → MuJoCo Follower Teleoperation (Dual Camera)")
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
    print("    Press B           → toggle front camera preview")
    print("    Press R           → reload MuJoCo XML and reopen viewer")
    print("    Close window      → quit")
    print("-" * 66 + "\n")

    reload_requested = True

    while reload_requested:
        reload_requested = False

        print(f"[INFO] Loading MuJoCo model: {xml_path}")
        try:
            model, data, act_ids, grip_l_act_id, grip_r_act_id = load_mujoco_bundle(xml_path)
        except Exception as exc:
            print(f"[ERROR] Failed to load MuJoCo XML: {exc}")
            break

        print("[OK] MuJoCo model loaded – arm + gripper actuators resolved")

        key_state = {"reload": False, "toggle_wrist": False, "toggle_front": False}
        camera_previews = {
            "wrist": {
                "camera_name": WRIST_CAMERA_NAME,
                "window_name": WRIST_CAMERA_WINDOW,
                "label": "Wrist camera preview",
                "title": "Intel RealSense-like RGB",
                "width": args.wrist_width,
                "height": args.wrist_height,
                "fps": args.wrist_fps,
                "enabled": args.show_wrist_camera,
                "window_x": WRIST_WINDOW_X,
                "window_y": WRIST_WINDOW_Y,
                "rotate_90_cw": True,
                "flip_code": -1,
                "debug_frame": WRIST_DEBUG_FRAME,
                "toggle_state_key": "toggle_wrist",
                "shortcut": "V",
                "init_message": "Initializing MuJoCo wrist camera...",
                "resume_message": "Wrist camera preview resumed",
            },
            "front": {
                "camera_name": FRONT_CAMERA_NAME,
                "window_name": FRONT_CAMERA_WINDOW,
                "label": "Front camera preview",
                "title": "Front Overview Camera",
                "width": args.front_width,
                "height": args.front_height,
                "fps": args.front_fps,
                "enabled": args.show_front_camera,
                "window_x": FRONT_WINDOW_X,
                "window_y": FRONT_WINDOW_Y,
                "rotate_90_cw": False,
                "flip_code": None,
                "debug_frame": FRONT_DEBUG_FRAME,
                "toggle_state_key": "toggle_front",
                "shortcut": "B",
                "init_message": "Initializing MuJoCo front camera...",
                "resume_message": "Front camera preview resumed",
            },
        }

        for preview in camera_previews.values():
            preview["renderer"] = create_camera_renderer(
                model,
                camera_name=preview["camera_name"],
                width=preview["width"],
                height=preview["height"],
                enabled=preview["enabled"],
                label=preview["label"],
            )
            preview["visible"] = preview["renderer"] is not None
            preview["closed_notice"] = False
            preview["render_interval"] = 1.0 / max(preview["fps"], 1.0)
            preview["last_render_time"] = 0.0
            preview["first_frame_saved"] = False
            preview["nonempty_frame_seen"] = False

            if preview["visible"]:
                open_camera_window(
                    preview["window_name"],
                    preview["width"],
                    preview["height"],
                    preview["window_x"],
                    preview["window_y"],
                    preview["rotate_90_cw"],
                )
                show_camera_placeholder(
                    preview["window_name"],
                    preview["title"],
                    preview["width"],
                    preview["height"],
                    preview["init_message"],
                    preview["window_x"],
                    preview["window_y"],
                    preview["rotate_90_cw"],
                )
                print(
                    f"[OK] {preview['label']} opened – "
                    f"{preview['width']}x{preview['height']} @ {preview['fps']:.0f} Hz"
                )
                print(
                    f"[INFO] {preview['label']} window moved to "
                    f"({preview['window_x']}, {preview['window_y']})"
                )

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
            elif key == "b":
                key_state["toggle_front"] = True

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
                    for preview in camera_previews.values():
                        close_camera_window(preview["window_name"])
                        close_fn = getattr(preview["renderer"], "close", None)
                        if callable(close_fn):
                            close_fn()
                    viewer.close()
                    break

                for preview in camera_previews.values():
                    toggle_state_key = preview["toggle_state_key"]
                    if not key_state[toggle_state_key]:
                        continue

                    key_state[toggle_state_key] = False
                    if preview["renderer"] is None:
                        print(f"[WARN] {preview['label']} is unavailable")
                        continue

                    preview["visible"] = not preview["visible"]
                    preview["closed_notice"] = False
                    if preview["visible"]:
                        open_camera_window(
                            preview["window_name"],
                            preview["width"],
                            preview["height"],
                            preview["window_x"],
                            preview["window_y"],
                            preview["rotate_90_cw"],
                        )
                        show_camera_placeholder(
                            preview["window_name"],
                            preview["title"],
                            preview["width"],
                            preview["height"],
                            preview["resume_message"],
                            preview["window_x"],
                            preview["window_y"],
                            preview["rotate_90_cw"],
                        )
                        print(f"[INFO] {preview['label']} enabled")
                    else:
                        close_camera_window(preview["window_name"])
                        print(f"[INFO] {preview['label']} hidden")

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

                now = time.perf_counter()
                for preview in camera_previews.values():
                    if preview["renderer"] is None or not preview["visible"]:
                        continue
                    if now - preview["last_render_time"] < preview["render_interval"]:
                        continue

                    preview["visible"], frame_is_nonempty, frame_bgr = render_camera_frame(
                        preview["renderer"],
                        data,
                        camera_name=preview["camera_name"],
                        window_name=preview["window_name"],
                        width=preview["width"],
                        height=preview["height"],
                        enabled=enabled,
                        overlay_title=preview["title"],
                        rotate_90_cw=preview["rotate_90_cw"],
                        flip_code=preview["flip_code"],
                    )
                    preview["nonempty_frame_seen"] = (
                        preview["nonempty_frame_seen"] or frame_is_nonempty
                    )
                    if not preview["first_frame_saved"] and frame_bgr is not None:
                        save_camera_debug_frame(preview["debug_frame"], frame_bgr)
                        preview["first_frame_saved"] = True
                        print(
                            f"[INFO] {preview['label']} debug frame saved to "
                            f"{preview['debug_frame']}"
                        )
                    if not preview["nonempty_frame_seen"] and preview["first_frame_saved"]:
                        print(
                            f"[WARN] {preview['label']} is rendering, "
                            "but the image is still fully black"
                        )
                        preview["nonempty_frame_seen"] = True
                    preview["last_render_time"] = now
                    if not preview["visible"] and not preview["closed_notice"]:
                        print(
                            f"[INFO] {preview['label']} window closed – "
                            f"press {preview['shortcut']} to reopen"
                        )
                        preview["closed_notice"] = True

                elapsed = time.perf_counter() - step_start
                if elapsed < render_interval:
                    precise_sleep(render_interval - elapsed)

        for preview in camera_previews.values():
            close_camera_window(preview["window_name"])
            close_fn = getattr(preview["renderer"], "close", None)
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
        description="Leader → MuJoCo Follower Teleoperation (Dual Camera)",
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
    parser.add_argument("--no_front_camera", dest="show_front_camera", action="store_false",
                        help="Disable the front overview camera preview window")
    parser.add_argument("--front_width", type=int, default=DEFAULT_FRONT_WIDTH,
                        help="Front camera preview width in pixels")
    parser.add_argument("--front_height", type=int, default=DEFAULT_FRONT_HEIGHT,
                        help="Front camera preview height in pixels")
    parser.add_argument("--front_fps", type=float, default=DEFAULT_FRONT_FPS,
                        help="Front camera preview refresh rate (Hz)")

    args = parser.parse_args()
    if args.wrist_width <= 0 or args.wrist_height <= 0:
        parser.error("--wrist_width and --wrist_height must be positive")
    if args.wrist_fps <= 0:
        parser.error("--wrist_fps must be positive")
    if args.front_width <= 0 or args.front_height <= 0:
        parser.error("--front_width and --front_height must be positive")
    if args.front_fps <= 0:
        parser.error("--front_fps must be positive")
    main(args)
