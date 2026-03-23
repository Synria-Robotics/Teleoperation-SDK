#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Synria Robotics Co., Ltd.
# Developer: Xuhui Zhou, Synria
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
"""
Demo: Realtime Leader arm monitoring with joint plots and input status panel.

Standalone version for Teleoperation-SDK.
Requires: alicia_d_sdk (pip install -e /home/ubuntu22/Alicia-D-SDK), matplotlib
"""

import math
import time
import argparse
import threading
from collections import deque
from pathlib import Path
import sys

# ── Make alicia_d_sdk importable (fallback if not pip-installed) ──
_SDK_PATH = str(Path("/home/ubuntu22/Alicia-D-SDK"))
if _SDK_PATH not in sys.path:
    sys.path.insert(0, _SDK_PATH)

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

# ── UI Color Palette ──────────────────────────────────────────────────────────
BG_MAIN    = '#0A0E17'   # window background
BG_PANEL   = '#0F1624'   # subplot / panel background
BG_CARD    = '#141C2E'   # card / inset background
CLR_BORDER = '#1E3D5C'   # subtle border
CLR_GRID   = '#172035'   # grid lines
CLR_DIM    = '#4A6070'   # dimmed / secondary text
CLR_CYAN   = '#00D4FF'   # primary accent
CLR_ORANGE = '#FF8C42'   # warning / right-button accent
CLR_GREEN  = '#2EFFA0'   # active / OK green
CLR_PURPLE = '#C77DFF'   # J5 accent
CLR_GOLD   = '#FFD166'   # J6 accent
CLR_RED    = '#FF3A5C'   # alert red

JOINT_COLORS = [CLR_CYAN] * 6

import alicia_d_sdk
from utils import precise_sleep


def _try_disable_torque(robot):
    """Disable torque using whichever SDK method is available."""
    if hasattr(robot, "torque_control"):
        robot.torque_control("off")
        return
    if hasattr(robot, "disable_torque"):
        robot.disable_torque()
        return
    if hasattr(robot, "torque_enable"):
        robot.torque_enable(False)
        return


# Leader trigger threshold:
# gripper range is 0-1000, where 1000 means released and 0 means fully pressed.
TRIGGER_THRESHOLD = 900


def _decode_handle_inputs(raw_status, run_status_text, gripper_value=None):
    """Decode Leader arm trigger and left/right button states.

    - Trigger: analog gripper channel, typically 1000 -> 0 while pressing
    - Left button: run_status bit4 (0x10) / "sync"
    - Right button: run_status bit0 (0x01) / "locked"
    - Both buttons: 0x11 / "sync_locked"
    """
    left_button = False
    right_button = False
    trigger = False

    if gripper_value is not None:
        trigger = gripper_value < TRIGGER_THRESHOLD

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

    return {
        "trigger": trigger,
        "button1": left_button,
        "button2": right_button,
    }


class StatusVisualizer:
    """Visual status panel — dark HUD / industrial style."""

    def __init__(self, ax):
        self.ax = ax
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, 10)
        self.ax.axis('off')
        self.ax.set_facecolor(BG_CARD)

        # ── Header bar ────────────────────────────────────────
        self.ax.add_patch(Rectangle((0, 9.35), 10, 0.65,
                                     facecolor=CLR_CYAN, linewidth=0))
        self.ax.text(5, 9.67, 'INPUT  STATUS', ha='center', va='center',
                     fontsize=9.5, fontweight='bold', color=BG_MAIN,
                     fontfamily='monospace')

        # corner bracket marks
        for x0, y0, dx, dy in [(0.08, 9.25, 0.5, 0), (0.08, 9.25, 0, -0.35),
                                 (9.92, 9.25, -0.5, 0), (9.92, 9.25, 0, -0.35)]:
            self.ax.plot([x0, x0 + dx], [y0, y0 + dy],
                         color=CLR_CYAN, lw=1.2, alpha=0.55)

        # ── Trigger card ──────────────────────────────────────
        self.ax.add_patch(Rectangle((0.35, 6.65), 9.3, 2.45,
                                     facecolor=BG_PANEL,
                                     edgecolor=CLR_BORDER, linewidth=1.2))
        self.ax.text(0.75, 8.82, 'TRIGGER', ha='left', va='center',
                     fontsize=6.5, fontweight='bold', color=CLR_DIM,
                     fontfamily='monospace')

        # gripper progress bar (background + fill)
        self.trig_bar_bg = Rectangle((0.5, 7.65), 9.0, 0.82,
                                      facecolor=BG_MAIN,
                                      edgecolor=CLR_BORDER, linewidth=0.8)
        self.ax.add_patch(self.trig_bar_bg)
        self.trig_bar = Rectangle((0.5, 7.65), 0.0, 0.82,
                                   facecolor=CLR_GREEN, linewidth=0)
        self.ax.add_patch(self.trig_bar)

        # gauge tick marks
        for k in range(11):
            tx = 0.5 + 9.0 * k / 10.0
            self.ax.plot([tx, tx], [7.65, 7.78],
                         color=CLR_DIM, lw=0.7, alpha=0.55)

        self.trigger_status = self.ax.text(
            5, 7.1, '○  OFF    0%  [0]',
            ha='center', va='center',
            fontsize=11, fontweight='bold', color=CLR_DIM,
            fontfamily='monospace')

        # ── Section divider ───────────────────────────────────
        self.ax.plot([0.35, 9.65], [6.55, 6.55],
                     color=CLR_BORDER, lw=0.8, alpha=0.8)
        self.ax.text(5, 6.3, 'B U T T O N S', ha='center', va='center',
                     fontsize=6.5, color=CLR_DIM, fontfamily='monospace',
                     alpha=0.7)

        # ── Left button card ──────────────────────────────────
        self.btn1_rect = Rectangle((0.35, 3.75), 3.95, 2.3,
                                    facecolor=BG_PANEL,
                                    edgecolor=CLR_BORDER, linewidth=1.5)
        self.ax.add_patch(self.btn1_rect)
        self.btn1_label = self.ax.text(
            2.33, 5.3, 'L', ha='center', va='center',
            fontsize=24, fontweight='bold', color=CLR_DIM,
            fontfamily='monospace')
        self.btn1_text = self.ax.text(
            2.33, 4.1, 'LEFT\n[0]', ha='center', va='center',
            fontsize=8, color=CLR_DIM, fontfamily='monospace')

        # ── Right button card ─────────────────────────────────
        self.btn2_rect = Rectangle((5.7, 3.75), 3.95, 2.3,
                                    facecolor=BG_PANEL,
                                    edgecolor=CLR_BORDER, linewidth=1.5)
        self.ax.add_patch(self.btn2_rect)
        self.btn2_label = self.ax.text(
            7.67, 5.3, 'R', ha='center', va='center',
            fontsize=24, fontweight='bold', color=CLR_DIM,
            fontfamily='monospace')
        self.btn2_text = self.ax.text(
            7.67, 4.1, 'RIGHT\n[0]', ha='center', va='center',
            fontsize=8, color=CLR_DIM, fontfamily='monospace')

        # ── Status bar ────────────────────────────────────────
        self.ax.add_patch(Rectangle((0.35, 2.35), 9.3, 1.1,
                                     facecolor=BG_PANEL,
                                     edgecolor=CLR_BORDER, linewidth=1.0))
        self.ax.text(0.75, 3.22, 'STATUS', ha='left', va='center',
                     fontsize=6, color=CLR_DIM, fontfamily='monospace')
        self.run_status_text = self.ax.text(
            5, 2.82, 'UNKNOWN', ha='center', va='center',
            fontsize=9, color=CLR_CYAN, fontfamily='monospace')

        # ── Raw / gripper data card ───────────────────────────
        self.ax.add_patch(Rectangle((0.35, 0.45), 9.3, 1.65,
                                     facecolor=BG_PANEL,
                                     edgecolor=CLR_BORDER, linewidth=1.0))
        self.ax.text(0.75, 1.87, 'RAW DATA', ha='left', va='center',
                     fontsize=6, color=CLR_DIM, fontfamily='monospace')
        self.raw_status_text = self.ax.text(
            5, 1.18, 'STATUS: --    GRIP: --', ha='center', va='center',
            fontsize=7.5, color=CLR_DIM, fontfamily='monospace')

        # bottom rule
        self.ax.plot([0.08, 9.92], [0.12, 0.12],
                     color=CLR_CYAN, lw=0.8, alpha=0.35)

        # state counters
        self.trigger_count = 0
        self.btn1_count    = 0
        self.btn2_count    = 0
        self.last_trigger  = False
        self.last_btn1     = False
        self.last_btn2     = False

    def update(self, trigger, button1, button2, run_status_text, raw_status, gripper_value):
        if trigger and not self.last_trigger:
            self.trigger_count += 1
        if button1 and not self.last_btn1:
            self.btn1_count += 1
        if button2 and not self.last_btn2:
            self.btn2_count += 1

        self.last_trigger = trigger
        self.last_btn1    = button1
        self.last_btn2    = button2

        grip_pct = max(0, min(100, int((1000 - (gripper_value or 1000)) / 10)))
        self.trig_bar.set_width(9.0 * grip_pct / 100.0)

        if trigger:
            self.trig_bar.set_facecolor(CLR_GREEN)
            self.trigger_status.set_text(
                f'●  ON    {grip_pct:3d}%  [{self.trigger_count}]')
            self.trigger_status.set_color(CLR_GREEN)
        else:
            self.trig_bar.set_facecolor(CLR_BORDER)
            self.trigger_status.set_text(
                f'○  OFF   {grip_pct:3d}%  [{self.trigger_count}]')
            self.trigger_status.set_color(CLR_DIM)

        if button1:
            self.btn1_rect.set_facecolor('#041830')
            self.btn1_rect.set_edgecolor(CLR_CYAN)
            self.btn1_label.set_color(CLR_CYAN)
            self.btn1_text.set_color(CLR_CYAN)
        else:
            self.btn1_rect.set_facecolor(BG_PANEL)
            self.btn1_rect.set_edgecolor(CLR_BORDER)
            self.btn1_label.set_color(CLR_DIM)
            self.btn1_text.set_color(CLR_DIM)
        self.btn1_text.set_text(f'LEFT\n[{self.btn1_count}]')

        if button2:
            self.btn2_rect.set_facecolor('#041830')
            self.btn2_rect.set_edgecolor(CLR_CYAN)
            self.btn2_label.set_color(CLR_CYAN)
            self.btn2_text.set_color(CLR_CYAN)
        else:
            self.btn2_rect.set_facecolor(BG_PANEL)
            self.btn2_rect.set_edgecolor(CLR_BORDER)
            self.btn2_label.set_color(CLR_DIM)
            self.btn2_text.set_color(CLR_DIM)
        self.btn2_text.set_text(f'RIGHT\n[{self.btn2_count}]')

        self.run_status_text.set_text(
            str(run_status_text).upper() if run_status_text else 'UNKNOWN')

        raw_text  = f'0x{raw_status:02X}' if isinstance(raw_status, int) else '--'
        grip_text = f'{gripper_value:.0f}'  if gripper_value is not None else '--'
        self.raw_status_text.set_text(f'STATUS: {raw_text}    GRIP: {grip_text}')


def main(args):
    robot = alicia_d_sdk.create_robot(
        port=args.port,
        variant=args.variant,
        gripper_type=args.gripper_type,
        debug_mode=args.debug,
    )

    print("\n" + "━" * 64)
    print("  ALICIA-D  ·  LEADER MONITOR  ·  JOINT ANGLES & INPUT STATUS")
    print("━" * 64)
    print(f"\n  ✓  CONNECTED  →  {args.port}")
    print(f"  ✓  VARIANT    →  {args.variant}")
    if args.disable_torque:
        print("  ⚠  TORQUE OFF — please support the arm by hand ...")
        _try_disable_torque(robot)
        print("  ✓  TORQUE DISABLED\n")
    else:
        print("  ✓  TORQUE ENABLED  (stable leader monitoring)\n")

    buffer_len = max(50, int(args.fps * args.history_sec))
    time_buf = deque(maxlen=buffer_len)
    joint_buf = [deque(maxlen=buffer_len) for _ in range(6)]

    status_lock = threading.Lock()
    status_info = {
        "run_status_text": "unknown",
        "run_status_raw": None,
        "gripper_value": None,
        "trigger": False,
        "button1": False,
        "button2": False,
    }

    stop_event = threading.Event()
    start_t = time.perf_counter()

    def collector():
        interval = 1.0 / args.fps
        while not stop_event.is_set():
            t0 = time.perf_counter()
            try:
                state = robot.get_robot_state("joint_gripper")
                if state is None:
                    values = [float("nan")] * 6
                else:
                    values = list(state.angles)[:6]
                    if len(values) < 6:
                        values.extend([float("nan")] * (6 - len(values)))

                run_status_text = getattr(state, "run_status_text", "unknown") if state is not None else "unknown"
                run_status_raw = getattr(robot.servo_driver.data_parser, "_run_status", None)
                gripper_value = getattr(state, "gripper", None) if state is not None else None

                decoded = _decode_handle_inputs(
                    run_status_raw,
                    run_status_text,
                    gripper_value=gripper_value,
                )
                
                with status_lock:
                    status_info["run_status_text"] = run_status_text
                    status_info["run_status_raw"] = run_status_raw
                    status_info["gripper_value"] = gripper_value
                    status_info["trigger"] = decoded["trigger"]
                    status_info["button1"] = decoded["button1"]
                    status_info["button2"] = decoded["button2"]

                if args.format == "deg":
                    values = [
                        math.degrees(v) if isinstance(v, (int, float)) and not math.isnan(v) else float("nan") 
                        for v in values
                    ]

                t_rel = time.perf_counter() - start_t
                time_buf.append(t_rel)
                for i in range(6):
                    joint_buf[i].append(values[i])
                    
            except Exception as e:
                if args.debug:
                    print(f"Error: {e}")
                t_rel = time.perf_counter() - start_t
                time_buf.append(t_rel)
                for i in range(6):
                    joint_buf[i].append(float("nan"))

            dt = time.perf_counter() - t0
            if dt < interval:
                precise_sleep(interval - dt)

    worker = threading.Thread(target=collector, daemon=True)
    worker.start()

    plt.style.use('dark_background')
    fig = plt.figure(figsize=(18, 9))
    fig.patch.set_facecolor(BG_MAIN)

    axes = []
    for i in range(6):
        row = i // 2
        col = i % 2
        ax = plt.subplot2grid((3, 3), (row, col), fig=fig)
        axes.append(ax)

    status_ax = plt.subplot2grid((3, 3), (0, 2), rowspan=3, fig=fig)
    status_ax.set_facecolor(BG_CARD)
    status_visualizer = StatusVisualizer(status_ax)

    lines = []
    y_unit = "deg" if args.format == "deg" else "rad"

    for i, ax in enumerate(axes):
        line, = ax.plot([], [], lw=2.0, color=JOINT_COLORS[i],
                        alpha=0.92, solid_capstyle='round')
        lines.append(line)
        ax.set_facecolor(BG_PANEL)
        ax.set_title(f'J{i + 1}', fontsize=11, fontweight='bold',
                     color=JOINT_COLORS[i], fontfamily='monospace', pad=4)
        ax.set_ylabel(y_unit, fontsize=8, color=CLR_DIM, fontfamily='monospace')
        ax.set_xlabel('time (s)', fontsize=8, color=CLR_DIM, fontfamily='monospace')
        ax.tick_params(axis='both', color=CLR_DIM, labelcolor=CLR_DIM, labelsize=7)
        ax.grid(True, color=CLR_GRID, linestyle='-', linewidth=0.6)
        for spine in ax.spines.values():
            spine.set_color(CLR_BORDER)
            spine.set_linewidth(1.0)

    plt.subplots_adjust(left=0.06, right=0.96, top=0.92, bottom=0.08,
                        hspace=0.42, wspace=0.32)

    fig.suptitle(
        f'ALICIA-D  ·  LEADER MONITOR  ·  {args.port}  ·  {args.fps:.0f} Hz',
        fontsize=12, fontweight='bold', color=CLR_CYAN,
        fontfamily='monospace', y=0.977,
    )
    fig.add_artist(Line2D([0.04, 0.96], [0.952, 0.952],
                           transform=fig.transFigure,
                           color=CLR_CYAN, linewidth=0.8, alpha=0.45))
    fig.text(0.975, 0.012, 'SYNRIA ROBOTICS', ha='right', va='bottom',
             fontsize=6.5, color=CLR_BORDER, fontfamily='monospace')

    def update(_frame):
        if len(time_buf) < 2:
            return lines

        xs = list(time_buf)
        x_min = max(0.0, xs[-1] - args.history_sec)
        x_max = max(args.history_sec, xs[-1])

        for i, line in enumerate(lines):
            ys = list(joint_buf[i])
            line.set_data(xs, ys)
            ax = axes[i]
            ax.set_xlim(x_min, x_max)

            valid = [v for v in ys if not math.isnan(v)]
            if valid:
                ymin = min(valid)
                ymax = max(valid)
                if abs(ymax - ymin) < 1e-6:
                    pad = 1.0
                else:
                    pad = (ymax - ymin) * 0.15
                ax.set_ylim(ymin - pad, ymax + pad)

        with status_lock:
            status_visualizer.update(
                trigger=status_info["trigger"],
                button1=status_info["button1"],
                button2=status_info["button2"],
                run_status_text=status_info["run_status_text"],
                raw_status=status_info["run_status_raw"],
                gripper_value=status_info["gripper_value"],
            )

        return lines

    ani = FuncAnimation(
        fig,
        update,
        interval=max(20, int(1000 / args.plot_fps)),
        blit=False,
        cache_frame_data=False,
    )

    print("  ✓  VISUALIZATION READY\n")
    print("  CONTROLS:")
    print("  →  Move Leader arm to observe joint trajectories")
    print("  →  L / R buttons  →  update button indicators")
    print("  →  Pull trigger   →  update trigger panel & gauge")
    print("  →  Close window   →  exit\n")
    print("━" * 64 + "\n")

    try:
        plt.show()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        stop_event.set()
        worker.join(timeout=1.0)
        robot.disconnect()
        del ani
        print("\n" + "━" * 64)
        print("  ✓  MONITORING STOPPED  ·  ROBOT DISCONNECTED")
        print("━" * 64 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Realtime monitor for six joints and Leader input states",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--port", type=str, default="/dev/ttyACM0")
    parser.add_argument("--gripper_type", type=str, default="50mm")
    parser.add_argument("--variant", type=str, default="leader", choices=["gripper_50mm", "gripper_100mm", "leader_ur", "leader", "vertical_50mm"])
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--plot_fps", type=float, default=15.0)
    parser.add_argument("--history_sec", type=float, default=15.0)
    parser.add_argument("--format", type=str, choices=["rad", "deg"], default="deg")
    parser.add_argument("--disable_torque", action='store_true')
    parser.add_argument("--debug", action='store_true')
    
    args = parser.parse_args()
    main(args)
