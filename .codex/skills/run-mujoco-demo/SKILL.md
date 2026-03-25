---
name: run-mujoco-demo
description: Launch and troubleshoot the Teleoperation-SDK MuJoCo follower demo in this repository. Use when a request asks to run, restart, or debug `02_demo_mujoco_follower.py`, the MuJoCo follower viewer, or the Alicia-D teleoperation demo, especially in the local `py311` conda environment on macOS where `mjpython` is required.
---

# Run Mujoco Demo

## Overview

Use the bundled launcher first. It validates the conda environment, checks that `mujoco` and `alicia_d_sdk` import correctly, prefers `mjpython` on macOS, auto-detects common serial device names, and starts `02_demo_mujoco_follower.py` from the repository root.

## Workflow

1. Run `scripts/start_mujoco_demo.sh` from this skill.
2. Let the script default to conda env `py311` unless the user explicitly wants another env.
3. Pass `--port <device>` when the user already knows the correct serial device.
4. Pass `--disable-torque`, `--debug`, or `--no-wrist-camera` only when the user asks for them or the failure mode clearly calls for them.
5. If the launch fails or the behavior is unclear, read [references/launch-notes.md](references/launch-notes.md).

## Commands

Use these patterns:

```bash
bash ./.codex/skills/run-mujoco-demo/scripts/start_mujoco_demo.sh
bash ./.codex/skills/run-mujoco-demo/scripts/start_mujoco_demo.sh --port /dev/cu.usbmodemXXXX
bash ./.codex/skills/run-mujoco-demo/scripts/start_mujoco_demo.sh --debug --disable-torque
bash ./.codex/skills/run-mujoco-demo/scripts/start_mujoco_demo.sh --dry-run
```

## Rules

- Use `mjpython` on macOS. Do not launch the viewer with plain `python` there.
- Keep the working directory at the repository root so relative asset paths resolve correctly.
- Expect the viewer to open even when no Leader arm is connected. In that case, explain that serial reconnect warnings are expected and teleoperation input will be unavailable.
- Prefer `--dry-run` before changing commands when you only need to inspect the resolved launcher, environment, port, or final command line.
- Treat `/dev/cu.usbmodem*`, `/dev/cu.usbserial*`, `/dev/ttyACM*`, `/dev/ttyUSB*`, and `/dev/ttyCH343USB*` as the primary port candidates.

## Resources

- `scripts/start_mujoco_demo.sh`: Default launcher for this repository.
- `references/launch-notes.md`: Failure modes, port patterns, and command examples.
