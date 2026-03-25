# Launch Notes

## Default flow

Use `scripts/start_mujoco_demo.sh` first. It runs from the repository root, defaults to conda env `py311`, checks `mujoco` and `alicia_d_sdk`, and chooses `mjpython` automatically on macOS.

## Common commands

```bash
bash ./.codex/skills/run-mujoco-demo/scripts/start_mujoco_demo.sh
bash ./.codex/skills/run-mujoco-demo/scripts/start_mujoco_demo.sh --port /dev/cu.usbmodemXXXX
bash ./.codex/skills/run-mujoco-demo/scripts/start_mujoco_demo.sh --debug --disable-torque
bash ./.codex/skills/run-mujoco-demo/scripts/start_mujoco_demo.sh --dry-run
```

## Port selection

Check these device names first:

- macOS: `/dev/cu.usbmodem*`, `/dev/cu.usbserial*`, `/dev/cu.wchusbserial*`
- Linux: `/dev/ttyACM*`, `/dev/ttyUSB*`, `/dev/ttyCH343USB*`

If no device exists, the MuJoCo viewer may still open, but the script will warn that teleoperation input is unavailable.

## Known failure modes

`RuntimeError: launch_passive requires that the Python script be run under mjpython on macOS`

- Cause: launched with plain `python` on macOS.
- Fix: rerun with the bundled script or use `conda run -n py311 mjpython 02_demo_mujoco_follower.py`.

`No available serial port device found` or repeated reconnect errors

- Cause: the Leader arm is disconnected, the wrong port was selected, or the port is busy.
- Fix: reconnect the device, identify the actual serial node, and relaunch with `--port`.

`ModuleNotFoundError` for `mujoco` or `alicia_d_sdk`

- Cause: wrong conda env or missing packages.
- Fix: verify the env, then rerun the bundled script so the import check fails early.
