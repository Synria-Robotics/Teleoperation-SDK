#!/usr/bin/env bash
set -euo pipefail

shopt -s nullglob

ENV_NAME="py311"
PORT=""
DISABLE_TORQUE=0
DEBUG=0
NO_WRIST_CAMERA=0
DRY_RUN=0
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  start_mujoco_demo.sh [options] [-- extra-args]

Options:
  --env NAME            Conda environment name. Default: py311
  --port DEVICE         Serial device passed to 02_demo_mujoco_follower.py
  --disable-torque      Pass --disable_torque
  --debug               Pass --debug
  --no-wrist-camera     Pass --no_wrist_camera
  --dry-run             Print the resolved command and exit
  -h, --help            Show this help text
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env)
      ENV_NAME="${2:?missing value for --env}"
      shift 2
      ;;
    --port)
      PORT="${2:?missing value for --port}"
      shift 2
      ;;
    --disable-torque)
      DISABLE_TORQUE=1
      shift
      ;;
    --debug)
      DEBUG=1
      shift
      ;;
    --no-wrist-camera|--no_wrist_camera)
      NO_WRIST_CAMERA=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

detect_port() {
  [[ -n "${PORT}" ]] && return 0

  local candidate
  if [[ "${OSTYPE:-}" == darwin* ]]; then
    for candidate in /dev/cu.usbmodem* /dev/cu.usbserial* /dev/cu.wchusbserial* /dev/tty.usbmodem* /dev/tty.usbserial*; do
      if [[ -e "${candidate}" ]]; then
        PORT="${candidate}"
        return 0
      fi
    done
  else
    for candidate in /dev/ttyACM* /dev/ttyUSB* /dev/ttyCH343USB*; do
      if [[ -e "${candidate}" ]]; then
        PORT="${candidate}"
        return 0
      fi
    done
  fi
}

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda is not available in PATH"
  exit 1
fi

detect_port

cd "${REPO_ROOT}"

conda run -n "${ENV_NAME}" python -c "import mujoco, alicia_d_sdk"

LAUNCHER="python"
if [[ "${OSTYPE:-}" == darwin* ]]; then
  conda run -n "${ENV_NAME}" which mjpython >/dev/null 2>&1
  LAUNCHER="mjpython"
fi

CMD=(conda run -n "${ENV_NAME}" "${LAUNCHER}" 02_demo_mujoco_follower.py)
if [[ -n "${PORT}" ]]; then
  CMD+=(--port "${PORT}")
fi
if [[ "${DISABLE_TORQUE}" -eq 1 ]]; then
  CMD+=(--disable_torque)
fi
if [[ "${DEBUG}" -eq 1 ]]; then
  CMD+=(--debug)
fi
if [[ "${NO_WRIST_CAMERA}" -eq 1 ]]; then
  CMD+=(--no_wrist_camera)
fi
if [[ "${#EXTRA_ARGS[@]}" -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

echo "[INFO] Repo root: ${REPO_ROOT}"
echo "[INFO] Conda env: ${ENV_NAME}"
echo "[INFO] Launcher: ${LAUNCHER}"
if [[ -n "${PORT}" ]]; then
  echo "[INFO] Serial port: ${PORT}"
else
  echo "[WARN] No serial port detected; the viewer may open but teleoperation input will be unavailable"
fi
printf "[INFO] Command:"
printf " %q" "${CMD[@]}"
printf "\n"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  exit 0
fi

exec "${CMD[@]}"
