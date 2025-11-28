#!/usr/bin/env bash

# H-SAE 실험(run_experiments.sh + python main.py) 정지 스크립트
# 사용 예시:
#   bash stop_experiments.sh

set -euo pipefail

echo "[STOP] Stopping H-SAE experiments..."

# 1) run_experiments.sh 를 실행 중인 쉘 프로세스들 종료
SCRIPT_PIDS="$(pgrep -f 'run_experiments.sh' || true)"
if [[ -n "${SCRIPT_PIDS}" ]]; then
  echo "[STOP] Killing run_experiments.sh PIDs: ${SCRIPT_PIDS}"
  kill ${SCRIPT_PIDS} || true
else
  echo "[STOP] No running run_experiments.sh processes found."
fi

# 2) 실험용 python main.py 프로세스들 종료
PYTHON_PIDS="$(pgrep -f 'python main.py' || true)"
if [[ -n "${PYTHON_PIDS}" ]]; then
  echo "[STOP] Killing python main.py PIDs: ${PYTHON_PIDS}"
  kill ${PYTHON_PIDS} || true
else
  echo "[STOP] No running 'python main.py' processes found."
fi

# 3) 잠시 대기 후, 남아있는 경우 강제 종료 시도
sleep 3

SCRIPT_PIDS_LEFT="$(pgrep -f 'run_experiments.sh' || true)"
PYTHON_PIDS_LEFT="$(pgrep -f 'python main.py' || true)"

if [[ -n "${SCRIPT_PIDS_LEFT}" || -n "${PYTHON_PIDS_LEFT}" ]]; then
  echo "[STOP] Some processes still alive, sending SIGKILL..."
  if [[ -n "${SCRIPT_PIDS_LEFT}" ]]; then
    echo "[STOP] kill -9 (run_experiments.sh): ${SCRIPT_PIDS_LEFT}"
    kill -9 ${SCRIPT_PIDS_LEFT} || true
  fi
  if [[ -n "${PYTHON_PIDS_LEFT}" ]]; then
    echo "[STOP] kill -9 (python main.py): ${PYTHON_PIDS_LEFT}"
    kill -9 ${PYTHON_PIDS_LEFT} || true
  fi
else
  echo "[STOP] All experiment processes appear to be stopped."
fi

echo "[STOP] Done."

