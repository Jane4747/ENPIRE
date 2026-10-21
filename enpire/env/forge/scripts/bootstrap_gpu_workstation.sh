#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# One-shot GPU workstation bootstrap for ENPIRE perception / planning / RL.
#
# Usage (from repo root):
#   bash enpire/env/forge/scripts/bootstrap_gpu_workstation.sh
#   bash enpire/env/forge/scripts/bootstrap_gpu_workstation.sh --start sam3
#   bash enpire/env/forge/scripts/bootstrap_gpu_workstation.sh --start perception
#   bash enpire/env/forge/scripts/bootstrap_gpu_workstation.sh --start cap-real
#   bash enpire/env/forge/scripts/bootstrap_gpu_workstation.sh --no-start
#
# What it does:
#   1. Verifies NVIDIA GPU + nvidia-smi
#   2. Ensures uv + Python 3.11
#   3. Initializes git submodules (cuRobo)
#   4. uv sync for vision / planning / cap / real-rl
#   5. Downloads facebook/sam3 from ModelScope into the HF hub cache
#   6. Syncs the isolated PLD RL runtime
#   7. Optionally starts services in tmux (default: sam3 only)
#
# Not installed automatically (external / licensed / hardware):
#   - AnyGrasp SDK + per-machine licence
#   - Real YAM station registration / calibration
#   - BundleSDF runtime
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${ROOT}"

START_MODE="sam3"   # sam3 | perception | cap-real | none
SKIP_WEIGHTS=0
SKIP_PLD=0
DRY_RUN_ONLY=0

usage() {
  cat <<'EOF'
Usage: bootstrap_gpu_workstation.sh [options]

  --start MODE     sam3 (default) | perception | cap-real | none
  --no-start       Same as --start none
  --skip-weights   Do not download SAM3
  --skip-pld       Do not sync the PLD RL runtime
  --dry-run-only   After install, only print service commands
  -h, --help       Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --start) START_MODE="${2:-}"; shift 2 ;;
    --no-start) START_MODE="none"; shift ;;
    --skip-weights) SKIP_WEIGHTS=1; shift ;;
    --skip-pld) SKIP_PLD=1; shift ;;
    --dry-run-only) DRY_RUN_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

case "${START_MODE}" in
  sam3|perception|cap-real|none) ;;
  *) echo "Invalid --start mode: ${START_MODE}" >&2; exit 2 ;;
esac

log() { printf '\n[bootstrap] %s\n' "$*"; }
die() { printf '[bootstrap] ERROR: %s\n' "$*" >&2; exit 1; }

# --- 0. PATH -----------------------------------------------------------------
export PATH="${HOME}/.local/bin:${PATH}"
mkdir -p "${HOME}/.local/bin" "${HOME}/.config/enpire"

# --- 1. GPU ------------------------------------------------------------------
log "Checking NVIDIA GPU"
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found. Install NVIDIA drivers first."
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
log "GPU OK: ${GPU_NAME}"

# --- 2. uv + Python 3.11 -----------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  log "Installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # shellcheck disable=SC1091
  source "${HOME}/.local/bin/env"
fi
log "uv $(uv --version)"

log "Ensuring Python 3.11"
uv python install 3.11

# --- 3. system packages (best-effort) ----------------------------------------
if command -v apt-get >/dev/null 2>&1 && [[ "$(id -u)" -eq 0 || -n "${BOOTSTRAP_USE_SUDO:-}" ]]; then
  log "Installing apt packages (git/tmux/ffmpeg)"
  SUDO=""
  [[ "$(id -u)" -ne 0 ]] && SUDO="sudo"
  ${SUDO} apt-get update -y
  ${SUDO} DEBIAN_FRONTEND=noninteractive apt-get install -y git git-lfs tmux ffmpeg curl
  git lfs install || true
else
  for bin in git tmux curl; do
    command -v "${bin}" >/dev/null 2>&1 || die "Missing required tool: ${bin}"
  done
  command -v ffmpeg >/dev/null 2>&1 || log "WARN: ffmpeg not found (optional for some workflows)"
fi

# --- 4. submodules -----------------------------------------------------------
log "Initializing git submodules (cuRobo)"
if [[ ! -f third_party/curobo/pyproject.toml ]]; then
  git submodule update --init --recursive
fi
[[ -f third_party/curobo/pyproject.toml ]] || die "third_party/curobo missing after submodule init"

# --- 5. root extras ----------------------------------------------------------
log "uv sync (vision / planning / cap / real-rl)"
uv sync \
  --extra dev \
  --extra vision \
  --extra vision-local \
  --extra planning \
  --extra planning-local \
  --extra cap \
  --extra real-rl \
  --extra pld

log "Installing modelscope for SAM3 weight download"
uv pip install modelscope

log "Running enpire doctor"
uv run enpire doctor || true

# --- 6. SAM3 weights ---------------------------------------------------------
if [[ "${SKIP_WEIGHTS}" -eq 0 ]]; then
  log "Downloading SAM3 from ModelScope into HF hub cache"
  uv run python enpire/env/forge/scripts/download_sam3.py --source modelscope
else
  log "Skipping SAM3 download (--skip-weights)"
fi

# --- 7. PLD runtime ----------------------------------------------------------
if [[ "${SKIP_PLD}" -eq 0 ]]; then
  log "Syncing isolated PLD RL runtime"
  uv sync --project enpire/policy/pld/runtime --extra dev
  uv run enpire rl learner --task pin_insertion --dry-run || true
  uv run enpire rl actor --task pin_insertion --dry-run || true
else
  log "Skipping PLD runtime (--skip-pld)"
fi

# --- 8. env file -------------------------------------------------------------
ENV_FILE="${HOME}/.config/enpire/services.env"
cat > "${ENV_FILE}" <<EOF
# Generated by bootstrap_gpu_workstation.sh — no secrets.
export ENPIRE_ROOT="${ROOT}"
export PATH="${HOME}/.local/bin:\${PATH}"
export SAM3_SERVER_HOST=127.0.0.1
export SAM3_SERVER_PORT=6767
export ANYGRASP_SERVER_HOST=127.0.0.1
export ANYGRASP_SERVER_PORT=8122
export ANYGRASP_SERVICE_URL="http://\${ANYGRASP_SERVER_HOST}:\${ANYGRASP_SERVER_PORT}"
# Fill after obtaining a per-machine AnyGrasp licence:
# export ANYGRASP_SDK_ROOT=/path/to/anygrasp_sdk
# export ANYGRASP_CHECKPOINT=/path/to/checkpoint_detection.tar
# export ANYGRASP_LICENSE_ZIP=/path/to/license.zip
export RL_DATA_PATH="${HOME}/.local/share/enpire/rl-data"
EOF
mkdir -p "${HOME}/.local/share/enpire/rl-data"
log "Wrote ${ENV_FILE}"

# shellcheck disable=SC1090
source "${ENV_FILE}"

# --- 9. verify torch CUDA ----------------------------------------------------
log "Verifying torch CUDA"
uv run python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available(), "count", torch.cuda.device_count())
assert torch.cuda.is_available(), "torch cannot see the GPU"
print("device0", torch.cuda.get_device_name(0))
PY

# --- 10. start / dry-run services --------------------------------------------
if [[ "${DRY_RUN_ONLY}" -eq 1 || "${START_MODE}" == "none" ]]; then
  log "Service dry-run (not starting)"
  if [[ "${START_MODE}" == "none" || "${START_MODE}" == "sam3" ]]; then
    uv run enpire services start --services sam3 --dry-run
  else
    uv run enpire services start --profile "${START_MODE}" --dry-run
  fi
  log "Bootstrap complete (services not started)."
  log "Start later with:"
  echo "  source ${ENV_FILE}"
  echo "  cd ${ROOT}"
  echo "  uv run enpire services start --services sam3"
  exit 0
fi

if tmux has-session -t enpire 2>/dev/null; then
  log "tmux session 'enpire' already exists — leaving it alone"
  uv run enpire services status || true
  exit 0
fi

log "Starting services: ${START_MODE}"
if [[ "${START_MODE}" == "sam3" ]]; then
  uv run enpire services start --services sam3
else
  uv run enpire services start --profile "${START_MODE}"
fi

sleep 2
uv run enpire services status || true
log "Bootstrap complete."
log "Attach: tmux attach -t enpire"
log "SAM3:   http://127.0.0.1:6767"
log "Env:    source ${ENV_FILE}"
