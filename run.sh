#!/bin/bash
# Start Qwen-Image 2.1 ComfyUI on http://<host>:8100 (docker, shared cu130 runtime image).
# Run from the package root. Requires: ./setup.sh preflight passing, i.e.
#   - runtime image comfy-h3-cu130:v1 present (from the companion H3 appliance)
#   - WORK= dir containing comfyui_qwen/ (ComfyUI 0.42) and ck_qwen/ (comfy-kitchen 0.2.39)
#   - CUDA-13 compat libs (COMPAT=) and the 3 models in ./models
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="${WORK:-$HERE}"                          # dir that holds comfyui_qwen/ + ck_qwen/
COMPAT="${COMPAT:-../comfy-h3-cu130/compat_libs/compat}"
DOCKER="${DOCKER:-docker}"
IMG="comfy-h3-cu130:v1"

$DOCKER image inspect "$IMG" >/dev/null 2>&1 || { echo "Missing runtime image $IMG (load the H3 appliance)."; exit 1; }
[ -f "$WORK/comfyui_qwen/main.py" ] || { echo "Missing $WORK/comfyui_qwen (ComfyUI 0.42). Set WORK= ; see setup.sh."; exit 1; }
[ -d "$COMPAT" ] || { echo "Missing CUDA-13 compat dir ($COMPAT). Set COMPAT= ; run ./setup.sh."; exit 1; }

MOUNTS="-v $COMPAT:/compat:ro -v /usr/lib:/host_lib:ro"
[ -d /usr/local/cuda-12.8/thor/lib64 ] && MOUNTS="$MOUNTS -v /usr/local/cuda-12.8/thor/lib64:/host_thor_lib:ro"
[ -d /usr/local/cuda-12.8/lib64 ]      && MOUNTS="$MOUNTS -v /usr/local/cuda-12.8/lib64:/host_cuda_lib:ro"

exec $DOCKER run --rm --name qwen-image --privileged --network host --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  $MOUNTS \
  -v "$WORK":/workspace \
  -v "$HERE/models":/workspace/models/h3 \
  -v "$HERE/app":/workspace/app:ro \
  -v /dev/nvgpu:/dev/nvgpu -v /dev/nvgpu-pci:/dev/nvgpu-pci \
  -v /dev/nvmap:/dev/nvmap -v /dev/tegradisp:/dev/tegradisp \
  -w /workspace --entrypoint bash "$IMG" /workspace/app/start_comfy_qwen.sh
