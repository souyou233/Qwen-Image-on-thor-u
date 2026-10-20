#!/bin/bash
# Qwen-Image 2.1 (uncensored) ComfyUI launcher — cu130 / Thor sm_101, port 8100.
# Runs inside the shared comfy-h3-cu130:v1 image (same env as the video backend).
# Uses the newer ComfyUI build (comfyui_qwen, v0.42) which has native Qwen-Image 2.1 support,
# with an isolated comfy-kitchen 0.2.39 shadowed via PYTHONPATH (so it does not fight the
# baked-in ck that the video tree needs).
#
# /compat = CUDA-13 forward-compat libcuda + JIT companions (MUST be first on LD_LIBRARY_PATH).
# System GPU userspace (libnvrm_*) comes from /host_lib. Do NOT copy host libcuda over compat.
export LD_LIBRARY_PATH=/compat:$LD_LIBRARY_PATH:/host_lib:/host_thor_lib:/host_cuda_lib
export PYTHONPATH=/workspace/ck_qwen:$PYTHONPATH
cd /workspace/comfyui_qwen
exec python3 main.py --listen 0.0.0.0 --port 8100 --enable-cors-header \
  --disable-xformers --disable-smart-memory --fast fp8_matrix_mult \
  --extra-model-paths-config /workspace/app/qwen_model_paths.yaml
