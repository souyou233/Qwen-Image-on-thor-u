#!/bin/bash
# Preflight for Qwen-Image 2.1 ComfyUI (cu130 / Thor sm_101). Run from the package root.
# This stack REUSES the companion comfy-h3-cu130 runtime image + CUDA-13 compat libs; it only
# needs its own ComfyUI 0.42 tree (comfyui_qwen), isolated ck (ck_qwen), and 3 models.
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
WORK="${WORK:-$HERE}"; COMPAT="${COMPAT:-../comfy-h3-cu130/compat_libs/compat}"; DOCKER="${DOCKER:-docker}"
IMG="comfy-h3-cu130:v1"
ok=1
say(){ printf "  [%s] %s\n" "$1" "$2"; }

echo "[1/4] runtime image ($IMG, shared with the H3 appliance) ..."
if $DOCKER image inspect "$IMG" >/dev/null 2>&1; then say OK "image present"; else say MISS "load comfy-h3-cu130:v1 from the H3 appliance"; ok=0; fi

echo "[2/4] CUDA-13 compat dir ($COMPAT) ..."
if [ -d "$COMPAT" ]; then say OK "compat present"; else say MISS "set COMPAT= to the compat dir from the H3 install"; ok=0; fi

echo "[3/4] code: comfyui_qwen (ComfyUI 0.42) + ck_qwen (comfy-kitchen 0.2.39) under WORK=$WORK ..."
if [ -f "$WORK/comfyui_qwen/main.py" ]; then
  V=$(grep -oE '0\.42[0-9.]*' "$WORK/comfyui_qwen/comfyui_version.py" 2>/dev/null | head -1)
  say OK "comfyui_qwen present (version ${V:-?})"; [ -n "$V" ] || say WARN "expected ComfyUI 0.42.x"
else say MISS "no $WORK/comfyui_qwen (ComfyUI 0.42 source tree)"; ok=0; fi
if [ -d "$WORK/ck_qwen" ]; then say OK "ck_qwen present"; else say MISS "no $WORK/ck_qwen (isolated comfy-kitchen 0.2.39)"; ok=0; fi

echo "[4/4] models ..."
for m in \
  diffusion_models/qwen-image-2.1-UC-fp8.safetensors \
  text_encoders/qwen3vl_8b_int8_convrot.safetensors \
  vae/qwen_image_2.1_vae_bf16.safetensors ; do
  if [ -f "models/$m" ]; then say OK "models/$m"; else say MISS "models/$m (see INSTALL.md §3)"; ok=0; fi
done

echo
if [ "$ok" = 1 ]; then echo "Preflight OK. Start with:  ./run.sh   (or install app/gen-image-comfy.service)";
else echo "Preflight incomplete (items above). Fix, then re-run ./setup.sh"; fi
