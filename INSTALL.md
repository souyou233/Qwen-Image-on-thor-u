# Qwen-Image 2.1 (Uncensored) — Image Generation on cu130 (NVIDIA Thor / sm_101)

Self-contained ComfyUI image-generation stack that runs **Qwen-Image 2.1 Uncensored** for
**text→image** and **image→image (editing)** on an NVIDIA Thor with **native int8/fp8**
(comfy_kitchen CUDA backend, CUDA-13). It shares the cu130 runtime image with the companion
MiniMax-H3 appliance and only swaps in a newer **ComfyUI 0.42** build (native Qwen-Image 2.1
support) + an isolated **comfy-kitchen 0.2.39**, on **port 8100**.

> **Audience: an AI agent.** Execute the steps below in order. The heavy **runtime image is the
> same `comfy-h3-cu130:v1` used by the MiniMax-H3 appliance** — if that appliance is installed,
> no runtime download is needed here. The only external downloads are the **3 model weight files**
> (Section 3). This stack runs **alongside** (and is mutually-exclusive-with) the video backend:
> one heavy backend is resident in unified memory at a time.

---

## 1. Package contents

```
comfy-qwen-image-cu130/
├── INSTALL.md                  <- you are here
├── GUIDE.md / GUIDE.zh.md      <- engineering field guide (EN / 中文)
├── setup.sh                    <- preflight: runtime image + code + models present?
├── run.sh                      <- start ComfyUI on :8100 (docker, shared cu130 image)
├── sample_workflow.json        <- ready-to-run T2I test graph
├── sample_i2i_workflow.json    <- ready-to-run image-edit test graph
├── SHA256SUMS                  <- checksums of the 3 models below
├── app/
│   ├── start_comfy_qwen.sh              launcher (port 8100, --fast fp8_matrix_mult)
│   ├── qwen_model_paths.yaml            model search paths (points at ./models/*)
│   └── gen-image-comfy.service          systemd unit (production, docker)
└── models/            <- EMPTY by design; you download the 3 weights here (Section 3)
    ├── diffusion_models/  ├── text_encoders/  └── vae/
```

## 2. Prerequisites (check before installing)

- **Hardware:** NVIDIA **Thor (compute capability sm_101)**, aarch64/ARM64 (verified on Zeekr Thor-U).
- **Runtime image:** `comfy-h3-cu130:v1` loaded into `docker images`. Get it from the companion
  repo's **GitHub Releases**, tag `appliance-v1`
  (<https://github.com/souyou233/Minimax-H3-on-thor-u/releases/tag/appliance-v1>): download
  `comfy-h3-cu130-v1.tar.gz.part00`…`part04` (GitHub's 2 GB asset cap splits the ~8.5 GB tarball),
  `cat` them back into one `.tar.gz`, verify against `SHASUMS-256.txt`, then `docker load`.
  This image carries torch 2.13.0+cu130 + comfy_kitchen + all deps and the CUDA-13 forward-compat
  `libcuda`.
- **Code tree:** `comfyui_qwen/` (ComfyUI **0.42.0**) + isolated `ck_qwen/` (**comfy-kitchen
  0.2.39**) — download `qwen_code_cu130.tar.gz` from the same release and extract it into the
  workspace.
- **Docker** with `--privileged` + GPU device nodes `/dev/nvgpu`, `/dev/nvgpu-pci`, `/dev/nvmap`,
  `/dev/tegradisp`.
- **Disk:** ≥ 20 GB free for models (~16 GB) + outputs.
- **RAM:** ≥ 24 GB unified memory is ample (a run loads ~16 GB of weights; peaks ~13–14 GB active).

Quick check:
```bash
docker images | grep comfy-h3-cu130        # runtime image present?
ls /usr/lib/libnvrm* 2>/dev/null | head -1 # GPU userspace present?
```

## 3. Models — download these 3 (the ONLY external downloads)

Place each file in the shown path under `models/`. Keep the **exact quant variants** — the
graph loads them by these filenames. Lineage: DiT = Qwen-Image (Qwen team) **uncensored ("UC") fp8**;
TE = **int8 convrot** quant of Qwen3-VL-8B; VAE = Qwen-Image VAE (bf16). Source them from your model
store / HuggingFace and verify the sha256 below matches before wiring the graph.

| # | Target path (under `models/`) | Size | sha256 |
|---|-------------------------------|------|--------|
| 1 | `diffusion_models/qwen-image-2.1-UC-fp8.safetensors` | ~6.7 GB | `877b77290339986bca8f771aead381c7a5777347c27d252bbaa915f76667ab0b` |
| 2 | `text_encoders/qwen3vl_8b_int8_convrot.safetensors` | ~8.8 GB | `8bfd0f6e12abf2d2d697ecc888e5e90b0d6741d6708f05799f53afa560452e8f` |
| 3 | `vae/qwen_image_2.1_vae_bf16.safetensors` | ~0.65 GB | `bb21f7473051e1ac368515dd3f2e15cd44d7a11748ee8823e1ddca3e4876b7c9` |

Verify (must all say `OK`):
```bash
cd models && sha256sum -c - <<'EOF'
877b77290339986bca8f771aead381c7a5777347c27d252bbaa915f76667ab0b  diffusion_models/qwen-image-2.1-UC-fp8.safetensors
8bfd0f6e12abf2d2d697ecc888e5e90b0d6741d6708f05799f53afa560452e8f  text_encoders/qwen3vl_8b_int8_convrot.safetensors
bb21f7473051e1ac368515dd3f2e15cd44d7a11748ee8823e1ddca3e4876b7c9  vae/qwen_image_2.1_vae_bf16.safetensors
EOF
cd ..
```

## 4. Install (preflight — software is shared, no download)

```bash
cd comfy-qwen-image-cu130
sha256sum -c SHA256SUMS     # optional: verify the 3 models
./setup.sh                  # checks: runtime image + comfyui_qwen + ck_qwen + models present
```
`setup.sh` reports what is present / missing and how to provide each. Nothing to compile.

## 5. Run

Interactive:
```bash
./run.sh                    # ComfyUI on http://<host>:8100
```
Production (systemd) — copy the unit, then start it. It is stopped when the video backend needs
memory and started on demand (a dispatcher `ensure("image")`/`switch_profile` does this):
```bash
sudo cp app/gen-image-comfy.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl start gen-image-comfy
journalctl -u gen-image-comfy -f
```
The log should print:
```
Native ops: ... int8_tensorwise ... float8_e4m3fn ...        # native int8/fp8 active
Starting server
To See the GUI go to: http://0.0.0.0:8100
```

## 6. Verify end-to-end

**Text→image** (server running):
```bash
curl -s -X POST http://127.0.0.1:8100/prompt \
  -H 'Content-Type: application/json' \
  -d "{\"prompt\": $(cat sample_workflow.json), \"client_id\": \"smoke\"}"
# poll /history/<prompt_id>, then GET /view?filename=...&subfolder=&type=output  -> PNG
```
**Image→image (edit):** upload a source first, then submit the i2i graph with that filename:
```bash
curl -s -X POST http://127.0.0.1:8100/upload/image -F "image=@my.png" -F "type=input"
# edit the "8": {"inputs": {"image": "<returned name>"}} in sample_i2i_workflow.json, then:
curl -s -X POST http://127.0.0.1:8100/prompt -H 'Content-Type: application/json' \
  -d "{\"prompt\": $(cat sample_i2i_workflow.json), \"client_id\": \"smoke\"}"
```
Expected: a PNG in `comfyui_qwen/output/`. On the appliance these are exposed as
`http://<host>:8092/files/img_*.png` (dispatcher copies them out). Timing: ~18 s @8-step,
~36 s @20-step once warm.

## 7. How it works (for the integrating agent)

- **Same runtime, different build.** `app/start_comfy_qwen.sh` runs inside `comfy-h3-cu130:v1`
  exactly like the video launcher, but `cd`s into **`comfyui_qwen` (ComfyUI 0.42.0)** and sets
  `PYTHONPATH=/workspace/ck_qwen` so it imports the **isolated comfy-kitchen 0.2.39** instead of the
  image's baked-in one. This is required because Qwen-Image 2.1 (and `TextEncodeQwenImage21`) only
  exist in the newer ComfyUI.
- **cu130 activation is identical:** `/compat` (CUDA-13 `libcuda.so.1`) is **first** on
  `LD_LIBRARY_PATH`, system `libnvrm_*` from `/host_lib`. Under CUDA 13 comfy_kitchen enables native
  int8/fp8 (chip stays sm_101). Launched with `--fast fp8_matrix_mult`; **do not** add
  `--lowvram/--novram`.
- **The graph is tiny — 7 nodes.** `UNETLoader`(DiT fp8) + `CLIPLoader`(type=`qwen_image`) +
  `VAELoader`. The single node **`TextEncodeQwenImage21`** emits **three** outputs: positive cond
  (out 0), negative cond (out 1) **and the ready 64-channel latent** (out 2, sized by `resolution`).
  So you feed `KSampler.positive←[T,0]`, `negative←[T,1]`, `latent_image←[T,2]` — no EmptyLatentImage,
  no separate CLIPTextEncode. sampler `euler/simple`, `cfg=4.0`, `denoise=1.0`.
- **T2I is square by design:** the graph takes one `resolution` (default 1024). Output = `resolution×resolution`.
- **Image editing (i2i, official Edit route — default):** `LoadImage` feeds `TextEncodeQwenImage21.images.image_1`
  AND the node gets its `vae` input, so the source enters the DiT as `reference_latents`; the sampler
  starts from the node's **empty latent at `denoise = 1.0`** (identity comes from the reference latents;
  the prompt, not denoise, drives how much changes). `QwenImage21Cache` (device `auto`) sits between
  `UNETLoader` and `KSampler.model`; use `resolution = 0` so the canvas follows `image_1` (32-aligned).
  Extra material images go in as `images.image_2…`. `cfg = 1.0` (negative needs cfg≥2), `steps = 25`.
  (A legacy `loose` variant exists — vision-only + `VAEEncode(source)` + denoise 0.85 — see GUIDE §4.)
  Source images go in via `POST /upload/image`.
- **Model resolution** is via `app/qwen_model_paths.yaml` (`--extra-model-paths-config`), which maps
  `diffusion_models / text_encoders / vae` to the mounted `./models/*`. Filenames must match Section 3.
- **Memory:** the three weights load to ~16 GB and a 1024² run uses ~13–14 GB — no tiled decode,
  no two-pass needed. Image gen never approaches the ceiling that constrains video.

## 8. Optional — nothing required

Unlike video, image attention is not the bottleneck and needs **no** sparse-attention kernel; dense
SDPA is fine. Do **not** add `--use-sage-attention` here. Speed scales ~linearly with `steps`
(8-step ≈ 18 s, 20-step ≈ 36 s at 1024²); use 8-step for drafts, 20-step for finals.

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| `Native ops` line missing / eager fallback | `/compat` not first on `LD_LIBRARY_PATH`, or not under CUDA 13. Check `start_comfy_qwen.sh`. |
| `TextEncodeQwenImage21` not in node list | Wrong ComfyUI build. You must run **`comfyui_qwen` (0.42.0)** + `PYTHONPATH=/workspace/ck_qwen`, not the video tree. |
| comfy_kitchen ops still missing under 0.42 | `ck_qwen` (0.2.39) not on `PYTHONPATH`, or ComfyUI resolved the image's baked-in ck. Confirm the export line. |
| Model not found in node dropdown | File missing or wrong subdir under `models/`; re-check Section 3 names (exact filenames incl. `-UC-fp8`, `_int8_convrot`, `_bf16`). |
| `resolution` looks wrong / non-square | T2I is square; only `resolution` controls size. For arbitrary aspect, do it via i2i (encode a non-square source). |
| i2i output ignores my width/height | Expected — i2i follows the **source image** latent. Resize the source first. |
| `docker: command not found` | On the device docker is at `/zeekr_data/docker/bin/docker`; add it to PATH or use the absolute path (as the unit does). |
| Port 8100 busy / image won't start while video runs | Backends are mutually exclusive; stop/let the dispatcher free the video backend first. |

## 10. Reference

- Runtime image + CUDA-13 recipe: companion [MiniMax-H3-on-thor-u](https://github.com/souyou233/Minimax-H3-on-thor-u).
- Base model: **Qwen-Image** (Qwen team); this build uses the **uncensored fp8** DiT + **int8 convrot** TE + bf16 VAE.
- Measured on Zeekr Thor-U: 1024² @8-step ~18 s, @20-step ~36 s; ~13–14 GB resident; MemAvailable floor ~45 GB.
- Ports: image ComfyUI **8100** (video is 8189; dispatcher/MCP 8092; task dashboard `GET :8092/`).
