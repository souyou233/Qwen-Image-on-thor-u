# Running Qwen-Image 2.1 (Uncensored) on NVIDIA Thor (sm_101) with CUDA 13 — a field guide

This is the image-generation sibling of the [MiniMax-H3 field guide](https://github.com/souyou233/Minimax-H3-on-thor-u/blob/main/GUIDE.md).
Same machine, same CUDA-13 runtime idea; a completely different (and far lighter) workload:
**Qwen-Image 2.1 Uncensored** for text→image and image→image editing, on an NVIDIA **Thor**
(60 GB unified memory, compute capability **sm_101**). It runs inside the **same `comfy-h3-cu130:v1`
docker image** as the video backend — only the ComfyUI build and launcher change.

> Two companion docs: **[GUIDE.zh.md](GUIDE.zh.md)** (中文), **[INSTALL.md](INSTALL.md)** (AI-driven install).

---

## 1. Hardware & baseline

| | |
|---|---|
| Device | Zeekr Thor-U — NVIDIA DRIVE **Thor**, **sm_101**, aarch64, **60 GB unified memory** (CPU+GPU share one pool) |
| Runtime | `comfy-h3-cu130:v1` — **torch 2.13.0 + cu130**, comfy_kitchen CUDA backend, CUDA-13 forward-compat `libcuda` |
| ComfyUI (image) | **`comfyui_qwen` — ComfyUI 0.42.0** (native Qwen-Image 2.1 nodes) |
| comfy_kitchen (image) | **0.2.39**, isolated via `PYTHONPATH=/workspace/ck_qwen` |
| Port | **8100** (video is 8189; MCP/dispatcher 8092; dashboard `GET :8092/`) |
| Weights | DiT `qwen-image-2.1-UC-fp8` (6.7 GB) · TE `qwen3vl_8b_int8_convrot` (8.8 GB) · VAE `qwen_image_2.1_vae_bf16` (0.65 GB) |

The whole point of this sibling repo: **video and image share the exact same cu130 environment.** If
you already run the H3 appliance, the image backend costs you **three weight files and a launcher** —
no second torch/CUDA stack to install.

## 2. Why a separate build: Qwen-Image 2.1 needs ComfyUI 0.42

The video backend ships **ComfyUI 0.34** (with the MiniMax-H3 nodes). Qwen-Image 2.1 — and the one
node that makes the graph trivial, **`TextEncodeQwenImage21`** — only exist in the **newer 0.42**
ComfyUI. Two constraints collide on a single device:

1. Both backends need the **same cu130 runtime** (torch 2.13+cu130, CUDA-13 compat) to get native
   int8/fp8 on sm_101 — that reasoning is identical to §2–§3 of the H3 guide.
2. They need **different ComfyUI source** and even **different `comfy-kitchen` versions**
   (0.34 wants the image's baked-in ck; 0.42 wants **0.2.39**).

The clean fix is **filesystem isolation, not a second image**: keep the runtime image, `cd` into a
different source tree (`comfyui_qwen`), and shadow the baked-in comfy-kitchen with a private copy on
`PYTHONPATH` (`/workspace/ck_qwen` → comfy-kitchen **0.2.39**). One runtime image, two Cozy
ComfyUIs, zero version fights. The launcher is the only real "new code" here:

```bash
# app/start_comfy_qwen.sh (paraphrased)
export LD_LIBRARY_PATH=/compat:$LD_LIBRARY_PATH:/host_lib:/host_thor_lib:/host_cuda_lib  # /compat first — same as video
export PYTHONPATH=/workspace/ck_qwen:$PYTHONPATH                                         # <- isolate comfy-kitchen 0.2.39
cd /workspace/comfyui_qwen                                                               # <- newer ComfyUI (0.42.0)
exec python3 main.py --listen 0.0.0.0 --port 8100 --enable-cors-header \
  --disable-xformers --disable-smart-memory --fast fp8_matrix_mult \
  --extra-model-paths-config /workspace/app/qwen_model_paths.yaml
```

`/compat` first on `LD_LIBRARY_PATH` (CUDA-13 `libcuda`) is what turns comfy_kitchen's CUDA backend
on → **native int8/fp8**; you must see `Native ops: … int8_tensorwise … float8_e4m3fn` at startup.
`--fast fp8_matrix_mult` selects the FP8 matmul path. **Do not** add `--lowvram/--novram`.

## 3. The graph is 7 nodes — `TextEncodeQwenImage21` does the heavy lifting

The trick that makes Qwen-Image in ComfyUI so small is a single node, `TextEncodeQwenImage21`. It
takes the prompt + negative + the **VAE** + a single `resolution`, and emits **three** outputs:

| output | meaning | feeds |
|---|---|---|
| `…[0]` | positive conditioning | `KSampler.positive` |
| `…[1]` | negative conditioning | `KSampler.negative` |
| `…[2]` | the ready **64-channel latent**, sized by `resolution` | `KSampler.latent_image` |

So there is **no `EmptyLatentImage`**, no `CheckpointLoader`, no separate `CLIPTextEncode` for the
negative. The entire text→image graph:

```
UNETLoader(DiT fp8)  CLIPLoader(type=qwen_image)  VAELoader
        │                    │                          │
        │             TextEncodeQwenImage21 ────────────┤   (prompt+negative+vae+resolution)
        │              out0=+cond  out1=−cond  out2=latent
        └────────────► KSampler(euler/simple, cfg=4.0, denoise=1.0) ─► VAEDecode ─► SaveImage
```

Defaults: sampler `euler`, scheduler `simple`, `cfg=4.0`, `steps=20` (8 for drafts),
negative `low quality, blurry, distorted, watermark, text`. **T2I is square** — one `resolution`
(default 1024) sets `resolution×resolution`.

## 4. Image→image (editing): the official Edit route (default), `loose` as legacy fallback

2.1 has **no dedicated Edit checkpoint** (that was the older 1.x `Qwen-Image-Edit-2509/2511`).
Editing is a mode of the same model: `TextEncodeQwenImage21` accepts optional reference images
(`image_1…image_16`).

**DEFAULT (since 2026-10-22) — the official Edit route** (ComfyUI template
`image_qwen_image_2_1_image_edit.json`, 0.42+): the node DOES get its `vae` input, so the source
image's latent is spliced into the DiT sequence as **`reference_latents`** (at the image slots); the
VLM vision tokens for those images are dropped (`keep_vision` logic, `comfy/text_encoders/qwen_image21.py`).
Sampling starts from the node's **empty latent at `denoise = 1.0`** — a full redraw in which identity
is carried by the reference latents, so **how much changes is driven by the prompt, not by denoise**.
`QwenImage21Cache` sits on the model line as the template ships it (device `auto|gpu|cpu|off`, default
`auto`; `off` recomputes the prefix and is the documented way to rule the cache out when debugging).
`resolution = 0` keeps every reference at its own pixel size (32-aligned) — the canvas follows
`image_1`, like it always has on this box.

```
UNETLoader ─► QwenImage21Cache(device=auto) ──────────────► KSampler.model
TextEncodeQwenImage21 (prompt + image_1 [+image_2..] + vae, resolution=0)  → +cond/−cond/empty latent
KSampler(euler/simple, cfg=1.0, steps=25, denoise=1.0, latent = TE out2) ─► VAEDecode ─► SaveImage
```

> Correction from an earlier version of this guide: colour-speckle noise once measured on this
> Thor/cu130 stack — which had us bypass the reference-latent splice — turned out to be an artifact
> of the **old 0.34-era cu130 stack**. On the current 0.42 build the official route renders clean.

**Legacy `loose` route** (dispatcher `edit_mode="loose"`; keep only to reproduce pre-2026-10-22
outputs or when you need the layout held near pixel-wise): no `vae` wiring — vision tokens carry the
edit — and the sampler starts at `VAEEncode(source)` with `denoise 0.85`. Because the source latent
locks the layout, this route *cannot* change what the source image hides.

- **Multi-reference editing:** extra material images go in as `images.image_2…` (tool param
  `ref_images`). Prompt addresses slots by number: `图1`/`image 1` = the image being edited,
  `图2`+ = materials ("the fox in image 1 wears the scarf from image 2").
- **Region targeting trick:** paint a coloured mark on the spot and name that colour in the prompt
  ("change the jacket in the RED area").
- **Output canvas follows `image_1`** (32-aligned); to change size, resize the source first.
- Source images are pushed with `POST /upload/image` (multipart); the returned `name` goes into the
  `LoadImage` node — exactly what the dispatcher's `image_to_image` tool does.

## 5. Results — speed & memory

Measured on Zeekr Thor-U, 1024×1024, same task-engine timing (wall clock of the running task):

| 配置 Config | 步 Steps | 时间 time | MemAvailable 最低 min | 常驻 resident |
|---|---|---|---|---|
| Cold (1st request, incl. boot) | 20 | **72 s** | ~45 GB | ~13–14 GB |
| Warm 预热 | 20 | **36 s** | ~45 GB | ~13–14 GB |
| Warm 预热 | 8 | **18 s** | ~45 GB | ~13–14 GB |

The headline is a **negative result**: image generation is **not memory-bound**. Three weight files
load to ~16 GB and a 1024² run stays around **13–14 GB active**; MemAvailable never left the mid-40s.
Compare the video stack on the *same machine*, which thrashes to a ~5 GB floor at 15 s@800×448. That
difference is why image needs **none** of the video gymnastics (two-pass TE eviction, tiled decode,
sparse attention) — it's just three loaders and a sampler.

Cost scales ~linearly with `steps`: ~2.2 s/step at 1024² (8-step ≈ 18 s, 20-step ≈ 36 s). Cold boot
(~35 s) is model load and is paid once per backend switch, amortized across a batch.

## 6. Integration — how the dispatcher drives it

On the Zeekr co-processor the image backend is one of several **mutually-exclusive** heavy backends
(one resident at a time in unified memory). The MCP dispatcher exposes:

- `generate_image {prompt, submitter*, width, steps, cfg, seed, priority?}` → **async task**, poll `get_task`.
- `image_to_image {prompt, image, ref_images?, edit_mode?, cache?, submitter*, denoise, steps, cfg, seed, priority?}` → async task.
  (`submitter` identifies the caller; `priority` — default 5, higher first — reorders the queue.
  `edit_mode=edit` = official Edit route and default; `loose` = legacy hybrid, see §4.)

Both go through a **single serial priority queue** and surface live in the **task dashboard**
(`GET http://<host>:8092/`). Because backends are serial, switching to `image` evicts `video` and
vice-versa — expect a cold boot (~35 s) whenever the backend flips, and near-zero switch cost when
batching images. Results are copied out of ComfyUI and served as `/files/img_*.png` (or `i2i_*.png`).

## 7. Pitfalls & gotchas

- **Wrong ComfyUI build.** If `TextEncodeQwenImage21` is missing from the node list, you are running
  the video tree (0.34). You must `cd comfyui_qwen` (0.42.0) **and** set `PYTHONPATH=/workspace/ck_qwen`.
- **comfy-kitchen not shadowed.** Under 0.42 the ops can disappear if the image's baked-in ck wins.
  Confirm the `PYTHONPATH=/workspace/ck_qwen` export precedes `main.py` and that it is 0.2.39.
- **Don't mix backends.** Running image while video holds memory will OOM or stall; the dispatcher's
  busy-check + `/free` handles this — don't hand-start both.
- **Filenames are load-bearing.** The graph references exact names (`…-UC-fp8`, `…_int8_convrot`,
  `…_bf16`). A rename in `models/` silently breaks the node dropdown.
- **Speckle noise was a 0.34-era artifact.** An earlier version of this guide blamed the
  reference-latent splice and shipped a workaround for it; on the current 0.42 build the official
  Edit route renders clean (§4). If you see colour speckle again, suspect the stack version first.
- **Multi-reference editing:** pass up to 10 extra images as `images.image_2…` (tool `ref_images`).
  On the official (default) route denoise is always 1.0 — just describe the fusion in the prompt.
  In `loose` mode, material-addition edits ("wear the scarf from image 2") need `denoise=1.0`: the
  0.85 pixel anchor suppresses added elements.
- **T2I is square; i2i follows the source.** There is no width/height in the T2I path beyond
  `resolution`; aspect-ratio work belongs in i2i (feed a non-square source) or a post-resize.
- **FP8 ≠ faster-forever.** `--fast fp8_matrix_mult` is the right default here; image is already fast
  — there is no attention lever to pull (see §8, and why video's SpargeAttn does **not** apply).

## 8. Why none of the video tricks apply

For completeness — everything the H3 guide spends chapters on is **irrelevant** to images:

| Video concern | Image reality |
|---|---|
| CUDA-13 gate → native int8/fp8 | **Applies** (same runtime, same `/compat` recipe) |
| Two-pass TE eviction (`/free`) | Not needed — ~16 GB weights, ~14 GB active, huge headroom |
| `VAEDecodeTiled` to cap decode | Not needed — single-frame decode |
| SpargeAttn block-sparse (~1.8×) | Not used — image attention isn't the bottleneck; keep dense SDPA |
| Turbo LoRA / sigma-shift | Not used — standard euler/simple, cfg 4.0, steps 8/20 |

## 9. Reproducing it

1. Install the [MiniMax-H3 appliance](https://github.com/souyou233/Minimax-H3-on-thor-u) so
   `comfy-h3-cu130:v1` + the CUDA-13 compat libs are in place (this repo reuses them).
2. Provide a `comfyui_qwen` (ComfyUI **0.42.0**) source tree and `ck_qwen` (comfy-kitchen **0.2.39**)
   where the launcher can mount them (they ride along in the same workspace on the device).
3. Download the 3 models into `models/` (INSTALL §3; verify sha256).
4. `./setup.sh` then `./run.sh` (or install `app/gen-image-comfy.service`).
5. Submit `sample_workflow.json`; you should get a 1024² PNG in ~36 s (warm).

## 10. Numbers appendix

- 1024² @20-step warm: **36 s**; @8-step warm: **18 s**; cold first request: **72 s** (boot ≈ 35 s).
- Resident ~13–14 GB; MemAvailable floor across runs **~45 GB**.
- Weights: DiT 6.7 GB · TE 8.8 GB · VAE 0.65 GB (≈16 GB total).
- sha256 (verified on device):
  - DiT `877b77290339986bca8f771aead381c7a5777347c27d252bbaa915f76667ab0b`
  - TE  `8bfd0f6e12abf2d2d697ecc888e5e90b0d6741d6708f05799f53afa560452e8f`
  - VAE `bb21f7473051e1ac368515dd3f2e15cd44d7a11748ee8823e1ddca3e4876b7c9`
