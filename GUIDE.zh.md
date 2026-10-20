# 在 NVIDIA Thor（sm_101）上用 CUDA 13 跑 Qwen-Image 2.1（无审查）—— 实战复盘

这是 [MiniMax-H3 实战复盘](https://github.com/souyou233/Minimax-H3-on-thor-u/blob/main/GUIDE.zh.md) 的**生图兄弟篇**。
同一台机器、同一套 CUDA-13 运行时的思路；但工作负载完全不同（也**轻得多**）：在 NVIDIA **Thor**
（60 GB 统一内存、算力 **sm_101**）上跑 **Qwen-Image 2.1（无审查）** 的文生图与图生图（编辑）。
它跑在**与视频后端完全相同的 `comfy-h3-cu130:v1` 镜像**里——只换了 ComfyUI 源码树和启动器。

> 配套两篇：**[GUIDE.md](GUIDE.md)**（英文）、**[INSTALL.md](INSTALL.md)**（给 AI 的自动安装）。

---

## 1. 硬件与基线

| | |
|---|---|
| 设备 | 极氪 Thor-U —— NVIDIA DRIVE **Thor**、**sm_101**、aarch64、**60 GB 统一内存**（CPU/GPU 同一内存池）|
| 运行时 | `comfy-h3-cu130:v1` —— **torch 2.13.0 + cu130**、comfy_kitchen CUDA 后端、CUDA-13 前向兼容 `libcuda` |
| ComfyUI（图） | **`comfyui_qwen` —— ComfyUI 0.42.0**（原生 Qwen-Image 2.1 节点）|
| comfy_kitchen（图） | **0.2.39**，用 `PYTHONPATH=/workspace/ck_qwen` 隔离 |
| 端口 | **8100**（视频 8189；MCP/调度器 8092；任务面板 `GET :8092/`）|
| 权重 | DiT `qwen-image-2.1-UC-fp8`(6.7 GB) · TE `qwen3vl_8b_int8_convrot`(8.8 GB) · VAE `qwen_image_2.1_vae_bf16`(0.65 GB) |

这个兄弟仓库的意义：**视频和图共用一模一样的 cu130 环境。** 只要装好 H3 一体机，生图对你就只是
**三个权重文件 + 一个启动器** 的事——不用再装第二套 torch/CUDA。

## 2. 为什么要单独的构建：Qwen-Image 2.1 需要 ComfyUI 0.42

视频后端用的是 **ComfyUI 0.34**（带 MiniMax-H3 节点）。而 Qwen-Image 2.1——以及把整张图变得极其精简
的那个节点 **`TextEncodeQwenImage21`**——只存在于**较新的 0.42** ComfyUI 里。设备上两个约束相互冲突：

1. 两个后端都需要**同一套 cu130 运行时**（torch 2.13+cu130、CUDA-13 兼容）才能在 sm_101 上吃到原生
   int8/fp8——道理与 H3 复盘 §2–§3 完全一致。
2. 它们需要**不同的 ComfyUI 源码**，甚至**不同的 `comfy-kitchen` 版本**（0.34 要镜像内置的 ck；
   0.42 要 **0.2.39**）。

干净的解法是**文件系统隔离，而非第二个镜像**：保留运行时镜像，`cd` 进另一棵源码树（`comfyui_qwen`），
并用 `PYTHONPATH` 上的一份私有 ck 遮蔽内置的（`/workspace/ck_qwen` → comfy-kitchen **0.2.39**）。
一个运行时镜像、两棵各得其所的 ComfyUI、零版本冲突。这里真正"新"的代码只有启动器：

```bash
# app/start_comfy_qwen.sh（大意）
export LD_LIBRARY_PATH=/compat:$LD_LIBRARY_PATH:/host_lib:/host_thor_lib:/host_cuda_lib  # /compat 在最前——与视频一致
export PYTHONPATH=/workspace/ck_qwen:$PYTHONPATH                                         # <- 隔离 comfy-kitchen 0.2.39
cd /workspace/comfyui_qwen                                                               # <- 较新的 ComfyUI (0.42.0)
exec python3 main.py --listen 0.0.0.0 --port 8100 --enable-cors-header \
  --disable-xformers --disable-smart-memory --fast fp8_matrix_mult \
  --extra-model-paths-config /workspace/app/qwen_model_paths.yaml
```

`/compat`（CUDA-13 `libcuda`）排在 `LD_LIBRARY_PATH` 最前，才会打开 comfy_kitchen 的 CUDA 后端 →
**原生 int8/fp8**；启动日志必须能看到 `Native ops: … int8_tensorwise … float8_e4m3fn`。
`--fast fp8_matrix_mult` 选 FP8 矩阵乘路径。**不要**加 `--lowvram/--novram`。

## 3. 图只有 7 个节点——`TextEncodeQwenImage21` 一人扛大梁

让 Qwen-Image 在 ComfyUI 里这么精简的，是一个节点 `TextEncodeQwenImage21`。它吃 提示词 + 负向 +
**VAE** + 一个 `resolution`，吐**三个**输出：

| 输出 | 含义 | 接到 |
|---|---|---|
| `…[0]` | 正向 conditioning | `KSampler.positive` |
| `…[1]` | 负向 conditioning | `KSampler.negative` |
| `…[2]` | 现成的 **64 通道 latent**（按 `resolution` 生成） | `KSampler.latent_image` |

所以**没有 `EmptyLatentImage`**、没有 `CheckpointLoader`、负向也不用单独的 `CLIPTextEncode`。
整张文生图长这样：

```
UNETLoader(DiT fp8)  CLIPLoader(type=qwen_image)  VAELoader
        │                    │                          │
        │             TextEncodeQwenImage21 ────────────┤   (prompt+negative+vae+resolution)
        │              out0=+cond  out1=−cond  out2=latent
        └────────────► KSampler(euler/simple, cfg=4.0, denoise=1.0) ─► VAEDecode ─► SaveImage
```

默认：采样器 `euler`、调度 `simple`、`cfg=4.0`、`steps=20`（打草稿用 8），负向
`low quality, blurry, distorted, watermark, text`。**文生图是正方形**——一个 `resolution`（默认 1024）
决定 `resolution×resolution`。

## 4. 图生图（编辑）：同图 + `VAEEncode`，`denoise = strength`

编辑复用完全相同的 loaders 和 `TextEncodeQwenImage21`，只换了 *latent 的来源*。加
`LoadImage → VAEEncode`，把编码后的 latent 送进 `KSampler.latent_image`，并设 **`denoise = strength`**
（0=几乎原图，1=完全重画；默认 **0.6**）。

```
… TextEncodeQwenImage21（同上）
LoadImage ─► VAEEncode ─► KSampler.latent_image ,  denoise=strength   （正负向不变）
```

两点务必记住：
- **输出尺寸跟随输入图**（由 VAEEncode 的 latent 决定），**不是**你传的宽高——要改尺寸先 resize 源图。
  这里的 `TextEncodeQwenImage21.resolution` 固定 1024，只影响（用不到的）文本 latent 尺寸。
- 源图用 `POST /upload/image`（multipart）推上去，返回的 `name` 填进 `LoadImage` 节点。这正是调度器
  给 `image_to_image` 工具做的事。

## 5. 结果 —— 速度与显存

在极氪 Thor-U、1024×1024、同一任务引擎计时（任务运行的挂钟时间）实测：

| 配置 | 步数 | 时间 | MemAvailable 最低 | 常驻 |
|---|---|---|---|---|
| 冷启动首图（含加载） | 20 | **72 s** | ~45 GB | ~13–14 GB |
| 预热 | 20 | **36 s** | ~45 GB | ~13–14 GB |
| 预热 | 8 | **18 s** | ~45 GB | ~13–14 GB |

最重要的结论是个**反直觉结果**：生图**不受内存限制**。三个权重加载约 16 GB，1024² 单次运行活跃约
**13–14 GB**，MemAvailable 一直在 40 多 GB。对比**同一台机器**上的视频栈——15s@800×448 会压到约
5 GB 的地板。正因如此，生图**不需要**视频那套折腾（两步卸 TE、tiled 解码、稀疏注意力）——它只是三个
loader 加一个 sampler。

耗时与步数近似线性：1024² 约 2.2 s/步（8 步≈18s，20 步≈36s）。冷启动（约 35s）是模型加载，每次切换
后端付一次，批量出图可摊薄。

## 6. 集成 —— 调度器怎么驱动它

在极氪协处理器上，图是几个**互斥**的重后端之一（统一内存里一次只驻留一个）。MCP 调度器提供：

- `generate_image {prompt, submitter*, width, steps, cfg, seed, priority?}` → **异步任务**，`get_task` 轮询。
- `image_to_image {prompt, image, submitter*, strength, steps, cfg, seed, priority?}` → 异步任务。
  （`submitter` 标识发起者；`priority` 默认 5，越大越先，用于队列重排。）

两者都走**同一个串行优先级队列**，并实时显示在**任务面板**（`GET http://<host>:8092/`）。因为后端
串行，切到 `image` 会挤掉 `video`（反之亦然）——后端翻转时会有一次冷启动（约 35s），批量出图时切换
开销近乎为零。结果从 ComfyUI 拷出后以 `/files/img_*.png`（或 `i2i_*.png`）提供。

## 7. 踩坑清单

- **跑错 ComfyUI 构建。** 节点列表里没有 `TextEncodeQwenImage21`，说明跑的是视频那棵（0.34）。必须
  `cd comfyui_qwen`（0.42.0）**且** 设 `PYTHONPATH=/workspace/ck_qwen`。
- **comfy-kitchen 没被遮蔽。** 0.42 下若内置 ck 抢占，算子会消失。确认 `PYTHONPATH=/workspace/ck_qwen`
  在 `main.py` 之前生效，且它是 0.2.39。
- **别两个后端一起跑。** 视频占着内存时再跑图会 OOM 或卡住；调度器的忙检测 + `/free` 会处理——别手动双开。
- **文件名是硬依赖。** 图里按精确名字引用（`…-UC-fp8`、`…_int8_convrot`、`…_bf16`）。`models/` 里改个名，
  节点下拉框就悄悄找不到。
- **文生图是方的；图生图跟随源图。** 文生图除 `resolution` 外没有宽高；宽高比的事放 i2i（喂非方源图）
  或出图后再 resize。
- **FP8 不是万能加速。** 这里 `--fast fp8_matrix_mult` 就是对的默认值；生图本来就快——没有注意力杠杆可拉
  （见 §8，以及为何视频的 SpargeAttn 在这里用不上）。

## 8. 为什么视频那套在这里统统用不上

为完整性——H3 复盘花了好几章的内容，对生图**全部无关**：

| 视频关注点 | 生图现实 |
|---|---|
| CUDA-13 门控 → 原生 int8/fp8 | **仍然适用**（同一运行时、同一 `/compat` 配方）|
| 两步卸 TE（`/free`）| 不需要——约 16G 权重、约 14G 活跃，余量巨大 |
| `VAEDecodeTiled` 封顶解码 | 不需要——单帧解码 |
| SpargeAttn 块稀疏（~1.8×）| 不用——生图注意力不是瓶颈；保持稠密 SDPA |
| Turbo LoRA / sigma-shift | 不用——标准 euler/simple、cfg 4.0、steps 8/20 |

## 9. 复现

1. 先装 [MiniMax-H3 一体机](https://github.com/souyou233/Minimax-H3-on-thor-u)，让 `comfy-h3-cu130:v1`
   + CUDA-13 兼容库就位（本仓库复用它们）。
2. 在启动器能挂载到的位置准备好 `comfyui_qwen`（ComfyUI **0.42.0**）源码树与 `ck_qwen`
   （comfy-kitchen **0.2.39**）——设备上它们随同一个 workspace 提供。
3. 把 3 个模型下到 `models/`（INSTALL §3；核对 sha256）。
4. `./setup.sh` 然后 `./run.sh`（或安装 `app/gen-image-comfy.service`）。
5. 提交 `sample_workflow.json`；预热后约 36s 得到一张 1024² PNG。

## 10. 数据附录

- 1024² @20步 预热 **36 s**；@8步 预热 **18 s**；冷启动首图 **72 s**（加载 ≈35 s）。
- 常驻约 13–14 GB；全程 MemAvailable 最低 **~45 GB**。
- 权重：DiT 6.7 GB · TE 8.8 GB · VAE 0.65 GB（合计约 16 GB）。
- sha256（设备上已校验）：
  - DiT `877b77290339986bca8f771aead381c7a5777347c27d252bbaa915f76667ab0b`
  - TE  `8bfd0f6e12abf2d2d697ecc888e5e90b0d6741d6708f05799f53afa560452e8f`
  - VAE `bb21f7473051e1ac368515dd3f2e15cd44d7a11748ee8823e1ddca3e4876b7c9`
