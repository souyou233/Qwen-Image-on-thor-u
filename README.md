<div align="center">

# Qwen-Image 2.1 (Uncensored) on NVIDIA Thor · CUDA 13
# 在 NVIDIA Thor 上跑 Qwen-Image 2.1（无审查）· CUDA 13

**Self-contained ComfyUI image-generation appliance + field guide** · **自包含 ComfyUI 生图一体机 + 技术复盘**

Runs **Qwen-Image 2.1 Uncensored** text→image **and** image→image (editing) on an NVIDIA Thor (sm_101) with native int8/fp8.
在 NVIDIA Thor（sm_101）上以原生 int8/fp8 跑 **Qwen-Image 2.1（无审查）** 文生图 **与** 图生图（编辑）。

</div>

---

## TL;DR · 一句话

| 🇨🇳 中文 | 🇬🇧 English |
|---|---|
| 与配套的 [MiniMax-H3 一体机](https://github.com/souyou233/Minimax-H3-on-thor-u) **共用同一套 CUDA-13 运行时镜像**（torch 2.13+cu130、comfy_kitchen CUDA 后端），只是换成**较新的 ComfyUI 0.42 源码树**（原生支持 Qwen-Image 2.1）+ **隔离的 comfy-kitchen 0.2.39** + 专用启动器。DiT/TE/VAE 三件套，端口 **8100**。生图很轻：1024² 8 步 **~18 秒**、20 步 **~36 秒**，常驻约 **13–14 GB**，**完全不吃内存**（和 60G 统一内存的视频栈是两个量级）。 | The image backend **reuses the exact CUDA-13 runtime image from the companion [MiniMax-H3 appliance](https://github.com/souyou233/Minimax-H3-on-thor-u)** (torch 2.13+cu130, comfy_kitchen CUDA backend); only the ComfyUI build changes — a **newer ComfyUI 0.42 source tree** (native Qwen-Image 2.1 support) + an **isolated comfy-kitchen 0.2.39** + a dedicated launcher. Three weight files (DiT/TE/VAE), port **8100**. Generation is light: 1024² @8-step **~18 s**, @20-step **~36 s**, ~**13–14 GB** resident — **memory is never the bottleneck** (a different order of magnitude from the video stack on the same 60 GB unified pool). |

## 性能对比 · Benchmark

*1024×1024 · euler/simple · Zeekr Thor-U (60 GB unified, sm_101) · CUDA 13*

| 配置 Config | 步 Steps | 出图时间 time | 常驻内存 resident | 备注 note |
|---|---|---|---|---|
| 冷启动首图 Cold (1st request) | 20 | **72 s** | ~13–14 GB | 含后端加载 ~35s model load included |
| 预热 Warm | 20 | **36 s** | ~13–14 GB | 生产默认 production default |
| 预热 Warm · 8 步 | 8 | **18 s** | ~13–14 GB | 快速预览 fast preview |

> 全程系统可用内存最低 **~45 GB**（即用约 13–14 GB）。生图不受内存限制；真正的成本只是**时间**。
> Lowest MemAvailable across runs was **~45 GB** (i.e. ~13–14 GB used). Image gen is never memory-bound; the only real cost is **time**.

## 快速开始 · Quick start

```bash
# 1) 下载 3 个模型到 models/（清单见 INSTALL.md §3）· download the 3 models into models/
# 2) 安装：确保 cu130 运行时镜像在位 + 校验模型（软件零额外下载）· install (runtime shared w/ H3 appliance)
./setup.sh
# 3) 启动 ComfyUI :8100 · start ComfyUI on :8100
./run.sh
#    （生产环境用 systemd：app/gen-image-comfy.service）
```

## 包内容 · Contents

| 路径 Path | 说明 | Description |
|---|---|---|
| `app/start_comfy_qwen.sh` | 端口 8100 启动器（cu130 / `--fast fp8_matrix_mult`） | port-8100 launcher (cu130, fp8) |
| `app/qwen_model_paths.yaml` | 模型搜索路径 | model search paths |
| `app/gen-image-comfy.service` | systemd 单元（生产；docker 方式） | systemd unit (production, docker) |
| `run.sh` / `setup.sh` | 一键启动 / 校验 | one-shot launch / preflight |
| `sample_workflow.json` | 现成的 **文生图** 测试图 | ready-to-run **T2I** test graph |
| `sample_i2i_workflow.json` | 现成的 **图生图/编辑** 测试图 | ready-to-run **image-edit** test graph |
| `models/` | **空**，需自下 3 个模型 | **empty**; download 3 models |
| `SHA256SUMS` | 3 个模型校验值 | model checksums |

> 📦 **运行时镜像** = 配套 `comfy-h3-cu130` 一体机的 `comfy-h3-cu130:v1`（**同一环境**，本仓库不重复打包）。镜像与代码树都在配套仓库的 **GitHub Releases**（tag `appliance-v1`）直接下载：镜像 `comfy-h3-cu130-v1.tar.gz.part00`~`part04`（cat 合并），代码树 `qwen_code_cu130.tar.gz`（含 `comfyui_qwen` ComfyUI 0.42 + `ck_qwen` comfy-kitchen 0.2.39），附 `SHASUMS-256.txt` 校验。
> 📦 The **runtime image** is `comfy-h3-cu130:v1` from the companion appliance (same environment; not re-shipped here). Image **and** code trees download straight from the companion repo's **GitHub Releases** (tag `appliance-v1`): `comfy-h3-cu130-v1.tar.gz.part00`…`part04` (cat them back) plus `qwen_code_cu130.tar.gz` (ships `comfyui_qwen` ComfyUI 0.42 + `ck_qwen` comfy-kitchen 0.2.39), with `SHASUMS-256.txt` for verification.

## 文档 · Documentation

| 🇨🇳 中文 | 🇬🇧 English |
|---|---|
| **[GUIDE.zh.md](GUIDE.zh.md)** — 完整技术复盘：为什么复用 cu130 镜像、0.42 + 隔离 ck、`TextEncodeQwenImage21` 图、速度/显存、踩坑 | **[GUIDE.md](GUIDE.md)** — full field guide: reusing the cu130 image, ComfyUI 0.42 + isolated ck, the `TextEncodeQwenImage21` graph, speed/memory, pitfalls |
| [INSTALL.md](INSTALL.md) — 给 AI 的自动安装（模型清单 + sha256 + 校验 + systemd） | [INSTALL.md](INSTALL.md) — AI-driven install (models + sha256 + verify + systemd) |

## 需要的模型 · Required models

**Qwen-Image 2.1 Uncensored** 三件套（DiT FP8 6.7 GB · int8-convrot 文本编码器 8.8 GB · Qwen-Image VAE bf16 645 MB）。
精确路径 / 大小 / sha256 见 [INSTALL.md §3](INSTALL.md)。 · Exact paths / sizes / sha256 in [INSTALL.md §3](INSTALL.md).

> 血统 lineage：DiT = Qwen-Image（Qwen team）**uncensored（"UC"）** fp8 量化变体；TE = **int8 convrot** 量化的 Qwen3-VL-8B；VAE = Qwen-Image VAE（bf16）。本设备从内置模型库提供；换源时请核对量化版本一致。
> On this device the three files are provisioned from the on-board model store; if you fetch elsewhere, keep the **same quant** variants.
