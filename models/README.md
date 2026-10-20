# Models — place these 3 files here before running

Keep the **exact quant variants** (the graph loads them by these filenames). Verify size and
sha256 (`sha256sum -c ../SHA256SUMS`, or see INSTALL.md §3). Lineage: DiT = Qwen-Image (Qwen team)
**uncensored ("UC") fp8**; TE = **int8 convrot** quant of Qwen3-VL-8B; VAE = Qwen-Image VAE (bf16).

```
models/
├── diffusion_models/
│   └── qwen-image-2.1-UC-fp8.safetensors          (~6.7 GB)   DiT (uncensored fp8)
├── text_encoders/
│   └── qwen3vl_8b_int8_convrot.safetensors        (~8.8 GB)   text encoder (int8 convrot)
└── vae/
    └── qwen_image_2.1_vae_bf16.safetensors        (~0.65 GB)  Qwen-Image VAE (bf16)
```

Exact paths + sha256 are in INSTALL.md (section "Models"). These are the ONLY external downloads;
the runtime image + ComfyUI 0.42 tree are shared with the companion comfy-h3-cu130 appliance.
