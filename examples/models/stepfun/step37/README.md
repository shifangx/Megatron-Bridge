# Step-3.7 (step37 / step3p7) Vision-Language SFT

This directory hosts example launch artifacts for the **Step-3.7** vision-language
model (HF: `stepfun-ai/step3p7`).

## Architecture

- **Language tower**: identical to Step-3.5-Flash — 45 decoder layers,
  hidden=4096, 64 attention heads with 8 KV groups, hybrid full/sliding
  attention pattern, per-layer RoPE base and per-layer SwiGLU clamp, 288-expert
  MoE on layers 3–44 (top-k=8, shared expert). Implemented in
  `src/megatron/bridge/models/stepfun/step35_provider.py`.
- **Vision tower**: Perception-Encoder PE-G/14 — 47 layers, hidden=1536,
  16 heads, image_size=728, patch_size=14, 2D RoPE, QuickGeLU activation.
  Implemented in
  `src/megatron/bridge/models/stepfun/step37/vision_model.py`.
- **Projector**: single `Linear(6144 → 4096, bias=False)` with Kaiming-normal
  init. Implemented in
  `src/megatron/bridge/models/stepfun/step37/projector.py`.
- **Fusion**: image features are spliced into the LLM token-embedding tensor
  at positions where `input_ids == 128001` (`<im_patch>`). 169 image tokens
  per image. Implemented in
  `src/megatron/bridge/models/stepfun/step37/step37_model.py`.

## Recipes

| Recipe | Use-case |
|---|---|
| `step37_smoke_sft_config` | 1-GPU functional smoke (TP=1 PP=1, 6 LLM layers, vision frozen) |
| `step37_321b_a38b_sft_config` | Full Step-3.7 SFT (TP=1 PP=8 CP=8 EP=8 SP=on) |

## Step function

`step37_step` — registered in `scripts/training/run_recipe.py`. Mirrors
`qwen3_vl_step` but routes through `Step37Model.forward` so vision tokens are
spliced via the embedding hook rather than passed through mRoPE.

## Quick start

```bash
# Smoke (single GPU)
bash Scripts-MBridge/7.5.step37_sft.sh

# Full-size SFT (8 GPUs, TP=1 PP=8 CP=8 EP=8)
NPROC=8 bash Scripts-MBridge/7.6.step37_sft_full.sh
```
