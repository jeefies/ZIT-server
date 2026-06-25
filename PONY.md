# Pony Support Branch

This branch keeps the existing service architecture:

- `image_service.py` is the Flask/task-queue process.
- `image_service_pony_pipeline.py` is the isolated Pony diffusers pipeline subprocess.
- The service still communicates with the pipeline by newline-delimited JSON over stdin and receives status through HTTP callbacks.

`i2i`/inpaint is intentionally not implemented for Pony. ZIT `i2i` remains available through the default model family. Pony requests return `501` when `mode` is not `t2i`.

## Default Model

Default model:

```text
AstraliteHeart/pony-diffusion-v6
```

Override it with:

```bash
export PONY_MODEL_ID=/path/or/huggingface/model-id
```

The pipeline keeps the old `local_files_only=True` behavior by default. Set this only when you explicitly want diffusers to fetch files:

```bash
export MODEL_LOCAL_FILES_ONLY=false
```

## API Compatibility

Use either `POST /generate` with `"model_family": "pony"` or `POST /generate/pony`. The existing `POST /generate` request shape remains compatible with ZIT defaults when `model_family` is omitted:

```json
{
  "mode": "t2i",
  "prompt": "score_9, score_8_up, 1girl",
  "negative_prompt": "score_4, score_5, score_6, lowres",
  "width": 1024,
  "height": 1024,
  "steps": 30,
  "guidance": 7.0,
  "seed": -1,
  "image_base64": null,
  "mask_base64": null
}
```

Two optional fields are accepted for Pony/SDXL:

```json
{
  "strength": 0.8,
  "clip_skip": null
}
```

`strength` is currently recorded for forward compatibility only because `i2i`/inpaint is disabled. `clip_skip` is passed only when the installed diffusers version supports it.

## Pony Defaults

Defaults in `image_service.py`:

- `width`: `1024`
- `height`: `1024`
- `steps`: `30`
- `guidance`: `7.0`
- `strength`: `0.8`
- `clip_skip`: disabled by default
- `negative_prompt`: `score_4, score_5, score_6, lowres, bad anatomy, bad hands, blurry, watermark, signature, text, censored`

Environment overrides:

```bash
export PONY_SERVICE_BASE_DIR=/home/jeefy/AV/Pony-service
export PONY_PIPELINE_SCRIPT=/path/to/image_service_pony_pipeline.py
export PONY_DEFAULT_WIDTH=1024
export PONY_DEFAULT_HEIGHT=1024
export PONY_DEFAULT_STEPS=30
export PONY_DEFAULT_GUIDANCE=7.0
export PONY_DEFAULT_STRENGTH=0.8
export PONY_DEFAULT_CLIP_SKIP=none
export PONY_DEFAULT_NEGATIVE_PROMPT="score_4, score_5, score_6, lowres"
```

## Pipeline Details

The diffusers classes are:

- `StableDiffusionXLPipeline` for `t2i`

Default runtime paths:

- Service data defaults remain under `/home/jeefy/AV/ZIT-service` unless overridden.
- Pony model/log defaults are under `/home/jeefy/AV/ZIT-service-pony`.
- Pipeline log: `/home/jeefy/AV/ZIT-service-pony/logs/pony_pipeline.log`

Memory options enabled:

- model CPU offload
- attention slicing
- VAE slicing when available
- VAE tiling when available

The default dtype is `float16`. Override with:

```bash
export PONY_TORCH_DTYPE=bfloat16
```
