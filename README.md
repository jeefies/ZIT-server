# ZIT-service

Z-Image-Turbo image generation service. Pony Diffusion XL is available as an optional extension that reuses the same queue/process architecture; see `PONY.md` for Pony-specific defaults and environment variables.

The default `/generate` behavior remains Z-Image-Turbo compatible. Pony can be selected with `model_family="pony"` or `POST /generate/pony`.

## 架构

```
Flask API (主进程, port 8765)
  ├── TaskProcessor: 任务队列 + 缓存 + 持久化 (TinyDB)
  └── PipelineManager: 管理 pipeline 子进程
        ├── ZImagePipeline / ZImageInpaintPipeline (default ZIT)
        └── StableDiffusionXLPipeline (optional Pony t2i)
```

主进程通过 `subprocess.Popen` 启动 pipeline 子进程，通过 stdin JSON 通信，通过 HTTP 回调同步状态。Pipeline 支持 t2i/i2i 双模式自动切换（需重新加载模型，约 20 秒）。同一模式下任务顺序处理，模式切换只在 pipeline 空闲时发生。

### 关键设计

- **单线程 Flask**: 串行处理请求，无并发问题
- **busy 标志**: pipeline 处理任务期间不切换模式、不推送新任务，防止正在处理的任务被杀死
- **缓存机制**: 相同参数的 completed 任务直接复用结果
- **自动重启**: 服务启动时恢复 interrupted 任务
- **GPU 显存管理**: CPU offload + attention slicing + 每次生成后激进清理
- **空闲超时**: 30 分钟无任务自动退出 pipeline 进程

## 依赖

- Python 3.10+ (conda env: `image`)
- Flask
- TinyDB
- diffusers (ZImagePipeline / ZImageInpaintPipeline, StableDiffusionXLPipeline for Pony)
- torch (CUDA, bfloat16)
- Pillow

## 运行

### 直接运行

```bash
python3 image_service.py
```

### systemd 服务

```bash
systemctl --user start ZIT-service
systemctl --user start ZIT-tunnel   # SSH 反向隧道暴露到远程
```

服务端口: **8765**

### 手动释放 pipeline

```bash
curl -X POST http://localhost:8765/pipeline_free
```

## 目录结构

```
ZIT-service/
├── image_service.py          # Flask 主程序
├── image_service_pipeline.py      # ZIT Pipeline 子进程
├── image_service_pony_pipeline.py # Pony Pipeline 子进程
├── data/                     # TinyDB + 生成图片 (gitignore)
├── logs/                     # 日志 (gitignore)
├── README.md
└── API.md
```

## Mask 约定

i2i 模式的 mask 遵循标准 diffusers 约定:
- **白色 (255) = inpaint 区域** (重绘)
- **黑色 (0) = preserve 区域** (保留)

不提供 mask 时自动生成全白 mask (全图重绘)。
