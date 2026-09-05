# ZIT-service — AI 编码代理须知

> 本文件面向 AI 编码代理。项目自然语言以中文为主，因此本文档使用中文撰写。

## 项目概述

ZIT-service 现已整合为**统一 AI 生成服务**，当前仅支持 Z-Image-Turbo 图片生成。
MiniMax-H3 视频生成（`fl2va` / `ref2va` / `t2va`）已暂时 deprecated，
因当前机器内存不足（30 GB）无法完成 INT4 模型加载。服务监听 `0.0.0.0:8765`。

### 支持的 pipeline 模式

| 模式 | 模型 | 说明 |
|------|------|------|
| `t2i` | Z-Image-Turbo (ZImagePipeline) | 文生图 |
| `i2i` | Z-Image-Turbo (ZImageInpaintPipeline) | 图生图/局部重绘 |
| `fl2va` | ~~MiniMax-H3 INT4~~ | **deprecated** — 内存不足无法加载 |
| `ref2va` | ~~MiniMax-H3 INT4~~ | **deprecated** — 内存不足无法加载 |

同一时刻只运行一个 pipeline 子进程（共享单 GPU）。当前仅 ZIT pipeline 可用，MH3 已禁用。

## 目录结构

```
ZIT-service/
├── image_service.py          # Flask 主程序（统一服务，管理 4 种模式）
├── image_service_pipeline.py # ZIT Pipeline 子进程（t2i/i2i）
├── data/
│   ├── image-gen-history.json # TinyDB 任务持久化数据库（所有任务共用）
│   └── images/                # 生成的 PNG 图片输出目录
├── logs/                      # 运行日志
│   ├── image-service.log      # 主进程日志
│   ├── image_pipeline.log     # ZIT pipeline 日志
│   └── mh3_pipeline.log       # MH3 pipeline 日志（实际在 MH3/logs/）
├── README.md
├── API.md
└── AGENTS.md                  # 本文件
```

MH3 pipeline 脚本位于 `/mnt/data/AV/MH3/mh3_pipeline.py`，模型权重在 `/mnt/data/AV/MH3/models/`。

## 技术栈

- **语言**：Python 3.10+
- **ZIT 虚拟环境**：conda 环境 `image`（硬编码解释器路径）
- **MH3 虚拟环境**：当前系统 Python（`sys.executable`）
- **Web 框架**：Flask（单线程模式运行）
- **持久化**：TinyDB（JSON 文件数据库）
- **ZIT 模型**：`diffusers`（`ZImagePipeline` / `ZImageInpaintPipeline`）
- **MH3 模型**：`diffusers`（`MiniMaxH3Pipeline`），INT4 量化

## 关键运行配置（代码中硬编码）

| 配置项 | 路径 |
|--------|------|
| ZIT 数据库 | `/mnt/data/AV/ZIT-service/data/image-gen-history.json` |
| ZIT 图片输出 | `/mnt/data/AV/ZIT-service/data/images/` |
| ZIT Pipeline 脚本 | `image_service_pipeline.py` |
| ZIT Python 解释器 | `/home/jeefy/openclaw-home/miniconda3/envs/image/bin/python3` |
| MH3 Pipeline 脚本 | `/mnt/data/AV/MH3/mh3_pipeline.py` |
| MH3 视频输出 | `/mnt/data/AV/MH3/data/videos/` |
| 服务端口 | `8765` |

## 启动与运行

```bash
python3 image_service.py
```

Flask 以单线程模式启动在 `0.0.0.0:8765`。

### systemd 服务

```bash
systemctl --user start ZIT-service
systemctl --user start ZIT-tunnel   # SSH 反向隧道
```

## 架构设计

```
Flask API 主进程 (port 8765)
  ├── TaskProcessor
  │     ├── TinyDB 任务持久化
  │     ├── t2i / i2i 内存队列（MH3 队列已禁用）
  │     └── 智能调度（2 队列，同模式优先）
  └── PipelineManager
        └── subprocess.Popen([conda python, image_service_pipeline.py, --mode t2i|i2i])
               ├── stdin  接收 JSON 任务
               └── HTTP POST /task_complete 回调状态
```

### 关键设计要点

1. **单线程 Flask**：`threaded=False`，请求串行处理。
2. **子进程隔离**：pipeline 运行在独立 Python 进程中。
3. **2 队列调度**：t2i/i2i 各有一个内存队列，优先处理当前模式的任务。
4. **ZIT 缓存复用**：已完成任务按参数匹配直接返回结果。
5. **自动重启**：启动时恢复被中断的 `processing`/`queued` 任务。
6. **空闲超时**：pipeline 子进程 30 分钟无任务自动退出。
7. **GPU 显存管理**：每次生成后执行 `gc` + `torch.cuda.empty_cache()`。

> **注意**：MH3 视频生成（fl2va/ref2va/t2va）已暂时禁用。
> 原因：当前机器 30 GB 内存不足以加载 INT4 量化模型（峰值需 ~36 GB）。
> 模型权重和 pipeline 脚本仍在 `/mnt/data/AV/MH3/`，待内存升级后可恢复。

## HTTP API 概览

完整接口定义参见 `API.md`。统一 API 端点（v1）：

### 统一端点

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/v1/tasks` | 提交生成任务（仅支持 `t2i`, `i2i`；`t2va`/`fl2va`/`ref2va` 已禁用） |
| GET  | `/v1/tasks/<id>` | 查询任务详情 |
| GET  | `/v1/tasks/<id>/output?type=image` | 下载生成图片 |
| GET  | `/v1/workflows` | 列出可用 workflow |
| GET  | `/v1/queue` | 队列状态 |
| GET  | `/v1/health` | 健康检查 |

### 向后兼容端点

旧端点保留，但新代码应使用 `/v1/*` 端点。

### 内部端点

- `POST /task_complete` — pipeline 子进程回调
- `POST /pipeline_status` — pipeline 加载/卸载状态回调
- `POST /pipeline_free` — 手动释放 pipeline
- `POST /pipeline_reset` — 强制重置 pipeline 状态（卡死恢复）
- `GET /pipeline_status` — pipeline 状态
- `GET /__restart` — 重启被中断任务

## 任务状态

- `queued`
- `processing`
- `completed`
- `failed`

## ZIT i2i Mask 约定

- **白色 (255)** = 重绘区域
- **黑色 (0)** = 保留区域
- 未提供 mask 时自动生成全白 mask（全图重绘）

## 代码组织

### `image_service.py`

- **配置区**：ZIT 的硬编码路径与常量。
- **`MODE_SCRIPT_MAP`**：模式到脚本/解释器的映射（仅 t2i/i2i）。
- **`PipelineManager`**：管理 ZIT pipeline 子进程生命周期。
- **`TaskProcessor`**：2 队列的任务调度、缓存、持久化。
- **Flask 路由**：ZIT 端点。
- **主程序**：启动日志、恢复中断任务、运行 Flask。

### `image_service_pipeline.py`（ZIT 子进程）

- `--mode {t2i,i2i}`
- 加载 Z-Image-Turbo pipeline
- 循环读取 stdin JSON 任务并生成图片
- 空闲 30 分钟自动退出

> **MH3 pipeline 已禁用**：`/mnt/data/AV/MH3/mh3_pipeline.py` 不再启动。
> 待内存升级（>= 64 GB）后可重新启用。

## 开发约定

- 日志包含 emoji 图标用于快速区分状态。
- 代码路径为硬编码绝对路径，环境迁移需同步修改源码。
- ZIT i2i 任务提交后会移除 `image_base64` / `mask_base64`，避免数据库膨胀。
- 回调状态统一处理：ZIT 发送 `success`，服务端视为完成。

## 部署与安全注意事项

1. **硬编码路径**：依赖 `/home/jeefy/...` 和 `/mnt/data/AV/...`，迁移需修改源码。
2. **单线程服务**：无并发能力，长时间生成会阻塞后续 API 请求。
3. **无认证授权**：API 无鉴权，公网暴露需通过反向代理或 SSH 隧道。
4. **内部端点暴露**：`/task_complete`、`/pipeline_status`、`/__restart` 无鉴权。
5. **GPU 显存**：ZIT (~6GB)，无需切换。
6. **Python 环境**：仅需 conda `image`。

## 常见维护操作

```bash
# 查看队列状态
curl http://localhost:8765/queue/status

# 健康检查
curl http://localhost:8765/health

# 查看日志
tail -f logs/image-service.log
tail -f logs/image_pipeline.log
```