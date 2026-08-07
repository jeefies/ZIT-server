# ZIT-service — AI 编码代理须知

> 本文件面向 AI 编码代理。项目自然语言以中文为主，因此本文档使用中文撰写。

## 项目概述

ZIT-service 现已整合为**统一 AI 生成服务**，同时支持 Z-Image-Turbo 图片生成和 MiniMax-H3 视频生成，
共享 GPU 资源。服务监听 `0.0.0.0:8765`。

### 支持的 pipeline 模式

| 模式 | 模型 | 说明 |
|------|------|------|
| `t2i` | Z-Image-Turbo (ZImagePipeline) | 文生图 |
| `i2i` | Z-Image-Turbo (ZImageInpaintPipeline) | 图生图/局部重绘 |
| `fl2va` | MiniMax-H3 INT4 (FL2VA checkpoint) | 文生视频(t2va) / 首尾帧生视频(fl2va) |
| `ref2va` | MiniMax-H3 INT4 (Ref2VA checkpoint) | 参考生视频 |

同一时刻只运行一个 pipeline 子进程（共享单 GPU），模式切换时关闭旧进程、启动新进程。

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
  │     ├── t2i / i2i / fl2va / ref2va 内存队列
  │     ├── ZIT 缓存命中复用（仅 t2i/i2i）
  │     └── 智能调度（4 队列，同模式优先）
  └── PipelineManager
        ├── subprocess.Popen([conda python, image_service_pipeline.py, --mode t2i|i2i])
        └── subprocess.Popen([python, mh3_pipeline.py, --mode fl2va|ref2va])
               ├── stdin  接收 JSON 任务
               └── HTTP POST /task_complete 回调状态
```

### 关键设计要点

1. **单线程 Flask**：`threaded=False`，请求串行处理。
2. **子进程隔离**：pipeline 运行在独立 Python 进程中。
3. **4 队列调度**：t2i/i2i/fl2va/ref2va 各有一个内存队列，优先处理当前模式的任务。
4. **模式切换**：切换模式时关闭旧进程、启动新进程（ZIT ~20s，MH3 ~30-60s）。
5. **ZIT 缓存复用**：已完成任务按参数匹配直接返回结果。
6. **自动重启**：启动时恢复被中断的 `processing`/`queued` 任务。
7. **空闲超时**：pipeline 子进程 30 分钟无任务自动退出。
8. **GPU 显存管理**：每次生成后执行 `gc` + `torch.cuda.empty_cache()`。

## HTTP API 概览

完整接口定义参见 `API.md`。常用端点：

### ZIT 端点（图片生成）

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/generate` | 提交图片生成任务（t2i/i2i） |
| GET  | `/status/<task_id>` | 查询任务详情 |
| GET  | `/status/<task_id>/image` | 下载生成图片 |
| GET  | `/queue/status` | 队列状态（含 4 种队列长度） |
| GET  | `/health` | 健康检查 |
| GET  | `/pipeline_status` | pipeline 状态 |
| POST | `/pipeline_free` | 手动释放 pipeline |
| GET  | `/history` | 任务历史（task_id + status） |
| GET  | `/__restart` | 重启被中断任务 |

### MH3 端点（视频生成）

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/v1/videos` | 提交视频生成任务（t2va/fl2va/ref2va） |
| GET  | `/v1/videos/<id>` | 查询视频任务状态 |
| GET  | `/v1/videos/<id>/content` | 下载生成视频 |
| GET  | `/v1/videos/<id>/audio` | 下载生成音频 |

### 内部端点

- `POST /task_complete` — pipeline 子进程回调（接受 `success` 和 `completed` 两种完成状态）
- `POST /pipeline_status` — pipeline 加载/卸载状态回调

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

- **配置区**：ZIT 和 MH3 的硬编码路径与常量。
- **`MODE_SCRIPT_MAP`**：模式到脚本/解释器的映射。
- **`PipelineManager`**：管理 4 种 pipeline 子进程生命周期。
- **`TaskProcessor`**：4 队列的任务调度、缓存、持久化。
- **Flask 路由**：ZIT 端点 + MH3 端点。
- **主程序**：启动日志、恢复中断任务、运行 Flask。

### `image_service_pipeline.py`（ZIT 子进程）

- `--mode {t2i,i2i}`
- 加载 Z-Image-Turbo pipeline
- 循环读取 stdin JSON 任务并生成图片
- 空闲 30 分钟自动退出

### `/mnt/data/AV/MH3/mh3_pipeline.py`（MH3 子进程）

- `--mode {fl2va,ref2va}`
- 加载 MiniMax-H3 INT4 量化模型
- t2va/fl2va 使用 FL2VA checkpoint，ref2va 使用 Ref2VA checkpoint
- 空闲 30 分钟自动退出

## 开发约定

- 日志包含 emoji 图标用于快速区分状态。
- 代码路径为硬编码绝对路径，环境迁移需同步修改源码。
- ZIT i2i 任务提交后会移除 `image_base64` / `mask_base64`，避免数据库膨胀。
- MH3 任务 ID 使用 `mh3_` 前缀，ZIT 任务使用 `gen_` 前缀。
- 回调状态统一处理：ZIT 发送 `success`，MH3 发送 `completed`，服务端均视为完成。

## 部署与安全注意事项

1. **硬编码路径**：依赖 `/home/jeefy/...` 和 `/mnt/data/AV/...`，迁移需修改源码。
2. **单线程服务**：无并发能力，长时间生成会阻塞后续 API 请求。
3. **无认证授权**：API 无鉴权，公网暴露需通过反向代理或 SSH 隧道。
4. **内部端点暴露**：`/task_complete`、`/pipeline_status`、`/__restart` 无鉴权。
5. **GPU 显存**：ZIT (~6GB) 和 MH3 (~15GB) 切换需完全卸载模型，耗时 20-60 秒。
6. **Python 环境**：ZIT 使用 conda `image`，MH3 使用系统 Python。两个环境都需要安装 `flask`、`tinydb`、`requests` 等依赖。

## 常见维护操作

```bash
# 查看队列状态（含 4 种队列）
curl http://localhost:8765/queue/status

# 健康检查
curl http://localhost:8765/health

# 释放显存
curl -X POST http://localhost:8765/pipeline_free

# 查看日志
tail -f logs/image-service.log
tail -f logs/image_pipeline.log
tail -f /mnt/data/AV/MH3/logs/mh3_pipeline.log
```