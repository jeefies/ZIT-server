# ZIT-service Unified API

服务地址: `http://localhost:8765`

当前支持 2 种 workflow: `t2i`, `i2i`。

> **注意**：`t2va`, `fl2va`, `ref2va` 已暂时 deprecated。
> 原因：当前机器 30 GB 内存无法加载 MiniMax-H3 INT4 量化模型。

---

## POST /v1/tasks

提交任意类型的生成任务。

### Request

```json
{
  "workflow": "t2i",
  "prompt": "a cat on a cushion",
  "negative_prompt": "blurry, low quality",
  "width": 1024,
  "height": 1024,
  "steps": 9,
  "guidance": 0.0,
  "seed": -1,
  "image_base64": null,
  "mask_base64": null,
  "conditions": [],
  "target": {}
}
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| workflow | string | `"t2i"` | 生成模式: `t2i`, `i2i`（`t2va`/`fl2va`/`ref2va` 已禁用） |
| prompt | string | `""` | 正向提示词 |
| negative_prompt | string | `""` | 负向提示词 (仅 ZIT/Pony) |
| width | int | `1024` | 图片/视频宽度 (仅 ZIT/Pony) |
| height | int | `1024` | 图片/视频高度 (仅 ZIT/Pony) |
| steps | int | `9` | 推理步数 (仅 ZIT/Pony) |
| guidance | float | `0.0` | CFG guidance scale (仅 ZIT/Pony) |
| seed | int | `-1` | 随机种子, `-1` 为随机生成 |
| image_base64 | string | `null` | i2i 模式必需, 输入原图 base64 |
| mask_base64 | string | `null` | i2i 掩码 base64 |
| conditions | array | `[]` | 已禁用 |
| target | object | `{}` | 已禁用 |

### Response (202 queued)

```json
{
  "id": "gen_20260617_011048_ceb6ab",
  "workflow": "t2va",
  "status": "queued",
  "created_at": "2026-06-17T01:10:48",
  "queue_position": 0
}
```

### Response (200 completed, cache hit)

```json
{
  "id": "gen_20260617_011048_ceb6ab",
  "workflow": "t2i",
  "status": "completed",
  "created_at": "2026-06-17T01:10:48",
  "completed_at": "2026-06-17T01:11:41",
  "error": null,
  "outputs": {
    "image": "/v1/tasks/gen_20260617_011048_ceb6ab/output?type=image"
  }
}
```

---

## GET /v1/tasks/:id

查询任务状态。

### Response

```json
{
  "id": "gen_20260617_011048_ceb6ab",
  "workflow": "t2i",
  "status": "completed",
  "created_at": "2026-06-17T01:10:48",
  "completed_at": "2026-06-17T01:11:41",
  "error": null,
  "outputs": {
    "image": "/v1/tasks/gen_20260617_011048_ceb6ab/output?type=image"
  }
}
```

任务状态: `queued` | `processing` | `completed` | `failed`

### Error Response (404)

```json
{"error": "Task not found"}
```

---

## GET /v1/tasks/:id/output

下载生成的文件。

### Query Parameters

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| type | string | `"image"` | 输出类型: `image`（`video`/`audio` 已禁用） |

### Response

- `type=image`: PNG 图片 (`Content-Type: image/png`)

### Error Response

| 状态码 | 说明 |
|--------|------|
| 404 | 任务不存在或文件不存在 |
| 400 | 任务未完成或类型不支持 |

---

## GET /v1/workflows

列出可用 workflow。

### Response

```json
{
  "workflows": [
    {"id": "t2i",   "name": "Text-to-Image",    "family": "zit"},
    {"id": "i2i",   "name": "Image-to-Image",    "family": "zit"}
  ]
}
```

---

## GET /v1/queue

查询队列状态。

### Response

```json
{
  "current_task": "gen_xxx",
  "current_mode": "t2i",
  "pipeline_type": "t2i",
  "pipeline_switching": false,
  "t2i_queue_length": 0,
  "i2i_queue_length": 1,
  "fl2va_queue_length": 0,
  "ref2va_queue_length": 0,
  "t2i_queue": [],
  "i2i_queue": ["gen_yyy"],
  "fl2va_queue": [],
  "ref2va_queue": []
}
```

---

## GET /v1/health

健康检查。

### Response

```json
{
  "status": "healthy",
  "service": "unified-image-video-service",
  "has_pending_tasks": false,
  "process_alive": true,
  "current_mode": "t2i"
}
```

---

## 内部端点 (不对外暴露)

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | /task_complete | Pipeline 子进程回调 |
| POST | /pipeline_status | Pipeline 加载/卸载状态回调 |
| POST | /pipeline_free | 手动释放 pipeline 进程和 GPU 显存 |
| POST | /pipeline_reset | 强制重置 pipeline 状态 (卡死恢复) |
| GET | /pipeline_status | 查询 pipeline 状态 |
| GET | /__restart | 重启被中断任务 |

---

## GET /pipeline_status

```json
{
  "current_type": "t2i",
  "pipeline_loaded": true,
  "busy": false,
  "process_alive": true,
  "last_activity": "2026-06-17T01:11:10.250892"
}
```

---

## 向后兼容端点

以下旧端点保留，但新代码应使用 `/v1/*` 端点:

| 方法 | 路径 | 新端点 |
|------|------|--------|
| POST | /generate | POST /v1/tasks |
| GET | /status/:id | GET /v1/tasks/:id |
| GET | /status/:id/image | GET /v1/tasks/:id/output?type=image |
| GET | /queue/status | GET /v1/queue |
| GET | /health | GET /v1/health |
| POST | /v1/videos | ~~POST /v1/tasks~~（已禁用） |
| GET | /v1/videos/:id | ~~GET /v1/tasks/:id~~（已禁用） |
| GET | /v1/videos/:id/content | ~~GET /v1/tasks/:id/output?type=video~~（已禁用） |
| GET | /v1/videos/:id/audio | ~~GET /v1/tasks/:id/output?type=audio~~（已禁用） |
| GET | /history | — (无替代) |

## i2i 接入说明

`i2i`（Image-to-Image）基于 `ZImageInpaintPipeline`，需要提供输入图片和可选的掩码。

### 请求示例

```bash
curl -X POST http://localhost:8765/v1/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "workflow": "i2i",
    "prompt": "a small red cube on a wooden table",
    "image_base64": "<BASE64_PNG>",
    "mask_base64": "<BASE64_PNG_OR_NULL>",
    "width": 512,
    "height": 512,
    "steps": 4,
    "guidance": 0.0,
    "seed": 42
  }'
```

### 参数说明

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `workflow` | string | 是 | 固定为 `"i2i"` |
| `prompt` | string | 是 | 正向提示词 |
| `image_base64` | string | 是 | 输入原图，Base64 编码的 PNG |
| `mask_base64` | string | 否 | 掩码图，Base64 编码的 PNG；不传则默认全图重绘 |
| `width` | int | 否 | 输出宽度，默认 `512` |
| `height` | int | 否 | 输出高度，默认 `512` |
| `steps` | int | 否 | 推理步数，默认 `4` |
| `guidance` | float | 否 | CFG scale，默认 `0.0` |
| `seed` | int | 否 | 随机种子，`-1` 为随机 |

### 掩码约定

- **白色 (255)** = 重绘区域
- **黑色 (0)** = 保留区域
- 未提供 `mask_base64` 时，服务端自动生成全白掩码，等价于全图重绘

### 响应

任务创建成功返回 `202`：

```json
{
  "id": "gen_20260906_001020_e7a399",
  "workflow": "i2i",
  "status": "queued",
  "created_at": "2026-09-06T00:10:20",
  "queue_position": 0
}
```

完成后查询：

```bash
curl http://localhost:8765/v1/tasks/gen_20260906_001020_e7a399
```

下载图片：

```bash
curl http://localhost:8765/v1/tasks/gen_20260906_001020_e7a399/output?type=image --output result.png
```