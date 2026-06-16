# ZIT-service API

服务地址: `http://localhost:8765`

---

## POST /generate

提交图片生成任务。

### Request

```json
{
  "mode": "t2i",
  "prompt": "a cat on a cushion",
  "negative_prompt": "blurry, low quality",
  "width": 1024,
  "height": 1024,
  "steps": 9,
  "guidance": 0.0,
  "seed": -1,
  "image_base64": null,
  "mask_base64": null
}
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| mode | string | `"t2i"` | 生成模式: `t2i` (文生图) 或 `i2i` (图生图) |
| prompt | string | `""` | 正向提示词 |
| negative_prompt | string | `""` | 负向提示词 (不希望出现的内容) |
| width | int | `1024` | 图片宽度 |
| height | int | `1024` | 图片高度 |
| steps | int | `9` | 推理步数 |
| guidance | float | `0.0` | CFG guidance scale |
| seed | int | `-1` | 随机种子, `-1` 为随机生成 |
| image_base64 | string | `null` | i2i 模式必需, 输入原图 base64 (支持 `data:image/png;base64,xxx` 格式) |
| mask_base64 | string | `null` | i2i 掩码 base64, 白=重绘区域, 黑=保留区域; 不提供则全图重绘 |

### Response (缓存命中)

```json
{
  "task_id": "gen_20260617_011048_ceb6ab",
  "status": "completed",
  "mode": "t2i",
  "message": "缓存命中：复用已完成任务 gen_xxx",
  "image_path": "/images/gen_xxx.png"
}
```

### Response (加入队列)

```json
{
  "task_id": "gen_20260617_011048_ceb6ab",
  "status": "queued",
  "mode": "t2i",
  "queue_position": 0,
  "message": "任务已加入队列"
}
```

### Response (i2i 参数错误)

```json
{
  "task_id": "gen_xxx",
  "status": "failed",
  "mode": "i2i",
  "error": "i2i 任务需要提供 image_base64"
}
```

---

## GET /status/:task_id

查询任务详情。

### Response

```json
{
  "task_id": "gen_20260617_011048_ceb6ab",
  "mode": "t2i",
  "prompt": "a cat on a cushion",
  "negative_prompt": "blurry, low quality",
  "width": 1024,
  "height": 1024,
  "steps": 9,
  "guidance": 0.0,
  "seed": 42,
  "status": "completed",
  "image_path": "/images/gen_20260617_011048_ceb6ab.png",
  "created_at": "2026-06-17T01:10:48.271551",
  "completed_at": "2026-06-17T01:11:41.837321",
  "error": null,
  "error_type": null
}
```

任务状态: `queued` | `processing` | `completed` | `failed`

### Error Response (404)

```json
{"error": "Task not found"}
```

---

## GET /status/:task_id/image

下载生成的图片 (PNG)。

### 条件

- 任务状态必须为 `completed`
- 图片文件必须存在

### Response

PNG 图片文件 (`Content-Type: image/png`)

### Error Response

| 状态码 | 说明 |
|--------|------|
| 404 | 任务不存在或图片文件不存在 |
| 400 | 任务未完成 (`Image not ready`) |

---

## GET /queue/status

查询任务队列状态。

### Response

```json
{
  "current_task": "gen_xxx",
  "current_mode": "t2i",
  "pipeline_type": "t2i",
  "pipeline_switching": false,
  "t2i_queue_length": 0,
  "i2i_queue_length": 1,
  "t2i_queue": [],
  "i2i_queue": ["gen_yyy"]
}
```

---

## GET /health

健康检查。

### Response

```json
{
  "status": "healthy",
  "service": "z-image-turbo-queue",
  "has_pending_tasks": false,
  "process_alive": true
}
```

---

## GET /pipeline_status

查询 pipeline 状态。

### Response

```json
{
  "current_type": "t2i",
  "pipeline_loaded": true,
  "busy": false,
  "process_alive": true,
  "last_activity": "2026-06-17T01:11:10.250892"
}
```

| 字段 | 说明 |
|------|------|
| current_type | 当前 pipeline 模式: `t2i` / `i2i` / `null` |
| pipeline_loaded | 模型是否已加载完成 |
| busy | pipeline 是否正在处理任务 |
| process_alive | 子进程是否存活 |
| last_activity | 最后活动时间 (ISO 8601) |

---

## POST /pipeline_free

手动释放 pipeline 进程和 GPU 显存。pipeline 空闲后下次任务会自动重新加载。

### Response (已释放)

```json
{"status": "freed", "message": "Pipeline 已释放"}
```

### Response (未运行)

```json
{"status": "already_free", "message": "Pipeline 未运行"}
```

---

## GET /history

查询所有任务历史 (仅 task_id + status)。

### Query Parameters

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| limit | int | `0` | 返回数量限制, `0` 为不限制 |

### Response

```json
[
  {"task_id": "gen_20260617_011048_ceb6ab", "status": "completed"},
  {"task_id": "gen_20260617_015851_f3bbb6", "status": "completed"}
]
```

按 task_id 倒序排列。

---

## POST /task_complete

Pipeline 子进程内部回调端点，用于通知主进程任务完成。**外部调用者不应直接使用此端点。**

### Request

```json
{
  "task_id": "gen_xxx",
  "status": "success",
  "mode": "t2i",
  "image_path": "/images/gen_xxx.png",
  "completed_at": "2026-06-17T01:11:41.837321"
}
```

---

## POST /pipeline_status

Pipeline 子进程内部回调端点，用于通知主进程 pipeline 加载/卸载状态。**外部调用者不应直接使用此端点。**

---

## GET /__restart

重启所有 interrupted (processing/queued) 任务。服务启动时自动调用。

### Response

```json
{"restart": "now"}
```
