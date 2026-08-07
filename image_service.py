#!/usr/bin/env python3
"""
Z-Image-Turbo + MiniMax-H3 统一服务 (主程序)

整合 ZIT 图片生成和 MH3 视频生成，共享 GPU 资源。
- 保留 ZIT 的接口和端口 (8765)
- 支持 4 种 pipeline 模式: t2i, i2i, fl2va, ref2va
- 同一时刻只运行一个 pipeline 子进程

架构设计：
- Flask API：单线程处理请求
- TaskProcessor：管理任务队列和持久化（4 个队列）
- PipelineManager：管理 pipeline 子进程（4 模式切换）
- subprocess：启动独立的 pipeline 进程，通过 stdin 通信
"""

import os
import sys
import uuid
import json
import subprocess
import requests
import gc
import hashlib
import random
from datetime import datetime

from flask import Flask, request, jsonify, send_file
from tinydb import TinyDB, Query

# ==================== 配置 ====================

def load_dotenv(path: str) -> None:
    """Load simple KEY=VALUE pairs without overriding existing environment values."""
    if not os.path.exists(path):
        return
    with open(path, encoding='utf-8-sig') as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value

def optional_int_env(name, default=None):
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip().lower()
    if value in {'', 'none', 'null', 'false', 'off', '0'}:
        return None
    return int(value)

load_dotenv(os.getenv('ENV_FILE', os.path.join(os.path.dirname(__file__), '.env')))

# --- ZIT (Z-Image-Turbo) ---
ZIT_BASE_DIR = os.getenv('ZIT_BASE_DIR', os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.getenv('DB_PATH', os.path.join(ZIT_BASE_DIR, 'data/image-gen-history.json'))
IMAGE_OUTPUT_DIR = os.getenv('IMAGE_OUTPUT_DIR', os.path.join(ZIT_BASE_DIR, 'data/images/'))
I2I_INPUT_DIR = os.getenv('I2I_INPUT_DIR', '/tmp/z-image-inputs/')
ZIT_PIPELINE_SCRIPT = os.getenv('ZIT_PIPELINE_SCRIPT', os.path.join(ZIT_BASE_DIR, 'image_service_pipeline.py'))
ZIT_PYTHON_BIN = os.getenv('ZIT_PYTHON_BIN', '/home/jeefy/openclaw-home/miniconda3/envs/image/bin/python3')
ZIT_PIPE_LOG = os.getenv('ZIT_PIPE_LOG', os.path.join(ZIT_BASE_DIR, 'logs/image_pipeline.log'))
SERVICE_PORT = int(os.getenv('PORT', '8765'))

# --- Pony (Pony Diffusion XL) ---
PONY_SERVICE_BASE_DIR = os.getenv('PONY_SERVICE_BASE_DIR', '/home/jeefy/AV/ZIT-service-pony')
PONY_PIPELINE_SCRIPT = os.getenv('PONY_PIPELINE_SCRIPT', os.path.join(os.path.dirname(__file__), 'image_service_pony_pipeline.py'))
PONY_LOG_FILE = os.getenv('PONY_PIPE_LOG_FILE', os.path.join(PONY_SERVICE_BASE_DIR, 'logs/pony_pipeline.log'))
PONY_PYTHON_BIN = os.getenv('PONY_PYTHON_BIN', '/home/jeefy/openclaw-home/miniconda3/envs/image/bin/python3')
PONY_MODEL_ID = os.getenv('PONY_MODEL_ID', 'AstraliteHeart/pony-diffusion-v6')

# --- MH3 (MiniMax-H3) ---
MH3_BASE_DIR = os.getenv('MH3_BASE_DIR', '/mnt/data/AV/MH3')
MH3_PIPELINE_SCRIPT = os.getenv('MH3_PIPELINE_SCRIPT', os.path.join(MH3_BASE_DIR, 'mh3_pipeline.py'))
MH3_VIDEO_OUTPUT_DIR = os.getenv('MH3_VIDEO_OUTPUT_DIR', os.path.join(MH3_BASE_DIR, 'data/videos'))
MH3_PIPE_LOG = os.getenv('MH3_PIPE_LOG', os.path.join(MH3_BASE_DIR, 'logs/mh3_pipeline.log'))
MH3_PYTHON_BIN = os.getenv('MH3_PYTHON_BIN', sys.executable)

# 确保目录存在
os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)
os.makedirs(I2I_INPUT_DIR, exist_ok=True)
os.makedirs(MH3_VIDEO_OUTPUT_DIR, exist_ok=True)

# --- Mode 映射 ---
DEFAULT_MODEL_FAMILY = os.getenv('MODEL_FAMILY', 'zit').lower()
SUPPORTED_MODEL_FAMILIES = {'zit', 'pony'}
ZIT_MODEL_ID = os.getenv('ZIT_MODEL_ID', 'z-image-turbo')

ZIT_DEFAULTS = {
    'width': int(os.getenv('ZIT_DEFAULT_WIDTH', '1024')),
    'height': int(os.getenv('ZIT_DEFAULT_HEIGHT', '1024')),
    'steps': int(os.getenv('ZIT_DEFAULT_STEPS', '9')),
    'guidance': float(os.getenv('ZIT_DEFAULT_GUIDANCE', '0.0')),
    'strength': float(os.getenv('ZIT_DEFAULT_STRENGTH', '0.8')),
    'negative_prompt': os.getenv('ZIT_DEFAULT_NEGATIVE_PROMPT', ''),
    'clip_skip': optional_int_env('ZIT_DEFAULT_CLIP_SKIP'),
}
PONY_DEFAULTS = {
    'width': int(os.getenv('PONY_DEFAULT_WIDTH', '1024')),
    'height': int(os.getenv('PONY_DEFAULT_HEIGHT', '1024')),
    'steps': int(os.getenv('PONY_DEFAULT_STEPS', '30')),
    'guidance': float(os.getenv('PONY_DEFAULT_GUIDANCE', '7.0')),
    'strength': float(os.getenv('PONY_DEFAULT_STRENGTH', '0.8')),
    'negative_prompt': os.getenv(
        'PONY_DEFAULT_NEGATIVE_PROMPT',
        'score_4, score_5, score_6, lowres, bad anatomy, bad hands, blurry, watermark, signature, text, censored'
    ),
    'clip_skip': optional_int_env('PONY_DEFAULT_CLIP_SKIP'),
}

def normalize_model_family(value=None):
    family = (value or DEFAULT_MODEL_FAMILY or 'zit').lower()
    if family not in SUPPORTED_MODEL_FAMILIES:
        raise ValueError(f"Unsupported model_family: {family}")
    return family

def defaults_for_family(family):
    return PONY_DEFAULTS if family == 'pony' else ZIT_DEFAULTS

def model_id_for_family(family):
    return PONY_MODEL_ID if family == 'pony' else ZIT_MODEL_ID

def pipeline_script_for_family(family):
    return PONY_PIPELINE_SCRIPT if family == 'pony' else ZIT_PIPELINE_SCRIPT

def log_file_for_family(family):
    return PONY_LOG_FILE if family == 'pony' else ZIT_PIPE_LOG

def python_bin_for_family(family):
    return PONY_PYTHON_BIN if family == 'pony' else ZIT_PYTHON_BIN

def pipeline_key_for_task(task_data):
    family = normalize_model_family(task_data.get('model_family'))
    return f"{family}:{task_data.get('mode', 't2i')}"

# ==================== 辅助函数 ====================

def save_i2i_images(task_id: str, image_base64: str, mask_base64: str | None) -> tuple:
    """保存 i2i 输入图片和掩码，返回路径"""
    import base64
    import re
    
    image_path = os.path.join(I2I_INPUT_DIR, f"{task_id}_input.png")
    mask_path = os.path.join(I2I_INPUT_DIR, f"{task_id}_mask.png")
    
    def decode_base64(data_url):
        if data_url and data_url.startswith('data:'):
            match = re.match(r'data:image/\w+;base64,(.+)', data_url)
            if match:
                return base64.b64decode(match.group(1))
        elif data_url:
            return base64.b64decode(data_url)
        return None
    
    image_data = decode_base64(image_base64)
    if image_data:
        with open(image_path, 'wb') as f:
            f.write(image_data)
    else:
        return None, None
    
    if mask_base64:
        mask_data = decode_base64(mask_base64)
        if mask_data:
            with open(mask_path, 'wb') as f:
                f.write(mask_data)
            return image_path, mask_path

    try:
        from PIL import Image
        img = Image.open(image_path)
        w, h = img.size
        mask_img = Image.new('L', (w, h), 255)
        mask_img.save(mask_path)
        print(f"🎨 自动生成全图重绘 mask: {mask_path} ({w}x{h})", file=sys.stderr)
        return image_path, mask_path
    except Exception as e:
        print(f"⚠️ 自动生成 mask 失败: {e}，使用无 mask 模式", file=sys.stderr)
        return image_path, None


MODE_SCRIPT_MAP = {
    'zit': {
        't2i':   (ZIT_PYTHON_BIN, ZIT_PIPELINE_SCRIPT, ZIT_PIPE_LOG),
        'i2i':   (ZIT_PYTHON_BIN, ZIT_PIPELINE_SCRIPT, ZIT_PIPE_LOG),
    },
    'pony': {
        't2i':   (PONY_PYTHON_BIN, PONY_PIPELINE_SCRIPT, PONY_LOG_FILE),
        'i2i':   (PONY_PYTHON_BIN, PONY_PIPELINE_SCRIPT, PONY_LOG_FILE),
    },
    'mh3': {
        'fl2va': (MH3_PYTHON_BIN, MH3_PIPELINE_SCRIPT, MH3_PIPE_LOG),
        'ref2va':(MH3_PYTHON_BIN, MH3_PIPELINE_SCRIPT, MH3_PIPE_LOG),
    },
}

def _mode_for_task(task_type: str) -> str:
    """Map task type to pipeline mode."""
    if task_type in ('t2va', 'fl2va'):
        return 'fl2va'
    if task_type == 'ref2va':
        return 'ref2va'
    return task_type  # t2i, i2i

def _pipeline_key(task_data: dict) -> str:
    """Determine the pipeline key (family:mode) from task data."""
    family = normalize_model_family(task_data.get('model_family', 'zit'))
    mode = _mode_for_task(task_data.get('task', task_data.get('mode', 't2i')))
    if mode in ('fl2va', 'ref2va'):
        return 'mh3', mode
    return family, mode


# ==================== PipelineManager ====================

class PipelineManager:
    """管理 pipeline 子进程，支持 t2i/i2i/fl2va/ref2va 四模式切换"""
    
    def __init__(self):
        self.process = None
        self.pipeline_loaded = False
        self.last_activity = datetime.now()
        self.current_type = None
        self.busy = False
        
    def get_current_type(self):
        return self.current_type
    
    def push_task(self, task_data):
        mode = task_data.get('mode', 't2i')
        self._ensure_pipeline_alive(mode)
        print(f"🔄 推送任务：{task_data['task_id']} (mode={mode})", file=sys.stderr)
        if self.process and self.process.stdin:
            try:
                self.process.stdin.write(json.dumps(task_data) + '\n')
                self.process.stdin.flush()
                self.busy = True
                print(f"✅ 任务已发送：{task_data['task_id']}", file=sys.stderr)
            except Exception as e:
                print(f"❌ 发送任务失败：{e}，尝试重启 pipeline", file=sys.stderr)
                self.process = None
                self._ensure_pipeline_alive(mode)
        else:
            print(f"❌ Pipeline 进程或 stdin 不可用", file=sys.stderr)
    
    def switch_to(self, mode):
        """切换到指定模式的 pipeline，支持 4 种模式"""
        if self.current_type == mode and self.pipeline_loaded:
            return True
            
        print(f"🔄 切换 pipeline 模式: {self.current_type} -> {mode}", file=sys.stderr)
        self.pipeline_loaded = False
        
        if self.process:
            try:
                self.process.stdin.close()
                self.process.wait(timeout=10)
            except:
                self.process.kill()
            self.process = None
        
        self.current_type = None
        success = self._start_pipeline_process(mode)
        return success
        
    def _ensure_pipeline_alive(self, mode='t2i'):
        if self.process is not None and self.process.poll() is None and self.current_type == mode:
            return
        
        if self.process:
            try:
                self.process.stdin.close()
                self.process.wait(timeout=5)
            except Exception:
                self.process.kill()
            self.process = None
        
        self._start_pipeline_process(mode)
    
    def _start_pipeline_process(self, mode='t2i'):
        """启动新的 pipeline 进程，根据模式选择正确的脚本和解释器"""
        self.current_type = mode
        self.pipeline_loaded = False
        
        # 优先从嵌套 MODE_SCRIPT_MAP 查找，兼容旧版 flat key
        info = None
        for family_map in MODE_SCRIPT_MAP.values():
            if mode in family_map:
                info = family_map[mode]
                break
        if not info:
            print(f"❌ 未知模式: {mode}", file=sys.stderr)
            self.current_type = None
            return False
        
        python_bin, script, log_file = info
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        try:
            pipe_log = open(log_file, "a+")
            pipe_log.write(f"\n--- Pipeline start (mode={mode}) at {datetime.now()} ---\n")
            pipe_log.flush()
            
            self.process = subprocess.Popen(
                [python_bin, script, "--mode", mode],
                stdin=subprocess.PIPE,
                stdout=pipe_log,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            print(f"✅ Pipeline 启动 (PID: {self.process.pid}, mode={mode}, interpreter={python_bin})", file=sys.stderr)
            return True
        except Exception as e:
            print(f"❌ Pipeline 启动失败: {e}", file=sys.stderr)
            self.process = None
            self.current_type = None
            return False

    def free_pipeline(self):
        if self.process:
            try:
                self.process.stdin.close()
                self.process.wait(timeout=10)
            except Exception:
                self.process.kill()
            self.process = None
        self.pipeline_loaded = False
        self.current_type = None
        self.busy = False
        self.last_activity = datetime.now()
        print(f"🧹 Pipeline 已手动释放", file=sys.stderr)

    def start_pipeline(self, mode='t2i'):
        return self.switch_to(mode)


# ==================== TaskProcessor ====================

class TaskProcessor:
    def __init__(self):
        self.db = TinyDB(DB_PATH)
        self.pipeline_manager = PipelineManager()
        # ZIT 队列
        self.t2i_queue = []
        self.i2i_queue = []
        # MH3 队列
        self.fl2va_queue = []  # t2va + fl2va 任务
        self.ref2va_queue = []  # ref2va 任务

    def _queue_for_mode(self, mode: str) -> list:
        if mode == 't2i':
            return self.t2i_queue
        if mode == 'i2i':
            return self.i2i_queue
        if mode == 'fl2va':
            return self.fl2va_queue
        if mode == 'ref2va':
            return self.ref2va_queue
        return None

    def add_task(self, task_data):
        """添加新任务到队列（支持 ZIT 和 MH3 任务）"""
        mode = task_data.get('mode', 't2i')
        task_type = task_data.get('task', mode)

        # 如果 seed == -1，生成随机种子
        if task_data.get('seed', -1) == -1:
            task_data['seed'] = random.randint(0, 2**31 - 1)
            print(f"🎲 随机种子：{task_data['seed']}", file=sys.stderr)

        # --- ZIT 任务处理 ---
        if mode in ('t2i', 'i2i'):
            if mode == 'i2i':
                task_data['image_hash'] = hashlib.md5(task_data.get('image_base64', '')[:1000].encode()).hexdigest()[:16] if task_data.get('image_base64') else None
                task_data['mask_hash'] = hashlib.md5(task_data.get('mask_base64', '')[:1000].encode()).hexdigest()[:16] if task_data.get('mask_base64') else 'full'

            # 缓存检查
            Task = Query()
            if mode == 't2i':
                cached = self.db.get(
                    (Task.mode == 't2i') &
                    (Task.prompt == task_data['prompt']) &
                    (Task.negative_prompt == task_data.get('negative_prompt', '')) &
                    (Task.width == task_data['width']) &
                    (Task.height == task_data['height']) &
                    (Task.steps == task_data['steps']) &
                    (Task.guidance == task_data['guidance']) &
                    (Task.seed == task_data['seed']) &
                    (Task.status == 'completed')
                )
            else:
                cached = self.db.get(
                    (Task.mode == 'i2i') &
                    (Task.prompt == task_data['prompt']) &
                    (Task.negative_prompt == task_data.get('negative_prompt', '')) &
                    (Task.width == task_data['width']) &
                    (Task.height == task_data['height']) &
                    (Task.steps == task_data['steps']) &
                    (Task.guidance == task_data['guidance']) &
                    (Task.seed == task_data['seed']) &
                    (Task.image_hash == task_data['image_hash']) &
                    (Task.mask_hash == task_data['mask_hash']) &
                    (Task.status == 'completed')
                )

            if cached:
                print(f"✅ 缓存命中：{task_data['task_id']} -> {cached['task_id']}", file=sys.stderr)
                cached_image_path = cached.get('image_path') or cached.get('output_path')
                db_record = {
                    'task_id': task_data['task_id'],
                    'mode': mode,
                    'prompt': task_data['prompt'],
                    'negative_prompt': task_data.get('negative_prompt', ''),
                    'width': task_data['width'],
                    'height': task_data['height'],
                    'steps': task_data['steps'],
                    'guidance': task_data['guidance'],
                    'seed': task_data['seed'],
                    'status': 'completed',
                    'image_path': cached_image_path,
                    'created_at': datetime.now().isoformat(),
                    'completed_at': datetime.now().isoformat(),
                    'error': None,
                    'error_type': None
                }
                if mode == 'i2i':
                    db_record['image_hash'] = task_data.get('image_hash')
                    db_record['mask_hash'] = task_data.get('mask_hash')
                self.db.insert(db_record)
                return {
                    'task_id': task_data['task_id'],
                    'status': 'completed',
                    'mode': mode,
                    'message': f'缓存命中：复用已完成任务 {cached["task_id"]}',
                    'image_path': cached_image_path
                }

            # i2i 图片处理
            if mode == 'i2i':
                image_base64 = task_data.get('image_base64')
                mask_base64 = task_data.get('mask_base64')
                if not image_base64:
                    return {
                        'task_id': task_data['task_id'],
                        'status': 'failed',
                        'mode': 'i2i',
                        'error': 'i2i 任务需要提供 image_base64'
                    }
                image_path, mask_path = save_i2i_images(task_data['task_id'], image_base64, mask_base64)
                if not image_path:
                    return {
                        'task_id': task_data['task_id'],
                        'status': 'failed',
                        'mode': 'i2i',
                        'error': '图片解码失败'
                    }
                task_data['input_image_path'] = image_path
                task_data['mask_path'] = mask_path
                task_data.pop('image_base64', None)
                task_data.pop('mask_base64', None)

        # 写入数据库
        task_data['status'] = 'queued'
        task_data['created_at'] = datetime.now().isoformat()
        if 'image_path' not in task_data:
            task_data['image_path'] = None
        task_data['error'] = None
        task_data['error_type'] = None
        self.db.insert(task_data)
        print(f"📝 任务已加入队列：{task_data['task_id']}", file=sys.stderr)

        # 加入对应类型的内存队列
        queue = self._queue_for_mode(mode)
        if queue is None:
            print(f"⚠️ 未知 mode: {mode}，使用 t2i 队列", file=sys.stderr)
            queue = self.t2i_queue
        queue_position = len(queue)
        queue.append(task_data)
        print(f"📥 任务加入 {mode} 队列，位置：{queue_position}", file=sys.stderr)

        # 触发调度
        self._schedule_next_task()

        return {
            'task_id': task_data['task_id'],
            'status': 'queued',
            'mode': mode,
            'queue_position': queue_position,
            'message': '任务已加入队列'
        }

    def _schedule_next_task(self):
        """调度下一个任务（支持 4 队列）"""
        pm = self.pipeline_manager
        process_alive = pm.process is not None and pm.process.poll() is None
        
        if not process_alive and pm.current_type is not None:
            pm.pipeline_loaded = False
            pm.current_type = None
            pm.busy = False
        
        if not pm.pipeline_loaded and pm.current_type is not None:
            return
        
        if pm.busy:
            return
        
        current_type = pm.get_current_type()
        queues = {
            't2i': self.t2i_queue,
            'i2i': self.i2i_queue,
            'fl2va': self.fl2va_queue,
            'ref2va': self.ref2va_queue,
        }
        
        # 优先处理当前类型的队列
        if current_type and queues.get(current_type) and queues[current_type]:
            task = queues[current_type].pop(0)
            pm.push_task(task)
            return
        
        # 切换到其他非空队列
        for mode in ['t2i', 'i2i', 'fl2va', 'ref2va']:
            if mode != current_type and queues.get(mode) and queues[mode]:
                if pm.switch_to(mode):
                    task = queues[mode].pop(0)
                    pm.push_task(task)
                    return
        
        # 没有队列有任务，如果 pipeline 在运行但空闲，可以保持
        if not current_type:
            for mode in ['t2i', 'i2i', 'fl2va', 'ref2va']:
                if queues.get(mode) and queues[mode]:
                    if pm.switch_to(mode):
                        task = queues[mode].pop(0)
                        pm.push_task(task)
                        return

    def update_task_status(self, result):
        """更新任务状态（被 /task_complete 调用）"""
        Task = Query()
        task = self.db.get(Task.task_id == result['task_id'])

        if task:
            raw_status = result.get('status', '')
            # 统一处理 "success" (ZIT) 和 "completed" (MH3) 两种完成状态
            if raw_status in ('success', 'completed'):
                task['status'] = 'completed'
                task['image_path'] = result.get('image_path', task.get('image_path'))
                task['video_path'] = result.get('video_path', task.get('video_path'))
                task['audio_path'] = result.get('audio_path', task.get('audio_path'))
                task['completed_at'] = result.get('completed_at', datetime.now().isoformat())
                print(f"✅ 任务完成：{result['task_id']}", file=sys.stderr)
            elif raw_status == 'failed':
                task['status'] = 'failed'
                task['error'] = result.get('error', 'unknown error')
                task['error_type'] = result.get('error_type', 'unknown')
                print(f"❌ 任务失败：{result['task_id']} - {task['error'][:50]}", file=sys.stderr)
            elif raw_status == 'processing':
                task['status'] = 'processing'
            else:
                print(f"⚠️ 未知状态：{raw_status}", file=sys.stderr)
            self.db.update(task, Task.task_id == task['task_id'])

            if raw_status in ('success', 'completed', 'failed'):
                self.pipeline_manager.busy = False
                self._schedule_next_task()
        else:
            print(f"⚠️ 未找到任务：{result['task_id']}", file=sys.stderr)

    def get_task(self, task_id):
        Task = Query()
        return self.db.get(Task.task_id == task_id)

    def get_queue_status(self):
        Task = Query()
        processing = self.db.search(Task.status == 'processing')
        
        return {
            'current_task': processing[0]['task_id'] if processing else None,
            'current_mode': processing[0].get('mode', 't2i') if processing else None,
            'pipeline_type': self.pipeline_manager.get_current_type(),
            'pipeline_switching': not self.pipeline_manager.pipeline_loaded,
            't2i_queue_length': len(self.t2i_queue),
            'i2i_queue_length': len(self.i2i_queue),
            'fl2va_queue_length': len(self.fl2va_queue),
            'ref2va_queue_length': len(self.ref2va_queue),
            't2i_queue': [t['task_id'] for t in self.t2i_queue],
            'i2i_queue': [t['task_id'] for t in self.i2i_queue],
            'fl2va_queue': [t['task_id'] for t in self.fl2va_queue],
            'ref2va_queue': [t['task_id'] for t in self.ref2va_queue],
        }

    def restart_interrupted_tasks(self):
        Task = Query()
        interrupted = self.db.search(Task.status == 'processing') + self.db.search(Task.status == 'queued')

        for task in interrupted:
            task['status'] = 'queued'
            self.db.update(task, Task.task_id == task['task_id'])
            print(f"🔄 重启中断任务：{task['task_id']}", file=sys.stderr)

            mode = task.get('mode', 't2i')
            queue = self._queue_for_mode(mode)
            if queue is None:
                queue = self.t2i_queue
            if not any(t['task_id'] == task['task_id'] for t in queue):
                queue.append(task)
                print(f"📥 中断任务加入 {mode} 队列", file=sys.stderr)

        self._schedule_next_task()


# ==================== Flask API ====================

app = Flask(__name__)
task_processor = TaskProcessor()

# ─── ZIT 原有端点 ─────────────────────────────────

def _submit_generation(family_override=None):
    data = request.get_json(silent=True) or {}
    task_id = f"gen_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

    try:
        family = normalize_model_family(family_override or data.get('model_family'))
    except ValueError as e:
        return jsonify({'task_id': task_id, 'status': 'failed', 'error': str(e)}), 400

    mode = data.get('mode', 't2i')
    if family == 'pony' and mode != 't2i':
        return jsonify({
            'task_id': task_id,
            'status': 'failed',
            'mode': mode,
            'model_family': family,
            'error': 'Only t2i is implemented for Pony. ZIT i2i remains available through model_family=zit.'
        }), 501

    defaults = defaults_for_family(family)
    task_data = {
        'task_id': task_id,
        'mode': mode,
        'prompt': data.get('prompt', ''),
        'negative_prompt': data.get('negative_prompt', defaults['negative_prompt']),
        'width': data.get('width', defaults['width']),
        'height': data.get('height', defaults['height']),
        'steps': data.get('steps', defaults['steps']),
        'guidance': data.get('guidance', defaults['guidance']),
        'strength': data.get('strength', defaults['strength']),
        'clip_skip': data.get('clip_skip', defaults['clip_skip']),
        'model_family': family,
        'model_id': model_id_for_family(family),
        'seed': data.get('seed', -1),
        'image_base64': data.get('image_base64'),
        'mask_base64': data.get('mask_base64'),
    }

    result = task_processor.add_task(task_data)
    return jsonify(result)

@app.route('/generate', methods=['POST'])
def generate():
    return _submit_generation()

@app.route('/generate/pony', methods=['POST'])
def generate_pony():
    return _submit_generation('pony')

@app.route('/batch_generate', methods=['POST'])
def batch_generate():
    raw = request.get_json()

    if isinstance(raw, list):
        tasks = raw
    elif isinstance(raw, dict):
        tasks = raw.get('tasks')
    else:
        tasks = None

    if not isinstance(tasks, list):
        return jsonify({
            'error': '请求体必须是任务列表（JSON array），或包含 "tasks" 字段的 JSON 对象'
        }), 400

    results = []
    queued_count = 0
    completed_count = 0
    failed_count = 0

    for idx, item in enumerate(tasks):
        if not isinstance(item, dict):
            results.append({'index': idx, 'status': 'failed', 'error': '任务项必须是 JSON 对象'})
            failed_count += 1
            continue

        task_id = f"gen_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        try:
            family = normalize_model_family(item.get('model_family'))
        except ValueError as e:
            results.append({'index': idx, 'task_id': task_id, 'status': 'failed', 'error': str(e)})
            failed_count += 1
            continue

        mode = item.get('mode', 't2i')
        if family == 'pony' and mode != 't2i':
            results.append({'index': idx, 'task_id': task_id, 'status': 'failed', 'mode': mode,
                           'model_family': family, 'error': 'Only t2i is implemented for Pony.'})
            failed_count += 1
            continue

        defaults = defaults_for_family(family)
        task_data = {
            'task_id': task_id,
            'mode': mode,
            'prompt': item.get('prompt', ''),
            'negative_prompt': item.get('negative_prompt', defaults['negative_prompt']),
            'width': item.get('width', defaults['width']),
            'height': item.get('height', defaults['height']),
            'steps': item.get('steps', defaults['steps']),
            'guidance': item.get('guidance', defaults['guidance']),
            'strength': item.get('strength', defaults['strength']),
            'clip_skip': item.get('clip_skip', defaults['clip_skip']),
            'model_family': family,
            'model_id': model_id_for_family(family),
            'seed': item.get('seed', -1),
            'image_base64': item.get('image_base64'),
            'mask_base64': item.get('mask_base64'),
        }

        try:
            result = task_processor.add_task(task_data)
        except Exception as e:
            result = {'task_id': task_id, 'status': 'failed', 'mode': task_data['mode'], 'error': f'提交任务异常: {str(e)}'}

        result['index'] = idx
        results.append(result)

        status = result.get('status')
        if status == 'failed':
            failed_count += 1
        elif status == 'completed':
            completed_count += 1
        else:
            queued_count += 1

    return jsonify({
        'results': results,
        'summary': {'total': len(tasks), 'queued': queued_count, 'completed': completed_count, 'failed': failed_count}
    })

@app.route('/status/<task_id>', methods=['GET'])
def get_status(task_id):
    task = task_processor.get_task(task_id)
    if task:
        return jsonify(task)
    return jsonify({'error': 'Task not found'}), 404

@app.route('/queue/status', methods=['GET'])
def queue_status():
    return jsonify(task_processor.get_queue_status())

@app.route('/health', methods=['GET'])
def health():
    status = task_processor.get_queue_status()
    pm = task_processor.pipeline_manager
    total_pending = status['t2i_queue_length'] + status['i2i_queue_length'] + status['fl2va_queue_length'] + status['ref2va_queue_length']
    return jsonify({
        'status': 'healthy',
        'service': 'unified-image-video-service',
        'default_model_family': DEFAULT_MODEL_FAMILY,
        'supported_model_families': sorted(SUPPORTED_MODEL_FAMILIES),
        'pony_model_id': PONY_MODEL_ID,
        'has_pending_tasks': total_pending > 0,
        'process_alive': pm.process is not None and pm.process.poll() is None,
        'current_mode': pm.current_type,
    })

@app.route('/task_complete', methods=['POST'])
def task_complete():
    result = request.json
    task_processor.update_task_status(result)
    return jsonify({'success': True})

@app.route('/history', methods=['GET'])
def get_history():
    tasks = task_processor.db.all()
    result = []
    for task in tasks:
        result.append({
            'task_id': task['task_id'],
            'status': 'completed' if task['status'] == 'completed' else task.get('status', 'queued')
        })
    result.sort(key=lambda x: x.get('task_id', ''), reverse=True)
    limit = int(request.args.get('limit', 0))
    return jsonify(result[:limit] if limit > 0 else result)

@app.route('/status/<task_id>/image', methods=['GET'])
def get_task_image(task_id):
    task = task_processor.get_task(task_id)
    if not task:
        return jsonify({'error': 'Task not found'}), 404
    if task['status'] != 'completed':
        return jsonify({'error': 'Image not ready'}), 400
    image_path = os.path.join(IMAGE_OUTPUT_DIR, f"{task_id}.png")
    if not os.path.exists(image_path):
        return jsonify({'error': 'Image file not found'}), 404
    return send_file(image_path, mimetype='image/png')

@app.route('/pipeline_status', methods=['POST'])
def update_pipeline_status():
    data = request.json
    status = data.get('status', 'unknown')

    if status == 'loaded':
        task_processor.pipeline_manager.pipeline_loaded = True
        task_processor.pipeline_manager.last_activity = datetime.now()
        print(f"🔄 Pipeline 已加载 (时间: {data.get('timestamp', 'N/A')})", file=sys.stderr)
        task_processor._schedule_next_task()
    elif status == 'unloaded':
        task_processor.pipeline_manager.pipeline_loaded = False
        print(f"🧹 Pipeline 已卸载 (时间: {data.get('timestamp', 'N/A')})", file=sys.stderr)
    elif status == 'error':
        error_type = data.get('error_type', 'unknown')
        task_processor.pipeline_manager.pipeline_loaded = False
        print(f"❌ Pipeline 错误: {error_type} - {data.get('message', 'N/A')}", file=sys.stderr)

    return jsonify({'success': True})

@app.route('/pipeline_free', methods=['POST'])
def pipeline_free():
    pm = task_processor.pipeline_manager
    if not pm.process and not pm.pipeline_loaded:
        return jsonify({'status': 'already_free', 'message': 'Pipeline 未运行'})
    pm.free_pipeline()
    return jsonify({'status': 'freed', 'message': 'Pipeline 已释放'})

@app.route('/pipeline_status', methods=['GET'])
def get_pipeline_status():
    pm = task_processor.pipeline_manager
    return jsonify({
        'current_type': pm.current_type,
        'pipeline_loaded': pm.pipeline_loaded,
        'busy': pm.busy,
        'process_alive': pm.process is not None and pm.process.poll() is None,
        'last_activity': pm.last_activity.isoformat() if pm.last_activity else None
    })

@app.route('/__restart')
def __restart_all_tasks():
    task_processor.restart_interrupted_tasks()
    return jsonify({"restart": "now"})

# ─── MH3 端点 ────────────────────────────────────

@app.route('/v1/videos', methods=['POST'])
def generate_video():
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "Invalid JSON body"}), 400

    task_type = data.get("task", "t2va")
    if task_type not in ("t2va", "fl2va", "ref2va"):
        return jsonify({"error": f"Unsupported task: {task_type}. Supported: t2va, fl2va, ref2va"}), 400

    mode = _mode_for_task(task_type)
    task_id = f"mh3_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

    task_data = {
        'task_id': task_id,
        'task': task_type,
        'mode': mode,
        'prompt': data.get('prompt', ''),
        'conditions': data.get('conditions', []),
        'target': data.get('target', {}),
        'seed': data.get('seed', -1),
        'callback_url': f"http://127.0.0.1:8765/task_complete",
        'created_at': datetime.now().isoformat(),
    }

    result = task_processor.add_task(task_data)

    # 统一响应格式为 MH3 风格
    if result.get('status') == 'queued':
        return jsonify({"id": task_id, "status": "queued"}), 202
    elif result.get('status') == 'completed':
        return jsonify({"id": task_id, "status": "completed"}), 200
    else:
        return jsonify({"id": task_id, "status": result.get('status', 'failed'), "error": result.get('error')}), 400

@app.route('/v1/videos/<task_id>', methods=['GET'])
def get_video_task(task_id):
    t = task_processor.get_task(task_id)
    if not t:
        return jsonify({"error": "Task not found"}), 404

    return jsonify({
        "id": t["task_id"],
        "status": t.get("status", "unknown"),
        "task": t.get("task"),
        "created_at": t.get("created_at"),
        "completed_at": t.get("completed_at"),
        "error": t.get("error"),
    })

@app.route('/v1/videos/<task_id>/content', methods=['GET'])
def get_video_content(task_id):
    t = task_processor.get_task(task_id)
    if not t:
        return jsonify({"error": "Task not found"}), 404
    if t.get("status") != "completed":
        return jsonify({"error": f"Task status is {t.get('status')}, not completed"}), 400

    video_path = t.get("video_path")
    if video_path and os.path.exists(video_path):
        return send_file(video_path, mimetype="video/mp4")

    frames_dir = os.path.join(MH3_VIDEO_OUTPUT_DIR, task_id)
    if os.path.isdir(frames_dir):
        import zipfile
        import io
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            for f in sorted(os.listdir(frames_dir)):
                zf.write(os.path.join(frames_dir, f), f)
        buf.seek(0)
        return send_file(buf, mimetype="application/zip", as_attachment=True,
                         download_name=f"{task_id}_frames.zip")

    return jsonify({"error": "No video file found"}), 404

@app.route('/v1/videos/<task_id>/audio', methods=['GET'])
def get_video_audio(task_id):
    t = task_processor.get_task(task_id)
    if not t:
        return jsonify({"error": "Task not found"}), 404
    audio_path = t.get("audio_path")
    if audio_path and os.path.exists(audio_path):
        return send_file(audio_path, mimetype="audio/wav")
    return jsonify({"error": "No audio file found"}), 404


# ==================== 主程序 ====================

if __name__ == '__main__':
    print("="*60, file=sys.stderr)
    print("🚀 统一 AI 生成服务启动", file=sys.stderr)
    print("="*60, file=sys.stderr)
    print(f"📁 ZIT 数据库：{DB_PATH}", file=sys.stderr)
    print(f"📁 ZIT 图片输出：{IMAGE_OUTPUT_DIR}", file=sys.stderr)
    print(f"📁 MH3 视频输出：{MH3_VIDEO_OUTPUT_DIR}", file=sys.stderr)
    print(f"🔧 支持模式：t2i / i2i / fl2va (t2va+fl2va) / ref2va", file=sys.stderr)
    print("="*60, file=sys.stderr)

    # 重启被中断的任务
    task_processor.restart_interrupted_tasks()

    try:
        app.run(host='0.0.0.0', port=8765, threaded=False)
    finally:
        if task_processor.pipeline_manager.process:
            task_processor.pipeline_manager.process.stdin.close()
            task_processor.pipeline_manager.process.wait(timeout=5)