#!/usr/bin/env python3
"""
Z-Image-Turbo Pipeline Process
独立的 pipeline 进程，通过 stdin 接收任务，通过 HTTP 回调状态

运行方式：
    python image_service_pony_pipeline.py

通信方式：
    - 从 stdin 读取 JSON 格式的任务
    - 通过 HTTP POST http://localhost:8765/task_complete 回调状态
"""

import os
import sys
import json
import gc
import torch
import requests
import logging
import time
import fcntl
import errno
import argparse
import inspect
from datetime import datetime
from pathlib import Path
from diffusers import StableDiffusionXLPipeline
from diffusers.utils import load_image
from PIL import Image
import numpy as np

# ==================== 参数解析 ====================

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

load_dotenv(os.getenv('ENV_FILE', os.path.join(os.path.dirname(__file__), '.env')))

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["t2i", "i2i"], default="t2i")
args = parser.parse_args()
MODE = args.mode

# ==================== 日志配置 ====================

SERVICE_BASE_DIR = os.getenv('PONY_SERVICE_BASE_DIR', '/home/jeefy/AV/Pony-service')
LOG_FILE = os.getenv('PIPE_LOG_FILE', os.path.join(SERVICE_BASE_DIR, 'logs', 'pony_pipeline.log'))
os.makedirs(Path(LOG_FILE).parent, exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] PID:%(process)d %(funcName)s:%(lineno)d - %(message)s',
)

logger = logging.getLogger(__name__)

# ==================== 配置 ====================

IMAGE_OUTPUT_DIR = Path(os.getenv("IMAGE_OUTPUT_DIR", os.path.join(SERVICE_BASE_DIR, "data", "images")))
SERVICE_BASE_URL = os.getenv('SERVICE_BASE_URL', 'http://localhost:8765').rstrip('/')
CALLBACK_URL = os.getenv('CALLBACK_URL', f'{SERVICE_BASE_URL}/task_complete')
PIPELINE_STATUS_URL = os.getenv('PIPELINE_STATUS_URL', f'{SERVICE_BASE_URL}/pipeline_status')
MODEL_ID = os.getenv("PONY_MODEL_ID", "AstraliteHeart/pony-diffusion-v6")
MODEL_LOCAL_FILES_ONLY = os.getenv("MODEL_LOCAL_FILES_ONLY", "true").lower() not in {"0", "false", "no"}
MODEL_VARIANT = os.getenv("PONY_MODEL_VARIANT", "").strip() or None
TORCH_DTYPE_NAME = os.getenv("PONY_TORCH_DTYPE", "float16").lower()
IDLE_TIMEOUT = 60 * 30  # 30 分钟

# 确保目录存在
os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)

# ==================== 工具函数 ====================

def _cleanup_after_generation():
    """每次生成后执行的激进清理"""
    # 强制 GC（多轮确保清理）
    for _ in range(3):
        gc.collect()

    # CUDA 清理
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.set_device(0)

    print("✅ 显存已彻底清理", file=sys.stderr, flush=True)

def _cleanup_pipeline():
    """退出前彻底清理"""
    _cleanup_after_generation()
    logger.info("Pipeline 已清理，进程退出")
    
    # 通知主进程 pipeline 状态
    try:
        requests.post(PIPELINE_STATUS_URL, 
                     json={'status': 'unloaded', 'timestamp': datetime.now().isoformat()}, 
                     timeout=5)
        logger.info("已通知主进程卸载状态")
    except Exception as e:
        logger.warning(f"通知主进程卸载状态失败: {e}")

def send_callback(result):
    """发送 HTTP 回调"""
    try:
        requests.post(CALLBACK_URL, json=result, timeout=10)
    except Exception as e:
        print(f"⚠️ 回调失败：{e}", file=sys.stderr, flush=True)

# ==================== 主函数 ====================

def _torch_dtype():
    if TORCH_DTYPE_NAME in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if TORCH_DTYPE_NAME in {"fp32", "float32"}:
        return torch.float32
    return torch.float16

def _load_kwargs():
    kwargs = {
        "torch_dtype": _torch_dtype(),
        "local_files_only": MODEL_LOCAL_FILES_ONLY,
        "use_safetensors": True,
    }
    if MODEL_VARIANT:
        kwargs["variant"] = MODEL_VARIANT
    return kwargs

def _enable_memory_optimizations(pipe):
    pipe.enable_model_cpu_offload()
    pipe.enable_attention_slicing("auto")
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    if hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()

def _load_t2i_pipeline():
    model_path = Path(MODEL_ID)
    if model_path.is_file() and model_path.suffix == '.safetensors':
        return StableDiffusionXLPipeline.from_single_file(str(model_path), **_load_kwargs())
    if model_path.is_dir():
        if (model_path / 'model_index.json').exists():
            return StableDiffusionXLPipeline.from_pretrained(MODEL_ID, **_load_kwargs())
        safetensors_files = sorted(model_path.glob('*.safetensors'))
        if safetensors_files:
            return StableDiffusionXLPipeline.from_single_file(str(safetensors_files[0]), **_load_kwargs())
        raise FileNotFoundError(f'No model_index.json or .safetensors found in {model_path}')
    return StableDiffusionXLPipeline.from_pretrained(MODEL_ID, **_load_kwargs())

def _pipeline_call(pipe, **kwargs):
    signature = inspect.signature(pipe.__call__)
    supported = set(signature.parameters)
    filtered = {key: value for key, value in kwargs.items() if key in supported}
    return pipe(**filtered)

def main():
    """Pipeline 进程入口"""
    
    # 内存优化配置
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # 设置 stdin 为非阻塞模式（解决 select 管道检测问题）
    try:
        flags = fcntl.fcntl(sys.stdin.fileno(), fcntl.F_GETFL)
        fcntl.fcntl(sys.stdin.fileno(), fcntl.F_SETFL, flags | os.O_NONBLOCK)
        logger.info("stdin 已设置为非阻塞模式")
    except Exception as e:
        logger.warning(f"设置 stdin 非阻塞失败: {e}")

    logger.info(f"加载 pipeline (mode={MODE})...")
    try:
        if MODE == "t2i":
            logger.info("加载 t2i pipeline...")
            pipe = _load_t2i_pipeline()
        else:  # i2i
            raise RuntimeError("Pony i2i/inpaint is not implemented in this branch.")
        logger.info("offload...")
        # 使用 sequential CPU offload 避免 hook pickle 问题
        _enable_memory_optimizations(pipe)
        logger.info(f"Pipeline 加载完成 (mode={MODE})")
        
        # 通知主进程 pipeline 状态
        try:
            requests.post(PIPELINE_STATUS_URL, 
                         json={'status': 'loaded', 'timestamp': datetime.now().isoformat()}, 
                         timeout=5)
            logger.info("已通知主进程加载状态")
        except Exception as e:
            logger.warning(f"通知主进程加载状态失败: {e}")
    except Exception as e:
        logger.fatal(f"❌ Pipeline 加载失败：{e}")
        send_callback({
            'status': 'error',
            'error': f'Pipeline 加载失败：{str(e)}'
        })
        return

    last_active = datetime.now()

    print(f"⏰ 空闲超时：{IDLE_TIMEOUT}秒", file=sys.stderr, flush=True)
    logger.info("开始从 stdin 读取任务...")
    pipeline_loaded_time = datetime.now()  # 记录 pipeline 加载时间

    while True:
        # 1. 检查是否超时
        idle_seconds = (datetime.now() - last_active).total_seconds()
        if idle_seconds > IDLE_TIMEOUT:
            logger.info(f"空闲 {idle_seconds:.0f} 秒，退出 pipeline")
            break

        # 2. 非阻塞读取 stdin（不再使用 select）
        try:
            line = sys.stdin.readline()

            if line:
                # 有数据，解析任务
                task = json.loads(line.strip())
                last_active = datetime.now()
                logger.info(f"开始生成：{task['task_id']}")
            else:
                # readline 返回空字符串
                # 检查 stdin 是否关闭（EOF）
                if sys.stdin.closed:
                    logger.info("stdin 已关闭，退出")
                    break
                # 否则只是无数据，等待后继续循环
                time.sleep(0.3)
                continue

        except IOError as e:
            # 非阻塞模式下的预期错误
            if hasattr(e, 'errno') and e.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                # 无数据可读，等待后继续
                time.sleep(0.3)
                continue
            else:
                # 其他 IO 错误
                logger.error(f"stdin IOError: {e}")
                break
        except json.JSONDecodeError as e:
            logger.error(f"JSON 解析错误：{e}")
            time.sleep(0.3)
            continue
        except Exception as e:
            logger.error(f"循环错误：{e}")
            # 等待后继续，避免快速失败
            time.sleep(0.3)
            continue

        # 3. 发送 processing 状态
        send_callback({
            'task_id': task['task_id'],
            'status': 'processing',
            'mode': task.get('mode', 't2i'),
        })

        # 4. 生成图片
        try:
            generator = torch.Generator("cuda").manual_seed(
                task['seed'] if task['seed'] >= 0 else torch.randint(0, 2**31, (1,)).item()
            )

            # 根据 mode 选择生成方式
            task_mode = task.get('mode', 't2i')

            if task_mode == 'i2i':
                # i2i 模式：需要原图和掩码
                width = task['width']
                height = task['height']

                # 读取原图（优先使用 input_image_path，向后兼容 image_path）
                input_path = task.get('input_image_path') or task.get('image_path')
                if not input_path:
                    raise ValueError("i2i 任务缺少输入图片路径")
                image = load_image(input_path).convert("RGB")
                # resize 到目标尺寸
                image = image.resize((width, height), Image.LANCZOS)

                # 处理掩码
                mask_path = task.get('mask_path')
                if mask_path and os.path.exists(mask_path):
                    mask = load_image(mask_path).convert("L")
                    mask = mask.resize((width, height), Image.LANCZOS)
                else:
                    # 无掩码，生成全白掩码（全图重绘）
                    mask = Image.new("L", (width, height), 255)

                # 调用 i2i pipeline
                image = _pipeline_call(
                    pipe,
                    image=image,
                    mask_image=mask,
                    prompt=task['prompt'],
                    negative_prompt=task.get('negative_prompt') or None,
                    width=width,
                    height=height,
                    num_inference_steps=task['steps'],
                    guidance_scale=task['guidance'],
                    strength=task.get('strength', 0.8),
                    clip_skip=task.get('clip_skip'),
                    generator=generator
                ).images[0]
            else:
                # t2i 模式：保持现有逻辑
                image = _pipeline_call(
                    pipe,
                    prompt=task['prompt'],
                    negative_prompt=task.get('negative_prompt') or None,
                    width=task['width'],
                    height=task['height'],
                    num_inference_steps=task['steps'],
                    guidance_scale=task['guidance'],
                    clip_skip=task.get('clip_skip'),
                    generator=generator
                ).images[0]

            # 保存图片
            image_path = os.path.join(IMAGE_OUTPUT_DIR, f"{task['task_id']}.png")
            image.save(image_path, "PNG")
            print(f"✅ 图片已保存：{image_path}", file=sys.stderr, flush=True)

            last_active = datetime.now()

            # 5. 发送成功回调
            send_callback({
                'task_id': task['task_id'],
                'status': 'success',
                'mode': task.get('mode', 't2i'),
                'image_path': f"/images/{task['task_id']}.png",
                'completed_at': datetime.now().isoformat()
            })

            # 6. 清理显存
            _cleanup_after_generation()

        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"OOM: {e}")
            _cleanup_after_generation()

            # 通知主进程错误状态
            try:
                requests.post(PIPELINE_STATUS_URL, json={
                    'status': 'error',
                    'error_type': 'oom',
                    'message': str(e),
                    'timestamp': datetime.now().isoformat()
                }, timeout=5)
                logger.info("已通知主进程错误状态")
            except Exception as callback_error:
                logger.warning(f"通知主进程错误状态失败: {callback_error}")

            send_callback({
                'task_id': task['task_id'],
                'status': 'failed',
                'mode': task.get('mode', 't2i'),
                'error': f'显存不足：{str(e)}',
                'error_type': 'oom_error',
                'retryable': False,
                'completed_at': datetime.now().isoformat()
            })

        except Exception as e:
            logger.error(f"❌ 生成失败：{e}")
            send_callback({
                'task_id': task['task_id'],
                'status': 'failed',
                'mode': task.get('mode', 't2i'),
                'error': str(e)[:500],
                'error_type': 'generation_error',
                'retryable': False,
                'completed_at': datetime.now().isoformat()
            })

    # 退出前清理
    _cleanup_pipeline()

if __name__ == '__main__':
    main()
