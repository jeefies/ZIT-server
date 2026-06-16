#!/usr/bin/env python3
"""
Z-Image-Turbo Image Service (主程序)
Flask + TaskProcessor + subprocess pipeline

架构设计：
- Flask API：单线程处理请求
- TaskProcessor：管理任务队列和持久化
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

DB_PATH = '/home/jeefy/AV/ZIT-service/data/image-gen-history.json'
IMAGE_OUTPUT_DIR = '/home/jeefy/AV/ZIT-service/data/images/'
I2I_INPUT_DIR = '/tmp/z-image-inputs/'
PIPELINE_SCRIPT = os.path.join(os.path.dirname(__file__), 'image_service_pipeline.py')
PYTHON_BIN = '/home/jeefy/openclaw-home/miniconda3/envs/image/bin/python3'
PIPE_LOG_FILE = '/home/jeefy/AV/ZIT-service/logs/image_pipeline.log'

# 确保目录存在
os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)
os.makedirs(I2I_INPUT_DIR, exist_ok=True)

# ==================== 辅助函数 ====================

def save_i2i_images(task_id: str, image_base64: str, mask_base64: str | None) -> tuple:
    """保存 i2i 输入图片和掩码，返回路径"""
    import base64
    import re
    
    image_path = os.path.join(I2I_INPUT_DIR, f"{task_id}_input.png")
    mask_path = os.path.join(I2I_INPUT_DIR, f"{task_id}_mask.png")
    
    # 解码 base64（处理 data:image/png;base64,xxx 格式）
    def decode_base64(data_url):
        if data_url and data_url.startswith('data:'):
            # 提取 base64 部分
            match = re.match(r'data:image/\w+;base64,(.+)', data_url)
            if match:
                return base64.b64decode(match.group(1))
        elif data_url:
            # 纯 base64
            return base64.b64decode(data_url)
        return None
    
    # 保存原图
    image_data = decode_base64(image_base64)
    if image_data:
        with open(image_path, 'wb') as f:
            f.write(image_data)
    else:
        return None, None  # 无效图片
    
    # 保存掩码（如果有）
    if mask_base64:
        mask_data = decode_base64(mask_base64)
        if mask_data:
            with open(mask_path, 'wb') as f:
                f.write(mask_data)
            return image_path, mask_path

    # 没有提供 mask，自动生成全图白色 mask
    try:
        from PIL import Image
        img = Image.open(image_path)
        w, h = img.size
        # 创建白色 mask (255 = inpaint区域, 0 = 保留区域) → 全图重绘
        mask_img = Image.new('L', (w, h), 255)
        mask_img.save(mask_path)
        print(f"🎨 自动生成全图重绘 mask: {mask_path} ({w}x{h})", file=sys.stderr)
        return image_path, mask_path
    except Exception as e:
        print(f"⚠️ 自动生成 mask 失败: {e}，使用无 mask 模式", file=sys.stderr)
        return image_path, None

# ==================== PipelineManager ====================

class PipelineManager:
    """管理 pipeline 子进程，支持 t2i/i2i 双模式切换"""
    
    def __init__(self):
        self.process = None
        self.pipeline_loaded = False
        self.last_activity = datetime.now()
        self.current_type = None
        self.busy = False
        
    def get_current_type(self):
        """获取当前 pipeline 类型"""
        return self.current_type
    
    def push_task(self, task_data):
        """推送任务到 pipeline 进程"""
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
        """切换到指定模式的 pipeline
        
        Args:
            mode: 't2i' 或 'i2i'
            
        Returns:
            bool: 切换是否成功
        """
        if self.current_type == mode and self.pipeline_loaded:
            return True
            
        print(f"🔄 切换 pipeline 模式: {self.current_type} -> {mode}", file=sys.stderr)
        self.pipeline_loaded = False
        
        # 关闭当前进程
        if self.process:
            try:
                self.process.stdin.close()
                self.process.wait(timeout=10)
            except:
                self.process.kill()
            self.process = None
        
        self.current_type = None
        
        # 启动新模式的进程
        success = self._start_pipeline_process(mode)
        return success
        
    def _ensure_pipeline_alive(self, mode='t2i'):
        """确保 pipeline 进程存活，模式不匹配时重启"""
        if self.process is not None and self.process.poll() is None and self.current_type == mode:
            return  # 进程存活且模式匹配
        
        # 终止旧进程
        if self.process:
            try:
                self.process.stdin.close()
                self.process.wait(timeout=5)
            except Exception:
                self.process.kill()
            self.process = None
        
        # 启动新进程
        self._start_pipeline_process(mode)
    
    def _start_pipeline_process(self, mode='t2i'):
        """启动新的 pipeline 进程
        
        Args:
            mode: 't2i' 或 'i2i'
            
        Returns:
            bool: 启动是否成功
        """
        self.current_type = mode
        self.pipeline_loaded = False
        try:
            self.process = subprocess.Popen(
                [PYTHON_BIN, PIPELINE_SCRIPT, "--mode", mode],
                stdin=subprocess.PIPE,
                stdout=open(PIPE_LOG_FILE, "a+"),
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            print(f"✅ Pipeline 启动 (PID: {self.process.pid}, mode: {mode})", file=sys.stderr)
            return True
        except Exception as e:
            print(f"❌ Pipeline 启动失败：{e}", file=sys.stderr)
            self.process = None
            self.current_type = None
            return False

    def free_pipeline(self):
        """手动释放 pipeline，关闭子进程并重置状态"""
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
        """启动 pipeline 进程（快速返回，不等待加载）"""
        return self.switch_to(mode)


# ==================== TaskProcessor ====================

class TaskProcessor:
    def __init__(self):
        self.db = TinyDB(DB_PATH)
        self.pipeline_manager = PipelineManager()  # 管理 pipeline 子进程
        self.t2i_queue = []  # t2i 任务队列
        self.i2i_queue = []  # i2i 任务队列

    def add_task(self, task_data):
        """添加新任务到队列"""
        mode = task_data.get('mode', 't2i')

        # 如果 seed == -1，生成随机种子
        if task_data.get('seed', -1) == -1:
            task_data['seed'] = random.randint(0, 2**31 - 1)
            print(f"🎲 随机种子：{task_data['seed']}", file=sys.stderr)

        # 为 i2i 任务计算 hash（用于缓存检查）
        if mode == 'i2i':
            task_data['image_hash'] = hashlib.md5(task_data.get('image_base64', '')[:1000].encode()).hexdigest()[:16] if task_data.get('image_base64') else None
            task_data['mask_hash'] = hashlib.md5(task_data.get('mask_base64', '')[:1000].encode()).hexdigest()[:16] if task_data.get('mask_base64') else 'full'

        # 检查缓存
        Task = Query()

        if mode == 't2i':
            # t2i 缓存检查
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
            # i2i 缓存检查
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
            
            # 将新 task_id 也写入数据库，避免 poller 查询 404
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
            print(f"📝 缓存命中任务已写入数据库：{task_data['task_id']}", file=sys.stderr)
            
            return {
                'task_id': task_data['task_id'],
                'status': 'completed',
                'mode': mode,
                'message': f'缓存命中：复用已完成任务 {cached["task_id"]}',
                'image_path': cached_image_path
            }

        # 处理 i2i 图片
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

            # 存储路径到任务数据（input_image_path 用于 i2i 输入，image_path 用于输出）
            task_data['input_image_path'] = image_path
            task_data['mask_path'] = mask_path

            # 移除 base64 数据（不存入数据库）
            task_data.pop('image_base64', None)
            task_data.pop('mask_base64', None)

        # 缓存未命中，加入队列
        task_data['status'] = 'queued'
        task_data['created_at'] = datetime.now().isoformat()
        if 'image_path' not in task_data:
            task_data['image_path'] = None  # 输出图片路径（完成后填充）
        task_data['error'] = None
        task_data['error_type'] = None

        self.db.insert(task_data)
        print(f"📝 任务已加入队列：{task_data['task_id']}", file=sys.stderr)

        # 加入对应类型的内存队列
        queue = self.t2i_queue if mode == 't2i' else self.i2i_queue
        queue_position = len(queue)
        queue.append(task_data)
        print(f"📥 任务加入 {mode} 队列，位置：{queue_position}", file=sys.stderr)

        # 触发智能调度
        self._schedule_next_task()

        return {
            'task_id': task_data['task_id'],
            'status': 'queued',
            'mode': mode,
            'queue_position': queue_position,
            'message': '任务已加入队列'
        }

    def _schedule_next_task(self):
        """调度下一个任务"""
        pm = self.pipeline_manager
        process_alive = pm.process is not None and pm.process.poll() is None
        
        # 如果进程已死，重置状态以便重新启动
        if not process_alive and pm.current_type is not None:
            pm.pipeline_loaded = False
            pm.current_type = None
            pm.busy = False
        
        # 如果正在切换/加载中，等待
        if not pm.pipeline_loaded and pm.current_type is not None:
            return
        
        # 如果 pipeline 正在处理任务，不推送新任务，留在队列中等待
        if pm.busy:
            return
        
        current_type = pm.get_current_type()
        
        # 优先处理当前类型的队列
        if current_type == 't2i' and self.t2i_queue:
            task = self.t2i_queue.pop(0)
            pm.push_task(task)
        elif current_type == 'i2i' and self.i2i_queue:
            task = self.i2i_queue.pop(0)
            pm.push_task(task)
        # pipeline 空闲，可切换模式
        elif current_type == 't2i' and not self.t2i_queue and self.i2i_queue:
            if pm.switch_to('i2i'):
                task = self.i2i_queue.pop(0)
                pm.push_task(task)
        elif current_type == 'i2i' and not self.i2i_queue and self.t2i_queue:
            if pm.switch_to('t2i'):
                task = self.t2i_queue.pop(0)
                pm.push_task(task)
        # 两个队列都空
        elif not current_type and (self.t2i_queue or self.i2i_queue):
            target = 't2i' if self.t2i_queue else 'i2i'
            if pm.switch_to(target):
                queue = self.t2i_queue if target == 't2i' else self.i2i_queue
                task = queue.pop(0)
                pm.push_task(task)

    def update_task_status(self, result):
        """更新任务状态（被 /task_complete 调用）"""
        Task = Query()
        task = self.db.get(Task.task_id == result['task_id'])

        if task:
            match result.get('status'):
                case 'success':
                    task['status'] = 'completed'
                    task['image_path'] = result['image_path']
                    task['completed_at'] = result.get('completed_at', datetime.now().isoformat())
                    print(f"✅ 任务完成：{result['task_id']}", file=sys.stderr)
                case 'failed':
                    task['status'] = 'failed'
                    task['error'] = result['error']
                    task['error_type'] = result.get('error_type', 'unknown')
                    print(f"❌ 任务失败：{result['task_id']} - {result['error'][:50]}", file=sys.stderr)
                case 'processing':
                    task['status'] = 'processing'
                case _:
                    print(f"⚠️ 未知状态：{result.get('status')}", file=sys.stderr)
            print("Updating: ", task, file=sys.stderr)
            self.db.update(task, Task.task_id == task['task_id'])

            # 任务完成后调度下一个
            if result.get('status') in ['success', 'failed']:
                self.pipeline_manager.busy = False
                self._schedule_next_task()
        else:
            print(f"⚠️ 未找到任务：{result['task_id']}", file=sys.stderr)

    def get_task(self, task_id):
        """获取任务信息"""
        Task = Query()
        return self.db.get(Task.task_id == task_id)

    def get_queue_status(self):
        """获取队列状态"""
        Task = Query()
        processing = self.db.search(Task.status == 'processing')
        
        return {
            'current_task': processing[0]['task_id'] if processing else None,
            'current_mode': processing[0].get('mode', 't2i') if processing else None,
            'pipeline_type': self.pipeline_manager.get_current_type(),
            'pipeline_switching': not self.pipeline_manager.pipeline_loaded,
            't2i_queue_length': len(self.t2i_queue),
            'i2i_queue_length': len(self.i2i_queue),
            't2i_queue': [t['task_id'] for t in self.t2i_queue],
            'i2i_queue': [t['task_id'] for t in self.i2i_queue],
        }

    def restart_interrupted_tasks(self):
        """重启被中断的任务（服务启动时调用）"""
        Task = Query()
        interrupted = self.db.search(Task.status == 'processing') + self.db.search(Task.status == 'queued')

        for task in interrupted:
            task['status'] = 'queued'
            self.db.update(task, Task.task_id == task['task_id'])
            print(f"🔄 重启中断任务：{task['task_id']}", file=sys.stderr)

            # 加入对应类型的内存队列（避免重复）
            mode = task.get('mode', 't2i')
            queue = self.t2i_queue if mode == 't2i' else self.i2i_queue
            if not any(t['task_id'] == task['task_id'] for t in queue):
                queue.append(task)
                print(f"📥 中断任务加入 {mode} 队列", file=sys.stderr)

        # 触发调度
        self._schedule_next_task()

# ==================== Flask API ====================

app = Flask(__name__)
task_processor = TaskProcessor()

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    task_id = f"gen_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

    task_data = {
        'task_id': task_id,
        'mode': data.get('mode', 't2i'),
        'prompt': data.get('prompt', ''),
        'negative_prompt': data.get('negative_prompt', ''),
        'width': data.get('width', 1024),
        'height': data.get('height', 1024),
        'steps': data.get('steps', 9),
        'guidance': data.get('guidance', 0.0),
        'seed': data.get('seed', -1),
        'image_base64': data.get('image_base64'),
        'mask_base64': data.get('mask_base64'),
    }

    result = task_processor.add_task(task_data)
    return jsonify(result)

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
    return jsonify({
        'status': 'healthy',
        'service': 'z-image-turbo-queue',
        'has_pending_tasks': (status['t2i_queue_length'] + status['i2i_queue_length']) > 0,
        'process_alive': task_processor.pipeline_manager.process is not None and task_processor.pipeline_manager.process.poll() is None
    })

@app.route('/task_complete', methods=['POST'])
def task_complete():
    result = request.json
    task_processor.update_task_status(result)
    return jsonify({'success': True})

@app.route('/history', methods=['GET'])
def get_history():
    """
    返回所有任务的状态列表
    
    返回格式：
    [
      {
        "task_id": "gen_xxx",
        "status": "completed|failed|processing|queued"
      }
    ]
    """
    # 从 TinyDB 读取所有任务
    tasks = task_processor.db.all()
    
    # 转换为统一格式（只返回 task_id 和 status）
    result = []
    for task in tasks:
        result.append({
            'task_id': task['task_id'],
            'status': 'completed' if task['status'] == 'completed' else task.get('status', 'queued')
        })
    
    # 按创建时间倒序排序
    result.sort(key=lambda x: x.get('task_id', ''), reverse=True)
    
    # 限制返回数量
    limit = int(request.args.get('limit', 0))
    return jsonify(result[:limit] if limit > 0 else result)

@app.route('/status/<task_id>/image', methods=['GET'])
def get_task_image(task_id):
    """
    下载任务生成的图片
    
    流程：
    1. 检查任务是否存在
    2. 检查任务是否完成
    3. 检查图片文件是否存在
    4. 返回图片文件
    """
    # 查询任务
    task = task_processor.get_task(task_id)
    if not task:
        return jsonify({'error': 'Task not found'}), 404
    
    # 检查状态
    if task['status'] != 'completed':
        return jsonify({'error': 'Image not ready'}), 400
    
    # 检查图片文件
    image_path = os.path.join(IMAGE_OUTPUT_DIR, f"{task_id}.png")
    if not os.path.exists(image_path):
        return jsonify({'error': 'Image file not found'}), 404
    
    # 返回图片
    return send_file(image_path, mimetype='image/png')

@app.route('/pipeline_status', methods=['POST'])
def update_pipeline_status():
    """Pipeline 子进程回调：更新 pipeline 状态"""
    data = request.json
    status = data.get('status', 'unknown')

    if status == 'loaded':
        task_processor.pipeline_manager.pipeline_loaded = True
        task_processor.pipeline_manager.last_activity = datetime.now()
        print(f"🔄 Pipeline 已加载 (时间: {data.get('timestamp', 'N/A')})", file=sys.stderr)
        # Pipeline 加载完成后，触发任务调度
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
    """手动释放 pipeline 进程和 GPU 显存"""
    pm = task_processor.pipeline_manager
    if not pm.process and not pm.pipeline_loaded:
        return jsonify({'status': 'already_free', 'message': 'Pipeline 未运行'})
    pm.free_pipeline()
    return jsonify({'status': 'freed', 'message': 'Pipeline 已释放'})

@app.route('/pipeline_status', methods=['GET'])
def get_pipeline_status():
    """获取 pipeline 状态"""
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

# ==================== 主程序 ====================

if __name__ == '__main__':
    print("="*60, file=sys.stderr)
    print("🚀 Z-Image-Turbo 文生图服务启动", file=sys.stderr)
    print("="*60, file=sys.stderr)
    print(f"📁 数据库：{DB_PATH}", file=sys.stderr)
    print(f"📁 图片输出：{IMAGE_OUTPUT_DIR}", file=sys.stderr)
    print(f"📜 Pipeline 脚本：{PIPELINE_SCRIPT}", file=sys.stderr)
    print("="*60, file=sys.stderr)

    # 重启被中断的任务
    task_processor.restart_interrupted_tasks()

    # 启动 Flask（单线程）
    try:
        app.run(host='0.0.0.0', port=8765, threaded=False)
    finally:
        # 关闭 pipeline 进程
        if task_processor.pipeline_manager.process:
            task_processor.pipeline_manager.process.stdin.close()
            task_processor.pipeline_manager.process.wait(timeout=5)
