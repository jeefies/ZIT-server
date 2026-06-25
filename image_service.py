#!/usr/bin/env python3
"""
Z-Image-Turbo Image Service (?
Flask + TaskProcessor + subprocess pipeline

?- Flask API
- TaskProcessor?- subprocess pipeline  stdin
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

# ====================  ====================

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

SERVICE_BASE_DIR = os.getenv('SERVICE_BASE_DIR', '/home/jeefy/AV/ZIT-service')
DB_PATH = os.getenv('DB_PATH', os.path.join(SERVICE_BASE_DIR, 'data', 'image-gen-history.json'))
IMAGE_OUTPUT_DIR = os.getenv('IMAGE_OUTPUT_DIR', os.path.join(SERVICE_BASE_DIR, 'data', 'images'))
I2I_INPUT_DIR = os.getenv('I2I_INPUT_DIR', '/tmp/z-image-inputs/')
PIPELINE_SCRIPT = os.getenv(
    'PIPELINE_SCRIPT',
    os.path.join(os.path.dirname(__file__), 'image_service_pipeline.py')
)
PYTHON_BIN = os.getenv('PYTHON_BIN', '/home/jeefy/openclaw-home/miniconda3/envs/image/bin/python3')
PIPE_LOG_FILE = os.getenv('PIPE_LOG_FILE', os.path.join(SERVICE_BASE_DIR, 'logs', 'image_pipeline.log'))
SERVICE_PORT = int(os.getenv('PORT', '8765'))

DEFAULT_MODEL_FAMILY = os.getenv('MODEL_FAMILY', 'zit').lower()
ZIT_MODEL_ID = os.getenv('ZIT_MODEL_ID', 'z-image-turbo')
PONY_SERVICE_BASE_DIR = os.getenv('PONY_SERVICE_BASE_DIR', '/home/jeefy/AV/ZIT-service-pony')
PONY_PIPELINE_SCRIPT = os.getenv(
    'PONY_PIPELINE_SCRIPT',
    os.path.join(os.path.dirname(__file__), 'image_service_pony_pipeline.py')
)
PONY_LOG_FILE = os.getenv('PONY_PIPE_LOG_FILE', os.path.join(PONY_SERVICE_BASE_DIR, 'logs', 'pony_pipeline.log'))
PONY_MODEL_ID = os.getenv('PONY_MODEL_ID', 'AstraliteHeart/pony-diffusion-v6')

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
    # Pony v6 produced near-blank textures with clip_skip=2 in validation.
    'clip_skip': optional_int_env('PONY_DEFAULT_CLIP_SKIP'),
}

SUPPORTED_MODEL_FAMILIES = {'zit', 'pony'}

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
    return PONY_PIPELINE_SCRIPT if family == 'pony' else PIPELINE_SCRIPT

def log_file_for_family(family):
    return PONY_LOG_FILE if family == 'pony' else PIPE_LOG_FILE

def pipeline_key_for_task(task_data):
    family = normalize_model_family(task_data.get('model_family'))
    return f"{family}:{task_data.get('mode', 't2i')}"

#
os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)
os.makedirs(I2I_INPUT_DIR, exist_ok=True)

# ====================  ====================

def save_i2i_images(task_id: str, image_base64: str, mask_base64: str | None) -> tuple:
    """Save i2i input image and mask, returning their filesystem paths."""
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
    if not image_data:
        return None, None

    with open(image_path, 'wb') as f:
        f.write(image_data)

    if mask_base64:
        mask_data = decode_base64(mask_base64)
        if mask_data:
            with open(mask_path, 'wb') as f:
                f.write(mask_data)
            return image_path, mask_path

    try:
        from PIL import Image
        img = Image.open(image_path)
        mask_img = Image.new('L', img.size, 255)
        mask_img.save(mask_path)
        print(f"Auto-generated full-image i2i mask: {mask_path} ({img.size[0]}x{img.size[1]})", file=sys.stderr)
        return image_path, mask_path
    except Exception as e:
        print(f"Failed to auto-generate i2i mask: {e}; continuing without mask", file=sys.stderr)
        return image_path, None
# ==================== PipelineManager ====================

class PipelineManager:
    """Manage the active pipeline subprocess."""

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
        family = normalize_model_family(task_data.get('model_family'))
        self._ensure_pipeline_alive(task_data)
        print(f"Push task: {task_data['task_id']} (family={family}, mode={mode})", file=sys.stderr)
        if self.process and self.process.stdin:
            try:
                self.process.stdin.write(json.dumps(task_data) + '\n')
                self.process.stdin.flush()
                self.busy = True
                print(f"Task sent: {task_data['task_id']}", file=sys.stderr)
            except Exception as e:
                print(f"Failed to send task: {e}; restarting pipeline", file=sys.stderr)
                self.process = None
                self._ensure_pipeline_alive(task_data)
        else:
            print("Pipeline process or stdin is unavailable", file=sys.stderr)

    def switch_to(self, task_data):
        mode = task_data.get('mode', 't2i')
        pipeline_key = pipeline_key_for_task(task_data)
        if self.current_type == pipeline_key and self.pipeline_loaded:
            return True

        print(f"Switch pipeline: {self.current_type} -> {pipeline_key}", file=sys.stderr)
        self.pipeline_loaded = False

        if self.process:
            try:
                self.process.stdin.close()
                self.process.wait(timeout=10)
            except Exception:
                self.process.kill()
            self.process = None

        self.current_type = None
        return self._start_pipeline_process(task_data)

    def _ensure_pipeline_alive(self, task_data):
        pipeline_key = pipeline_key_for_task(task_data)
        if self.process is not None and self.process.poll() is None and self.current_type == pipeline_key:
            return

        if self.process:
            try:
                self.process.stdin.close()
                self.process.wait(timeout=5)
            except Exception:
                self.process.kill()
            self.process = None

        self._start_pipeline_process(task_data)

    def _start_pipeline_process(self, task_data):
        mode = task_data.get('mode', 't2i')
        family = normalize_model_family(task_data.get('model_family'))
        pipeline_key = pipeline_key_for_task(task_data)
        script = pipeline_script_for_family(family)
        log_file = log_file_for_family(family)
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        self.current_type = pipeline_key
        self.pipeline_loaded = False
        try:
            self.process = subprocess.Popen(
                [PYTHON_BIN, script, "--mode", mode],
                stdin=subprocess.PIPE,
                stdout=open(log_file, "a+"),
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            print(f"Pipeline started (PID: {self.process.pid}, family={family}, mode={mode})", file=sys.stderr)
            return True
        except Exception as e:
            print(f"Pipeline start failed: {e}", file=sys.stderr)
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
        print("Pipeline manually freed", file=sys.stderr)

    def start_pipeline(self, mode='t2i'):
        return self.switch_to({'mode': mode, 'model_family': DEFAULT_MODEL_FAMILY})

# ==================== TaskProcessor ====================

class TaskProcessor:
    def __init__(self):
        self.db = TinyDB(DB_PATH)
        self.pipeline_manager = PipelineManager()
        self.t2i_queue = []
        self.i2i_queue = []

    def add_task(self, task_data):
        mode = task_data.get('mode', 't2i')
        task_data['model_family'] = normalize_model_family(task_data.get('model_family'))

        if task_data.get('seed', -1) == -1:
            task_data['seed'] = random.randint(0, 2**31 - 1)
            print(f"Random seed: {task_data['seed']}", file=sys.stderr)

        if mode == 'i2i':
            task_data['image_hash'] = hashlib.md5(task_data.get('image_base64', '')[:1000].encode()).hexdigest()[:16] if task_data.get('image_base64') else None
            task_data['mask_hash'] = hashlib.md5(task_data.get('mask_base64', '')[:1000].encode()).hexdigest()[:16] if task_data.get('mask_base64') else 'full'

        Task = Query()
        common_cache = (
            (Task.mode == mode) &
            (Task.prompt == task_data['prompt']) &
            (Task.negative_prompt == task_data.get('negative_prompt', '')) &
            (Task.width == task_data['width']) &
            (Task.height == task_data['height']) &
            (Task.steps == task_data['steps']) &
            (Task.guidance == task_data['guidance']) &
            (Task.clip_skip == task_data.get('clip_skip')) &
            (Task.model_family == task_data.get('model_family')) &
            (Task.model_id == task_data.get('model_id')) &
            (Task.seed == task_data['seed']) &
            (Task.status == 'completed')
        )
        if mode == 'i2i':
            common_cache = common_cache & (Task.strength == task_data.get('strength')) & (Task.image_hash == task_data['image_hash']) & (Task.mask_hash == task_data['mask_hash'])

        cached = self.db.get(common_cache)
        if cached:
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
                'strength': task_data.get('strength'),
                'clip_skip': task_data.get('clip_skip'),
                'model_family': task_data.get('model_family'),
                'model_id': task_data.get('model_id'),
                'seed': task_data['seed'],
                'status': 'completed',
                'image_path': cached_image_path,
                'created_at': datetime.now().isoformat(),
                'completed_at': datetime.now().isoformat(),
                'error': None,
                'error_type': None,
            }
            if mode == 'i2i':
                db_record['image_hash'] = task_data.get('image_hash')
                db_record['mask_hash'] = task_data.get('mask_hash')
            self.db.insert(db_record)
            return {
                'task_id': task_data['task_id'],
                'status': 'completed',
                'mode': mode,
                'message': f'Cache hit: reused completed task {cached["task_id"]}',
                'image_path': cached_image_path,
            }

        if mode == 'i2i':
            image_base64 = task_data.get('image_base64')
            mask_base64 = task_data.get('mask_base64')
            if not image_base64:
                return {'task_id': task_data['task_id'], 'status': 'failed', 'mode': 'i2i', 'error': 'i2i requires image_base64'}
            image_path, mask_path = save_i2i_images(task_data['task_id'], image_base64, mask_base64)
            if not image_path:
                return {'task_id': task_data['task_id'], 'status': 'failed', 'mode': 'i2i', 'error': 'image decode failed'}
            task_data['input_image_path'] = image_path
            task_data['mask_path'] = mask_path
            task_data.pop('image_base64', None)
            task_data.pop('mask_base64', None)

        task_data['status'] = 'queued'
        task_data['created_at'] = datetime.now().isoformat()
        task_data.setdefault('image_path', None)
        task_data['error'] = None
        task_data['error_type'] = None
        self.db.insert(task_data)

        queue = self.t2i_queue if mode == 't2i' else self.i2i_queue
        queue_position = len(queue)
        queue.append(task_data)
        print(f"Task queued: {task_data['task_id']} family={task_data['model_family']} mode={mode} position={queue_position}", file=sys.stderr)
        self._schedule_next_task()

        return {
            'task_id': task_data['task_id'],
            'status': 'queued',
            'mode': mode,
            'queue_position': queue_position,
            'message': 'Task queued',
        }

    def _schedule_next_task(self):
        pm = self.pipeline_manager
        process_dead = pm.process is not None and pm.process.poll() is not None
        if process_dead:
            pm.pipeline_loaded = False
            pm.current_type = None
            pm.busy = False

        if not pm.pipeline_loaded and pm.current_type is not None:
            return
        if pm.busy:
            return

        queues = [self.t2i_queue, self.i2i_queue]
        selected_queue = None
        if pm.current_type:
            for queue in queues:
                if queue and pipeline_key_for_task(queue[0]) == pm.current_type:
                    selected_queue = queue
                    break
        if selected_queue is None:
            selected_queue = next((queue for queue in queues if queue), None)
        if not selected_queue:
            return

        task = selected_queue[0]
        if pm.current_type != pipeline_key_for_task(task):
            if not pm.switch_to(task):
                return
        selected_queue.pop(0)
        pm.push_task(task)

    def update_task_status(self, result):
        Task = Query()
        task = self.db.get(Task.task_id == result['task_id'])
        if not task:
            print(f"Task not found: {result['task_id']}", file=sys.stderr)
            return

        status = result.get('status')
        if status == 'success':
            task['status'] = 'completed'
            task['image_path'] = result['image_path']
            task['completed_at'] = result.get('completed_at', datetime.now().isoformat())
        elif status == 'failed':
            task['status'] = 'failed'
            task['error'] = result.get('error')
            task['error_type'] = result.get('error_type', 'unknown')
        elif status == 'processing':
            task['status'] = 'processing'
        else:
            print(f"Unknown task status: {status}", file=sys.stderr)

        self.db.update(task, Task.task_id == task['task_id'])
        if status in ['success', 'failed']:
            self.pipeline_manager.busy = False
            self._schedule_next_task()

    def get_task(self, task_id):
        Task = Query()
        return self.db.get(Task.task_id == task_id)

    def get_queue_status(self):
        Task = Query()
        processing = self.db.search(Task.status == 'processing')
        return {
            'current_task': processing[0]['task_id'] if processing else None,
            'current_mode': processing[0].get('mode', 't2i') if processing else None,
            'current_model_family': processing[0].get('model_family') if processing else None,
            't2i_queue_length': len(self.t2i_queue),
            'i2i_queue_length': len(self.i2i_queue),
            't2i_queue': [task['task_id'] for task in self.t2i_queue],
            'i2i_queue': [task['task_id'] for task in self.i2i_queue],
            'pipeline_type': self.pipeline_manager.current_type,
            'pipeline_switching': self.pipeline_manager.current_type is not None and not self.pipeline_manager.pipeline_loaded,
        }

    def restart_interrupted_tasks(self):
        Task = Query()
        interrupted = self.db.search(Task.status == 'processing') + self.db.search(Task.status == 'queued')
        for task in interrupted:
            task['status'] = 'queued'
            task['error'] = None
            task['error_type'] = None
            self.db.update(task, Task.task_id == task['task_id'])
            mode = task.get('mode', 't2i')
            queue = self.t2i_queue if mode == 't2i' else self.i2i_queue
            queue.append(task)
        if interrupted:
            print(f"Restarted {len(interrupted)} interrupted tasks", file=sys.stderr)
            self._schedule_next_task()
app = Flask(__name__)
task_processor = TaskProcessor()

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
        'default_model_family': DEFAULT_MODEL_FAMILY,
        'supported_model_families': sorted(SUPPORTED_MODEL_FAMILIES),
        'pony_model_id': PONY_MODEL_ID,
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
    ?
    ?    [
      {
        "task_id": "gen_xxx",
        "status": "completed|failed|processing|queued"
      }
    ]
    """
    # ?TinyDB ?    tasks = task_processor.db.all()

    #  task_id ?status?    result = []
    for task in tasks:
        result.append({
            'task_id': task['task_id'],
            'status': 'completed' if task['status'] == 'completed' else task.get('status', 'queued')
        })

    #
    result.sort(key=lambda x: x.get('task_id', ''), reverse=True)

    #
    limit = int(request.args.get('limit', 0))
    return jsonify(result[:limit] if limit > 0 else result)

@app.route('/status/<task_id>/image', methods=['GET'])
def get_task_image(task_id):
    """
    ?
    ?    1. ?    2. ?    3. ?    4.
    """
    #
    task = task_processor.get_task(task_id)
    if not task:
        return jsonify({'error': 'Task not found'}), 404

    # ?    if task['status'] != 'completed':
        return jsonify({'error': 'Image not ready'}), 400

    # ?    image_path = os.path.join(IMAGE_OUTPUT_DIR, f"{task_id}.png")
    if not os.path.exists(image_path):
        return jsonify({'error': 'Image file not found'}), 404

    #
    return send_file(image_path, mimetype='image/png')

@app.route('/pipeline_status', methods=['POST'])
def update_pipeline_status():
    data = request.json or {}
    status = data.get('status', 'unknown')

    if status == 'loaded':
        task_processor.pipeline_manager.pipeline_loaded = True
        task_processor.pipeline_manager.last_activity = datetime.now()
        task_processor._schedule_next_task()
    elif status == 'unloaded':
        task_processor.pipeline_manager.pipeline_loaded = False
    elif status == 'error':
        task_processor.pipeline_manager.pipeline_loaded = False
        print(f"Pipeline error: {data.get('error_type', 'unknown')} - {data.get('message', 'N/A')}", file=sys.stderr)

    return jsonify({'success': True})

@app.route('/pipeline_free', methods=['POST'])
def pipeline_free():
    pm = task_processor.pipeline_manager
    if not pm.process and not pm.pipeline_loaded:
        return jsonify({'status': 'already_free', 'message': 'Pipeline not running'})
    pm.free_pipeline()
    return jsonify({'status': 'freed', 'message': 'Pipeline freed'})

@app.route('/pipeline_status', methods=['GET'])
def get_pipeline_status():
    pm = task_processor.pipeline_manager
    return jsonify({
        'current_type': pm.current_type,
        'pipeline_loaded': pm.pipeline_loaded,
        'busy': pm.busy,
        'process_alive': pm.process is not None and pm.process.poll() is None,
        'last_activity': pm.last_activity.isoformat() if pm.last_activity else None,
    })
@app.route('/__restart')
def __restart_all_tasks():
    task_processor.restart_interrupted_tasks()
    return jsonify({"restart": "now"})

# ==================== ?====================

if __name__ == '__main__':
    print("=" * 60, file=sys.stderr)
    print("ZIT image generation service starting", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(f"DB: {DB_PATH}", file=sys.stderr)
    print(f"Image output: {IMAGE_OUTPUT_DIR}", file=sys.stderr)
    print(f"Default pipeline: {PIPELINE_SCRIPT}", file=sys.stderr)
    print(f"Pony pipeline: {PONY_PIPELINE_SCRIPT}", file=sys.stderr)
    print(f"Port: {SERVICE_PORT}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    task_processor.restart_interrupted_tasks()

    try:
        app.run(host='0.0.0.0', port=SERVICE_PORT, threaded=False)
    finally:
        if task_processor.pipeline_manager.process:
            task_processor.pipeline_manager.process.stdin.close()
            task_processor.pipeline_manager.process.wait(timeout=5)
