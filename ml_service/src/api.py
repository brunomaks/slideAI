"""
ML Training API Server

Provides HTTP endpoints for triggering and monitoring training jobs.
This allows the web app to start training without needing Docker access.
"""
import os
import uuid
import threading
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="ML Training API", version="1.0.0")

# In-memory store for training jobs (in production, use Redis/DB)
training_jobs: Dict[str, Dict[str, Any]] = {}


# Pydantic models for request/response validation
class TrainingConfig(BaseModel):
    epochs: int = 10
    batch_size: int = 32
    image_size: int = 128
    version: Optional[str] = None


class TrainingJobResponse(BaseModel):
    job_id: str
    status: str
    message: str


class HealthResponse(BaseModel):
    status: str
    service: str


def run_training(job_id: str, config: dict):
    """Run training in a subprocess and update job status."""
    try:
        training_jobs[job_id]['status'] = 'running'
        training_jobs[job_id]['started_at'] = datetime.now().isoformat()
        
        # Build command
        cmd = [
            'python', '-u', 'src/train.py',  # -u for unbuffered output
            '--epochs', str(config.get('epochs', 10)),
            '--batch-size', str(config.get('batch_size', 32)),
            '--img-size', str(config.get('image_size', 128)),
            '--set-active',
        ]
        
        if config.get('version'):
            cmd.extend(['--version', config['version']])
        
        # Run training with streaming output capture
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd='/workspace',
            bufsize=1  # Line buffered
        )
        
        # Stream output to job record
        stdout_lines = []
        for line in iter(process.stdout.readline, ''):
            stdout_lines.append(line)
            training_jobs[job_id]['stdout'] = ''.join(stdout_lines)
        
        process.wait()
        
        training_jobs[job_id]['return_code'] = process.returncode
        
        if process.returncode == 0:
            training_jobs[job_id]['status'] = 'completed'
            # Attach metrics if available
            try:
                version = config.get('version') or ''
                metrics_path = Path('/models') / f"gesture_model_{version}.metrics.json"
                if metrics_path.exists():
                    import json
                    with open(metrics_path, 'r') as f:
                        training_jobs[job_id]['metrics'] = json.load(f)
                # Always attach model file name
                training_jobs[job_id]['model_file'] = f"gesture_model_{version}.keras" if version else training_jobs[job_id]['metrics'].get('model_file') if training_jobs[job_id].get('metrics') else None
            except Exception:
                pass
        else:
            training_jobs[job_id]['status'] = 'failed'
            training_jobs[job_id]['error'] = 'Training failed with exit code ' + str(process.returncode)
            
    except Exception as e:
        training_jobs[job_id]['status'] = 'failed'
        training_jobs[job_id]['error'] = str(e)
    finally:
        training_jobs[job_id]['completed_at'] = datetime.now().isoformat()


@app.get('/health', response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(status='healthy', service='ml-training')


@app.post('/train', response_model=TrainingJobResponse, status_code=202)
async def start_training(config: TrainingConfig):
    """
    Start a new training job.
    
    Accepts training configuration and returns job ID for status tracking.
    """
    # Generate job ID
    job_id = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    
    # Initialize job record
    training_jobs[job_id] = {
        'id': job_id,
        'status': 'pending',
        'config': config.model_dump(),
        'created_at': datetime.now().isoformat(),
        'started_at': None,
        'completed_at': None,
        'stdout': '',
        'stderr': '',
        'error': None,
    }
    
    # Start training in background thread
    thread = threading.Thread(target=run_training, args=(job_id, config.model_dump()))
    thread.daemon = True
    thread.start()
    
    return TrainingJobResponse(
        job_id=job_id,
        status='pending',
        message='Training job started'
    )


@app.get('/train/{job_id}')
async def get_training_status(job_id: str):
    """Get status of a training job."""
    job = training_jobs.get(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail='Job not found')
    
    return job


@app.get('/train/{job_id}/logs')
async def get_training_logs(job_id: str):
    """Get logs for a training job."""
    job = training_jobs.get(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail='Job not found')
    
    return {
        'job_id': job_id,
        'stdout': job.get('stdout', ''),
        'stderr': job.get('stderr', ''),
    }


@app.get('/train')
async def list_training_jobs():
    """List all training jobs."""
    return {
        'jobs': list(training_jobs.values())
    }


@app.get('/models')
async def list_models():
    """List available models."""
    model_path = Path('/models')
    models = []
    
    if model_path.exists():
        for model_file in model_path.glob('*.keras'):
            models.append({
                'name': model_file.name,
                'path': str(model_file),
                'size_mb': round(model_file.stat().st_size / (1024 * 1024), 2),
                'modified': datetime.fromtimestamp(model_file.stat().st_mtime).isoformat()
            })
    
    # Check active model
    active_model_file = model_path / 'active_model.txt'
    active_model = None
    if active_model_file.exists():
        active_model = active_model_file.read_text().strip()
    
    return {
        'models': models,
        'active_model': active_model
    }


if __name__ == '__main__':
    import uvicorn
    port = int(os.environ.get('PORT', 8003))
    uvicorn.run(app, host='0.0.0.0', port=port)
