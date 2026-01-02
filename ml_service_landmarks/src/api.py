"""
ML Training API Server

Provides HTTP endpoints for triggering and monitoring training jobs.
This allows the web app to start training without needing Docker access.
"""
import os
import uuid
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from preprocess_data import init_database, ingest_raw_landmarks, ingest_normalized_landmarks

DB_PATH = Path(os.getenv("DATABASE_PATH"))
RAW_IMAGES_PATH = Path(os.getenv("RAW_IMAGES_PATH"))
LANDMARK_DETECTOR_PATH = Path(os.getenv("LANDMARK_DETECTOR_PATH"))

app = FastAPI(title="ML Training API", version="1.0.0")

# In-memory store for training jobs (in production, use Redis/DB)
training_jobs: Dict[str, Dict[str, Any]] = {}

preprocessing_jobs: Dict[str, Dict[str, Any]] = {}


# Pydantic models for request/response validation
class TrainingConfig(BaseModel):
    epochs: int = 10
    batch_size: int = 32
    version: Optional[str] = None


class TrainingJobResponse(BaseModel):
    job_id: str
    status: str
    message: str

class PreprocJobRequest(BaseModel): # for preprocessing
    dataset_version: str
    zip_filename: str

class HealthResponse(BaseModel):
    status: str
    service: str


def run_training(job_id: str, config: dict):
    """Run training directly and update job status."""
    import sys
    from io import StringIO
    
    try:
        training_jobs[job_id]['status'] = 'running'
        training_jobs[job_id]['started_at'] = datetime.now().isoformat()
        
        # Import and run training directly (avoids subprocess + file I/O)
        from train import train_model
        import argparse
        
        # Build args object matching train.py expectations
        class TrainingArgs:
            def __init__(self, cfg):
                self.epochs = cfg.get('epochs', 10)
                self.batch_size = cfg.get('batch_size', 32)
                self.version = cfg.get('version')
                self.no_set_active = False  # Set active by default
                self.model_output_path = '/models'
        
        args = TrainingArgs(config)
        
        # Capture stdout during training
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()
        
        try:
            # train_model now returns (test_accuracy, metrics_dict)
            result = train_model(args)
            if isinstance(result, tuple):
                test_accuracy, metrics = result
            else:
                # Fallback for backward compatibility
                test_accuracy = result
                metrics = None
            
            training_jobs[job_id]['stdout'] = captured_output.getvalue()
            training_jobs[job_id]['status'] = 'completed'
            training_jobs[job_id]['return_code'] = 0
            
            if metrics:
                training_jobs[job_id]['metrics'] = metrics
            
            version = config.get('version') or ''
            training_jobs[job_id]['model_file'] = f"gesture_model_{version}.keras" if version else (metrics.get('model_file') if metrics else None)
            
            # Trigger inference service reload if active_model.json was updated
            inference_url = os.getenv('INFERENCE_API_URL')
            if inference_url:
                print(f"Triggering inference reload at {inference_url}/reload")
                import requests
                try:
                    r = requests.post(f"{inference_url}/reload", timeout=5)
                    r.raise_for_status()
                    training_jobs[job_id]['reload_status'] = 'success'
                except Exception as re:
                    print(f"Failed to reload inference service: {re}")
                    training_jobs[job_id]['reload_status'] = f'failed: {re}'
                    
        except Exception as train_error:
            training_jobs[job_id]['stdout'] = captured_output.getvalue()
            training_jobs[job_id]['status'] = 'failed'
            training_jobs[job_id]['error'] = str(train_error)
            training_jobs[job_id]['return_code'] = 1
        finally:
            sys.stdout = old_stdout
            
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

@app.post("/preprocess", response_model=TrainingJobResponse)
async def run_preprocessing(request: PreprocJobRequest):
    job_id = f"preprocess_{uuid.uuid4().hex[:8]}"

    preprocessing_jobs[job_id] = {
        "status": "pending",
        "message": "",
    }

    def task():
        import shutil
        import zipfile
        
        try:
            preprocessing_jobs[job_id]["status"] = "running"
            
            # Paths
            # RAW_IMAGES_PATH is /images (mapped volume)
            zip_path = RAW_IMAGES_PATH / request.zip_filename
            
            if not zip_path.exists():
                raise FileNotFoundError(f"ZIP file not found: {zip_path}")

            # Temp extraction path (local to container, fast I/O)
            temp_extract_path = Path("/tmp") / f"extract_{job_id}"
            temp_extract_path.mkdir(parents=True, exist_ok=True)
            
            try:
                print(f"Extracting {zip_path} to {temp_extract_path}")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_extract_path)
                
                # Verify structure (handle optional root folder)
                # If everything is in one subfolder, perform processing on that subfolder
                content = list(temp_extract_path.iterdir())
                target_root = temp_extract_path
                if len(content) == 1 and content[0].is_dir():
                    target_root = content[0]

                init_database(DB_PATH)
                raw_stats = ingest_raw_landmarks(DB_PATH, LANDMARK_DETECTOR_PATH, target_root, request.dataset_version)
                normalized_stats = ingest_normalized_landmarks(DB_PATH, request.dataset_version)
                
                preprocessing_jobs[job_id]["status"] = "completed"
                preprocessing_jobs[job_id]["message"] = f"Raw: {raw_stats}, Normalized: {normalized_stats}"
                
            finally:
                # Cleanup temp files
                if temp_extract_path.exists():
                    shutil.rmtree(temp_extract_path)
                    
                # Cleanup source ZIP (landmarks are now in DB, raw images no longer needed)
                if zip_path.exists():
                    try:
                        zip_path.unlink()
                        print(f"Deleted processed ZIP: {zip_path}")
                    except Exception as del_err:
                        print(f"Warning: Could not delete ZIP {zip_path}: {del_err}")

        except Exception as e:
            print(f"Preprocessing error: {e}")
            preprocessing_jobs[job_id]["status"] = "failed"
            preprocessing_jobs[job_id]["message"] = str(e)

    threading.Thread(target=task, daemon=True).start()

    return TrainingJobResponse(job_id=job_id, status="pending", message="Preprocessing started")

@app.get("/preprocess/{job_id}")
async def get_preprocessing_status(job_id: str):
    job = preprocessing_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Preprocessing job not found")
    return job



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
    active_model_file = model_path / 'active_model.json'
    active_model = None
    if active_model_file.exists():
        import json
        try:
            with open(active_model_file, 'r') as f:
                active_data = json.load(f)
            active_model = active_data.get('model_file')
        except Exception:
            pass
    
    return {
        'models': models,
        'active_model': active_model
    }


if __name__ == '__main__':
    import uvicorn
    port = int(os.environ.get('PORT', 8003))
    uvicorn.run(app, host='0.0.0.0', port=port)
