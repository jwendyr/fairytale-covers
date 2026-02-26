#!/usr/bin/env python3
"""
Fairytale Cover Generator — runs on vast.ai GPU instances.
Generates covers from batch.json, pushes results back to GitHub.
Supports multiple models for comparison (FLUX.1-dev, SD3.5, flux.1-lite).
"""

import json
import os
import sys
import time
import subprocess
import traceback
import gc
import signal
from pathlib import Path

# Safety timeout: auto-exit after 24 hours to prevent runaway costs
MAX_RUNTIME_SECONDS = 24 * 3600

def _timeout_handler(signum, frame):
    print(f"\n\nSAFETY TIMEOUT: {MAX_RUNTIME_SECONDS}s reached. Exiting to prevent runaway cost.")
    sys.exit(1)

signal.signal(signal.SIGALRM, _timeout_handler)
signal.alarm(MAX_RUNTIME_SECONDS)

REPO_URL = "git@github.com:jwendyr/fairytale-covers.git"
WORK_DIR = "/workspace/fairytale-covers"
BATCH_FILE = "batch.json"
OUTPUT_DIR = "output"
STATUS_FILE = "status.json"

# Default model if not specified in batch.json
DEFAULT_MODEL = "Freepik/flux.1-lite-8B-alpha"

# Model configs: pipeline class, default params
MODEL_CONFIGS = {
    "black-forest-labs/FLUX.1-dev": {
        "pipeline": "FluxPipeline",
        "steps": 28,
        "guidance": 3.5,
        "max_seq_len": 512,
    },
    "Freepik/flux.1-lite-8B-alpha": {
        "pipeline": "FluxPipeline",
        "steps": 24,
        "guidance": 3.5,
        "max_seq_len": 512,
    },
    "stabilityai/stable-diffusion-3.5-large": {
        "pipeline": "StableDiffusion3Pipeline",
        "steps": 28,
        "guidance": 4.5,
    },
}


def run_cmd(cmd, cwd=None, timeout=300):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                            cwd=cwd, timeout=timeout)
    if result.returncode != 0:
        print(f"CMD FAILED: {cmd}\nSTDERR: {result.stderr[:500]}", file=sys.stderr)
    return result


def git_push(cwd, message):
    """Add, commit, push — retry once on failure."""
    for attempt in range(2):
        r = run_cmd(f'git add -A && git commit -m "{message}" --allow-empty && git push',
                    cwd=cwd, timeout=120)
        if r.returncode == 0:
            return True
        if attempt == 0:
            time.sleep(5)
    return False


def update_status(status_data):
    status_path = os.path.join(WORK_DIR, STATUS_FILE)
    status_data["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(status_path, "w") as f:
        json.dump(status_data, f, indent=2)
    git_push(WORK_DIR, f'status: {status_data.get("phase", "update")}')


def setup_ssh():
    key_b64 = os.environ.get("GITHUB_SSH_KEY_B64", "")
    if not key_b64:
        print("ERROR: GITHUB_SSH_KEY_B64 not set", file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.expanduser("~/.ssh"), exist_ok=True)

    import base64
    key_data = base64.b64decode(key_b64).decode("utf-8")
    key_path = os.path.expanduser("~/.ssh/id_ed25519")
    with open(key_path, "w") as f:
        f.write(key_data)
    os.chmod(key_path, 0o600)

    # Embed GitHub host keys directly — ssh-keyscan FAILS on most vast.ai machines (port 22 blocked)
    known_hosts = os.path.expanduser("~/.ssh/known_hosts")
    with open(known_hosts, "a") as f:
        f.write("github.com ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOMqqnkVzrm0SdG6UOoqKLsabgH5C9okWi0dh2l9GKJl\n")
        f.write("github.com ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBEmKSENjQEezOmxkZMy7opKgwFB9nkt5YRrYMjNuG5N87uRgg6CLrbo5wAdT/y6v0mKV0U2w0WZ2YB/++Tpockg=\n")
    run_cmd('git config --global user.email "gpu-worker@fairytale.ucok.org"')
    run_cmd('git config --global user.name "Fairytale GPU Worker"')


def clone_repo():
    global WORK_DIR

    # If we're already inside the repo (launched by onstart), just use it
    batch_here = os.path.join(WORK_DIR, BATCH_FILE)
    if os.path.exists(batch_here):
        print(f"Repo already exists at {WORK_DIR} with {BATCH_FILE}, skipping clone.")
        return

    # Also check if we're running from inside the repo already (different path)
    try:
        cwd = os.getcwd()
    except OSError:
        cwd = "/"
    cwd_batch = os.path.join(cwd, BATCH_FILE)
    if cwd != "/" and os.path.exists(cwd_batch) and cwd != WORK_DIR:
        WORK_DIR = cwd
        print(f"Running from repo at {WORK_DIR}, skipping clone.")
        return

    # Fresh clone needed — cd out first to avoid deleting our cwd
    os.chdir("/tmp")
    if os.path.exists(WORK_DIR):
        run_cmd(f"rm -rf {WORK_DIR}")

    result = run_cmd(f"git clone {REPO_URL} {WORK_DIR}", timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to clone repo: {result.stderr}")


def load_model(model_id):
    import torch

    config = MODEL_CONFIGS.get(model_id, MODEL_CONFIGS[DEFAULT_MODEL])
    pipeline_type = config["pipeline"]

    print(f"Loading model {model_id} (pipeline: {pipeline_type})...")

    if pipeline_type == "StableDiffusion3Pipeline":
        from diffusers import StableDiffusion3Pipeline
        pipe = StableDiffusion3Pipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
        )
    else:
        from diffusers import FluxPipeline
        pipe = FluxPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
        )

    pipe.enable_model_cpu_offload()
    print(f"Model {model_id} loaded successfully.")
    return pipe, config


def unload_model(pipe):
    """Free GPU memory between models."""
    import torch
    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("GPU memory cleared.")


def self_push_progress(work_dir, batch_id, models, total, completed, failed,
                       results, start_time, model_short):
    """Push intermediate progress to GitHub (inside generation loop)."""
    manifest = {
        "batch_id": batch_id,
        "models": models,
        "total": total,
        "completed": completed,
        "failed": failed,
        "results": results,
        "elapsed_seconds": int(time.time() - start_time),
    }
    with open(os.path.join(work_dir, "results.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    git_push(work_dir, f"{model_short}: {completed}/{total}")


def generate_image(pipe, config, prompt, output_path, seed=None):
    import torch

    generator = None
    if seed is not None:
        generator = torch.Generator("cpu").manual_seed(seed)

    kwargs = {
        "prompt": prompt,
        "height": 1024,
        "width": 1024,
        "guidance_scale": config["guidance"],
        "num_inference_steps": config["steps"],
        "generator": generator,
    }

    # FLUX models support max_sequence_length
    if "max_seq_len" in config:
        kwargs["max_sequence_length"] = config["max_seq_len"]

    image = pipe(**kwargs).images[0]
    image.save(output_path)
    return output_path


def main():
    start_time = time.time()

    print("=" * 60)
    print("FAIRYTALE COVER GENERATOR - GPU WORKER")
    print("=" * 60)

    # Phase 1: Setup
    print("\n[Phase 1] Setting up environment...")
    setup_ssh()
    clone_repo()

    batch_path = os.path.join(WORK_DIR, BATCH_FILE)
    if not os.path.exists(batch_path):
        print("ERROR: No batch.json found. Nothing to do.")
        sys.exit(1)

    with open(batch_path) as f:
        batch = json.load(f)

    jobs = batch.get("jobs", [])
    batch_id = batch.get("batch_id", "unknown")
    models = batch.get("models", [DEFAULT_MODEL])
    print(f"Batch ID: {batch_id}, Jobs: {len(jobs)}, Models: {len(models)}")
    for m in models:
        print(f"  - {m}")

    if not jobs:
        update_status({"phase": "complete", "batch_id": batch_id, "total": 0, "completed": 0})
        sys.exit(0)

    total_images = len(jobs) * len(models)

    # Phase 2+3: Load each model and generate
    output_dir = os.path.join(WORK_DIR, OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    results = []
    completed = 0
    failed = 0

    for model_idx, model_id in enumerate(models):
        model_short = model_id.split("/")[-1]
        model_dir = os.path.join(output_dir, model_short)
        os.makedirs(model_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"MODEL {model_idx+1}/{len(models)}: {model_id}")
        print(f"{'='*60}")

        update_status({
            "phase": "loading_model",
            "batch_id": batch_id,
            "model": model_id,
            "model_index": model_idx + 1,
            "total_models": len(models),
            "total": total_images,
            "completed": completed,
        })

        try:
            pipe, config = load_model(model_id)
        except Exception as e:
            print(f"FAILED to load model {model_id}: {e}")
            traceback.print_exc()
            for job in jobs:
                results.append({
                    "id": job.get("id", "?"),
                    "story_path": job.get("story_path", ""),
                    "model": model_id,
                    "status": "failed",
                    "error": f"Model load failed: {e}",
                })
                failed += 1
            continue

        for i, job in enumerate(jobs):
            job_id = job.get("id", f"job_{i}")
            prompt = job.get("prompt", "")
            story_path = job.get("story_path", "")
            seed = job.get("seed")

            safe_name = story_path.replace("/", "__")
            output_path = os.path.join(model_dir, f"{safe_name}.png")

            print(f"\n[{model_short}] [{i+1}/{len(jobs)}] Generating: {story_path}")
            print(f"  Prompt: {prompt[:100]}...")

            try:
                generate_image(pipe, config, prompt, output_path, seed)
                file_size = os.path.getsize(output_path)
                print(f"  OK: {file_size} bytes")

                results.append({
                    "id": job_id,
                    "story_path": story_path,
                    "model": model_id,
                    "filename": f"{model_short}/{safe_name}.png",
                    "status": "completed",
                    "file_size": file_size,
                })
                completed += 1

            except Exception as e:
                print(f"  FAILED: {e}")
                traceback.print_exc()
                results.append({
                    "id": job_id,
                    "story_path": story_path,
                    "model": model_id,
                    "status": "failed",
                    "error": str(e),
                })
                failed += 1

            # Push progress every 25 images or on last image
            if (i + 1) % 25 == 0 or (i + 1) == len(jobs):
                self_push_progress(WORK_DIR, batch_id, models, total_images,
                                   completed, failed, results, start_time,
                                   model_short)

        # Always push after each model finishes
        manifest = {
            "batch_id": batch_id,
            "models": models,
            "total": total_images,
            "completed": completed,
            "failed": failed,
            "results": results,
            "elapsed_seconds": int(time.time() - start_time),
        }
        with open(os.path.join(WORK_DIR, "results.json"), "w") as f:
            json.dump(manifest, f, indent=2)

        git_push(WORK_DIR, f"{model_short}: {completed}/{total_images}")

        update_status({
            "phase": "generating",
            "batch_id": batch_id,
            "model": model_id,
            "model_index": model_idx + 1,
            "total_models": len(models),
            "total": total_images,
            "completed": completed,
            "failed": failed,
            "elapsed_seconds": int(time.time() - start_time),
        })

        # Unload model before loading next
        unload_model(pipe)

    # Phase 4: Done
    elapsed = int(time.time() - start_time)
    print(f"\n[Phase 4] Complete! {completed}/{total_images} images in {elapsed}s")

    update_status({
        "phase": "complete",
        "batch_id": batch_id,
        "models": models,
        "total": total_images,
        "completed": completed,
        "failed": failed,
        "elapsed_seconds": elapsed,
    })

    with open(os.path.join(WORK_DIR, "DONE"), "w") as f:
        f.write(f"completed={completed} failed={failed} elapsed={elapsed}s\n")
    git_push(WORK_DIR, "DONE")

    print("GPU worker finished. Instance can be destroyed.")


if __name__ == "__main__":
    main()
