#!/usr/bin/env python3
"""
Fairytale Cover Generator — runs on vast.ai GPU instances.
Pulls FLUX.1 Dev from HuggingFace, generates covers from batch.json,
pushes results back to GitHub.
"""

import json
import os
import sys
import time
import subprocess
import traceback
from pathlib import Path

REPO_URL = "git@github.com:jwendyr/fairytale-covers.git"
WORK_DIR = "/workspace/fairytale-covers"
BATCH_FILE = "batch.json"
OUTPUT_DIR = "output"
STATUS_FILE = "status.json"
MODEL_ID = "black-forest-labs/FLUX.1-dev"


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

    run_cmd('ssh-keyscan github.com >> ~/.ssh/known_hosts 2>/dev/null')
    run_cmd('git config --global user.email "gpu-worker@fairytale.ucok.org"')
    run_cmd('git config --global user.name "Fairytale GPU Worker"')


def clone_repo():
    # If we're already inside the repo (launched by onstart), just use it
    batch_here = os.path.join(WORK_DIR, BATCH_FILE)
    if os.path.exists(batch_here):
        print(f"Repo already exists at {WORK_DIR} with {BATCH_FILE}, skipping clone.")
        return

    # Also check if we're running from inside the repo already
    cwd_batch = os.path.join(os.getcwd(), BATCH_FILE) if os.getcwd() != "/" else None
    if cwd_batch and os.path.exists(cwd_batch) and os.getcwd() != WORK_DIR:
        # We're in the repo but at a different path — update WORK_DIR
        global WORK_DIR
        WORK_DIR = os.getcwd()
        print(f"Running from repo at {WORK_DIR}, skipping clone.")
        return

    # Fresh clone needed — make sure we're not deleting our own cwd
    if os.path.exists(WORK_DIR):
        try:
            os.chdir("/workspace")
        except Exception:
            os.chdir("/tmp")
        run_cmd(f"rm -rf {WORK_DIR}")

    result = run_cmd(f"git clone {REPO_URL} {WORK_DIR}", timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to clone repo: {result.stderr}")


def load_model():
    import torch
    from diffusers import FluxPipeline

    print(f"Loading model {MODEL_ID}...")
    pipe = FluxPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()
    print("Model loaded successfully.")
    return pipe


def generate_image(pipe, prompt, output_path, seed=None):
    import torch

    generator = None
    if seed is not None:
        generator = torch.Generator("cpu").manual_seed(seed)

    image = pipe(
        prompt,
        height=1024,
        width=1024,
        guidance_scale=3.5,
        num_inference_steps=30,
        max_sequence_length=512,
        generator=generator,
    ).images[0]

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
    print(f"Batch ID: {batch_id}, Jobs: {len(jobs)}")

    if not jobs:
        update_status({"phase": "complete", "batch_id": batch_id, "total": 0, "completed": 0})
        sys.exit(0)

    update_status({
        "phase": "loading_model",
        "batch_id": batch_id,
        "total": len(jobs),
        "completed": 0,
    })

    # Phase 2: Load model
    print("\n[Phase 2] Loading model...")
    pipe = load_model()

    # Phase 3: Generate images
    output_dir = os.path.join(WORK_DIR, OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    results = []
    completed = 0
    failed = 0

    for i, job in enumerate(jobs):
        job_id = job.get("id", f"job_{i}")
        prompt = job.get("prompt", "")
        story_path = job.get("story_path", "")
        seed = job.get("seed")

        print(f"\n[{i+1}/{len(jobs)}] Generating: {story_path}")
        print(f"  Prompt: {prompt[:100]}...")

        safe_name = story_path.replace("/", "__")
        output_path = os.path.join(output_dir, f"{safe_name}.png")

        try:
            generate_image(pipe, prompt, output_path, seed)
            file_size = os.path.getsize(output_path)
            print(f"  OK: {file_size} bytes")

            results.append({
                "id": job_id,
                "story_path": story_path,
                "filename": f"{safe_name}.png",
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
                "status": "failed",
                "error": str(e),
            })
            failed += 1

        # Push progress every 5 images or on last
        if (i + 1) % 5 == 0 or (i + 1) == len(jobs):
            manifest = {
                "batch_id": batch_id,
                "total": len(jobs),
                "completed": completed,
                "failed": failed,
                "results": results,
                "elapsed_seconds": int(time.time() - start_time),
            }
            with open(os.path.join(WORK_DIR, "results.json"), "w") as f:
                json.dump(manifest, f, indent=2)

            git_push(WORK_DIR, f"batch {completed}/{len(jobs)}")

            update_status({
                "phase": "generating",
                "batch_id": batch_id,
                "total": len(jobs),
                "completed": completed,
                "failed": failed,
                "current_job": i + 1,
                "elapsed_seconds": int(time.time() - start_time),
            })

    # Phase 4: Done
    elapsed = int(time.time() - start_time)
    print(f"\n[Phase 4] Complete! {completed}/{len(jobs)} images in {elapsed}s")

    update_status({
        "phase": "complete",
        "batch_id": batch_id,
        "total": len(jobs),
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
