#!/usr/bin/env python3
"""Launcher that uses the pre-installed venv on della to run Kadabra training.

Fixes namespace conflict: gpu_manager clones to a work_dir that has metamon/
as a namespace package, which shadows the properly installed editable package.
We clean PYTHONPATH and sys.path before exec'ing into the venv Python.

Launches a local Pokemon Showdown server for eval (compute nodes have no internet,
but the server + poke-env guest auth works fully offline).
"""
import os
import sys
import subprocess
import time
import signal
import socket
import atexit

VENV_BIN = "/scratch/gpfs/CHIJ/milkkarten/metamon_ref/.venv/bin"
VENV_PYTHON = os.path.join(VENV_BIN, "python")
ACCELERATE = os.path.join(VENV_BIN, "accelerate")
METAMON_DIR = "/scratch/gpfs/CHIJ/milkkarten/metamon_ref"
CACHE_DIR = "/scratch/gpfs/CHIJ/milkkarten/.pokemon_cache"
SAVE_DIR = "/scratch/gpfs/CHIJ/milkkarten/metamon_ckpts"
NODE_BIN = "/home/sk9014/.nvm/versions/node/v22.21.0/bin"
SHOWDOWN_DIR = "/scratch/gpfs/CHIJ/milkkarten/metamon_ref/server/pokemon-showdown"
SHOWDOWN_PORT = 8000

# Global reference to showdown process for cleanup
_showdown_proc = None


def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def start_showdown():
    """Start local Pokemon Showdown server for eval."""
    global _showdown_proc

    if is_port_in_use(SHOWDOWN_PORT):
        print(f"WARNING: Port {SHOWDOWN_PORT} already in use, assuming Showdown is running")
        return

    node_path = os.path.join(NODE_BIN, "node")
    showdown_script = os.path.join(SHOWDOWN_DIR, "pokemon-showdown")

    print(f"Starting Pokemon Showdown server on port {SHOWDOWN_PORT}...")
    _showdown_proc = subprocess.Popen(
        [node_path, showdown_script, "--skip-build"],
        cwd=SHOWDOWN_DIR,
        env={**os.environ, "PATH": f"{NODE_BIN}:{os.environ.get('PATH', '')}"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    # Wait for server to be ready
    for i in range(30):
        time.sleep(1)
        if is_port_in_use(SHOWDOWN_PORT):
            print(f"Showdown server ready on port {SHOWDOWN_PORT} (took {i+1}s)")
            return
        # Check if process died
        if _showdown_proc.poll() is not None:
            output = _showdown_proc.stdout.read().decode() if _showdown_proc.stdout else ""
            print(f"Showdown server failed to start (exit code {_showdown_proc.returncode})")
            print(f"Output: {output}")
            _showdown_proc = None
            raise RuntimeError("Showdown server failed to start")

    print("WARNING: Showdown server did not respond within 30s, continuing anyway")


def stop_showdown():
    """Stop the local Showdown server."""
    global _showdown_proc
    if _showdown_proc is not None and _showdown_proc.poll() is None:
        print("Stopping Showdown server...")
        _showdown_proc.terminate()
        try:
            _showdown_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _showdown_proc.kill()
            _showdown_proc.wait()
        print("Showdown server stopped.")
        _showdown_proc = None


# Detect GPU count from CUDA_VISIBLE_DEVICES
cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
num_gpus = len(cuda_devices.split(",")) if cuda_devices else 1

# Clean environment to avoid namespace conflicts
env = os.environ.copy()
env.pop("PYTHONPATH", None)
env["METAMON_CACHE_DIR"] = CACHE_DIR
env["HF_HOME"] = os.path.join(CACHE_DIR, "huggingface")
env["HF_HUB_OFFLINE"] = "1"  # Don't try to download anything on compute node
env["WANDB_MODE"] = "offline"  # No internet on compute nodes
env["PYTHONUNBUFFERED"] = "1"  # Flush output for SLURM log visibility

# Default training args for Kadabra (medium_multitaskagent + binary_rl)
default_args = [
    "--run_name", "kadabra_reference",
    "--save_dir", SAVE_DIR,
    "--model_gin_config", "medium_multitaskagent.gin",
    "--train_gin_config", "binary_rl.gin",
    "--obs_space", "TeamPreviewObservationSpace",
    "--action_space", "DefaultActionSpace",
    "--tokenizer", "DefaultObservationSpace-v1",
    "--parsed_replay_dir", os.path.join(CACHE_DIR, "parsed-replays"),
    "--eval_gens", "9",  # Eval on gen9ou against heuristic baselines
    "--epochs", "50",
    "--batch_size_per_gpu", "64",
    "--dloader_workers", "4",
    "--formats", "gen9ou",
    "--use_cached_filenames",
    "--log",
]

# Allow overriding with CLI args
args = sys.argv[1:] if len(sys.argv) > 1 else default_args

# Check if eval is enabled (look for --eval_gens with actual gen values)
eval_enabled = False
for i, arg in enumerate(args):
    if arg == "--eval_gens":
        # Check if next arg exists and is a number (not another flag)
        if i + 1 < len(args) and not args[i + 1].startswith("--"):
            eval_enabled = True
        break

# Start Showdown server if eval is enabled
if eval_enabled:
    start_showdown()
    atexit.register(stop_showdown)
    # Also handle SIGTERM (sent by SLURM on job cancellation)
    signal.signal(signal.SIGTERM, lambda sig, frame: (stop_showdown(), sys.exit(128 + sig)))

# Use accelerate launch for multi-GPU support
cmd = [
    ACCELERATE, "launch",
    "--num_processes", str(num_gpus),
    "--mixed_precision", "no",
    "-m", "metamon.rl.train",
] + args

print(f"GPUs: {num_gpus} (CUDA_VISIBLE_DEVICES={cuda_devices})")
print(f"Eval enabled: {eval_enabled}")
print(f"Running: {' '.join(cmd)}")
print(f"CWD: {METAMON_DIR}")
sys.stdout.flush()

os.chdir(METAMON_DIR)
# Can't use os.execve since we need atexit cleanup for showdown
# Use subprocess instead, forwarding exit code
result = subprocess.run(cmd, env=env)
stop_showdown()
sys.exit(result.returncode)
