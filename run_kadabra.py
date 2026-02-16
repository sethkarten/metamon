#!/usr/bin/env python3
"""Launcher that uses the pre-installed venv on della to run Kadabra training.

Fixes namespace conflict: gpu_manager clones to a work_dir that has metamon/
as a namespace package, which shadows the properly installed editable package.
We clean PYTHONPATH and sys.path before exec'ing into the venv Python.
"""
import os
import subprocess
import sys

VENV_PYTHON = "/scratch/gpfs/CHIJ/milkkarten/metamon_ref/.venv/bin/python"
METAMON_DIR = "/scratch/gpfs/CHIJ/milkkarten/metamon_ref"
CACHE_DIR = "/scratch/gpfs/CHIJ/milkkarten/.pokemon_cache"
SAVE_DIR = "/scratch/gpfs/CHIJ/milkkarten/metamon_ckpts"

# Clean environment to avoid namespace conflicts
env = os.environ.copy()
env.pop("PYTHONPATH", None)
env["METAMON_CACHE_DIR"] = CACHE_DIR
env["HF_HOME"] = os.path.join(CACHE_DIR, "huggingface")
env["HF_HUB_OFFLINE"] = "1"  # Don't try to download anything on compute node

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
    "--eval_gens",  # Empty = no eval (compute nodes can't run showdown)
    "--epochs", "50",
    "--batch_size_per_gpu", "64",
    "--dloader_workers", "4",
    "--log",
]

# Allow overriding with CLI args
args = sys.argv[1:] if len(sys.argv) > 1 else default_args

cmd = [VENV_PYTHON, "-m", "metamon.rl.train"] + args
print(f"Running: {' '.join(cmd)}")
print(f"CWD: {METAMON_DIR}")
sys.stdout.flush()

os.chdir(METAMON_DIR)
os.execve(VENV_PYTHON, cmd, env)
