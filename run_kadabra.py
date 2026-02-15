#!/usr/bin/env python3
"""Launcher that uses the pre-installed venv on della to run Kadabra training."""
import os
import sys

VENV_PYTHON = "/scratch/gpfs/CHIJ/milkkarten/metamon_ref/.venv/bin/python"
METAMON_DIR = "/scratch/gpfs/CHIJ/milkkarten/metamon_ref"

os.chdir(METAMON_DIR)
os.execv(VENV_PYTHON, [VENV_PYTHON, "-m", "metamon.rl.train"] + sys.argv[1:])
