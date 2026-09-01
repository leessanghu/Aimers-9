#!/usr/bin/env bash
set -euo pipefail

python --version
python -m pip install --upgrade pip
python -m pip install -r envs/latest_tabular/requirements-colab-gpu.txt
python envs/latest_tabular/check_env.py --track colab-gpu

