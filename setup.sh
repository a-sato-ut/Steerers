#!/usr/bin/env bash

set -eu

pip install git+https://github.com/georg-bn/DeDoDe.git
pip install -e .
bash download_weights.sh

sudo apt-get update
sudo apt-get install libgl1-mesa-glx

pip uninstall -y torch torchvision
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

exit 0
