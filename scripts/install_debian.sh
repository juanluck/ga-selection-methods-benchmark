#!/usr/bin/env bash
set -euo pipefail

sudo apt update
sudo apt install -y \
  python3 \
  python3-venv \
  python3-pip \
  git \
  build-essential

echo
echo "System dependencies installed."
echo "Recommended next step:"
echo "  bash scripts/setup_venv.sh"
