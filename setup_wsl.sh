#!/bin/bash
# One-time setup of DreamerV3 (GPU) inside WSL2 Ubuntu.
# Usage (from WSL):  bash "/mnt/c/Users/<you>/.../Lab 8/setup_wsl.sh"
set -e

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$HOME/lab8"

# 1. uv (gives us Python 3.11 without touching the system Python)
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

# 2. Copy project to the Linux filesystem (much faster than /mnt/c)
mkdir -p "$DEST"
tar -C "$SRC" --exclude=.venv --exclude=.git --exclude=uv.lock --exclude=__pycache__ -cf - . | tar -C "$DEST" -xf -
cd "$DEST"

# 3. Python 3.11 venv + GPU requirements (jax[cuda12]==0.4.33 works on Linux/WSL2)
uv venv --clear --python 3.11 .venv
source .venv/bin/activate
#    Extras not listed in requirements.txt: ruamel.yaml + imageio (inference
#    videos) and dm_control (DMC Walker Walk).
uv pip install -r requirements.txt ruamel.yaml imageio imageio-ffmpeg dm_control

# 4. Verify GPU and headless MuJoCo rendering
export MUJOCO_GL=egl
python -c "import jax; print(jax.__version__, jax.devices())"
python - <<'EOF'
from dm_control import suite
env = suite.load('walker', 'walk')
env.reset()
print('walker render ok', env.physics.render(64, 64, camera_id=0).shape)
EOF
