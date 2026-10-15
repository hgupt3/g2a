#!/usr/bin/env bash
set -euo pipefail

# Linux x86_64, Python 3.11, Isaac Sim 5.1 and the pinned Isaac Lab 2.3.2 release.
g2a_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
g2a_env="${1:-$g2a_root/.venv}"
g2a_lab_dir="$g2a_root/.deps/IsaacLab"
g2a_lab_revision="37ddf626871758333d6ed89cf64ad702aef127d0"

if ! command -v uv >/dev/null; then
    echo "Install uv, then run scripts/setup.sh again."
    exit 1
fi
if [[ ! -x "$g2a_env/bin/python" ]]; then
    uv venv --python 3.11 "$g2a_env"
fi
g2a_python="$g2a_env/bin/python"
"$g2a_python" -c 'import sys; assert sys.version_info[:2] == (3, 11), "Isaac Sim 5.1 requires Python 3.11"'
uv pip install --python "$g2a_python" torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$g2a_python" 'isaacsim[all,extscache]==5.1.0' --extra-index-url https://pypi.nvidia.com
if [[ ! -d "$g2a_lab_dir/.git" ]]; then
    mkdir -p "$g2a_root/.deps"
    git clone --depth=1 --branch v2.3.2 https://github.com/isaac-sim/IsaacLab.git "$g2a_lab_dir"
fi
if [[ "$(git -C "$g2a_lab_dir" rev-parse HEAD)" != "$g2a_lab_revision" ]]; then
    echo "Expected Isaac Lab $g2a_lab_revision in $g2a_lab_dir. Use a separate checkout for modified SDK code."
    exit 1
fi
uv pip install --python "$g2a_python" -e "$g2a_lab_dir/source/isaaclab" -e "$g2a_lab_dir/source/isaaclab_rl"
uv pip install --python "$g2a_python" -e "$g2a_root[rl,visual]"
"$g2a_python" -m g2a validate
echo "Installed. See README.md for optimization, training, evaluation and visualization commands."
