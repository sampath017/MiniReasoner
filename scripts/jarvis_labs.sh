#!/usr/bin/env bash
set -euo pipefail

repo_url=https://github.com/sampath017/MiniReasoner.git
script_parent=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if [[ -f "$script_parent/notebooks/main.ipynb" && -e "$script_parent/.git" ]]; then
    project_dir=$script_parent
else
    project_dir=/home/MiniReasoner
fi

missing=()
for tool in git curl nvtop; do
    if ! command -v "$tool" >/dev/null; then
        missing+=("$tool")
    fi
done
if (( ${#missing[@]} )); then
    apt=(apt-get)
    if (( EUID != 0 )); then
        apt=(sudo apt-get)
    fi
    "${apt[@]}" update
    "${apt[@]}" install -y --no-install-recommends ca-certificates "${missing[@]}"
fi

if [[ ! -e "$project_dir/.git" ]]; then
    git clone "$repo_url" "$project_dir"
fi
cd -- "$project_dir"
if [[ ! -f notebooks/main.ipynb || ! -f uv.lock ]]; then
    echo "Expected the MiniReasoner notebook and uv.lock in $project_dir." >&2
    exit 1
fi

export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null; then
    curl -fsSL https://astral.sh/uv/install.sh | UV_INSTALL_DIR="$HOME/.local/bin" sh
fi
uv sync --locked --python 3.12

if [[ ! -e .env && ! -L .env ]]; then
    (umask 077 && set -C && cat .env.example > .env)
fi

echo "Ready in $project_dir. Set your DagsHub credentials in .env."
echo "For first-time Drive setup, run $project_dir/.venv/bin/rclone config and name the remote gdrive."
echo "Open notebooks/main.ipynb with $project_dir/.venv/bin/python as the kernel."
