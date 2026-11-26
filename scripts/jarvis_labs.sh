#!/usr/bin/env bash
set -euo pipefail

repo_url=https://github.com/sampath017/MiniReasoner.git
script_parent=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if [[ -f "$script_parent/src/main.py" && -f "$script_parent/pyproject.toml" && -e "$script_parent/.git" ]]; then
    project_dir=$script_parent
else
    project_dir=/home/MiniReasoner
fi

missing=()
for tool in git curl nvtop tmux; do
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
if [[ ! -f src/main.py || ! -f uv.lock ]]; then
    echo "Expected the MiniReasoner source (src/main.py) and uv.lock in $project_dir." >&2
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
if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env 2>/dev/null || true
    set +a
fi

rclone="$project_dir/.venv/bin/rclone"
config_hint="$project_dir/.venv/minireasoner-rclone-path"
config_file=$("$rclone" config file | tail -n 1)
mkdir -p "$(dirname "$config_file")"

# Automatically deploy rclone config from RCLONE_CONFIG_BASE64 in .env
if [[ -z ${RCLONE_CONFIG:-} ]]; then
    if [[ -n "${RCLONE_CONFIG_BASE64:-}" ]]; then
        printf '%s' "$RCLONE_CONFIG_BASE64" | base64 -d > "$config_file"
        export RCLONE_CONFIG="$config_file"
    elif [[ -r "$config_hint" ]]; then
        IFS= read -r saved_config < "$config_hint"
        if [[ -f "$saved_config" ]]; then
            export RCLONE_CONFIG="$saved_config"
        fi
    fi
fi
if [[ -z ${RCLONE_CONFIG:-} ]]; then
    export RCLONE_CONFIG=$(realpath -m -- "$config_file")
fi
remotes=$("$rclone" listremotes 2>/dev/null || true)
if ! grep -Fxq 'gdrive:' <<< "$remotes"; then
    if [[ ! -t 0 || ! -t 1 ]]; then
        echo "Drive connection 'gdrive' is missing. Rerun this script in a VM terminal to authorize Google Drive, or set RCLONE_CONFIG to an authorized config file on the VM and rerun." >&2
        exit 1
    fi
    echo "Set up gdrive with your own Google OAuth client ID and secret when prompted."
    echo "Follow the headless sign-in instructions on a computer with a browser. Choose the Google account that holds MiniReasoner/data."
    "$rclone" config create gdrive drive scope drive config_is_local false --all --no-output
fi
if ! "$rclone" lsf gdrive: --dirs-only --max-depth 1 > /dev/null; then
    echo "Cannot read gdrive. Check the Drive connection in $RCLONE_CONFIG and rerun setup." >&2
    printf 'For an expired login, run %q --config %q config reconnect gdrive:\n' "$rclone" "$RCLONE_CONFIG" >&2
    exit 1
fi
# Save only the config path. rclone manages the credential in its original file.
hint_tmp=$(mktemp "$config_hint.XXXXXX")
printf '%s\n' "$RCLONE_CONFIG" > "$hint_tmp"
mv -f -- "$hint_tmp" "$config_hint"

echo "Ready in $project_dir. Ensure Azure Storage, Databricks, and RCLONE_CONFIG_BASE64 are set in .env."
echo "Verified Google Drive access for gdrive."
echo "Start training with 'bash scripts/train.sh', or open evaluation notebooks/ (`phase3_...` through `phase4c_...`) with $project_dir/.venv/bin/python as the kernel."
