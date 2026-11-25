#!/usr/bin/env bash
# scripts/prepare_sft_data.sh - Generate, inspect, save, and cache Phase 4 arithmetic SFT splits
#
# Examples:
#   bash scripts/prepare_sft_data.sh
#   bash scripts/prepare_sft_data.sh --max-digits 4
#   bash scripts/prepare_sft_data.sh --ops add sub mul div mod --max-digits 3
#   bash scripts/prepare_sft_data.sh --nonnegative-only --samples-per-op 5000
#   bash scripts/prepare_sft_data.sh --no-drive --overwrite

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_dir=$(cd -- "$script_dir/.." && pwd)

cd "$project_dir"

if [[ -x "$project_dir/.venv/bin/python" ]]; then
    exec "$project_dir/.venv/bin/python" -u src/main.py prepare-sft-data "$@"
else
    exec uv run --no-sync python -u src/main.py prepare-sft-data "$@"
fi
