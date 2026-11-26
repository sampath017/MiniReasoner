#!/usr/bin/env bash
# scripts/prepare_data.sh - Generate, inspect, save, and cache Phase 1 corpus, Phase 4 SFT splits, or Phase 5 failed RL prompts
#
# Examples:
#   bash scripts/prepare_data.sh                                    # Normal direct SFT data (arith_v2)
#   bash scripts/prepare_data.sh --format cot --multistep           # SFT data with Chain-of-Thought scratchpad + multi-step (arith_v3)
#   bash scripts/prepare_data.sh --stage rl                         # Mine & cache Phase 5 hard RL dataset of failed SFT train problems
#   bash scripts/prepare_data.sh --stage corpus                     # Phase 1 SimpleStories tokenized corpus & BPE tokenizer
#   bash scripts/prepare_data.sh --max-digits 4 --ops add sub mul div mod
#   bash scripts/prepare_data.sh --nonnegative-only --samples-per-op 5000
#   bash scripts/prepare_data.sh --no-drive --overwrite

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_dir=$(cd -- "$script_dir/.." && pwd)

cd "$project_dir"

if [[ -x "$project_dir/.venv/bin/python" ]]; then
    exec "$project_dir/.venv/bin/python" -u src/main.py prepare-data "$@"
else
    exec uv run --no-sync python -u src/main.py prepare-data "$@"
fi
