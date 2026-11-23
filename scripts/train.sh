#!/usr/bin/env bash
# scripts/train.sh - Tmux-based background training runner with per-run logging
#
# Commands:
#   bash scripts/train.sh [options...]       # Starts training inside tmux session 'train' (auto-attaches if interactive)
#   bash scripts/train.sh attach             # Attaches to running tmux session
#   bash scripts/train.sh status             # Checks job status and GPU utilization
#   bash scripts/train.sh logs               # Streams live training logs (tail -f logs/latest.log)
#   bash scripts/train.sh stop               # Gracefully terminates training and saves checkpoint
#
# Examples:
#   bash scripts/train.sh
#   bash scripts/train.sh --preset base
#   bash scripts/train.sh --resume --run-id <RUN_ID>
#   bash scripts/train.sh --stop-after 1000

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_dir=$(cd -- "$script_dir/.." && pwd)
logs_dir="$project_dir/logs"
session_name="train"
stop_marker="$logs_dir/.stop_requested"

mkdir -p "$logs_dir"

session_exists() {
    tmux has-session -t "$session_name" 2>/dev/null
}

python_command() {
    if [[ -x "$project_dir/.venv/bin/python" ]]; then
        printf '%q' "$project_dir/.venv/bin/python"
    else
        printf 'uv run --no-sync python'
    fi
}

cmd_help() {
    cat <<'EOF'
Usage: bash scripts/train.sh {start|attach|status|logs|stop|restart|help} [train options...]

Commands:
  [start] [options...]   Start training in tmux session 'train' (attaches if interactive)
  attach                 Attach to the running tmux session 'train'
  status                 Show tmux status, GPU utilization, and recent logs
  logs                   Follow logs/latest.log without attaching to tmux
  stop                   Send Ctrl+C so training saves a checkpoint before closing tmux
  restart [options...]   Stop any active run and start a new tmux training run
  help                   Show this help message and src/main.py options
EOF
    echo ""
    local py_cmd
    py_cmd=$(python_command)
    cd "$project_dir"
    eval "$py_cmd src/main.py --help"
}

cmd_start() {
    if ! command -v tmux >/dev/null 2>&1; then
        echo "Error: tmux is not installed on this system. Install with: apt-get install -y tmux" >&2
        exit 1
    fi

    if session_exists; then
        echo "Tmux session '$session_name' is already running."
        if [[ -t 0 && -t 1 ]]; then
            echo "Attaching to existing session '$session_name'..."
            exec tmux attach-session -t "$session_name"
        else
            echo "Use 'bash scripts/train.sh attach' or 'bash scripts/train.sh logs' to monitor."
            exit 0
        fi
    fi

    echo "Starting MiniReasoner training inside tmux session '$session_name'..."
    cd "$project_dir"

    local py_cmd
    py_cmd=$(python_command)
    local quoted_args=""
    if (( $# )); then
        printf -v quoted_args ' %q' "$@"
    fi

    rm -f "$stop_marker"
    local q_dir q_marker inner_cmd
    q_dir=$(printf '%q' "$project_dir")
    q_marker=$(printf '%q' "$stop_marker")
    inner_cmd="cd $q_dir && $py_cmd -u src/main.py${quoted_args}; rc=\$?; if [[ -f $q_marker ]]; then rm -f $q_marker; exit \$rc; fi; echo ''; echo \"Training exited with code \$rc. Press Enter to exit tmux.\"; read -r"

    tmux new-session -d -s "$session_name" -c "$project_dir" "bash -lc $(printf '%q' "$inner_cmd")"

    echo "=================================================================="
    echo "MiniReasoner Training Started in tmux session '$session_name'"
    echo "  Session:  $session_name"
    echo "  Logs:     logs/ (each run logged to a separate file; latest: logs/latest.log)"
    echo "  Losses:   Formatted to 4 decimal places (.4f)"
    echo "=================================================================="
    echo "To detach from tmux without stopping training: Press Ctrl+B, then D."
    echo "=================================================================="

    if [[ -t 0 && -t 1 ]]; then
        exec tmux attach-session -t "$session_name"
    else
        echo "Running detached. Attach anytime with: bash scripts/train.sh attach"
    fi
}

cmd_attach() {
    if ! session_exists; then
        echo "No active tmux session named '$session_name' found."
        echo "Start one with: bash scripts/train.sh"
        exit 1
    fi
    exec tmux attach-session -t "$session_name"
}

cmd_status() {
    echo "==================== MiniReasoner Job Status ===================="
    if session_exists; then
        echo "Status: RUNNING inside tmux session '$session_name'"
        echo ""
        tmux list-panes -t "$session_name" -F "Pane #{pane_index}: PID #{pane_pid} (active=#{pane_active})" 2>/dev/null || true
        echo ""
        if command -v nvidia-smi >/dev/null 2>&1; then
            echo "GPU Status:"
            nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv 2>/dev/null || true
            echo ""
        fi
        echo "Recent log output (last 15 lines from logs/latest.log):"
        echo "------------------------------------------------------------------"
        if [[ -f "$logs_dir/latest.log" ]]; then
            tail -n 15 "$logs_dir/latest.log"
        fi
        echo "------------------------------------------------------------------"
    else
        echo "Status: NOT RUNNING (no tmux session '$session_name')"
        if [[ -f "$logs_dir/latest.log" ]]; then
            echo ""
            echo "Last 15 lines from previous run (logs/latest.log):"
            echo "------------------------------------------------------------------"
            tail -n 15 "$logs_dir/latest.log"
            echo "------------------------------------------------------------------"
        fi
    fi
    echo "================================================================="
}

cmd_logs() {
    local target="$logs_dir/latest.log"
    if [[ ! -f "$target" ]]; then
        target=$(ls -t "$logs_dir"/*.log 2>/dev/null | head -n 1 || true)
    fi
    if [[ -z "$target" || ! -f "$target" ]]; then
        echo "No log files found in $logs_dir yet."
        exit 1
    fi
    echo "Streaming $target (Ctrl+C to detach; training continues in tmux)..."
    tail -n 40 -f "$target"
}

cmd_stop() {
    if ! session_exists; then
        echo "No active training session named '$session_name' is running."
        rm -f "$stop_marker"
        exit 0
    fi

    echo "Sending graceful stop signal (Ctrl+C) to tmux session '$session_name'..."
    touch "$stop_marker"
    tmux send-keys -t "$session_name" C-c

    # Wait up to 90 seconds so evaluation and Azure checkpoint upload finish cleanly
    local count=0
    while session_exists; do
        sleep 1
        count=$((count + 1))
        if (( count >= 90 )); then
            echo "Session did not close after 90 seconds; killing tmux session..."
            tmux kill-session -t "$session_name" 2>/dev/null || true
            break
        fi
    done
    rm -f "$stop_marker"

    echo "Training job stopped."
}

action="${1:-}"

case "$action" in
    help|-h|--help)
        cmd_help
        ;;
    attach)
        cmd_attach
        ;;
    status)
        cmd_status
        ;;
    logs|log|tail)
        cmd_logs
        ;;
    stop)
        cmd_stop
        ;;
    restart)
        shift
        cmd_stop
        cmd_start "$@"
        ;;
    start)
        shift
        cmd_start "$@"
        ;;
    *)
        cmd_start "$@"
        ;;
esac
