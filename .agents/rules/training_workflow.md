# Training Workflow Rules

- **Headless Execution**: Use `tmux` for detached background training on GPU VMs via `scripts/train.sh`.
- **Survives Terminal Disconnect**: Training jobs must run detached so SSH disconnects or closing the laptop do not stop execution.
- **Loss Formatting**: Every loss printout (training loss, interval mean, validation loss, step 0 loss) must format strictly with 4 decimal digits (`.4f`), e.g. `Step 1 | train loss 8.1102`.
- **Per-Run Logging**: All training runs must output logs to their own dedicated timestamped file inside `logs/` (e.g., `logs/train_<TIMESTAMP>_<PRESET>.log`) and update `logs/latest.log`.
- **Checkpointing**: Upload snapshot checkpoints to Azure Blob Storage at every evaluation interval and upon receiving graceful stop signals (`SIGINT`/`SIGTERM`).
