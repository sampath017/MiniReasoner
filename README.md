# MiniReasoner

MiniReasoner is a decoder-only Transformer trained to predict the next token in English SimpleStories. It uses the full published training split, with approximately 99% of stories for training and 1% held out for validation (`2,094,643` train stories / `853,498,695` tokens; `21,053` val stories / `8,598,126` tokens). The byte-level BPE tokenizer learns only from training stories, uses a target vocabulary of 1,024 tokens, and keeps digits separate.

- **Dataset Citation:** [SimpleStories/SimpleStories](https://huggingface.co/datasets/SimpleStories/SimpleStories) ([arXiv:2504.09184](https://arxiv.org/abs/2504.09184)) · Hugging Face snapshot commit SHA: `e63b8adc3b1a1bdc7cac5b500d150b71346b0628` (`2,115,696` source `train` rows)

## Model and training settings

Choose `base` or `small` via `--preset` when launching training (defaults to `base`). Parameter counts assume a 1,024-token vocabulary and count the shared input embedding and output weights once. [Preset definitions](src/settings.py)

| Setting                            |   Base (default) |          Small |
| ---------------------------------- | ---------------: | -------------: |
| Trainable parameters               |       32,049,152 |        165,632 |
| Positional embeddings              |  RoPE (θ = 10,000)|  RoPE (θ = 10,000)|
| Transformer blocks                 |               10 |              2 |
| Attention heads                    |                8 |              4 |
| Embedding dimension                |              512 |             64 |
| Context length (tokens per window) |            1,024 |             64 |
| Batch size (windows per step)      |               32 |             32 |
| Prediction targets per full step   |           32,768 |          2,048 |
| Training duration                  | 1 epoch (~26,047)|       60 steps |
| Learning rate (peak → minimum)     |0.00065 → 0.000065| 0.001 → 0.0001 |
| Warmup steps                       |              700 |              5 |
| Validation interval                |        500 steps |       20 steps |

Both presets train on the VM's CUDA GPU using AdamW with weight decay 0.1 and Rotary Positional Embeddings (RoPE). Training uses BF16 on modern GPUs (e.g. NVIDIA A30) and FP32 otherwise.

The base preset trains a 32M RoPE model from scratch for 1 epoch (~26,047 steps at 32,768 tokens/step) on the ~853.5M-token SimpleStories training corpus. The small preset starts a fresh 60-step run.

## Loss metrics

The training CLI prints the current batch loss at every step (4 decimal digits, `.4f`) and mirrors each run's full output to its own file under `logs/`. At each validation check it also prints the interval's mean training loss beside validation loss. MLflow records all three against the optimizer step.

| MLflow metric              | What it measures                                                                                                                                   |
| -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `train_loss`               | Next-token cross-entropy for the current training batch.                                                                                           |
| `val_loss`                 | Mean cross-entropy over a fresh random sample of the packed validation targets (`eval_token_fraction`). Measured at step 0, each validation interval, and the stopping step. |
| `train_loss_interval_mean` | Mean training loss since the previous check, weighted by the number of prediction targets.                                                         |

## Start a run on the VM

Run both presets on a CUDA VM. On JarvisLabs, run the [setup script](scripts/jarvis_labs.sh) from your MiniReasoner checkout.

```bash
bash scripts/jarvis_labs.sh
```

For a fresh VM, download the script and run it to clone the project into `/home/MiniReasoner`, install `nvtop`, and sync the locked Python environment.

```bash
curl -fsSLo /tmp/jarvis_labs.sh https://raw.githubusercontent.com/sampath017/MiniReasoner/main/scripts/jarvis_labs.sh
bash /tmp/jarvis_labs.sh
```

Use an existing private **Azure Blob Storage** container named `minireasoner` (or change `AZURE_CONTAINER` in `src/settings.py`). Hot storage with locally redundant storage (LRS) suits frequently restored checkpoints. Give the VM access to read, list, and write blobs. Deleting blobs is needed only if you enable checkpoint pruning.

Fill in the entries in the project-root `.env` from [.env.example](.env.example).

| Variable | Value |
| --- | --- |
| `AZURE_STORAGE_CONNECTION_STRING` | Connection string for the account containing the private checkpoint container |
| `DATABRICKS_HOST` | HTTPS workspace URL |
| `DATABRICKS_TOKEN` | Workspace token with permission to log runs under `/Shared/MiniReasoner` |
| `RCLONE_CONFIG_BASE64` | Base64-encoded `rclone.conf` for unattended `gdrive:` access (optional if `rclone` is already configured) |

For container-scoped access, generate a SAS on the `minireasoner` container with Read, Write, and List permissions and HTTPS only. Set `AZURE_STORAGE_CONNECTION_STRING` to `BlobEndpoint=https://<account>.blob.core.windows.net;SharedAccessSignature=<token>`. Use the token text without its leading question mark. Enable Delete only if the VM will prune checkpoints. [Azure SAS connection strings](https://learn.microsoft.com/en-us/azure/storage/common/storage-configure-connection-string#create-a-connection-string-using-a-shared-access-signature)

### Headless background training via `tmux` (survives terminal disconnects)

To start long-running training inside a detached `tmux` session (`train`) that keeps running when you close your SSH terminal, browser, or laptop:

```bash
# Start training in tmux and attach immediately (Ctrl+B, then D to detach)
bash scripts/train.sh --preset base

# Re-attach to the running tmux session
bash scripts/train.sh attach

# Monitor status, GPU stats, and recent log lines
bash scripts/train.sh status

# Stream live training logs without attaching to tmux
bash scripts/train.sh logs

# Stop gracefully when needed (saves checkpoint to Azure before exiting)
bash scripts/train.sh stop
```

You can also run [src/main.py](src/main.py) directly:
```bash
python src/main.py --preset base
# Or resume an existing run:
python src/main.py --preset base --resume --run-id <MLFLOW_RUN_ID>
```

### Evaluation & Inference Notebook

Use [notebooks/evaluation.ipynb](notebooks/evaluation.ipynb) to inspect checkpoints from Azure Blob Storage, plot convergence curves, evaluate perplexity/BPB and Story Cloze accuracy, test arithmetic baselines, and generate stories on both CPU and GPU. See [COOKBOOK.md](COOKBOOK.md) for the complete end-to-end build, training, post-training SFT/RL, and release plan.

## Reuse the corpus and save checkpoints

Jarvis setup installs `tmux` and checks that it can list Google Drive using the `gdrive` connection before it reports ready. When `RCLONE_CONFIG_BASE64` is present in `.env`, `scripts/jarvis_labs.sh` configures `gdrive:` automatically with no interactive prompts. Otherwise, run setup in a VM terminal and authorize Google Drive interactively. [Google OAuth client setup](https://rclone.org/drive/#making-your-own-client-id) · [rclone remote setup](https://rclone.org/remote_setup/)

| Location | Contents |
| --- | --- |
| Google Drive · `My Drive/MiniReasoner/data/raw/` | SimpleStories training text saved during fresh preparation |
| Google Drive · `My Drive/MiniReasoner/data/tokenized/full-val-0.01/` | Tokenizer, packed training and validation tokens, and their manifest |
| Azure · `minireasoner/checkpoints/Shared/MiniReasoner/<run-id>/` | Checkpoints (`step-*.pt`) and run logs (`logs/train_<TIMESTAMP>_<PRESET>.log`) |
| Databricks · `/Shared/MiniReasoner` | Per-run curves, settings, `logs/` artifacts, and Azure checkpoint/log tags |
| Local VM · `logs/` | One local log per run (`logs/train_<TIMESTAMP>_<PRESET>.log`, no local aliases) |

A fresh VM downloads the prepared corpus from Drive and verifies its file sizes and checksums before training. An existing folder with `manifest.json`, `tokenizer.json`, `train.bin`, and `val.bin` works directly when its manifest matches the corpus settings. The VM reads training batches from its disk cache. If preparation is needed, it reuses raw training stories from Drive or fetches them from Hugging Face, fits BPE on the training portion, and saves the finished corpus to Drive.

Uploads run in the background after validation. The training runner waits for pending uploads on completion or `SIGINT`/`SIGTERM`. `CHECKPOINT_LIMIT = None` keeps every checkpoint. To resume on a GPU VM, pass `--resume --run-id <MLFLOW_RUN_ID>` to `bash scripts/train.sh` or `python src/main.py` using the same storage container, experiment path, corpus, and training settings.

When the run is complete, open [notebooks/evaluation.ipynb](notebooks/evaluation.ipynb) with the project's `.venv` kernel (on either a local CPU machine or a CUDA GPU VM) to auto-select the best checkpoint from Azure, inspect metrics, and run inference benchmarks.

To remove Azure checkpoints for runs you deleted in the Databricks MLflow UI:

```bash
# Safe dry-run preview (lists active, deleted, and missing runs)
uv run python src/main.py cleanup

# Delete orphaned Azure blobs for runs deleted in Databricks MLflow
uv run python src/main.py cleanup --delete
```

## CPU inference

Open [notebooks/evaluation.ipynb](notebooks/evaluation.ipynb) on your local machine (it automatically detects CPU vs CUDA). Leave `RUN_ID = None` to select the latest Azure run's lowest-loss checkpoint, set `RUN_ID = "<MLFLOW_RUN_ID>"`, or set `LOCAL_CHECKPOINT` to a downloaded `.pt` file for offline inference. You can also call `src.generate.complete()` directly from Python.
