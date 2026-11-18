# MiniReasoner

MiniReasoner is a decoder-only Transformer trained to predict the next token in English SimpleStories. It uses the full published training split, with approximately 99% of stories for training and 1% held out for validation. The byte-level BPE tokenizer learns only from training stories, uses a target vocabulary of 1,024 tokens, and keeps digits separate. [Dataset](https://huggingface.co/datasets/SimpleStories/SimpleStories)

## Model and training settings

Choose `base` or `small` in the notebook. Parameter counts assume a 1,024-token vocabulary and count the shared input embedding and output weights once. [Preset definitions](src/settings.py)

| Setting                            |   Base (default) |          Small |
| ---------------------------------- | ---------------: | -------------: |
| Trainable parameters               |       32,180,224 |        169,728 |
| Transformer blocks                 |               10 |              2 |
| Attention heads                    |                8 |              4 |
| Embedding dimension                |              512 |             64 |
| Context length (tokens per window) |              256 |             64 |
| Batch size (windows per step)      |               64 |             32 |
| Prediction targets per full step   |           16,384 |          2,048 |
| Training duration                  |          1 epoch |       60 steps |
| Learning rate (peak → minimum)     | 0.0005 → 0.00005 | 0.001 → 0.0001 |
| Warmup steps                       |               50 |              5 |
| Validation interval                |        500 steps |       20 steps |

Both presets use AdamW with weight decay 0.1. Training requires a CUDA GPU and uses BF16 when supported or FP32 on a T4. Inference defaults to CPU.

The base notebook pauses at optimizer step 12,000 on its one-epoch learning-rate schedule. On resume, use the original MLflow run ID and training settings. Set `stop_after` to a later step to continue beyond 12,000.

## Loss metrics

The notebook prints batch training loss at every step and validation loss at each check. MLflow records both against the optimizer step.

| MLflow metric              | What it measures                                                                                                                                   |
| -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `train_loss`               | Next-token cross-entropy for the current training batch.                                                                                           |
| `val_loss`                 | Mean cross-entropy over a fresh random half of the packed validation targets. Measured at step 0, each validation interval, and the stopping step. |
| `train_loss_interval_mean` | Mean training loss since the previous check, weighted by the number of prediction targets.                                                         |

## Start a run

Leave `resume = False` and `mlflow_run_id = None` to start from scratch. Save the run ID the notebook prints. MLflow tracks parameters and losses. Its `checkpoint_location` tag identifies the Google Drive folder holding the model and optimizer.

**Colab.** Select a GPU and mount Drive when prompted. Click **Open MLflow** in the cell just above **Train and validate**. It opens the tracking dashboard in a new browser tab. Select the `minireasoner` experiment to follow your run. The default link works while the notebook and runtime remain connected. MLflow reads its SQLite database on the runtime disk. To retain the same metrics across runtime resets, set `MLFLOW_TRACKING_URI` to a persistent tracking service before training. Drive keeps your prepared corpus and checkpoints under `MyDrive/MiniReasoner/`. [Open in Colab](https://colab.research.google.com/github/sampath017/MiniReasoner/blob/main/notebooks/MiniReasoner_Colab.ipynb)

**JarvisLabs or another GPU VM.** From the project directory, sync the environment. On Linux x86-64 and ARM64 this installs the rclone CLI. On first setup, create a Google Drive remote named `gdrive` and grant it write access. Existing remotes appear in `uv run rclone listremotes`. If you use a writable Drive mount instead, set `MINIREASONER_CHECKPOINT_ROOT` to its checkpoint directory and skip the rclone configuration step. [Google Drive setup for rclone](https://rclone.org/drive/) · [Authorizing a headless VM](https://rclone.org/remote_setup/)

```bash
uv sync --locked --python 3.12
uv run rclone config
```

With **JarvisLabs in your browser**, set `MINIREASONER_JARVIS_ORIGIN` in the project's `.env` to the HTTPS origin from your notebook's address bar, e.g. `https://YOUR_INSTANCE.notebooksn.jarvislabs.net`. Restart the notebook kernel after updating the project or environment and run the cells in order. Click the newly printed MLflow link. The notebook adds the current `/proxy/<port>/` path and configures MLflow to accept that hostname.

With **VS Code Remote – SSH** connected to JarvisLabs, select the project's `.venv` kernel and run the notebook. The dashboard cell starts MLflow on the VM and shows an **Open MLflow** link. If forwarding is needed, use **Ports → Forward a Port**, enter the port printed by the cell (normally `5000`), and open its **Local Address** in your browser. The VM stores metrics in `runs/mlflow.db` by default. Use a persistent disk or set `MLFLOW_TRACKING_URI` to a shared tracking server when moving between machines. [VS Code port forwarding](https://code.visualstudio.com/docs/remote/ssh#_forwarding-a-port-creating-ssh-tunnel)

To reuse your Colab data on a GPU VM, set `MINIREASONER_CORPUS_DRIVE_URL` in the project's `.env` to the Google Drive link for the `full-val-0.01` folder. The folder must allow anyone with the link to view and download its contents. Run the notebook to download `train.bin`, `val.bin`, `tokenizer.json`, and `manifest.json` into `data/tokenized/full-val-0.01/`. It verifies the files before training and reuses the local copy on later runs.

**Checkpoints.** The default `CHECKPOINT_LIMIT = None` in [settings.py](src/settings.py) keeps every checkpoint on Drive, including across resumes. A positive limit enables pruning. For example, `5` keeps the latest checkpoint and the four best earlier steps by validation loss, with interval-mean training loss breaking ties. Uploads overlap training, and the notebook waits for the final upload at a planned stop. To resume, set `resume = True` and the saved `mlflow_run_id` with the same tracking backend and Drive root. CPU inference can use `latest`, `best`, or `step-N` directly from Drive.
