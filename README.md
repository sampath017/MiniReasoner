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

Both presets use AdamW with weight decay 0.1. CUDA training uses BF16 when supported or FP32 on a T4. For a local FP32 CPU run, set `preset = "small"` and `training_device = "cpu"` in the notebook. Inference defaults to CPU.

The base notebook pauses at optimizer step 12,000 on its one-epoch learning-rate schedule. On resume, use the original MLflow run ID and training settings. Set `stop_after` to a later step to continue beyond 12,000.

## Loss metrics

The notebook prints the current batch loss at every step. At each validation check it also prints the interval's mean training loss beside validation loss. MLflow records all three against the optimizer step.

| MLflow metric              | What it measures                                                                                                                                   |
| -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `train_loss`               | Next-token cross-entropy for the current training batch.                                                                                           |
| `val_loss`                 | Mean cross-entropy over a fresh random half of the packed validation targets. Measured at step 0, each validation interval, and the stopping step. |
| `train_loss_interval_mean` | Mean training loss since the previous check, weighted by the number of prediction targets.                                                         |

## Start a run on the VM

Leave `resume = False` and `mlflow_run_id = None` to start from scratch. Save the run ID the notebook prints. MLflow tracks parameters and losses. Its `checkpoint_location` tag identifies the Google Drive folder holding the model and optimizer.

Open [notebooks/main.ipynb](notebooks/main.ipynb) in the VM checkout using the project's `.venv` kernel. On a fresh checkout, copy `.env.example` to `.env` in the project root.

**DagsHub charts and Google Drive checkpoints.** Set these three values in the VM's `.env`. Use a DagsHub access token with write access to the repository.

```dotenv
MLFLOW_TRACKING_URI=https://dagshub.com/YOUR_ACCOUNT/YOUR_REPOSITORY.mlflow
MLFLOW_TRACKING_USERNAME=YOUR_ACCOUNT
MLFLOW_TRACKING_PASSWORD=YOUR_DAGSHUB_ACCESS_TOKEN
```

DagsHub keeps each run's settings, tags and loss history after the VM stops. Google Drive stores its model and optimizer under `MiniReasoner/checkpoints/<run-id>/`. [DagsHub MLflow setup](https://dagshub.com/docs/integration_guide/mlflow_tracking/)

**SQLite tracking.** With `MLFLOW_TRACKING_URI` blank, the notebook backs up its local database to `MiniReasoner/tracking/mlflow.db` on Drive during training and when a run closes. A fresh runtime restores the most recent backup. Use one training runtime at a time with this database.

**JarvisLabs or another GPU VM.** From the project directory, sync the environment. On Linux x86-64 and ARM64 this installs the rclone CLI. On first setup, create a Google Drive remote named `gdrive` and grant it write access. Existing remotes appear in `uv run rclone listremotes`. If you use a writable Drive mount instead, set `MINIREASONER_CHECKPOINT_ROOT` to its checkpoint directory and skip the rclone configuration step. [Google Drive setup for rclone](https://rclone.org/drive/) · [Authorizing a headless VM](https://rclone.org/remote_setup/)

```bash
uv sync --locked --python 3.12
uv run rclone config
```

For the **SQLite dashboard in a JarvisLabs browser**, set `MINIREASONER_JARVIS_ORIGIN` to the notebook's HTTPS origin, e.g. `https://YOUR_INSTANCE.notebooksn.jarvislabs.net`. Restart the kernel after updating the project or its environment. The dashboard cell generates the `/proxy/<port>/` link.

With **VS Code Remote – SSH**, select the project's `.venv` kernel. For the SQLite dashboard, use **Ports → Forward a Port** if prompted and open the **Local Address** in your browser. For DagsHub, **Open MLflow** goes directly to your hosted dashboard. [VS Code port forwarding](https://code.visualstudio.com/docs/remote/ssh#_forwarding-a-port-creating-ssh-tunnel)

The notebook reuses the verified corpus in `data/tokenized/full-val-0.01/`. To restore it on a fresh VM, set `MINIREASONER_CORPUS_DRIVE_URL` in the project's `.env` to the Google Drive link for the folder containing `train.bin`, `val.bin`, `tokenizer.json`, and `manifest.json`. The folder must allow anyone with the link to download its contents.

**Checkpoints.** The default `CHECKPOINT_LIMIT = None` in [settings.py](src/settings.py) keeps every checkpoint on Drive, including across resumes. A positive limit enables pruning. For example, `5` keeps the latest checkpoint and the four best earlier steps by validation loss, with interval-mean training loss breaking ties. Uploads overlap training, and the notebook waits for the final upload at a planned stop. To resume, set `resume = True` and the saved `mlflow_run_id` with the same tracking backend and Drive root. CPU inference can use `latest`, `best`, or `step-N` directly from Drive.
