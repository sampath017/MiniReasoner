# MiniReasoner · Phase 1

Pretrain a decoder-only Transformer from random weights on English SimpleStories. The notebook draws up to 60,000 training stories and 1,024 validation stories from the published training split. The published test split is reserved for final evaluation. The byte-level BPE tokenizer learns from the local training stories and splits digits individually. Arithmetic is reserved for later supervised fine-tuning and reinforcement learning.

[SimpleStories dataset](https://huggingface.co/datasets/SimpleStories/SimpleStories) · [SimpleStories paper](https://arxiv.org/abs/2504.09184)

## Project layout

| Directory | Contents |
| --- | --- |
| `src/` | Model, presets, and reusable tokenizer, training, and sampling functions |
| `notebooks/` | Data preparation, training, and checkpoint sampling |
| `data/` | Raw stories, packed token files, tokenizer, and corpus manifest |
| `models/` | Generated checkpoints |

## Set up with uv

Training requires a CUDA GPU. Run these commands from a writable copy of the project:

```bash
uv venv --python 3.12
uv pip install --torch-backend=auto -r pyproject.toml
```

On Kaggle, enable a GPU accelerator and internet access. Use a writable directory such as `/kaggle/working`. In notebook cells, use `%cd` to change directories and prefix shell commands with `!`.

On Colab, upload [MiniReasoner_Colab.ipynb](notebooks/MiniReasoner_Colab.ipynb), select a GPU runtime, and run the cells in order.

[uv's PyTorch setup guide](https://docs.astral.sh/uv/guides/integration/pytorch/#automatic-backend-selection)

## Prepare data and train

To launch Jupyter locally after setting up the project environment, run:

```bash
uv pip install --group dev
uv run --no-sync jupyter notebook
```

For authenticated Hugging Face downloads, copy [`.env.example`](.env.example) to `.env` at the project root, uncomment `HF_TOKEN`, and set your token. If shard reads keep timing out, uncomment `HF_HUB_DOWNLOAD_TIMEOUT` as well. The notebook loads `.env` before importing Hugging Face libraries. Restart the notebook kernel after changing the file. Git ignores `.env`.

Open [prepare_data.ipynb](notebooks/prepare_data.ipynb) in Jupyter, select the project's Python environment, and run the cells in order. The notebook summarizes story lengths for each local split and previews a training story. It calls [src/tokenizer.py](src/tokenizer.py) to train the tokenizer and pack the splits. Edit story limits in the notebook and shared paths in [src/settings.py](src/settings.py). Restart the notebook kernel after changing paths.

Set `VOCAB_SIZE` in `src/settings.py` to change the tokenizer's target vocabulary size. The default is 1,024. After changing it, restart the notebook kernel and regenerate the tokenizer and packed corpus. Training sizes the model from the actual vocabulary size recorded in the corpus manifest. Byte-level BPE starts with all 256 byte values and learns frequent merges from the training stories. Common words or word pieces can become single tokens, while rare words can still be encoded with smaller pieces. Each digit is kept separate for later arithmetic training.

The download cell caches only the published training shards in Arrow format under `data/raw/`. It shuffles the training stories with a fixed seed, sets aside validation stories first, and excludes exact copies of them from the local training split. The export cell saves `train.bin`, `val.bin`, `tokenizer.json`, and `manifest.json` under `data/tokenized/`. The final cell verifies EOS at every packed story boundary.

## Train and sample in Jupyter

After exporting the corpus, open [train.ipynb](notebooks/train.ipynb) and run the cells in order. Choose the preset in the settings cell. The data and checkpoint paths come from `src/settings.py`. The notebook checks that both local splits came from the published training split, logs training and validation loss, and saves `base.pt` under `models/simplestories/` by default. Edit the prompt in the final cell to sample from that checkpoint.

The model uses nanoGPT's naming with `context_length` for the context setting. The base preset in `src/settings.py` sets `n_layer = 10` for the number of Transformer blocks, `context_length = 256` for the context length, `n_embd = 512` for the embedding dimension, and `n_head = 8` for attention heads. Each head has 64 dimensions. With a 1,024-token vocabulary, the model has 32,180,224 trainable parameters.

Each optimizer step uses one batch. The base preset processes 64 sequences per step and 24,576,000 target tokens over 1,500 steps, including repeated windows. For a shorter GPU run, set `preset = "small"`. It uses 32 sequences per step. Give the local run a separate `RUN_DIR`, such as `models/simplestories-small`. With automatic precision, training uses BF16 on GPUs with native support and FP32 on GPUs such as the T4.

## Inference only

Set `train_model = False` in either training notebook to generate from a saved checkpoint without preparing the corpus. The default `inference_device = "cpu"` runs generation on CPU. Set it to `"cuda"` to use a GPU. For a local install on a CPU-only machine, use `--torch-backend=cpu` in the install command above.

## Pause and resume

Set `stop_after = 100` in the settings cell to save and pause at step 100. To continue, set `resume = True`, clear `stop_after` by setting it to `None`, and rerun from the settings cell with the same preset, corpus, and checkpoint directory. On Kaggle, save `base.pt` and the four files from `data/tokenized/` before the notebook session ends.

Checkpoints saved before the switch to one batch per step still work for inference. Start a fresh training run in a separate checkpoint directory to use the current presets.
