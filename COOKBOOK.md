# MiniReasoner Full Build Cookbook & Training Plan

A complete end-to-end engineering playbook for **MiniReasoner**—covering environment provisioning, corpus preparation, RoPE pre-training on a GPU VM, post-training evaluation & storage hygiene, arithmetic Supervised Fine-Tuning (SFT), Verifiable-Reward RL (GRPO), and final release verification.

---

## Phase 0 · Before Training (Pre-Flight & Environment Setup)

### 0.1 Credentials (`.env`-Only Configuration)
Keep all secrets strictly in the untracked project-root `.env` file (seeded from [.env.example](.env.example); never commit `.env`):

| Variable | Purpose | Notes |
| --- | --- | --- |
| `AZURE_STORAGE_CONNECTION_STRING` | Azure Blob Storage container (`minireasoner`) | Container-scoped SAS (`sp=rwdl`, HTTPS only, no leading `?`) |
| `DATABRICKS_HOST` | Databricks workspace URL | `https://<workspace>.cloud.databricks.com` |
| `DATABRICKS_TOKEN` | MLflow tracking token | Write permissions under `/Shared/MiniReasoner` |
| `RCLONE_CONFIG_BASE64` | Headless Google Drive authorization | Base64 of `rclone.conf` with your own Google OAuth Client ID/Secret so VMs never throttle or prompt interactively |

### 0.2 Provision a GPU VM (JarvisLabs / Ubuntu CUDA)
Recommended hardware: **NVIDIA A30 (24 GB VRAM)** or Ampere+ GPU (`BF16` native support).

```bash
# On a fresh VM: clones repo to /home/MiniReasoner, installs git/curl/nvtop/tmux, syncs Python 3.12 via uv
curl -fsSLo /tmp/jarvis_labs.sh https://raw.githubusercontent.com/sampath017/MiniReasoner/main/scripts/jarvis_labs.sh
bash /tmp/jarvis_labs.sh
```

Copy your local `.env` to `/home/MiniReasoner/.env` (`chmod 600`, LF line endings), then re-run `bash scripts/jarvis_labs.sh` from `/home/MiniReasoner` to auto-configure `gdrive:` in `rclone` from `RCLONE_CONFIG_BASE64` and cache the config path in `.venv/minireasoner-rclone-path`.

### 0.3 Pre-Flight Sanity Checks (Run Before Starting GPU Billing Long Runs)
```bash
uv run python tests/test_rope_upgrade.py
uv run python tests/test_azure_cleanup.py
uv run python src/main.py --help
```

Confirm:
- Parameter count matches **`32,049,152`** (`base` preset, RoPE, `context_length = 1,024`, `VOCAB_SIZE = 1,024`, tied embeddings).
- Causality test passes (`F.scaled_dot_product_attention(..., is_causal=True)` + FP32 RoPE rotations).
- `B=32, T=1024` fits cleanly in BF16 VRAM (`~6.9–7.7 GB` on a 24 GB A30).

---

## Phase 1 · Corpus Preparation & Tokenizer (`SimpleStories`)

Implemented in [src/tokenizer.py](src/tokenizer.py), [src/drive_corpus.py](src/drive_corpus.py), and `ensure_tokenized_corpus()` in [src/train.py](src/train.py):

1. **Deterministic Hash Split (`story_hash_v1`)**:
   - **Dataset Snapshot:** [`SimpleStories/SimpleStories`](https://huggingface.co/datasets/SimpleStories/SimpleStories) ([arXiv:2504.09184](https://arxiv.org/abs/2504.09184)), Hugging Face commit SHA **`e63b8adc3b1a1bdc7cac5b500d150b71346b0628`** (`2025-12-19 02:08:07 UTC`, `2,115,696` source `train` rows).
   - Splits only the published `train` split (`seed = 42`, `VALIDATION_FRACTION = 0.01`), producing **`2,094,643`** training stories (**`853,498,695`** packed tokens) and **`21,053`** validation stories (**`8,598,126`** packed tokens).
   - **Rule:** Keep the published `SimpleStories` `test` split strictly reserved for final evaluation (never touch it during corpus fitting, training, or hyperparameter tuning).
2. **Tokenizer Invariants**:
   - Byte-level BPE (`VOCAB_SIZE = 1,024`), keeping single digits (`0`–`9`) isolated as individual tokens so Phase 4/5 arithmetic reasoning does not fight multi-digit merges.
   - Verified EOS boundaries (`<|eos|>`) and SHA-256 checksums recorded in `manifest.json`.
3. **Google Drive Cache (`gdrive:MiniReasoner/data`)**:
   - Fresh VMs download and SHA-256 verify the `~1.7 GB` packed `tokenized/full-val-0.01/` directory automatically rather than re-tokenizing.

---

## Phase 2 · Pre-Training (`32.05M` RoPE Base Model)

### 2.1 Hyperparameters (`PRESETS["base"]` in [src/settings.py](src/settings.py))
- **Model:** 10 layers, 8 heads (`head_dim = 64`), `n_embd = 512`, `RoPE` (`theta = 10000.0`), `context_length = 1,024` (RoPE is built into pre-training from the start so context length can be extended up to **`2,048` tokens** if needed for SFT)
- **Batching:** `batch_size = 32` windows/step (`32,768` tokens/step) → **`26,047` optimizer steps** per 1-epoch pass (`~853.5M` prediction targets)
- **Optimizer & Schedule:** Fused `AdamW` (`betas=(0.9, 0.95)`, `weight_decay = 0.1` for 2D+ params), `warmup_steps = 700`, peak LR `0.00065` → cosine decay to `0.000065`
- **Validation & Checkpoints:** Every `500` steps (`eval_token_fraction = 0.1`), plus final step; full state (`model`, `optimizer`, `tokenizer_json`, `metrics`) queued asynchronously to Azure Blob Storage (`CHECKPOINT_LIMIT = None`).

### 2.2 Running Detached in `tmux` & Logging
Always launch training via [scripts/train.sh](scripts/train.sh) so SSH disconnects or laptop sleep never kill the job. Every run writes **one local log file** under `logs/` (`logs/train_<TIMESTAMP>_<PRESET>.log`, with no local aliases or symlinks), formats all losses to **4 decimal digits (`.4f`)**, and saves the log remotely in both **Azure Blob Storage** (`checkpoints/Shared/MiniReasoner/<run_id>/logs/train_<TIMESTAMP>_<PRESET>.log`) and **Databricks MLflow** (under Artifacts `logs/`, tagged via `log_location`):

```bash
# Start fresh 1-epoch base run inside tmux session 'train'
bash scripts/train.sh --preset base

# Detach anytime without stopping: Ctrl+B, then D
# Re-attach / stream / inspect GPU / stop cleanly (saves checkpoint before exit)
bash scripts/train.sh attach
bash scripts/train.sh logs
bash scripts/train.sh status
bash scripts/train.sh stop
```

### 2.3 Resuming After Spot Preemption or VM Restart
`EpochBatchSampler` deterministically reconstructs window permutations from `(seed, epoch)` and `step`, and `TrainingRun` skips already-recorded MLflow steps cleanly:

```bash
bash scripts/train.sh --preset base --resume --run-id <MLFLOW_RUN_ID>
```

### 2.4 Recorded Pre-Training Runs & Milestones
| Run ID | Dates | Architecture / Context | Steps | Final Train Mean | Final / Best Val Loss | Throughput (A30) | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| **`23a72f3aa8d6487ca401b92b22ae229c`** | 24–25 Nov 2026 | **32.05M `RoPE`, `1,024` ctx, `bs=32`** | **26,047** (1 epoch) | **`1.1231`** | **`1.1302`** (Step 25,000) / `1.1303` (Step 26,047) | ~137k–144k tok/s | Resumed cleanly from Azure Step `25,000` after VM restart at `25,240`; `853.5M` tok |

---

## Phase 3 · After Pre-Training (Evaluation, Baseline & Cleanup)

### 3.1 Evaluate & Select the Base Checkpoint ([notebooks/evaluation.ipynb](notebooks/evaluation.ipynb))
Run [notebooks/evaluation.ipynb](notebooks/evaluation.ipynb) locally (CPU FP32) or on GPU (CUDA BF16/FP32). It automatically:
1. Discovers active runs in Azure container `minireasoner` (`RUN_ID = None` selects the newest active run) and caches the chosen checkpoint under `checkpoints/<run_id>/`.
2. Inspects RoPE weights, parameter counts (`32,049,152`), and plots the convergence & learning rate curves.
3. Scores the model on **held-out SimpleStories `test` split stories**:
   - **Cross-entropy loss, Perplexity ($\exp(\mathcal{L})$), and Bits-Per-Byte (`BPB`)** (tokenizer-invariant metric for comparing against `4k`/`32k`/`50k`-vocab research models).
   - **4-Choice Story Cloze Accuracy (`story_mc_acc`)**: HellaSwag-style length-normalized log-likelihood scoring over real next sentences vs. 3 distractors from other held-out stories (`25.0%` random baseline).
   - **RoPE Position-Bucket Scaling (`1–256` / `257–512` / `513–1,024` tokens)** and long-prompt continuation across the `1,024`-token context window.
4. Runs qualitative story generation (greedy + nucleus sampling) across the 5 canonical prompts (e.g., Mia and the key / Lily / Leo) and measures inference throughput (`tok/s`).
5. **Freezes the Pre-SFT Arithmetic Baseline** (accuracy + format rate before any arithmetic training).

*(When comparing close candidates such as Step `25,000` at `1.1302` vs. Step `26,047` at `1.1303`, score both on identical held-out windows and prompts before locking the SFT initialization checkpoint.)*

### 3.2 Clean Up Deleted MLflow Runs in Azure
Deleting a run in the Databricks MLflow UI only marks it `deleted` in MLflow—it does not delete your private Azure blobs. Run the sync command after pruning failed/aborted runs in Databricks:

```bash
# Preview orphaned Azure blobs (safe read-only dry-run)
uv run python src/main.py cleanup

# Permanently remove Azure blobs whose Databricks MLflow run is deleted (interactive confirm)
uv run python src/main.py cleanup --delete
```

Before deleting any VM instance, also verify your local repo, remote `origin/main`, `.env`, Databricks run URL, and Azure checkpoint hash are synced, and preserve the chosen base checkpoint + `tokenizer_json`.

---

## Phase 4 · Arithmetic Supervised Fine-Tuning (SFT)

### 4.1 Problem Design & Leakage Prevention
- Start with addition and subtraction over nonnegative **1- and 2-digit operands** (covering zero, equal operands, carries, and borrows; nonnegative results first; track negatives, longer operands, multiplication, and short word problems separately later).
- Build answers using exact Python integer arithmetic.
- **Deduplicated Splits:** Split by the underlying canonical problem **before** generating wording variants: keep `7 + 8`, `8 + 7`, and `"What is seven plus eight?"` in the same split (preserve order for subtraction). Maintain three disjoint splits: SFT/RL train, development, and a sealed final test set (recording generator version, split seed, and category counts).
- Standardize prompt/target format, e.g. `What is 47 + 8?\nAnswer:` → `55<|eos|>`.

### 4.2 SFT Training Protocol
- Initialize from the chosen Phase 3 base checkpoint (`weights` + `tokenizer_json`); start a **fresh optimizer** and **new MLflow run** tagged with the parent base run ID and checkpoint step.
- **Optional Context Extension (`1,024 → 2,048`):** Because pre-training uses RoPE (`theta = 10000.0`) from the start, SFT can extend `context_length` up to **`2,048` tokens** if longer multi-step reasoning or chain-of-thought traces require extra headroom.
- Mask cross-entropy loss on the prompt tokens (`-100`) and compute loss **only on the answer + `<|eos|>`**.
- Mix or monitor a small SimpleStories retention slice alongside arithmetic SFT so narrative coherence does not collapse.
- Run a small pilot first; inspect failure modes (operand copying, missed carries/borrows, trailing gibberish past the number), expand demonstrations in weak categories, and repeat dev comparisons.
- **Promotion Gate:** Promote the best SFT checkpoint as the reference model for Phase 5. If SFT already achieves near-ceiling accuracy on 1–2 digits, escalate difficulty (e.g., 3-digit carries, chained operations, or short word problems) before spending GPU budget on RL.

---

## Phase 5 · Verifiable-Reward RL (GRPO)

- Initialize policy and frozen reference model from the selected Phase 4 SFT checkpoint ([GRPO paper](https://arxiv.org/abs/2402.03300)).
- Generate $G$ rollouts (e.g., $G = 4$) per training prompt.
- **Deterministic Verifier (`{0, 1}` reward):**
  - Award **`1.0`** only if the completion contains **strictly one valid integer** matching the ground-truth answer, cleanly terminated by `<|eos|>` (or agreed stop) with no extra guesses, conflicting numbers, or story drift.
  - Award **`0.0`** otherwise (explicitly test the verifier on `0`, negative signs if enabled, multiple numbers, whitespace, and empty outputs).
- Focus RL prompts on categories where SFT produces mixed groups (both right and wrong rollouts provide gradient variance). Keep KL regularization against the frozen SFT reference and monitor mean answer length, invalid-format rate, held-out greedy arithmetic accuracy, and story retention.
- **Promotion Gate:** Run paired greedy evaluation (`RL` vs. `SFT` on identical dev problems), counting wins (`RL fixed`) vs. regressions (`RL broke`) by category. Promote RL only if gains clearly exceed paired noise without degrading story quality beyond tolerance.

---

## Phase 6 · Final Release & Verification Checklist

1. Freeze the winning model config, weights, `tokenizer_json`, prompt template, decoding settings, and verifier.
2. Run **once** on:
   - The **sealed arithmetic test set** (exact-match accuracy, valid-format rate, per-category breakdown, generalization to unseen wordings/lengths, failure inspection).
   - The **published `SimpleStories` test split** (`test_loss`, `test_ppl`, `test_bpb`, `story_mc_acc`, and 5 canonical story prompts) to confirm zero catastrophic forgetting.
3. Verify offline CPU loading and deterministic greedy generation in [notebooks/evaluation.ipynb](notebooks/evaluation.ipynb) (`DEVICE_PREFERENCE = "cpu"`).
4. Record final release artifact URIs (Azure blob name + SHA-256, Databricks run ID + parent lineage, Google Drive corpus manifest) before pruning experimental checkpoints or deleting the GPU VM.

---

## Decision & Milestone Log

| Date / Milestone | Evidence Recorded | Decision |
| --- | --- | --- |
| 25 Nov 2026 | Run `23a72f3a...` (`1,024` ctx, `RoPE`) finished 1 full epoch (`26,047` steps): best val **`1.1302`** at Step `25,000`, final **`1.1303`** at Step `26,047` | Promote `RoPE` `1,024`-ctx base model (extensible up to `2,048` for SFT); evaluate Step `25,000` vs `26,047` in `evaluation.ipynb` |
| Phase 3 (Next) | Held-out SimpleStories test PPL/BPB, 4-choice Story Cloze accuracy, RoPE position-bucket scaling (`1–1,024`), and pre-SFT arithmetic baseline | Lock base checkpoint, clean up deleted MLflow Azure blobs (`c26cf6e5...`), and record pre-SFT baseline |
| Phase 4 (Planned) | Dev arithmetic accuracy by category + story retention after SFT pilot | Expand SFT demonstrations, move to GRPO pilot, or escalate problem difficulty |
| Phase 5–6 (Planned) | Paired GRPO vs. SFT dev wins/regressions, sealed arithmetic test, final CPU load check | Select final release checkpoint and archive provenance |
