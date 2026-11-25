"""Phase 2 SimpleStories pre-training loop."""

from dataclasses import asdict
import math
from pathlib import Path
from time import perf_counter

import torch

from .common import (
    EpochBatchSampler,
    TokenStream,
    TrainConfig,
    configure_optimizer,
    evaluate,
    loss_for,
    open_run_log,
    require_training_gpu,
    resolve_precision,
    tracked_training_session,
)
from ..core.checkpoint import load_checkpoint
from ..core.model import GPT, GPTConfig
from ..core.tokenizer import verify_corpus_files
from ..data.corpus import ensure_tokenized_corpus
from ..infra.checkpoint_store import CheckpointStore
from ..infra.tracking import download_checkpoint
from ..settings import (
    AZURE_CONTAINER,
    DATASET_CACHE_DIR,
    DRIVE_DATA_ROOT,
    PRESETS,
    TOKENIZED_DATA_DIR,
    VALIDATION_FRACTION,
    VOCAB_SIZE,
)


def run_training(
    *,
    preset: str = "base",
    resume: bool = False,
    mlflow_run_id: str | None = None,
    stop_after: int | None = None,
    precision_request: str = "auto",
    seed: int = 42,
    dataset_name: str = "SimpleStories/SimpleStories",
    source_splits: dict[str, str] | None = None,
    azure_container: str | None = None,
    tokenizer_device: str = "auto",
    loss_decimals: int = 4,
    device: torch.device | None = None,
    logs_dir: Path | None = None,
) -> dict:
    """Run or resume Phase 2 autoregressive pre-training on packed SimpleStories."""
    with open_run_log(preset, logs_dir=logs_dir) as (log_path, log_lock, is_stop_requested):
        if preset not in PRESETS:
            raise ValueError(f"Unknown preset {preset!r}. Choose from: {list(PRESETS.keys())}")
        settings = PRESETS[preset]

        if azure_container is None:
            azure_container = AZURE_CONTAINER
        if device is None:
            device = require_training_gpu()
        precision = resolve_precision(precision_request, device)
        if resume and not mlflow_run_id:
            raise ValueError("Set mlflow_run_id when resuming a previous run.")

        tokenized_path = TOKENIZED_DATA_DIR
        manifest = ensure_tokenized_corpus(
            dataset_name=dataset_name,
            source_splits=source_splits,
            seed=seed,
            validation_fraction=VALIDATION_FRACTION,
            target_vocab_size=VOCAB_SIZE,
            cache_dir=DATASET_CACHE_DIR,
            tokenized_path=tokenized_path,
            drive_data_root=DRIVE_DATA_ROOT,
            tokenizer_device=tokenizer_device,
        )

        verify_corpus_files(tokenized_path, manifest)
        tokenizer_json = (tokenized_path / "tokenizer.json").read_text(encoding="utf-8")
        vocab_size = manifest["vocab_size"]

        model_config = GPTConfig(vocab_size=vocab_size, **settings["model"])
        head_dim = model_config.n_embd // model_config.n_head
        print(
            f"MiniReasoner {preset} model: vocab={model_config.vocab_size:,}, "
            f"layers={model_config.n_layer}, heads={model_config.n_head}, "
            f"n_embd={model_config.n_embd:,}, head_dim={head_dim}, "
            f"context_length={model_config.context_length:,}, pos_encoding={model_config.pos_encoding}",
            flush=True,
        )

        streams = {
            split: TokenStream(tokenized_path / f"{split}.bin", manifest[split]["tokens"], model_config.context_length)
            for split in ("train", "val")
        }

        training_settings = settings["training"].copy()
        epochs = training_settings.pop("epochs", None)
        sampler = EpochBatchSampler(streams["train"], training_settings["batch_size"], training_settings["seed"])
        if epochs is not None:
            if not isinstance(epochs, int) or epochs < 1:
                raise ValueError("epochs must be a positive integer")
            training_settings["steps"] = epochs * sampler.steps_per_epoch

        config = TrainConfig(**training_settings)
        val_data = streams["val"]
        val_targets_per_check = math.ceil(val_data.target_count * config.eval_token_fraction)
        stop = config.steps if stop_after is None else stop_after
        if not 1 <= stop <= config.steps:
            raise ValueError("stop_after must be between 1 and the configured total steps")

        train_data = streams["train"]
        scheduled_targets = sampler.targets_through(stop)
        print(
            f"Run schedule: batch_size={config.batch_size:,}, steps_per_epoch={sampler.steps_per_epoch:,}, "
            f"steps={stop:,} ({scheduled_targets / train_data.target_count:.4f} epochs)",
            flush=True,
        )

        torch.manual_seed(config.seed)
        model = GPT(model_config).to(device)
        optimizer = configure_optimizer(
            model,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            device=device,
        )

        run_identity = {
            "model_config": asdict(model_config),
            "train_config": asdict(config),
            "sampling": "shuffled_windows_v1",
            "data_manifest": manifest,
        }

        start = 0
        history: list[dict] = []
        if resume:
            print(f"Resuming run {mlflow_run_id} from Azure container {azure_container}...", flush=True)
            resume_store = CheckpointStore(azure_container, mlflow_run_id)
            with download_checkpoint(resume_store, alias="latest") as ckpt_path:
                saved = load_checkpoint(ckpt_path)
            if saved.get("version") != 1:
                raise ValueError(f"Unsupported checkpoint version: {saved.get('version')}")
            if any(saved.get(key) != value for key, value in run_identity.items()):
                raise ValueError("Azure checkpoint configuration or data differs from this notebook.")
            if saved["tokenizer_json"] != tokenizer_json:
                raise ValueError("Tokenizer in Azure checkpoint differs from disk corpus.")
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            start = int(saved["step"])
            history = saved["metrics"]
            del saved
            print(f"Restored Azure checkpoint from step {start:,}. Resuming to step {stop:,}.", flush=True)

        parameters = sum(p.numel() for p in model.parameters())
        print(
            f"Parameters: {parameters:,} | Runtime: {device.type} | {precision} | Steps {start + 1:,} to {stop:,}",
            flush=True,
        )

        train_loss = float("nan")
        val_loss = float("nan")
        last_step = start

        with tracked_training_session(
            run_params={"preset": preset, "device": device.type, "precision": precision, **run_identity},
            log_path=log_path,
            log_lock=log_lock,
            is_stop_requested=is_stop_requested,
            azure_container=azure_container,
            resume_run_id=mlflow_run_id if resume else None,
            tags={"stage": "pretrain"},
        ) as (session, check_stop):
            run = session.run
            store = session.store
            active_run_id = run.id

            if resume:
                store.prune()
                for row in history:
                    run.log(row, step=row["step"])
                if history:
                    val_loss = history[-1].get("val_loss", val_loss)
            else:
                print("Evaluating validation loss at step 0 before first training batch...", flush=True)
                initial_loss = evaluate(model, val_data, config, device, precision, step=0)
                val_loss = initial_loss
                history = [{"step": 0, "tokens": 0, "val_loss": initial_loss, "val_targets": val_targets_per_check}]
                print(f"Step 0 | mean train loss N/A | validation loss {initial_loss:.{loss_decimals}f}", flush=True)
                run.log(history[0], step=0)

            interval_start = perf_counter()
            interval_start_targets = sampler.targets_through(start)
            interval_loss_sum = 0.0

            for step in range(start + 1, stop + 1):
                if check_stop():
                    print(f"Stopping early as requested at step {last_step:,}.", flush=True)
                    break

                lr = config.lr_at(step)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.zero_grad(set_to_none=True)

                starts = sampler.starts(step)
                x, y = streams["train"].batch(starts, device)
                loss = loss_for(model, x, y, precision)
                tokens_seen = sampler.targets_through(step)
                batch_targets = tokens_seen - sampler.targets_through(step - 1)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                last_step = step

                train_loss = loss.detach().item()
                if not math.isfinite(train_loss):
                    raise FloatingPointError(f"Non-finite training loss at step {step}.")
                print(f"Step {step:,} | train loss {train_loss:.{loss_decimals}f}", flush=True)
                interval_loss_sum += train_loss * batch_targets

                metrics = {"step": step, "train_loss": train_loss}
                is_eval_step = (step % config.eval_every == 0) or (step == stop) or check_stop()
                if not is_eval_step:
                    run.log(metrics, step=step)
                    continue

                interval_targets = tokens_seen - interval_start_targets
                interval_mean = interval_loss_sum / interval_targets
                throughput = interval_targets / max(1e-6, perf_counter() - interval_start)
                val_loss = evaluate(model, val_data, config, device, precision, step=step)
                if not math.isfinite(val_loss):
                    raise FloatingPointError(f"Non-finite validation loss at step {step}.")

                print(
                    f"Step {step:,} | mean train loss {interval_mean:.{loss_decimals}f} | "
                    f"validation loss {val_loss:.{loss_decimals}f} | {round(throughput):,} tok/s",
                    flush=True,
                )
                metrics.update({
                    "tokens": tokens_seen,
                    "epoch": tokens_seen / streams["train"].target_count,
                    "train_loss_interval_mean": interval_mean,
                    "val_loss": val_loss,
                    "val_targets": val_targets_per_check,
                    "lr": lr,
                    "train_tokens_per_second": round(throughput),
                })
                run.log(metrics, step=step)
                history.append(metrics)

                print(f"Saving checkpoint snapshot at step {step:,}...", flush=True)
                session.save_checkpoint(
                    {
                        "version": 1,
                        **run_identity,
                        "step": step,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "tokenizer_json": tokenizer_json,
                        "metrics": history,
                    },
                )
                interval_loss_sum = 0.0
                interval_start_targets = tokens_seen
                interval_start = perf_counter()

                if check_stop():
                    print(f"Stopping early after saving checkpoint at step {last_step:,}.", flush=True)
                    break

            status = "Completed" if not check_stop() and last_step >= config.steps else "Stopped"
            print(
                f"Training {status} at step {last_step:,}. Final train loss: {train_loss:.{loss_decimals}f}, "
                f"val loss: {val_loss:.{loss_decimals}f}",
                flush=True,
            )

        return {
            "status": status,
            "step": last_step,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "mlflow_run_id": active_run_id,
            "log_path": str(log_path),
        }
