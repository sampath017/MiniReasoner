from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
VALIDATION_FRACTION = 0.01
DATASET_CACHE_SUBDIR = Path("data") / "raw"
TOKENIZED_DATA_SUBDIR = Path("data") / "tokenized" / f"full-val-{VALIDATION_FRACTION}"
DATASET_CACHE_DIR = PROJECT_ROOT / DATASET_CACHE_SUBDIR
TOKENIZED_DATA_DIR = PROJECT_ROOT / TOKENIZED_DATA_SUBDIR
SIMPLESTORIES_TRAIN_FILES = "data/train-*.parquet"

MLFLOW_EXPERIMENT = "minireasoner"
MLFLOW_DB_SUBPATH = Path("runs") / "mlflow.db"
RCLONE_CHECKPOINT_ROOT = "gdrive:MiniReasoner/checkpoints"
CHECKPOINT_LIMIT: int | None = None  # Keep every checkpoint unless a limit is set.
VOCAB_SIZE = 1_024


PRESETS = {
    "base": {
        "model": {
            "context_length": 256,
            "n_layer": 10,
            "n_head": 8,
            "n_embd": 512,
        },
        "training": {
            "batch_size": 64,
            "epochs": 1,
            "warmup_steps": 50,
            "learning_rate": 0.0005,
            "min_learning_rate": 0.00005,
            "weight_decay": 0.1,
            "eval_every": 500,
            "eval_token_fraction": 0.5,
            "seed": 42,
        },
    },
    "small": {
        "model": {
            "context_length": 64,
            "n_layer": 2,
            "n_head": 4,
            "n_embd": 64,
        },
        "training": {
            "batch_size": 32,
            "steps": 60,
            "warmup_steps": 5,
            "learning_rate": 0.001,
            "min_learning_rate": 0.0001,
            "weight_decay": 0.1,
            "eval_every": 20,
            "eval_token_fraction": 0.5,
            "seed": 42,
        },
    },
}
