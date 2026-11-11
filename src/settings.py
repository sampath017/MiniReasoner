from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATASET_CACHE_DIR = DATA_DIR / "raw"
TOKENIZED_DATA_DIR = DATA_DIR / "tokenized"
SIMPLESTORIES_TRAIN_FILES = "data/train-*.parquet"

RUN_DIR = PROJECT_ROOT / "models" / "simplestories"
CHECKPOINT_PATH = RUN_DIR / "base.pt"

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
            "steps": 1500,
            "warmup_steps": 50,
            "learning_rate": 0.0005,
            "min_learning_rate": 0.00005,
            "weight_decay": 0.1,
            "eval_every": 100,
            "eval_windows": 64,
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
            "eval_windows": 32,
            "seed": 42,
        },
    },
}
