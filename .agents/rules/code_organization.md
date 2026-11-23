# Code Organization Rules

- **Source Code Directory**: All Python runtime source files must reside strictly inside `src/`.
- **CLI Entrypoint**: `src/main.py` serves as the primary command-line interface for training and runtime execution.
- **No Loose Scripts**: Do not place `.py` files in the repository root or `scripts/`.
- **Tests**: Keep automated test scripts strictly inside `tests/`.
- **Notebooks**: Keep exploratory and evaluation notebooks inside `notebooks/` (e.g., `notebooks/evaluation.ipynb`).
- **Configuration**: Rely exclusively on `.env` (with `RCLONE_CONFIG_BASE64`, `AZURE_STORAGE_CONNECTION_STRING`, `DATABRICKS_HOST`, `DATABRICKS_TOKEN`) for environment credentials. Do not require standalone `rclone.conf` files.
