"""Display links to training runs in the Databricks MLflow UI."""

from html import escape
from urllib.parse import quote

from .metrics import databricks_host


def open_mlflow_dashboard(*, experiment_id: str | None = None, run_id: str | None = None) -> str:
    """Display a workspace, experiment or run link in the notebook."""
    from IPython.display import HTML, display

    if run_id and not experiment_id:
        raise ValueError("A Databricks run URL requires its experiment ID.")
    url = f"{databricks_host()}/ml/experiments"
    label = "Open Databricks experiments"
    if experiment_id:
        url += f"/{quote(experiment_id, safe='')}"
        label = "Open experiment in Databricks"
    if run_id:
        url += f"/runs/{quote(run_id, safe='')}"
        label = "Open training run in Databricks"
    display(HTML(f'<a href="{escape(url, quote=True)}" target="_blank" rel="noopener noreferrer">'
                 f'{label}</a>'))
    return url
