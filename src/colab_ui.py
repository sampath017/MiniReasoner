"""Show the notebook's MLflow dashboard through Colab, JarvisLabs, or an SSH tunnel."""

import atexit
from html import escape
import os
import re
import signal
import socket
import subprocess
import sys
from tempfile import NamedTemporaryFile
from time import monotonic, sleep
from urllib.parse import urlsplit, urlunsplit
from urllib.request import urlopen


_servers: dict[tuple[str, bool, str], tuple[subprocess.Popen, str]] = {}


def _stop(process: subprocess.Popen) -> None:
    if process.poll() is None:
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
        except ProcessLookupError:
            pass


def _serve(tracking_uri: str, *, in_colab: bool) -> str:
    origin = "" if in_colab else os.getenv("MINIREASONER_JARVIS_ORIGIN", "").strip()
    if origin:
        # Host allowlists accept wildcards and commas. Accept only a single literal DNS host.
        match = re.fullmatch(r"https://(?:[a-zA-Z0-9-]+\.)+[a-zA-Z0-9-]+(?::(\d{1,5}))?/?", origin)
        if not match or (match[1] is not None and not 1 <= int(match[1]) <= 65535):
            raise ValueError("Set MINIREASONER_JARVIS_ORIGIN to the HTTPS origin of the notebook, "
                             "for example https://INSTANCE.notebooksn.jarvislabs.net")
        origin = origin.rstrip("/").lower()
    key = (tracking_uri, in_colab, origin)
    previous = _servers.get(key)
    if previous is not None and previous[0].poll() is None:
        return previous[1]

    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", 0 if in_colab else 5000))
        except OSError:
            # Leave an existing service on port 5000 alone.
            sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://localhost:{port}"
    command = [
        sys.executable, "-m", "mlflow", "server", "--host", "127.0.0.1",
        "--port", str(port), "--workers", "1",
    ]
    if in_colab:
        from google.colab import output

        url = output.eval_js(f"google.colab.kernel.proxyPort({port})")
    elif origin:
        # Jupyter strips /proxy/PORT/ before forwarding. The trailing slash keeps the
        # browser's relative MLflow asset and API requests under the same proxy.
        url = f"{origin}/proxy/{port}/"
    if in_colab or origin:
        proxy = urlsplit(url or "")
        if proxy.scheme != "https" or not proxy.hostname:
            raise RuntimeError("Could not determine the HTTPS proxy URL for this notebook.")
        command.extend([
            "--allowed-hosts", f"127.0.0.1:*,localhost:*,{proxy.netloc}",
            "--cors-allowed-origins", f"{proxy.scheme}://{proxy.netloc}",
        ])
    # The backend URI can include credentials. Pass it through the environment.
    with NamedTemporaryFile(prefix="minireasoner-mlflow-", suffix=".log", delete=False) as log:
        process = subprocess.Popen(
            command, env={**os.environ, "MLFLOW_BACKEND_STORE_URI": tracking_uri},
            stdout=log, stderr=subprocess.STDOUT, start_new_session=os.name == "posix",
        )
    try:
        deadline = monotonic() + 60
        while process.poll() is None and monotonic() < deadline:
            try:
                with urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as response:
                    if response.status == 200 and process.poll() is None:
                        _servers[key] = (process, url)
                        atexit.register(_stop, process)
                        return url
            except OSError:
                pass
            sleep(0.3)
        raise RuntimeError(f"MLflow did not start. See the runtime log at {log.name}.")
    except BaseException:
        _stop(process)
        raise


def open_mlflow_dashboard(tracking_uri: str, *, in_colab: bool) -> str | None:
    """Display a new-tab link to the dashboard for the notebook's tracking backend."""
    from IPython.display import HTML, display

    parts = urlsplit(tracking_uri)
    if parts.scheme in {"http", "https"} and parts.hostname:
        # Only the browser-facing location belongs in saved notebook outputs.
        url = urlunsplit((parts.scheme, parts.netloc.rsplit("@", 1)[-1], parts.path, "", ""))
    elif parts.scheme.split("+")[0] in {"", "file", "sqlite", "postgresql", "mysql", "mssql"}:
        url = _serve(tracking_uri, in_colab=in_colab)
    else:
        print("Open the dashboard supplied by your configured MLflow tracking service.")
        return None
    dashboard = urlsplit(url)
    if not in_colab and dashboard.hostname in {"localhost", "127.0.0.1", "::1"}:
        port = dashboard.port or (443 if dashboard.scheme == "https" else 80)
        print(f"VS Code Remote SSH: forward port {port} in the Ports panel. "
              "If the local port differs, open the Local Address shown there.")
    display(HTML(f'<a href="{escape(url, quote=True)}" target="_blank" rel="noopener noreferrer">'
                 'Open MLflow dashboard in a new tab</a>'))
    return url


def open_colab_dashboard(tracking_uri: str) -> str | None:
    return open_mlflow_dashboard(tracking_uri, in_colab=True)
