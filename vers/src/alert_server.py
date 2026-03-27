"""Mock backend server for receiving VERS alerts.

Phase-1 Alignment
-----------------
- Communication layer: mock dispatch integration endpoint
- Methodology #7: Receives structured JSON alerts from the live pipeline
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from flask import Flask, jsonify, request

try:
    from rich import print as rprint
    from rich.panel import Panel
except Exception:  # pragma: no cover - optional dependency
    rprint = print
    Panel = None  # type: ignore[assignment]

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

app = Flask(__name__)

logging.getLogger("werkzeug").setLevel(logging.ERROR)


def _render_payload(payload: dict) -> None:
    if Panel is not None:
        body = (
            f"[bold red]Severity:[/bold red] {payload.get('Severity', 'Unknown')}\n"
            f"[bold cyan]Message:[/bold cyan] {payload.get('Message', 'No message provided')}\n\n"
            f"[dim]{json.dumps(payload, indent=2)}[/dim]"
        )
        rprint(Panel(body, title=f"VERS Alert: {payload.get('AlertID', 'UNKNOWN')}", border_style="red"))
    else:
        print(json.dumps(payload, indent=2))


@app.route("/alert", methods=["POST"])
def receive_alert():
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"status": "error", "message": "Empty JSON payload"}), 400

    _render_payload(payload)
    return jsonify({"status": "received", "message": "Alert processed by mock server"}), 200


def main() -> None:
    rprint("[bold blue]Starting mock VERS Alert Server at http://127.0.0.1:8000[/bold blue]")
    app.run(host="127.0.0.1", port=8000, debug=False)


if __name__ == "__main__":
    main()
