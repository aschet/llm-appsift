# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT
"""Runtime configuration, assembled from command-line arguments and the environment."""
from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_OUTPUT = "report.html"


def data_dir(output: str | os.PathLike) -> Path:
    """Where the records for a report belong: beside it, named after it.

    A report and the applications it was made from are one result, so they are kept
    together and two reports do not share a store. `report.html` puts them in
    `report_data`. A report already given as `report_data/report.html` is already
    inside that directory, so it is reused rather than nested again -- otherwise
    `report_data/report_data` is the same mistake typed out.
    """
    output = Path(output)
    want = f"{output.stem}_data"
    if output.parent.name == want:
        return output.parent
    return output.parent / want


def normalise_host(host: str) -> str:
    host = host.rstrip("/")
    if not host.startswith(("http://", "https://")):
        host = "http://" + host
    return host


@dataclass
class Config:
    host: str = DEFAULT_HOST
    results_dir: Path = None  # type: ignore[assignment]
    models: list[str] = field(default_factory=list)
    timeout: int = 2400

    def __post_init__(self) -> None:
        self.host = normalise_host(self.host)
        self.results_dir = Path(self.results_dir if self.results_dir is not None
                                else data_dir(DEFAULT_OUTPUT))

    @property
    def is_remote(self) -> bool:
        return not any(h in self.host for h in ("localhost", "127.0.0.1", "::1"))

    def path(self, name: str) -> Path:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        return self.results_dir / name

    def resolve_models(self) -> list[str]:
        """Explicit models if given, otherwise everything installed on the server."""
        if self.models:
            return list(self.models)
        return installed_models(self.host)


def installed_models(host: str) -> list[str]:
    with urllib.request.urlopen(normalise_host(host) + "/api/tags", timeout=30) as r:
        return sorted(m["name"] for m in json.loads(r.read())["models"])


def read_model_file(path: str | os.PathLike) -> list[str]:
    """One model per line; blank lines and # comments ignored."""
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.append(line)
    return out
