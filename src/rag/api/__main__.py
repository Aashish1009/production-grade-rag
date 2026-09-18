"""Server launcher: ``rag-server`` / ``python -m rag.api``.

Kept deliberately thin -- everything it does is configure uvicorn and hand it
the app from :func:`rag.api.app.create_app`. The previous console script was
a full Typer command set; this one starts a server and has no commands,
because the UI is the interface now.
"""

from __future__ import annotations

import uvicorn

from rag.api.app import create_app
from rag.config import get_settings


def main() -> None:
    settings = get_settings()
    # reload is off and workers is left at its default of 1: the embedded
    # Qdrant store allows exactly one process, so a second worker would fail
    # to open it rather than share it.
    uvicorn.run(
        create_app(settings),
        host=settings.api_host,
        port=settings.api_port,
        log_config=None,  # let setup_logging own the format
    )


if __name__ == "__main__":  # pragma: no cover
    main()
