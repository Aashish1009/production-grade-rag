"""HTTP API and bundled web UI.

The pipeline is a library first -- the graphs in :mod:`rag.graphs` are ordinary
callables -- and this package is one client of it. It exists because the
embedded Qdrant store is single-process: the only safe way to drive the
pipeline from a browser is for one process to own the store and expose it over
localhost, which is what :mod:`rag.api.app` does. The former Typer CLI was
deleted rather than kept alongside, because a second process would deadlock on
the store's file lock.

Nothing is re-exported here on purpose: importing this package should not pull
in the server, so a script that wants only the graphs pays nothing for it.
"""
