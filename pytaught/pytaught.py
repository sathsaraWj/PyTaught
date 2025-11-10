"""
Legacy module that exposes the CLI entrypoint for backward compatibility.

The implementation now lives in more modular files inside the pytaught package.
"""

from .cli import main  # noqa: F401

__all__ = ["main"]


if __name__ == "__main__":
    main()
