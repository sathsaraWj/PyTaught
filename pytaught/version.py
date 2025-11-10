import os

try:
    import importlib.metadata as importlib_metadata
except ImportError:  # pragma: no cover
    import importlib_metadata  # type: ignore

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    try:
        import tomli as tomllib  # type: ignore
    except ModuleNotFoundError:
        tomllib = None


def _resolve_version() -> str:
    try:
        return importlib_metadata.version("pytaught")
    except importlib_metadata.PackageNotFoundError:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        pyproject_path = os.path.join(project_root, "pyproject.toml")
        if tomllib is not None and os.path.exists(pyproject_path):
            try:
                with open(pyproject_path, "rb") as f:
                    data = tomllib.load(f)
                return data.get("project", {}).get("version", "unknown")
            except Exception:
                pass
    return "unknown"


__all__ = ["__version__"]
__version__ = _resolve_version()

