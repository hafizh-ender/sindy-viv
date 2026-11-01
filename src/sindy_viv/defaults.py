"""Read-only numerical defaults distributed with the package.

Callers may supply their own configuration files. Built-in defaults never search
the working directory or depend on the surrounding repository layout.
"""
from pathlib import Path


def default_config_path(*parts: str) -> Path:
    """Return a bundled JSON preset, retaining its filename for model identities."""
    directory = Path(__file__).resolve().with_name('presets')
    path = directory.joinpath(*parts).resolve()
    if not path.is_relative_to(directory):
        raise ValueError('Default configuration must be inside the package presets.')
    if not path.is_file():
        raise FileNotFoundError(f'No bundled configuration: {"/".join(parts)}')
    return path
