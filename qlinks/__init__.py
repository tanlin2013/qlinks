import logging as _logging
from importlib import metadata as _metadata

# -- Version -----------------------------------------------------------------

try:
    __version__ = _metadata.version(__name__)
except _metadata.PackageNotFoundError:  # pragma: no cover - source-tree fallback
    __version__ = "0+unknown"

# -- Define logger and the associated formatter and handler ------------------

_formatter = _logging.Formatter(
    "%(asctime)s [%(filename)s] %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

_handler = _logging.StreamHandler()
_handler.setLevel(_logging.INFO)
_handler.setFormatter(_formatter)

logger = _logging.getLogger("qlinks")
logger.setLevel(_logging.INFO)
logger.addHandler(_handler)

__all__ = [
    "__version__",
    "logger",
]
