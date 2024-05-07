# Append shared library path to sys.path
from . import __path__ as _PYTSA_DIR
from .logger import logger
import sys
from pathlib import Path
sharedpath = Path(_PYTSA_DIR[0]) / "shared"
logger.info(f"Loading shared libraries")
sys.path.append(sharedpath.as_posix())
del sharedpath, Path, _PYTSA_DIR

from .tsea.search_agent import SearchAgent, TimePosition, TargetShip
from .structs import BoundingBox, ShipType
from .trajectories import Inspector, ExampleRecipe
from .decoder import decode_from_file, decode
from .visualization import register_plot_dir
from . import visualization


__version__ = "2.3.4"
__author__ = "Niklas Paulig <niklas.paulig@tu-dresden.de>"
logger.info(f"You are using PyTSA version {__version__}")