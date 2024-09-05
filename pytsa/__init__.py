from .logger import logger

from .tsea.search_agent import SearchAgent, TimePosition, TargetShip
from .tsea.split import TREXMethod
from .structs import BoundingBox, ShipType
from .trajectories import Inspector, ExampleRecipe
from .decoder import decode_from_file, decode
from .visualization import register_plot_dir
from . import visualization

__version__ = "2.3.13"
__author__ = "Niklas Paulig <niklas.paulig@tu-dresden.de>"
logger.info(f"You are using PyTSA version {__version__}")