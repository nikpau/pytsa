from .tsea.search_agent import SearchAgent, TimePosition, TargetShip
from .structs import BoundingBox, ShipType
from .trajectories import Inspector, ExampleRecipe
from .decoder import decode_from_file, decode
from .visualization import register_plot_dir
from . import visualization

__version__ = "2.2.2"
__author__ = "Niklas Paulig <niklas.paulig@tu-dresden.de>"
print(f"You are using PyTSA version {__version__}")