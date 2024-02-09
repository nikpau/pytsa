from .tsea.search_agent import SearchAgent, TimePosition, TargetShip
from .structs import BoundingBox, ShipType
from .tsea.targetship import TrajectoryMatcher
from .trajectories import Inspector, ExampleRecipe
from .decode import decode_from_file, decode
from .visualization import register_plot_dir
from . import visualization
import tomli, pathlib
from . import __path__ as projpath

# Parse version from pyproject.toml
__ppj = pathlib.Path(projpath[0]).parent / "pyproject.toml"
with open(__ppj,"rb") as tml:
    __v = tomli.load(tml)["tool"]["poetry"]["version"]

__version__ = __v
__author__ = "Niklas Paulig <niklas.paulig@tu-dresden.de>"
print(__version__)