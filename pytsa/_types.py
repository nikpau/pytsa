from .tsea.search_agent import TargetShip
from .structs import AISMessage

# Type aliases
MMSI = int
Targets = dict[MMSI,TargetShip]
Track = list[AISMessage]
