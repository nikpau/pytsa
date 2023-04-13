import pytsa
from pathlib import Path

# Global geographic search area.
# Outside these bounds, no search will be commenced
frame = pytsa.BoundingBox(
    LATMIN = 52.2, # [°N]
    LATMAX = 56.9, # [°N]
    LONMIN = 6.3,  # [°E]
    LONMAX = 9.5,  # [°E]
)

# File containing AIS messages
sourcefile = Path("data/testdata.csv")

# Instantiate the search agent with the source file 
# and the search area
search_agent = pytsa.SearchAgent(sourcefile,frame)

# Provide a position and time for which the search
# will be carried out
tpos = pytsa.TimePosition(
    timestamp="2021-07-03T12:03:00.000Z",
    lat=52.245,
    lon=9.878
)

# Buffer AIS messages dependent on location
search_agent.init(tpos.position)

# Search for TargetVessels
target_ships = search_agent.get_ships(tpos)

# Extract the current position, speed and
# course for all found target vessels.
for ship in target_ships:
    ship.observe()