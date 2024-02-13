import pytsa
from pathlib import Path
import matplotlib.pyplot as plt

# Global geographic search area.
# Outside these bounds, no search will be commenced
frame = pytsa.BoundingBox(
    LATMIN = 51.85,
    LATMAX = 60.49,
    LONMIN = 4.85,
    LONMAX = 14.3,
)

# File containing AIS messages
# Replace with the path to your own data ---v
dynamic_data = Path("path/to/dynamic_data.csv")
static_data = Path("path/to/static_data.csv")

# Instantiate the search agent with the source file 
# and the search area
search_agent = pytsa.SearchAgent(
    dynamic_paths=dynamic_data,
    static_paths=static_data,
    frame=frame,
    preprocessor= lambda df: df[df["speed"] > 0]
)

# Provide a position and time for which the search
# will be carried out
tpos = pytsa.TimePosition(
    timestamp="2021-04-01T08:00:00Z",
    lat=54.00,
    lon=8.26
)

# Track the target vessels for 100 iterations
# of 3 seconds each and print the observed
# position, speed and course
for _ in range(0,100):
    # Search for TargetVessels
    tpos.timestamp += 3 # 3 seconds
    target_ships = search_agent.freeze(tpos,search_radius=40)

    # Extract the current position, speed and
    # course for all found target vessels.
    for ship in target_ships.values():
        print(ship.observe())