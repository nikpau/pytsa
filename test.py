import pytsa
from pathlib import Path
import matplotlib.pyplot as plt

# Global geographic search area.
# Outside these bounds, no search will be commenced
frame = pytsa.LatLonBoundingBox(
    LATMIN = 51.85,
    LATMAX = 60.49,
    LONMIN = 4.85,
    LONMAX = 14.3,
)

# File containing AIS messages
dynamic_data = Path("data/dynamic_data.csv")
static_data = Path("data/static_data.csv")

# Instantiate the search agent with the source file 
# and the search area
search_agent = pytsa.SearchAgent(
    msg12318file=dynamic_data,
    msg5file=static_data,
    frame=frame,
    preprocessor= lambda df: df[df["speed"] > 0]
)

# Provide a position and time for which the search
# will be carried out
tpos = pytsa.TimePosition(
    timestamp="2021-07-01T00:00:00",
    lat=54.00,
    lon=7.6
)
search_agent.init(tpos)

r = search_agent.get_all_ships()

def plot_tracks(ships):
    
    f, ax = plt.subplots(1,1,figsize=(10,10))
    for ship in ships.values():
        for track in ship.tracks:
            ax.plot([m.lon for m in track], 
                    [m.lat for m in track], color="black", alpha=0.1)
    plt.show()
    
 #plot_tracks(r)

for _ in range(0,500):
    # Search for TargetVessels
    tpos.timestamp += 1 # 1 second
    target_ships = search_agent.get_ships(tpos,search_radius=40)

    # Extract the current position, speed and
    # course for all found target vessels.
    for ship in target_ships.values():
        print(ship.observe())