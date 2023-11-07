# Target ship search agent based on extracted AIS records

Simulation studies in maritime contexts often lack a suitable model for vessel traffic around the simulated vessel. This module aims at providing such a model, by using AIS records to extract the locations and trajectories of ships around a user-provided location.

## Installation
Install the package via pip:
```shell
$ pip install git+https://github.com/nikpau/pytsa
```

## Instantiation
The search agent must be instantiated with three components: Its _BoundingBox_, _msg12318files_ and _msg5files_:

- _BoundingBox_: Reference frame containing the spatial extent of the searchable area in degrees of latitude and longitude. 

    - If you want your output to be of the form [northing, easting, SOG, COG] you can call the `.to_utm()` method on the bounding box.

- _msg12318files_: File path to a `.csv` file containing dynamic AIS messages (Types 1,2,3 and 18 only) to consider for the search procedure. See the next section for details on the data structure.
- _msg5files_: File path to the corresponding `.csv` file contatining static AIS messages (message type 5)

Example instantiation for a small area in the North Sea:

```py
import pytsa
from pathlib import Path

# Lat-Lon Box with [lat,lon, SOG, COG] outputs
frame = pytsa.LatLonBoundingBox(
    LATMIN = 52.2, # [°N]
    LATMAX = 56.9, # [°N]
    LONMIN = 6.3,  # [°E]
    LONMAX = 9.5,  # [°E]
)

# UTM Box with [northing, easting, SOG, COG] outputs
frame = pytsa.LatLonBoundingBox(
    LATMIN = 52.2, # [°N]
    LATMAX = 56.9, # [°N]
    LONMIN = 6.3,  # [°E]
    LONMAX = 9.5,  # [°E]
).to_utm()

dynamic_data = Path("data/dynamic_test.csv")
static_data = Path(data/static_test.csv)

search_agent = pytsa.SearchAgent(
    msg12318file = dynamic_data,
    msg5file = static_data
    frame = frame
)
```

## Data structure
The AIS records must be present in the `.csv` file format and need to at least contain `timestamp,lat,lon,speed,course,rotation` fields, whereby:

- _timestamp_: ISO 8601 parseable date format (e.g. "_2021-07-03T00:00:00.000Z_")
- _lat_: Latitude as float (51.343)
- _lon_: Longitude as float (12.45)
- _speed_: Speed over ground (SOG) in knots
- _course_: Course over ground (COG) in degrees (0-360)
- _rotation_: Rate of rotation (-127 - 127)

Three lines of example data could look like this: 

| timestamp | MMSI | originator |	 speed | lon | lat | course |
| --- | --- | --- | --- | --- | --- | --- |
2021-07-03T00:00:01.000Z | 219020102|	SWE|	0|	12.312933|	56.125557|	24.8|
2021-07-03T00:00:01.000Z|	219017737|	DNK|	0.1|	12.056323|	55.836347|	29.5|
2021-07-03T00:00:01.000Z|	266474000|	SWE|	12.8|	9.406958|	58.19693|	121.7|

## Search

To commence a search for ships around a given location, the `SearchAgent` class must be initialized to this position. During initialization, the data will be loaded into the agent and the area around the agent is buffered. It is recommended to use a `TimePosition` object to store the position and time at which to start the search shall be commenced simultaneously. Example:

```py
from pytsa import TimePosition

tpos = TimePosition(
    timestamp="2021-07-03T12:03:00.000Z",
    lat=52.245,
    lon=9.878
)

search_agent.init(tpos)
```

After initialization, a search can be commenced by

```py
target_ships = search_agent.get_ships(tpos)
```
yielding a list of `TargetShip` objects (see `pytsa/targetship.py` for more information).
To get the current `Latitude, Longitude, SOG, COG, ROT dROT` for each `TargetShip` object at the provided timestamp, the `observe_at_query()` method is to be called on every `TargetShip`:

```py
for ship in target_ships:
    ship.observe_at_query()

# Example output for one ship
>>> np.array([52.232,9.847,12.34,223.4])
```

## Full example
```py
import pytsa
from pathlib import Path

# Global geographic search area.
# Outside these bounds, no search will be commenced
frame = pytsa.LatLonBoundingBox(
    LATMIN = 52.2, # [°N]
    LATMAX = 56.9, # [°N]
    LONMIN = 6.3,  # [°E]
    LONMAX = 9.5,  # [°E]
)

# File containing AIS messages
dynamic_data = Path("data/dynamic_test.csv")
static_data = Path(data/static_test.csv)

# Instantiate the search agent with the source file 
# and the search area
search_agent = pytsa.SearchAgent(
    msg12318file = dynamic_data,
    msg5file = static_data
    frame = frame
)

# Provide a position and time for which the search
# will be carried out
tpos = pytsa.TimePosition(
    timestamp="2021-07-03T12:03:00.000Z",
    lat=52.245,
    lon=9.878
)

# Buffer AIS messages dependent on location
search_agent.init(tpos)

# Search for TargetVessels
target_ships = search_agent.get_ships(tpos)

# Extract the current position, speed and
# course for all found target vessels.
for ship in target_ships:
    ship.observe_at_query()

# Example output for one ship
>>> np.array([52.232,9.847,12.34,223.4,2.5,0.3])
```
