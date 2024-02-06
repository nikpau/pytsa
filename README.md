# Python Trajectory Search Agent (PyTSA) for raw AIS records

This module provides a set of functionalities around Automatic Identification System (AIS) messages, such as

- Decoding raw AIS messages
- Extracting clean, practical and interpolated trajectories from the data, based on various (also user-defined) filters.
- Providing an easy-to-use interface for observing target ships and their state around a given time and position. 

## Motivation

Simulation studies in maritime contexts often lack an easy-to-use model for vessel traffic extraction around a simulated vessel, as the large amounts of AIS records make it challenging and time-consuming to pinpoint the exact vessels to be monitored. 

Also, for validating path-following, or collision avoidance systems, it is beneficial to use real-world trajectories, as they often provide a larger variety of movement patterns than simulated trajectories.
However, the exact process of extracting the relevant information from the raw AIS data is often not sufficiently documented, thus making it difficult to reproduce the results.

Therefore, this module aims to provide a unified, open-source solution for extracting relevant trajectories from raw AIS data, as well as providing an easy-to-use interface for observing target ships around a given position and time.

## Installation
Install the package via pip:
```shell
$ pip install pytsa-ais
```

## Usage
### Raw AIS data
One file of raw AIS records must only contain dynamic AIS messages (Types 1,2,3 and 18) or static AIS messages (Type 5). A combination of both is not supported. The data must be provided in the `.csv` format and *must be named YYYY_MM_DD.csv*. Other file names are not supported.
> This is done to avoid extra-day sorting of the data, which would be necessary if the data was not sorted by date. Intra-day sorting is done regardless of the file name.

Individual files must contain the following columns:

- _timestamp_: ISO 8601 parseable date format (e.g. "_2021-07-03T00:00:00.000Z_")
- _message_id_: AIS message type (1,2,3,5,18)

For dynamic AIS messages (Types 1,2,3,18) additionally

- _raw_message_: For messages of type 1,2,3,18, the raw message consists of a single AIVDM sentence. 

For static AIS messages (Type 5) additionally:

- _raw_message1_: First AIVDM sentence
- _raw_message2_: Second AIVDM sentence

#### Example Table for dynamic AIS messages
| timestamp | message_id | raw_message |
|-----------|---------|-------------|
| 2021-07-03T00:00:00.000Z | 1 | "!ABVDM,1,1,,B,177PhT001iPWhwJPsK=9DoQH0<>i,0*7C"

#### Example Table for static AIS messages
| timestamp | message_id | raw_message1 | raw_message2 |
|-----------|---------|-------------|-------------|
| 2021-07-03T00:00:00.000Z | 5 | "!ABVDM,2,1,5,A,53aQ5aD2;PAQ0@8l000lE9LD8u8L00000000001??H<886?80@@C1F0CQ4R@,0*35" | "!ABVDM,2,2,5,A,@0000000000,2*5A"

For more information on the AIS message structure, see [here](https://gpsd.gitlab.io/gpsd/AIVDM.html).

#### Decoding AIS messages

Once your raw AIS data is in the correct format, you can decode the AIS messages by calling the `decode()` function. The function takes as arguments the path to a directory containing the raw AIS data, as well as the path to the output directory. The function will then decode all `.csv` files in the input directory and save the decoded data to the output directory under the same file name.

```py
from pytsa import decode

decode(
    source = "path/to/raw_dir",
    dest = "path/to/decoded_dir",
    njobs = 1
)
```

> For decoding AIS messages, you can choose between single-processing and multi-processing decoding. The multi-processing decoding is recommended for large datasets containing multiple files, as it is significantly faster than single-process decoding. However, during decoding, the files are loaded into memory in their entirety, which may lead to memory issues for large datasets or a large number of jobs.  Therefore, it is recommended to use single-processing decoding for smaller datasets or if you encounter memory issues. Parallel decoding may also not be avialable on Windows systems (due to the lack of testing on Windows systems, this is not guaranteed, sorry...)

### Decoded AIS data
In case you already have decoded AIS messages, you have to make sure, that the fields of your `.csv` file at least partially match `Msg12318Columns` and `Msg5Columns` at `pytsa/decode/filedescriptor.py`. 

In case you have a different data structure, you can either adapt the `Msg12318Columns` and `Msg5Columns` classes, or you can adapt the column names of your `.csv` file to match the column names of the `Msg12318Columns` and `Msg5Columns` classes.

## Using the `SearchAgent` for extracting target ships

The central object of the module is the `SearchAgent` class, which provides an easy-to-use interface for extracting target ships around a given position and time.

Possible applications include:

- Tracking traffic around a simulated route
- Monitoring traffic around a fixed location
- Extracting trajectories 

The Search Agent must be instantiated with three components: Its _BoundingBox_, _msg12318files_ and _msg5files_:

- _BoundingBox_: Reference frame containing the spatial extent of the searchable area in degrees of latitude and longitude. 

- _msg12318files_: File path to a `.csv` file containing *decoded* dynamic AIS messages (Types 1,2,3 and 18 only) to consider for the search procedure. See the next section for details on the data structure.
- _msg5files_: File path to the corresponding `.csv` file containing *decoded* static AIS messages (message type 5)

Example instantiation for a small area in the North Sea:

```py
import pytsa
from pathlib import Path

# Lat-Lon Box with [lat,lon, SOG, COG] outputs
frame = pytsa.BoundingBox(
    LATMIN = 52.2, # [°N]
    LATMAX = 56.9, # [°N]
    LONMIN = 6.3,  # [°E]
    LONMAX = 9.5,  # [°E]
)

dynamic_data = Path("/path/to/dynamic.csv")
static_data = Path("/path/to/static.csv")

search_agent = pytsa.SearchAgent(
    msg12318file = dynamic_data,
    msg5file = static_data
    frame = frame
)
```
### Monitoring vessel traffic around a given position

To commence a search for ships around a given location, it is mandatory to use a `TimePosition` object to store the position and time at which the search shall be commenced simultaneously. Example:

```py
from pytsa import TimePosition

tpos = TimePosition(
    timestamp="2021-07-03T12:03:00.000Z",
    lat=52.245,
    lon=9.878
)
```

After defining a TimePosition, a search can be commenced by freezing the search agent at the given position and time 

```py
target_ships = search_agent.freeze(tpos)
```
yielding a list of `TargetShip` objects (see `pytsa/targetship.py` for more information).

By default, the resulting TargetShip objects used linear interpolation to estimate the current position, speed and course of the target ships. If instead, cubic spline interpolation is desired, the interpolation option can be set to `spline`. Additionally, the `search_radius` can be set to a custom value in nautical miles.

```py
target_ships = search_agent.freeze(
    tpos, 
    interpolation="spline", 
    search_radius=5 # [nm]
)
```

To get the current `Latitude, Longitude, SOG, COG` for each `TargetShip` object at the provided timestamp, the `observe()` method can be used, returning a numpy array with the current position, speed and course.

```py
for ship in target_ships:
    ship.observe()

# Example output for one ship
# 
# Interpolated COG ---------------
# Interpolated SOG -----------    |
# Interpolated Longitude-|   |    |
# Interpolated Latitude  |   |    |
#                v       v   v    v
>>> np.array([52.232,9.847,12.34,223.4])
```

### Full example
```py
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
dynamic_data = Path("/path/to/dynamic.csv")
static_data = Path("/path/to/static.csv")

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

# Search for TargetVessels with 
# default settings: 
#   Linear interpolation, 
#   20 nm search radius
target_ships = search_agent.freeze(tpos)

# Extract the current position, speed and
# course for all found target vessels.
for ship in target_ships:
    ship.observe()

# Example output for one ship
>>> np.array([52.232,9.847,12.34,223.4])
```
### Extracting trajectories 

If instead of observing target ships around a given position, you want to extract trajectories from the data, you can use the `SearchAgent.extract_all()`.

By default, the `extract_all()` method walks through the entire dataset and extracts all trajectories that are within the search area utilizing the split-point approach from Section 4 in our original paper. The method returns a dictionary with the MMSI as keys and the corresponding `TargetShip` objects as values.
    
```py
all_ships = search_agent.extract_all()
```

To skip the split-point approach you can set the `skip_tsplit` parameter to `True`. This will result in TargetShip objects that only contain a single trajectory, which is the raw, time-ordered set of AIS messages for the given MMSI. 

```py
all_ships = search_agent.extract_all(skip_tsplit=True)
```
> The `extract_all()` method used 4-core parallel processing by default. This can be adjusted by setting the `njobs` parameter to a custom value. Note, that high `njobs` values may lead to a slowdown due to the overhead of splitting the data into chunks and reassembling the results.

The trajectories of each `TargetShip` object can be accessed by the `tracks` attribute, which is of type `list[Track]`. Each `Track` within the `tracks` list contains the AIS messages for a single trajectory. 
See the `pytsa.structs.AISMessage` module for more information on the fields of the AIS messages.

```py
# Example for printing the positions for each trajectory
for ship in all_ships.values():
    for track in ship.tracks:
        for msg in track:
            print(msg.lat, msg.lon)
```

The trajectories extracted via the `extract_all()` method are not interpolated by default. To manually interpolate them, you can use the `interpolate()` method of the `TargetShip` object. 

```py
for ship in all_ships.values():
    ship.interpolate(mode="linear") # or "spline"
```
Refer also to the function documentation for further details.

#### Refining trajectories using the `Inspector` class

Once the `TargetShips` with its trajectories are extracted, PyTSA provides a flexible interface for refining the trajectories using the `Inspector` class. The output of the Inspector is two dictionaries [_accepted_,_rejected_], of type `dict[MMSI,TargetShip]`. The first dictionary contains the TargetShip objects that passed the inspection, while the second dictionary contains the TargetShip objects that failed the inspection.
> Note: It is possible that the same MMSI is present in both dictionaries. If so, the TargetShip object in the _rejected_ dictionary will contain only rejected trajectories, while the TargetShip object in the _accepted_ dictionary will contain only accepted trajectories.

The `Inspector` works with a set of _rules_, that must be combined into a `Recipe` object, which is then passed to the Inspector object. 

Before we show an example, let's explain the concept of _rules_ and _recipes_:

##### Rules

A _rule_ is a function following the signature `rule(track: Track) -> bool`. It takes a single `Track` object as input and returns a boolean value. _Rules_ are set to act as a negative filter, meaning that if a _rule_ returns _True_, the corresponding `Track` will be _removed_ from the `TargetShip` object.
> It is possible for rules to have more than one argument, like `rule(track: Track, *args, **kwargs) -> bool`, however, for constructing a recipe, all other arguments must be pre-set, for example by using a lambda function, or the `functools.partial` function.

A simple rule that removes all tracks with less than 10 AIS messages would look like this:

```py
from pytsa import Track

def track_too_short(track: Track) -> bool:
    return len(track) < 10
```
A rule filtering trajectories whose latitude is outside given bounds could look like this:

```py
def lat_outside_bounds(track: Track, latmin: float, latmax: float) -> bool:
    return any([msg.lat < latmin or msg.lat > latmax for msg in track])
```

Feel free to define your own rules, or use the ones provided in the `pytsa.trajectories.rules` module.

##### Recipes

A _recipe_ is a list of _rules_ that are combined into a single function using the `Recipe` class.

```py
from pytsa import Recipe
from fuctools import partial

# Create a recipe with the 
# two rules defined above.
recipe = Recipe(
    track_too_short,
    partial(lat_outside_bounds, latmin=52.2, latmax=56.9)
)
```
#### Applying the recipe to the `Inspector`

Once the recipe is created, it can be passed to the `Inspector` object, which will then apply the recipe to the `TargetShip` objects, filtering out the trajectories that do not pass the rules.

```py
from pytsa import Inspector

inspector = Inspector(all_ships, recipe)
accepted, rejected = inspector.inspect()
```

## Visualizing AIS data

This module provides various functions for visualizing AIS data. Currently, there exist two groups of functions:

- Functions for visualizing the empirical distribution functions used in the split-point approach in the original paper. These functions are located in the `pytsa.visualization.ecdf` module. They are intended to both make the results of the split-point approach more transparent and to provide a tool for adapting the split-point approach to different datasets.

- Miscellaneous functions can be found in the `pytsa.visualization.misc` module. Currently, the following functionalities are provided:

    - Plotting the trajectories on a map
    - Comparing trajectories based on different σ_ssd ranges (see Figure 12 in the original paper)
    - Plotting all trajectories as a heatmap
    - Generating a pixel map of average smoothness as a function of the number of messages in a trajectory and the spatial standard deviation of the trajectory (Figure 14 in the paper)

## Issues and Contributing

Currently, this project is developed by a single person and is therefore not thoroughly tested.

If you encounter any issues or have any suggestions for improvements, you are invited to open an issue or a pull request.

## Citation

If you use this module in your research, please consider citing this repository as follows:

```
@misc{pytsa2024,
  author = {Paulig, Niklas},
  title = {{PyTSA}: A Python Module for Extracting Trajectories from Raw AIS Data},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/nikpau/pytsa}},
}
```

## Appendix

### Split-point procedure

The split-point procedure takes place in the `pytsa.tsea.split` module. Its main function, `is_split_point()`, will be called on every pair of AIS messages in the dataset. The function returns a boolean value, indicating whether the pair of messages is a split point or not. 

In case you want to adapt the split-point procedure to your dataset, you can use the `pytsa.tsea.split` module as a starting point.