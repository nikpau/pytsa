"""
Utility functions for pytsa
"""
import math
from pathlib import Path
import vincenty as _vincenty
import ciso8601

def m2nm(m: float) -> float:
    """Convert meters to nautical miles"""
    return m/1852

def nm2m(nm: float) -> float:
    """Convert nautical miles to meters"""
    return nm*1852

def s2h(s: float) -> float:
    """Convert seconds to hours"""
    return s/3600

def mi2nm(mi: float) -> float:
    """Convert miles to nautical miles"""
    return mi/1.151

def vincenty(lon1, lat1, lon2, lat2, miles = True) -> float:
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees)
    """
    p1 = (lat1,lon1)
    p2 = (lat2,lon2)
    return _vincenty.vincenty_inverse(p1,p2,miles=miles)

def haversine(lon1, lat1, lon2, lat2, miles = True):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = (
        math.sin(dlat/2)**2 + math.cos(lat1) * 
        math.cos(lat2) * math.sin(dlon/2)**2
        )
    c = 2 * math.asin(math.sqrt(a)) 
    r = 3956 if miles else 6371 # Radius of earth in kilometers or miles
    return c * r

def greater_circle_distance(lon1, lat1, lon2, lat2, miles = True, method = "haversine"):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees)
    """
    assert method in ["haversine", "vincenty"], "Invalid method: {}".format(method)
    if method == "haversine":
        return haversine(lon1, lat1, lon2, lat2, miles=miles)
    return vincenty(lon1, lat1, lon2, lat2, miles=miles)

def heading_change(h1,h2):
    """
    calculate the change between two headings
    such that the smallest angle is returned.
    """

    diff = abs(h1-h2)
    if diff > 180:
        diff = 360 - diff
    if (h1 + diff) % 360 == h2:
        return diff
    else:
        return -diff

def _date_transformer(datefile: Path) -> float:
    """
    Transform a date string to a float.

    Parameters
    ----------
    date : str
        The date string to transform.

    Returns
    -------
    float
        The date as a float.
    """
    return ciso8601.parse_datetime(datefile.stem.replace("_", "-"))

def align_data_files(dyn: list[Path], stat: list[Path]) -> bool:
    """
    Align dynamic and static data files by sorting them
    according to their date, and removing all files that
    are not present in both lists.

    This function assumes that the dynamic and static data files
    are named according to the following convention:
    - dynamic data files: YYYY_MM_DD.csv
    - static data files: YYYY_MM_DD.csv
    
    In case your input data files are not named according to this
    convention, you are advised to either rename them accordingly
    or adapt the `_date_transformer` function, which is used to
    sort the files by date.
    """
    if len(dyn) != len(stat):

        print(
            "Number of dynamic and static messages do not match."
            f"Dynamic: {len(dyn)}, static: {len(stat)}\n"
            "Processing only common files."
        )
        # Find the difference
        d = set([f.stem for f in dyn])
        s = set([f.stem for f in stat])
        
        # Find all files that are in d and s
        common = list(d.intersection(s))
        
        # Remove all files that are not in common
        dyn = [f for f in dyn if f.stem in common]
        stat = [f for f in stat if f.stem in common]
        
    # Sort the files by date
    dyn = sorted(dyn, key=_date_transformer)
    stat = sorted(stat, key=_date_transformer)

    assert all([d.stem == s.stem for d,s in zip(dyn, stat)]),\
        "Dynamic and static messages are not in the same order."

    return True