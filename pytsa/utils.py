"""
Utility functions for pytsa
"""
import math
import vincenty as _vincenty # store the original vincenty function

def m2nm(m: float) -> float:
    """Convert meters to nautical miles"""
    return m/1852

def nm2m(nm: float) -> float:
    """Convert nautical miles to meters"""
    return nm*1852

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