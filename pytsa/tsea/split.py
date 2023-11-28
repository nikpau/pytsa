"""
Auxiliary functions for determining trajectory splitting points.
"""
from ..structs import AISMessage
from .. import utils

# Empirical quantiles for the
# change in heading [°] between two 
# consecutive messages.
# These values have been obtained
# from the AIS data set.
HQUANTILES = {
    99: [-174.1,163.1],
    95: [-109.3,56.8],
    90: [-44.9,25.1]
}

# Empirical quantiles for the
# change in speed [kn] between two
# consecutive messages.
# These values have been obtained
# from the AIS data set.
SQUANTILES = {
    99: 7.4,
    95: 2.7,
    90: 1.4
}
    
# Empirical quantiles for the
# distance [mi] between two consecutive
# messages.
# These values have been obtained
# from the AIS data set.
DQUANTILES = {
    99: 1.81,
    95: 1.14,
    90: 0.9
}

def speed_change_too_large(msg_t0: AISMessage, 
                           msg_t1: AISMessage) -> bool:
    """
    Return True if the change in speed between two AIS Messages
    is larger than the 95% quantile of the speed change distribution.
    """
    return (
        abs(msg_t1.SOG - msg_t0.SOG) > SQUANTILES[95]
    )
    
def heading_change_too_large(msg_t0: AISMessage, 
                             msg_t1: AISMessage) -> bool:
    """
    Return True if the change in heading between two AIS Messages
    is larger than the 95% quantile of the heading change distribution.
    """
    return not (
        HQUANTILES[95][0] < utils.heading_change(msg_t0.COG,msg_t1.COG) < HQUANTILES[95][1]
    )

def distance_too_large(msg_t0: AISMessage, 
                       msg_t1: AISMessage,
                       method: str) -> bool:
    """
    Return True if the spatial difference between two AIS Messages
    is larger than the 95% quantile of the distance distribution.
    """
    assert method in ["haversine", "vincenty"], "Invalid method: {}".format(method)
    d = utils.greater_circle_distance(
        msg_t0.lon,msg_t0.lat,msg_t1.lon,msg_t1.lat,method=method)
    return d > DQUANTILES[95]

def speed_from_position(msg_t0: AISMessage, 
                        msg_t1: AISMessage) -> float:
    """
    Return the speed [kn] between two AIS Messages
    calculated from the spatial difference and time difference.
    """
    d = utils.greater_circle_distance(
        msg_t0.lon,msg_t0.lat,msg_t1.lon,msg_t1.lat)
    t = msg_t1.timestamp - msg_t0.timestamp
    return utils.mi2nm(d) / utils.s2h(t)

def avg_speed(msg_t0: AISMessage, 
              msg_t1: AISMessage) -> float:
    """
    Return the average speed [kn] between two AIS Messages
    as reported by the AIS.
    """
    return (msg_t0.SOG + msg_t1.SOG) / 2

def is_split_point(msg_t0: AISMessage,
                   msg_t1: AISMessage) -> bool:
    """
    Pipeline function for checking whether a given
    AIS Message pair is a valid split point.
    """
    return (
        distance_too_large(msg_t0,msg_t1) or
        speed_change_too_large(msg_t0,msg_t1) or
        heading_change_too_large(msg_t0,msg_t1)
    )