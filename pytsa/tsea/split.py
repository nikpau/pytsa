"""
Auxiliary functions for determining trajectory splitting points.
"""
from ..structs import AISMessage
from .. import utils

# Empirical quantiles for the
# change in heading [°] between two 
# consecutive messages.
# These values have been obtained
# from one day of the AIS data set.
HQUANTILES = {
    99: [-174.1,163.1],
    95: [-109.3,56.8],
    90: [-44.9,25.1]
}

# Empirical quantiles for the
# change in speed [kn] between two
# consecutive messages.
# These values have been obtained
# from one day of the AIS data set.
SQUANTILES = {
    99: 7.4,
    95: 2.7,
    90: 1.4
}
    
# Empirical quantiles for the
# distance [mi] between two consecutive
# messages.
# These values have been obtained
# from one day of the AIS data set.
DQUANTILES = {
    99: 1.81,
    95: 1.14,
    90: 0.9
}
# Empirical quantiles for the
# difference between the reported
# speed [kn] and the speed calculated
# from the spatial difference and time difference.
# These values have been obtained
# from one day of the AIS data set.
RMCSQUANTILES = {
    99: [-25.83,14.20],
    95: [-3.61,7.52],
    90: [-1.47,3.97]
}

# Empirical quantiles for the
# time difference [s] between two
# consecutive messages.
# These values have been obtained
# from one day of the AIS data set.
TQUANTILES = {
    99: 1936.55,
    55: 392.00,
    90: 361.00
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
                       msg_t1: AISMessage) -> bool:
    """
    Return True if the spatial difference between two AIS Messages
    is larger than the 95% quantile of the distance distribution.
    """
    d = utils.greater_circle_distance(
        msg_t0.lon,msg_t0.lat,msg_t1.lon,msg_t1.lat,method="haversine")
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

def deviation_from_reported_too_large(msg_t0: AISMessage,
                                      msg_t1: AISMessage) -> bool:
        """
        Return True if the difference between the reported speed [kn] 
        from the AIS record and the speed calculated from the spatial 
        difference and time difference is larger than the 99% quantile 
        of the deviation distribution.
        """
        msgs = (msg_t0,msg_t1)
        diff = avg_speed(*msgs) - speed_from_position(*msgs)
        return not (
            RMCSQUANTILES[99][0] < diff < RMCSQUANTILES[99][1]
        )
        
def time_difference_too_large(msg_t0: AISMessage,
                              msg_t1: AISMessage) -> bool:
        """
        Return True if the time difference between two AIS Messages
        is larger than the 95% quantile of the time difference distribution.
        """
        return not (
            TQUANTILES[99] > (msg_t1.timestamp - msg_t0.timestamp)
        )

def is_split_point(msg_t0: AISMessage,
                   msg_t1: AISMessage) -> bool:
    """
    Pipeline function for checking whether a given
    AIS Message pair is a valid split point.
    """
    return (
        deviation_from_reported_too_large(msg_t0,msg_t1) or
        time_difference_too_large(msg_t0,msg_t1) or
        # distance_too_large(msg_t0,msg_t1) or
        speed_change_too_large(msg_t0,msg_t1) or
        heading_change_too_large(msg_t0,msg_t1)
    )