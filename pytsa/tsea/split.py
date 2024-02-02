"""
Auxiliary functions for determining trajectory splitting points.
"""
import numpy as np
import pickle

from ..structs import AISMessage
from .. import utils
from ..data.quantiles import __path__ as _DATA_DIR

# Load empirical quantiles
# from pickle file. The quantiles
# are numpy arrays with shape (1001,) to
# allow for quantiles from 0% up to 99.9%
# in 0.1% steps.
DATA_DIR = _DATA_DIR[0]
with open(f"{DATA_DIR}/dquants.pkl","rb") as f:
    _DQUANTILES = pickle.load(f)
with open(f"{DATA_DIR}/hquants.pkl","rb") as f:
    _HQUANTILES = pickle.load(f)
with open(f"{DATA_DIR}/squants.pkl","rb") as f:
    _SQUANTILES = pickle.load(f)
with open(f"{DATA_DIR}/diffquants.pkl","rb") as f:
    _RMCSQUANTILES = pickle.load(f)
with open(f"{DATA_DIR}/tquants.pkl","rb") as f:
    _TQUANTILES = pickle.load(f)
    
# Convert quantiles to dictionaries
# with the quantile as key and the
# quantile value as value. 
# ==================================
_NQ = 1001 # Number of quantiles
_QVALUES = np.linspace(0,100,1001) # Quantile values
# Empirical quantiles for the
# change in heading [°] between two 
# consecutive messages.
HQUANTILES = {_QVALUES[k]: _HQUANTILES[k] for k in range(_NQ)}

# Empirical quantiles for the
# change in speed [kn] between two
# consecutive messages.
SQUANTILES = {_QVALUES[k]: _SQUANTILES[k] for k in range(_NQ)}

# Empirical quantiles for the
# distance [mi] between two consecutive
# messages. NOTE: NOT USED in original paper.
DQUANTILES = {_QVALUES[k]: _DQUANTILES[k] for k in range(_NQ)}

# Empirical quantiles for the
# difference between the reported
# speed [kn] and the speed calculated
# from the spatial difference and time difference.
RMCSQUANTILES = {_QVALUES[k]: _RMCSQUANTILES[k] for k in range(_NQ)}

# Empirical quantiles for the
# time difference [s] between two
# consecutive messages.
TQUANTILES = {_QVALUES[k]: _TQUANTILES[k] for k in range(_NQ)}

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
        HQUANTILES[2.5] < utils.heading_change(msg_t0.COG,msg_t1.COG) < HQUANTILES[97.5]
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
            RMCSQUANTILES[2.5] < diff < RMCSQUANTILES[97.5]
        )
        
def time_difference_too_large(msg_t0: AISMessage,
                              msg_t1: AISMessage) -> bool:
        """
        Return True if the time difference between two AIS Messages
        is larger than the 95% quantile of the time difference distribution.
        """
        return not (
            TQUANTILES[95.0] > (msg_t1.timestamp - msg_t0.timestamp)
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
        distance_too_large(msg_t0,msg_t1) or
        speed_change_too_large(msg_t0,msg_t1) or
        heading_change_too_large(msg_t0,msg_t1)
    )