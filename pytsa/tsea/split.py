"""
Auxiliary functions for determining trajectory splitting points.
"""
import numpy as np
import pickle

from ..structs import AISMessage
from .. import utils
from ..logger import logger
from ..data.quantiles import __path__ as _DATA_DIR

# Load empirical quantiles
# from pickle file. The quantiles
# are numpy arrays with shape (1001,) to
# allow for quantiles from 0% up to 99.9%
# in 0.1% steps.
DATA_DIR = _DATA_DIR[0]
with open(f"{DATA_DIR}/dquants.pkl","rb") as f:
    _DQUANTILES = pickle.load(f)
with open(f"{DATA_DIR}/trquants.pkl","rb") as f:
    _TRQUANTILES = pickle.load(f)
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
_NQ = 10001 # Number of quantiles
# We have to round the quantiles to
# avoid floating point errors when using
# them as keys in dictionaries.
_QVALUES = np.round(np.linspace(0,1,_NQ),4) # Quantile values
# Empirical quantiles for the
# turning rate [°/s] between two 
# consecutive messages.
TRQUANTILES = {_QVALUES[k]: _TRQUANTILES[k] for k in range(_NQ)}

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


class Splitter:
    """
    Class for determining split points in a trajectory.
    
    Besides containing the functions for determining split points,
    the class also tracks internal statistics about the split points
    found.
    
    Parameters:
    - alpha (float): The significance level for the quantiles used
        in the split point detection. Default is 0.05.
    """
    
    def __init__(self, alpha: float = 0.05) -> None:
        self._split_points = []
        self._speed_change = 0
        self._turning_rate = 0
        self._distance = 0
        self._deviation = 0
        self._time_difference = 0

        # Cutoff values for the quantiles
        self.ST = 1 - alpha
        self.DTL = alpha / 2
        self.DTU = 1 - self.DTL

        
    def __len__(self) -> int:
        return len(self._split_points)
    
    
    def speed_change_too_large(self,
                               msg_t0: AISMessage, 
                               msg_t1: AISMessage) -> bool:
        """
        Return True if the change in speed between two AIS Messages
        is larger than the 95% quantile of the speed change distribution.
        """
        too_large =  (
            abs(msg_t1.SOG - msg_t0.SOG) > SQUANTILES[self.ST]
        )
        if too_large:
            self._speed_change += 1
            return too_large
        return too_large
        
    def turning_rate_too_large(self,
                               msg_t0: AISMessage, 
                               msg_t1: AISMessage) -> bool:
        """
        Return True if the change in heading between two AIS Messages
        is larger than the 95% quantile of the heading change distribution.
        """
        col = TRQUANTILES[self.DTL]
        cou = TRQUANTILES[self.DTU]
        hc = utils.heading_change(msg_t0.COG,msg_t1.COG)
        td = msg_t1.timestamp - msg_t0.timestamp
        too_large = not col < (hc/td) < cou
        if too_large:
            self._turning_rate += 1
            return too_large
        return too_large
    
    def distance_too_large(self,
                           msg_t0: AISMessage, 
                           msg_t1: AISMessage) -> bool:
        """
        Return True if the spatial difference between two AIS Messages
        is larger than the 95% quantile of the distance distribution.
        """
        d = utils.greater_circle_distance(
            msg_t0.lon,msg_t0.lat,msg_t1.lon,msg_t1.lat,method="haversine")
        too_large = d > DQUANTILES[self.ST]
        if too_large:
            self._distance += 1
            return too_large
        return too_large

    @staticmethod
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

    @staticmethod
    def avg_speed(msg_t0: AISMessage, 
                msg_t1: AISMessage) -> float:
        """
        Return the average speed [kn] between two AIS Messages
        as reported by the AIS.
        """
        return (msg_t0.SOG + msg_t1.SOG) / 2


    def deviation_from_reported_too_large(self,
                               msg_t0: AISMessage,
                                           msg_t1: AISMessage) -> bool:
            """
            Return True if the difference between the reported speed [kn] 
            from the AIS record and the speed calculated from the spatial 
            difference and time difference is larger than the 99% quantile 
            of the deviation distribution.
            """
            msgs = (msg_t0,msg_t1)
            diff = self.avg_speed(*msgs) - self.speed_from_position(*msgs)
            col = RMCSQUANTILES[self.DTL]
            cou = RMCSQUANTILES[self.DTU]
            too_large = not col < diff < cou
            if too_large:
                self._deviation += 1
                return too_large
            return too_large
            
    def time_difference_too_large(self,
                                  msg_t0: AISMessage,
                                  msg_t1: AISMessage) -> bool:
            """
            Return True if the time difference between two AIS Messages
            is larger than the 95% quantile of the time difference distribution.
            """
            co = TQUANTILES[self.ST]
            too_large = not (
                co > (msg_t1.timestamp - msg_t0.timestamp)
            )
            if too_large:
                self._time_difference += 1
                return too_large
            return too_large
                
    def is_split_point(self,
                    msg_t0: AISMessage,
                    msg_t1: AISMessage) -> bool:
        """
        Pipeline function for checking whether a given
        AIS Message pair is a valid split point.
        """
        split = []
        for f in (
            self.deviation_from_reported_too_large,
            self.time_difference_too_large,
            self.distance_too_large,
            self.speed_change_too_large,
            self.turning_rate_too_large
        ):
            split.append(f(msg_t0,msg_t1))
        return any(split)
        
    def print_split_stats(self) -> None:
        """
        Print the internal statistics about the split points found.
        """
        # Printout styling
        title = "Split Point Summary"
        separator = "-" * 50
        header = f"{'Metric':<30}{'Value':>20}"
        
        # Printout
        logger.info(title)
        logger.info(separator)
        logger.info(header)
        logger.info(separator)
        logger.info(f"{'Speed change too large':<30}{self._speed_change:>20}")
        logger.info(f"{'Turning rate too large':<30}{self._turning_rate:>20}")
        logger.info(f"{'Distance too large':<30}{self._distance:>20}")
        logger.info(f"{'Rep. - Calc. speed too large':<30}{self._deviation:>20}")
        logger.info(f"{'Time difference too large':<30}{self._time_difference:>20}")
        logger.info(separator)
        
        