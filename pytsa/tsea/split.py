"""
Auxiliary functions for determining trajectory splitting points.
"""
import pickle

from ..structs import AISMessage, LENGTH_BINS
from .. import utils
from ..logger import logger
from ..data.quantiles import __path__ as _DATA_DIR

# Load empirical quantiles
# from pickle file. The quantiles
# are numpy arrays with shape (1001,) to
# allow for quantiles from 0% up to 99.9%
# in 0.1% steps.
DATA_DIR = _DATA_DIR[0]

FNAMES = [
    "dquants",    # Distance quantiles
    "trquants",   # Turning rate quantiles
    "squants",    # Speed quantiles
    "diffquants", # Difference between reported and calculated speed quantiles
    "tquants"     # Time difference quantiles
]

# Load the quantiles
with open(f"{DATA_DIR}/quantiles.pkl","rb") as f:
    QUANTILES = pickle.load(f)
    
with open(f"{DATA_DIR}/independent/iquantiles.pkl","rb") as f:
    IQUANTILES = pickle.load(f)


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
                               msg_t1: AISMessage,
                               length_bin:str | None) -> bool:
        """
        Return True if the change in speed between two AIS Messages
        is larger than the 95% quantile of the speed change distribution.
        """
        if length_bin is None:
            q = IQUANTILES["squants"][self.ST]
        else:
            q = QUANTILES["squants"][length_bin][self.ST]
        too_large =  (
            abs(msg_t1.SOG - msg_t0.SOG) > q
        )
        if too_large:
            self._speed_change += 1
            return too_large
        return too_large
        
    def turning_rate_too_large(self,
                               msg_t0: AISMessage, 
                               msg_t1: AISMessage,
                               length_bin: str | None) -> bool:
        """
        Return True if the change in heading between two AIS Messages
        is larger than the alpha-quantile of the heading change distribution.
        """
        if length_bin is None:
            col = IQUANTILES["trquants"][self.DTL]
            cou = IQUANTILES["trquants"][self.DTU]
        else:
            col = QUANTILES["trquants"][length_bin][self.DTL]
            cou = QUANTILES["trquants"][length_bin][self.DTU]
        hc = utils.heading_change(msg_t0.COG,msg_t1.COG)
        td = msg_t1.timestamp - msg_t0.timestamp
        too_large = not col < (hc/td) < cou
        if too_large:
            self._turning_rate += 1
            return too_large
        return too_large
    
    def distance_too_large(self,
                           msg_t0: AISMessage, 
                           msg_t1: AISMessage,
                           length_bin: str | None) -> bool:
        """
        Return True if the spatial difference between two AIS Messages
        is larger than the alpha-quantile of the distance distribution.
        """
        d = utils.greater_circle_distance(
            msg_t0.lon,msg_t0.lat,msg_t1.lon,msg_t1.lat,method="haversine")
        if length_bin is None:
            too_large = d > IQUANTILES["dquants"][self.ST]
        else:
            too_large = d > QUANTILES["dquants"][length_bin][self.ST]
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


    def deviation_from_reported_too_large(
        self,
        msg_t0: AISMessage,
        msg_t1: AISMessage,
        length_bin: str | None) -> bool:
            """
            Return True if the difference between the reported speed [kn] 
            from the AIS record and the speed calculated from the spatial 
            difference and time difference is larger than the alpha-quantile 
            of the deviation distribution.
            """
            msgs = (msg_t0,msg_t1)
            diff = self.avg_speed(*msgs) - self.speed_from_position(*msgs)
            if length_bin is None:
                col = IQUANTILES["diffquants"][self.DTL]
                cou = IQUANTILES["diffquants"][self.DTU]
            else:
                col = QUANTILES["diffquants"][length_bin][self.DTL]
                cou = QUANTILES["diffquants"][length_bin][self.DTU]
            too_large = not col < diff < cou
            if too_large:
                self._deviation += 1
                return too_large
            return too_large
            
    def time_difference_too_large(self,
                                  msg_t0: AISMessage,
                                  msg_t1: AISMessage,
                                  length_bin: str | None) -> bool:
            """
            Return True if the time difference between two AIS Messages
            is larger than the 95% quantile of the time difference distribution.
            """
            co = IQUANTILES["tquants"][self.ST]
            too_large = not (
                co > (msg_t1.timestamp - msg_t0.timestamp)
            )
            if too_large:
                self._time_difference += 1
                return too_large
            return too_large
                
    def is_split_point(self,
                    msg_t0: AISMessage,
                    msg_t1: AISMessage,
                    ship_length: str | None) -> bool:
        """
        Pipeline function for checking whether a given
        AIS Message pair is a valid split point.
        """
        split = []
        if ship_length is None:
            lengthbin = None
        else: lengthbin = get_length_bin(ship_length)
        for f in (
            self.deviation_from_reported_too_large,
            self.time_difference_too_large,
            self.distance_too_large,
            self.speed_change_too_large,
            self.turning_rate_too_large
        ):
            split.append(f(msg_t0,msg_t1,lengthbin))
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
        
def get_length_bin(ship_length: float) -> str:
    """
    Return the length bin for a given ship length.
    """
    for b1, b2 in zip(LENGTH_BINS,LENGTH_BINS[1:]):
        if b1 <= ship_length < b2:
            return f"{b1}-{b2}"
    return f"{LENGTH_BINS[-2]}-{LENGTH_BINS[-1]}"