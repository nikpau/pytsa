"""
Auxiliary functions for determining trajectory splitting points.
"""
import pickle
import time

from itertools import pairwise
from enum import Enum

from pytsa.tsea.targetship import TargetShip

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
class TrackRejected(Exception):
    pass

class _TREXStatCollector:
    """
    Simple class for collecting statistics 
    about the split points found.
    """
    def __init__(self) -> None:
        self._n_rejoined_tracks = 0
        self._speed_change = 0
        self._turning_rate = 0
        self._distance = 0
        self._deviation = 0
        self._time_difference = 0
        self._n_split_points = 0
        
    def reset(self) -> None:
        self._n_rejoined_tracks = 0
        self._speed_change = 0
        self._turning_rate = 0
        self._distance = 0
        self._deviation = 0
        self._time_difference = 0
        self._n_split_points = 0

class GuoTREX(_TREXStatCollector):
    """
    
    Trajectory extraction part of Guo et al. (2021).
    
    https://doi.org/10.1016/j.oceaneng.2021.109256
    
    Part of the abstract:
    
    In this paper, an improved kinematic interpolation is presented for 
    AIS trajectory reconstruction, which integrates data pre-processing 
    and interpolation that considers the ships' kinematic information. 
    The improved kinematic reconstruction method includes four steps: 
    (1) data preprocessing, 
    (2) analysis of time interval distribution, 
    (3) abnormal data detection and removal, 
    (4) kinematic interpolation that takes the kinematic feature of ships 
    (i.e.,velocity and acceleration) into account, adding forward and 
    backward track points to help correct the acceleration function of 
    reconstruction points. 

    -----------------------------------------------

    Since this framework is a pure trajectory extractions method, we will
    only implement the first three steps. The fourth step is not relevant
    for our purposes.
    
    Controllable parameters:
    - vlim (float): The cutoff value for the speed change rate.
    - clim (float): The cutoff value for the change in course over ground.
    """
    
    NAME = "Guo et al. (2021)"
    
    def __init__(self, vlim: float = 30, clim: float = 2):
        """
        Trajetory extraction method of Guo et al. (2021).
        
        Parameters:
        - vlim (float): The cutoff value for the speed change rate.
        - clim (float): The cutoff value for the change in course over ground.
        """
        super().__init__()
        self.vlim = vlim
        self.clim = clim

    def iterative_abnormal_data_detection_and_removal(
        self,
        tracks:list[list[AISMessage]]) -> tuple[list[float], list[float]]:
        """
        Step 3 of Guo et al.'s method.
        
        Section 4.2.2 of the paper:
        """
        track = tracks[0]
        def average_change_rate(track:list[AISMessage]) -> list[float]:
            """
            Calculate the average change rate of the track.
            """
            roc = []
            for msg1, msg2 in pairwise(track):
                cogdiff = msg2.COG - msg1.COG
                if cogdiff <= 180:
                    roc.append(cogdiff / (msg2.timestamp - msg1.timestamp))
                else:
                    roc.append((360 - cogdiff) / (msg2.timestamp - msg1.timestamp))
            return roc
        
        v_bar = [speed_from_position(msg1, msg2) for msg1, msg2 in pairwise(track)]
        delta_c = average_change_rate(track)
        return  v_bar, delta_c

    def trex(self, ship: TargetShip) -> None:
        """
        Perform the trajectory extraction.
        """
        if ship._trex_applied:
            logger.warning(
                "Trajectory extraction has already been applied to this ship. Skipping..."
            )
            return
        raw_tracks = ship.tracks
        v_bar, delta_c = self.iterative_abnormal_data_detection_and_removal(raw_tracks)
        
        # Step 4
        out = []
        for i in range(len(raw_tracks[0])-1):
            if v_bar[i] < self.vlim or delta_c[i] < self.clim:
                out.append(raw_tracks[0][i])
                
        ship.tracks = [out]

        return

class ZhaoTREX(_TREXStatCollector):
    """
    Trajectory extraction method of Zhao et al. (2018)
    
    https://doi.org/10.1017/S0373463318000188
    
    """
    NAME = "Zhao et al. (2018)"
    
    def __init__(self):
        super().__init__()
        pass
        
    def pyhsical_integrety(self, track: list[list[AISMessage]]) -> list[list[AISMessage]] | TrackRejected:
        """
        Check the physical integrity of the trajectory.
        
        Section 3.1 of the paper:
        
        As described in Section 2.1.2, we judge the completeness of a track based on the num-
        ber of track points and whether there is corresponding additional information. It is easy
        to collect large amounts of AIS data over a short time, because the AIS data sampling
        rate is very high (2 s - 10 s). However, there are reasons which may cause a shortage of
        track points, such as signal loss and a brief stay in the research area. We believe that short
        tracks whose number of track points is less than 100 cannot characterise the movement
        of a ship. The threshold of completeness in pre-processing space data is empirically set
        as 100. That is to say, the track will be discarded if the number of track points is less
        than 100.
        """
        if len(track) < 100:
            raise TrackRejected("Track too short")
        
        return track
    def speed_change_too_large(self, msg_t0: AISMessage, msg_t1: AISMessage) -> bool:
        """
        Return True if the change in speed between two AIS Messages
        is larger than 15 knots.
        """
        if abs(msg_t1.SOG - msg_t0.SOG) > 15:
            self._speed_change += 1
            return True
        return False

    def time_difference_too_large(self,msg_t0: AISMessage, msg_t1: AISMessage) -> bool:
        """
        Return True if the time difference between two AIS Messages
        is larger than 600 seconds.
        """
        if msg_t1.timestamp - msg_t0.timestamp > 600:
            self._time_difference += 1
            return True
        return False
    
    def spatial_logical_integrety(self, track: list[AISMessage]) -> list[AISMessage]:
        """
        Check the spatial logical integrity of the trajectory.
        
        Section 3.2 of the paper:
        
        To solve the problems in Section 2.2 such as abnormal individual 
        points, tracks that lack relevance and sharing of MMSIs, a method which
        can quickly process large amounts of historical AIS data is proposed, see Algorithm 1.
        The method processes the time data ﬁrst, and then the space data, which consists of three
        parts, as shown in Figure 6. The ﬁrst part is the partition, and a breakpoint is found to
        split the track into sub-tracks (see Algorithm 2 and the partition part in Figure 6). The
        breakpoint is determined based on the thresholds of space (speed gate: 15 knots) and time
        (10 min), which are set empirically. A sub-track can be an individual point or a track seg-
        ment. The second part is the association of sub-tracks. All the sub-tracks will be judged
        by the threshold of space based on the last track point and the ﬁrst track point of all the
        following sub-tracks (see Algorithm 3 and the association part in Figure 6. The big arrow
        between two sub-tracks means they can be associated). If the judgment condition is met,
        the sub-tracks can be associated with each other. The judgment of the new sub-track will
        continue until the individual association is done. The third part is ﬁltering tracks that lack
        completeness, as described in Section 3.1 (see ﬁltering part in Figure 6(a), the track that
        lacks completeness is marked with a note ‘outlier’).
        """
        
        subtracks = [[track[0]]]
        for msg in track[1:]:
            if self.speed_change_too_large(subtracks[-1][-1],msg) \
                or self.time_difference_too_large(subtracks[-1][-1],msg):
                subtracks.append([msg])
                self._n_split_points += 1
            else:
                subtracks[-1].append(msg)
                
        # Subtrack association
        out = [subtracks[0]]
        for i in range(len(subtracks)-1):
            if speed_from_position(out[-1][-1],subtracks[i+1][0]) <= 15:
                out[-1].extend(subtracks[i+1])
            else: out.append(subtracks[i+1])
    
        return out

    def accuracy_of_time(self, track: list[AISMessage]) -> list[AISMessage]:
        """
        Temporal accuracy of the trajectory.
        
        Section 3.3 of the paper:
        
        [...]. "The data whose ﬁelds of generated time are all zero 
        and where the absolute value of deviation is larger than 
        ﬁve will be discarded." [...]
        """
        out = []
        def abs_sec_diff(recorded, generated):
            """
            Calculate the absolute difference between the recorded and
            generated seconds, with respect to wrapping around the minute.
            """
            if recorded == 0:
                recorded = 60
            diff = abs(recorded - generated)
            if diff > 30: # half a minute
                return 60 - diff
            else:
                return diff
        
        for msg in track:
            recorded = time.localtime(msg.timestamp)
            generated = time.localtime(msg.second)
            
            recorded_second = recorded.tm_sec
            if abs_sec_diff(recorded_second, generated.tm_sec) > 5:
                continue
            else:
                out.append(msg)
            
        return out
        
    def trex(self, ship: TargetShip) -> None:
        """
        Perform the trajectory extraction.
        """
        if ship._trex_applied:
            logger.warning(
                "Trajectrory extraction has already been applied to this ship. Skipping..."
            )
            return 
        raw_tracks = ship.tracks 
        try:
            tracks = self.pyhsical_integrety(raw_tracks)
        except TrackRejected:
            logger.info(
                "Track rejected due to length."
            )
            ship.tracks = [[]]
            ship._trex_applied = True
            return
        tracks = self.accuracy_of_time(tracks)
        tracks = self.spatial_logical_integrety(tracks)
        
        ship.tracks = tracks
        ship._trex_applied = True
        return 


class PauligTREX(_TREXStatCollector):
    """
    Class for determining split points in a trajectory.
    
    Besides containing the functions for determining split points,
    the class also tracks internal statistics about the split points
    found.
    
    Parameters:
    - alpha (float): The cutoff level for the quantiles used
        in the split point detection. Default is 0.05.
    """
    NAME = "Paulig & Okhrin (2024)"
    
    def __init__(self, alpha: float = 0.05) -> None:
        super().__init__()
        self._split_points = []

        # Cutoff values for the quantiles
        self.ST = 1 - alpha
        self.DTL = alpha / 2
        self.DTU = 1 - self.DTL

        
    def __len__(self) -> int:
        return len(self._split_points)
    
    def trex(self,ship: TargetShip) -> None:
        """
        Determine the split points in the trajectory,
        based on the alpha-method described in the paper.
        """
        logger.debug(
            f"Processing target ship {ship.mmsi} "
        )
        for track in ship.tracks:
            track.sort(key=lambda x: x.timestamp)
            _itracks = [[ship.tracks[0][0]]] # Intermediary track
            for msg_t0,msg_t1 in pairwise(track):
                if self.is_split_point(msg_t0,msg_t1,ship.ship_length):
                    _itracks.append([msg_t1])
                    self._n_split_points += 1
                else:    
                    _itracks[-1].append(msg_t1)
        # Recombine tracks
        tracks = [track for track in _itracks if len(track) > 1]
        # If no tracks are left, return an empty list
        if not tracks:
            logger.debug(
                f"Target ship {ship.mmsi} has no tracks left after filtering."
            )
            return [[]]
        
        ship.tracks = tracks
        ship._trex_applied = True
        return self._rejoin_tracks(ship)

    def _rejoin_tracks(self, target: TargetShip) -> None:
        """
        During the application of the split-point method,
        we split a trajectory into multiple tracks if two 
        consecutive AIS messages do not fall into the 
        95th percentile bounds of the defined metrics 
        in the paper.
        
        In case of a single erroneous AIS message, the
        split-point method will split the trajectory into
        a first part containing everything before the
        erroneous message, a second part containing only
        the erroneous message and a third part containing
        everything after the erroneous message. If the 
        erroneous part in the middle has only one or two 
        AIS messages, it is removed, leaving us with two 
        separate tracks.
        
        This leads to a situation where a single erroneous
        message can lead to the creation of two or three 
        separate tracks, of which the first and the last 
        logically belong together.
        
        This function determines whether the last AIS message
        of the first track and the first AIS message of the
        second track are close enough to be considered as
        part of the same track. If so, the two tracks are
        joined together.
        
        Judgment is based on the same split-point metrics
        as used for the initial split.
        """
        rejoined = []
        for i, track in enumerate(target.tracks):
            if i == 0:
                rejoined.append(track)
                continue
            if not self.is_split_point(rejoined[-1][-1],track[0],target.ship_length):
                rejoined[-1].extend(track)
                self._n_rejoined_tracks += 1
            else:
                rejoined.append(track)
        target.tracks = rejoined
        return
    
    
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
            diff = avg_speed(*msgs) - speed_from_position(*msgs)
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

        
def get_length_bin(ship_length: float) -> str:
    """
    Return the length bin for a given ship length.
    """
    for b1, b2 in zip(LENGTH_BINS,LENGTH_BINS[1:]):
        if b1 <= ship_length < b2:
            return f"{b1}-{b2}"
    return f"{LENGTH_BINS[-2]}-{LENGTH_BINS[-1]}"

class TREXMethod(Enum):
    """
    Selection of the different trajectory extraction methods.
    """
    __order__ = "ZHAO GUO PAULIG"
    ZHAO = ZhaoTREX
    GUO = GuoTREX
    PAULIG = PauligTREX
    
ANY_SPLITTER = PauligTREX | ZhaoTREX | GuoTREX

def print_split_stats(method: ANY_SPLITTER) -> None:
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
    logger.info(f"{'Speed change too large':<30}{method._speed_change:>20}")
    logger.info(f"{'Turning rate too large':<30}{method._turning_rate:>20}")
    logger.info(f"{'Distance too large':<30}{method._distance:>20}")
    logger.info(f"{'Rep. - Calc. speed too large':<30}{method._deviation:>20}")
    logger.info(f"{'Time difference too large':<30}{method._time_difference:>20}")
    logger.info(separator)