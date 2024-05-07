from __future__ import annotations

from pathlib import Path
from typing import Callable, Generator, List, Union
from more_itertools import pairwise
import multiprocessing as mp
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from ..logger import logger
from ..structs import (
    BoundingBox, TimePosition,
    UNIX_TIMESTAMP, ShipType
)
from . import split
from ..decoder.filedescriptor import (
    BaseColumns, Msg12318Columns, Msg5Columns
)
from .targetship import (
    TargetShip, AISMessage, 
    InterpolationError, Track, Targets
)
from ..utils import DataLoader

def _identity(x):
    return x


class NeighborhoodTreeSearch:
    """
    A class for performing neighborhood searches 
    on AIS message data using a kd-tree.

    This class takes AIS messages, filters them 
    based on a time window, and uses a kd-tree
    for efficient spatial searches to find neighboring 
    ships within a specified search radius.

    Attributes:
        `data` (DataLoader): Instance of DataLoader 
                             to load AIS message data.
        `time_delta` (int): Time delta in minutes for 
                            filtering messages around a 
                            specific time.
        `max_tgt_ships` (int): Maximum number of target 
                               ships to return in the 
                               neighborhood search.

    Methods:
        `get_neighbors`: Retrieves neighboring AIS messages 
                         within a specified search radius.
    """
    def __init__(self, 
                 data_loader: DataLoader,
                 time_delta: int,
                 max_tgt_ships: int) -> None:
        
        self.data = data_loader # Load data into memory
        self.time_delta = time_delta
        self.max_tgt_ships = max_tgt_ships
    
    def _build_kd_tree(self, data: pd.DataFrame) -> cKDTree:
        """
        Private method to build a kd-tree from latitude 
        and longitude columns of a pandas DataFrame.

        Args:
            `data` (pd.DataFrame): DataFrame containing 
                                   latitude and longitude data.

        Returns:
            `cKDTree`: A scipy cKDTree object built from 
                       the latitude and longitude data.

        Raises:
            AssertionError: If the input DataFrame lacks 
                            latitude or longitude columns.
        """
        assert Msg12318Columns.LAT.value in data and Msg12318Columns.LON.value in data, \
            "Input dataframe has no `lat` or `lon` columns"
        lat=data[Msg12318Columns.LAT.value].values
        lon=data[Msg12318Columns.LON.value].values
        return cKDTree(np.column_stack((lat,lon)))
            

    def get_neighbors(self, 
                      tpos: TimePosition, 
                      search_radius: float = 20) -> pd.DataFrame:
        """
        Return all AIS messages that are no more than
        `self.search_radius` [nm] away from the given position.

        Args:
            tpos: TimePosition object of postion and time for which 
                    neighbors shall be found        

        """
        if not self.data.loaded:
            self.data.load() # Load data into memory
        
        filtered = self._time_filter(
            self.data.dynamic_data,tpos.timestamp,self.time_delta
        )
        # Check if filterd result is empty
        if filtered.empty:
            logger.warning("No AIS messages found in time-filtered cell.")
            return filtered
        tree = self._build_kd_tree(filtered)
        # The conversion to degrees is only accurate at the equator.
        # Everywhere else, the distances get smaller as lines of 
        # Longitude are not parallel. Therefore, 
        # this is a conservative estimate.
        if not search_radius == np.inf:
            sr =  search_radius/60 # Convert to degrees
        else:
            sr = np.inf
        d, indices = tree.query(
            tpos.position.as_list,
            k=self.max_tgt_ships,
            distance_upper_bound=sr
        )
        # Sort out any infinite distances
        res = [indices[i] for i,j in enumerate(d) if j != float("inf")]

        return filtered.iloc[res]
    
    @staticmethod
    def _time_filter(df: pd.DataFrame, 
                     date: UNIX_TIMESTAMP, 
                     delta: int) -> pd.DataFrame:
        """
        Filter a pandas dataframe to only return 
        rows whose `Timestamp` is not more than 
        `delta` minutes apart from imput `date`.
        """
        assert BaseColumns.TIMESTAMP.value in df, "No `timestamp` column found"
        timezone = df[BaseColumns.TIMESTAMP.value].iloc[0].tz.zone # Get timezone
        date = pd.to_datetime(int(date),unit="s").tz_localize(timezone)
        dt = pd.Timedelta(delta, unit="minutes")
        mask = (
            (df[BaseColumns.TIMESTAMP.value] > (date-dt)) & 
            (df[BaseColumns.TIMESTAMP.value] < (date+dt))
        )
        return df.loc[mask]
class SearchAgent:
    """
    Class searching for target ships
    near a user-provided location.

    To initialize the Agent, you have to specify a 
    datapath, where source AIS Messages are saved as csv files.
    You also have to specify a global frame 
    for the search agent. Outside its borders, no
    search will be commenced.
    """
    
    def __init__(self, 
                 dynamic_paths: Union[Path,List[Path]],
                 frame: BoundingBox,
                 static_paths: Union[Path,List[Path]],
                 max_tgt_ships: int = 200,
                 preprocessor: Callable[[pd.DataFrame],pd.DataFrame] = _identity,
                 decoded: bool = True,
                 high_accuracy: bool = False) -> None:
       
        """ 
        frame: BoundingBox object setting the search space
                for the taget ship extraction process.
                AIS records outside this BoundingBox are 
                not eligible for TargetShip construction.
        dynamic_paths: path or list of paths to a csv file 
                       containing AIS messages of type 1,2,3 and 18.
        static_paths: path or list of paths to a csv file 
                      containing AIS messages of type 5.
        max_tgt_ships: maximum number of target ships to retrun

        preproceccor: Preprocessor function for input data. 
                Defaults to the identity.
        decoded: boolean flag indicating whether the input data
                is already decoded or not. If `decoded` is False,
                the input data is decoded on the fly. Please note
                that this process is magnitudes slower than using
                decoded data.
        high_accuracy: boolean flag indicating whether Vincenty's
                formula should be used for distance calculations.
                If `high_accuracy` is False, the Haversine formula
                is used. Defaults to False.
        """
        # Exhaust generators 
        # This is possible if Path().glob() is used
        if isinstance(dynamic_paths, Generator):
            dynamic_paths = list(dynamic_paths)
        elif not isinstance(dynamic_paths,list):
            dynamic_paths = [dynamic_paths]
            dynamic_paths = [Path(p) for p in dynamic_paths]
        else:
            dynamic_paths = [Path(p) for p in dynamic_paths]
        
        if isinstance(static_paths, Generator):
            static_paths = list(static_paths)
        elif not isinstance(static_paths,list):
            static_paths = [static_paths]
            static_paths = [Path(p) for p in static_paths]
        else:
            static_paths = [Path(p) for p in static_paths]

        self.high_accuracy = high_accuracy
        
        spatial_filter = (
            f"{Msg12318Columns.LON.value} > {frame.LONMIN} and "
            f"{Msg12318Columns.LON.value} < {frame.LONMAX} and "
            f"{Msg12318Columns.LAT.value} > {frame.LATMIN} and "
            f"{Msg12318Columns.LAT.value} < {frame.LATMAX}"
        )

        self.data_loader = DataLoader(
            dynamic_paths,
            static_paths,
            preprocessor,
            spatial_filter
        )

        # Maximum number of target ships to extract
        self.max_tgt_ships = max_tgt_ships
        
        # Maximum temporal deviation of target
        # ships from provided time in `init()`.
        # This is used internally to catch
        # all messages around the queried time,
        # for building and interpolating trajectories.
        self.time_delta = 30 # in minutes

        self.neighborhood = NeighborhoodTreeSearch(
            self.data_loader,
            self.time_delta,
            self.max_tgt_ships
        )
        
        # Flag indicating whether the input data
        # is already decoded or not
        self.decoded = decoded
        
        # Length of the original AIS message dataframe
        self._n_original = 0
        # Length of the filtered AIS message dataframe
        self._n_filtered = 0
        
        # Number of times, the _speed_correction function
        # was called
        self._n_speed_correction = 0
        
        # Number of times, the _position_correction function
        # was called
        self._n_position_correction = 0
        
    def freeze(self, 
               tpos: TimePosition, 
               search_radius: float = 20, # in nautical miles
               interpolation: str = "linear") -> Targets:
        """
        Freeze around a given position and time and return 
        a list of target ships present in the neighborhood 
        around the given position. 
        
        tpos: TimePosition object of agent for which 
                neighbors shall be found
        
        search_radius: Radius around agent in which taget vessel 
                search is rolled out [nm]

        overlap_tpos: boolean flag indicating whether
                we only want to return vessels, whose track overlaps
                with the queried timestamp. If `overlap_tpos` is False,
                all vessels whose track is within the time delta of the
                queried timestamp are returned.
        """
        assert interpolation in ["linear","spline","auto"], \
            "Interpolation method must be either 'linear', 'spline' or 'auto'"
        # Get neighbors
        neighbors = self.neighborhood.get_neighbors(tpos,search_radius)

        constructor = TargetShipConstructor(
            self.data_loader,
            split.Splitter()
        )
        tgts = constructor._sp_construct_target_vessels(neighbors,tpos,True)
        # Contruct Splines for all target ships
        tgts = self._interpolate_trajectories(tgts,mode=interpolation)
        return tgts
    
    def extract_all(self,
                    njobs: int = 4, 
                    alpha: float = 0.05,
                    skip_tsplit: bool = False) -> Targets:
        """
        Extracts a dictionary of all target ships in the
        frame of the search agent using the split-point
        method as described in the paper. 
        
        Parameters:
        ----------
        njobs: int
            Number of jobs to use for multiprocessing.
        alpha: float
            The significance level for the quantiles used
            in the split point detection. Default is 0.05.
        skip_tsplit: bool
            If True, the split-point method is not used
            to split the tracks of the target ships.
            Instead, all vessels will only have one track,
            containing all time-ordered AIS messages
            comning from the same MMSI.
            
        Returns:
        --------
        dict: A dictionary of dict[MMSI,TargetShip]. 
        """
        splitter = split.Splitter(alpha)
        constructor = TargetShipConstructor(
            self.data_loader,
            splitter
        )
        targets = constructor._mp_construct_target_vessels(
            njobs,skip_tsplit
        )
        constructor.print_trex_stats()
        splitter.print_split_stats()
        return targets

    def _interpolate_trajectories(self, 
                           tgts: Targets,
                           mode: str = "auto") -> Targets:
        """
        Interpolate all target ships' trajectories.
        """
        for mmsi,tgt in list(tgts.items()):
            try:
                tgt.interpolate(mode) # Construct splines
            except InterpolationError as e:
                logger.warn(e)
                del tgts[mmsi]
        return tgts


class TargetShipConstructor:
    """
    A class for constructing TargetShip 
    objects from raw AIS messages.

    Attributes:
        `data_loader`: 
            An instance of the `pytsa.utils.DataLoader` 
            class used to load AIS message data.
        `splitter`:
            An instance of the `pytsa.tsea.split.Splitter`
            class used to determine split points in
            trajectories.

    Methods:
        `construct_target_vessels`: 
            Constructs TargetVessel objects from 
            a given DataFrame.
    """
    def __init__(self, 
                 data_loader: DataLoader,
                 splitter: split.Splitter) -> None:
        
        self.data_loader = data_loader
        self.splitter = splitter
        
        # Statistics
        self._n_split_points = 0
        self._n_rejoined_tracks = 0
        self._n_duplicates = 0
        self._n_obs_raw = 0
        self._n_trajectories = 0

    def _distribute(self, df: pd.DataFrame, nparts: int) -> list[pd.DataFrame]:
        """
        Distribute a given dataframe into `nparts`
        equally sized smaller data frames.
        
        Parameters:
        -----------------
        df (pd.DataFrame): Pandas DataFrame containing the
            data to be split
        nparts (int): number of parts into which `df` is
            going to be split.
    
        Returns:
        -----------------
        List of DataFrames: A list of DataFrames, 
        each being a part of the original DataFrame.
        """
        # Check if the DataFrame can be split 
        # into the desired number of parts
        if nparts <= 0 or df is None:
            raise ValueError(
                "Number of parts must be a positive"
                "integer and DataFrame must not be None."
            )

        # Use numpy.array_split to split the 
        # DataFrame into nearly equal parts
        return list(np.array_split(df, nparts))
        
    
    def _impl_construct_target_vessel(self, 
                                      dyn: pd.DataFrame,
                                      stat: pd.DataFrame) -> Targets:
        """
        Construct a single TargetVessel object from a given dataframe.
        """
        MMSI = int(dyn[Msg12318Columns.MMSI.value].iloc[0])
        ts = dyn[BaseColumns.TIMESTAMP.value].iloc[0].to_pydatetime().timestamp()
        # Initialize TargetVessel object
        tv =  TargetShip(
            ts = None,
            mmsi=MMSI,
            ship_type=self._get_ship_type(stat,MMSI,ts),
            length=self._get_ship_length(stat,MMSI),
            tracks=[[]]
        )
        first = True
                
        dyn = dyn.sort_values(by=BaseColumns.TIMESTAMP.value)
        
        for ts,lat,lon,sog,cog in zip(
            dyn[BaseColumns.TIMESTAMP.value], 
            dyn[Msg12318Columns.LAT.value],  
            dyn[Msg12318Columns.LON.value],       
            dyn[Msg12318Columns.SPEED.value],
            dyn[Msg12318Columns.COURSE.value]):
            
            ts: pd.Timestamp # make type hinting happy
            
            msg = AISMessage(
                sender=MMSI,
                timestamp=int(ts.timestamp()), # Convert to unix
                lat=lat,lon=lon,
                COG=cog,SOG=sog
            )
            if first:
                tv.tracks[-1].append(msg)
                first = False
            else:
                # Check if message is a duplicate, i.e. had been 
                # received by multiple AIS stations
                if msg.timestamp == tv.tracks[-1][-1].timestamp:
                    continue
                tv.tracks[-1].append(msg)

        return {MMSI:tv}
    
    def _mp_construct_target_vessels(self,
                                     njobs: int = 4,
                                     skip_tsplit: bool = False) -> Targets:
        """
        Adaption from `_construct_target_vessels` with multiprocessing.
        The initial dataframe is split into smaller dataframes, each
        containing only messages from a single MMSI. These dataframes
        are then processed in parallel and the results are merged.
        """
        self.reset_stats()
        singles: list[Targets] = []
        with mp.Pool(njobs) as pool:
            results = []
            for dyn, stat in self.data_loader.iterate_chunks():
                self._n_obs_raw += len(dyn)
                single_frames = self._distribute(dyn,njobs)
                res = pool.starmap_async(
                    self._impl_construct_target_vessel,
                    [(dframe,stat) for dframe in single_frames]
                )
                results.append(res)
            for result in results:
                singles.extend(result.get())
        targets = self._merge_targets(*singles)
        targets = self._remove_duplicates(targets)
        if not skip_tsplit:
            targets = self._rejoin_tracks(
                self._determine_split_points(targets)
            )
        self._n_trajectories = sum([len(tgt.tracks) for tgt in targets.values()])
        return targets
        
    def _remove_duplicates(self, targets: Targets) -> Targets:
        """
        Remove duplicate positions and timestamps.
        Duplicate positions are removed by the 
        set operation, as equality of AISMessage
        objects is defined by their lat/lon.
        
        Duplicate timestamps are removed by sorting
        the list of AISMessage objects by their
        timestamp and then removing all but the
        first occurence of a timestamp.
        """
        for tgt in targets.values():
            # Rm pos dups
            rmpos = list(set(tgt.tracks[0]))
            self._n_duplicates += len(tgt.tracks[0]) - len(rmpos)
            # Check for timestamp dups
            ts_sorted_idx = np.unique(
                [msg.timestamp for msg in rmpos],
                return_index=True)[1]
            self._n_duplicates += len(rmpos) - len(ts_sorted_idx)
            tgt.tracks[0] = [rmpos[i] for i in ts_sorted_idx]
            
        return targets
    
    def _merge_targets(self,
                       *singles: Targets) -> Targets:
        """
        Join tracks of multiple TargetShip objects
        into a single TargetShip object.
        
        Note: The input TargetShip objects are only 
        consisting of a single track each, as they
        just have been constructed.
        """
        targets: Targets = {}
        for single in singles:
            for mmsi,tgt in single.items():
                if mmsi not in targets:
                    targets[mmsi] = tgt
                else:
                    targets[mmsi].tracks[0].extend(
                        tgt.tracks[0]
                    )
        return targets
    
    def _rejoin_tracks(self,
                       targets: Targets) -> Targets:
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
        logger.info("Rejoining tracks...")
        for tgt in targets.values():
            rejoined = []
            for i, track in enumerate(tgt.tracks):
                if i == 0:
                    rejoined.append(track)
                    continue
                if not self.splitter.is_split_point(rejoined[-1][-1],track[0]):
                    rejoined[-1].extend(track)
                    self._n_rejoined_tracks += 1
                else:
                    rejoined.append(track)
            tgt.tracks = rejoined
        logger.info(f"Rejoined {self._n_rejoined_tracks} tracks.")
        return targets
                    
    def _determine_split_points(self,
                                targets: Targets) -> Targets:
        """
        Determine split points for all target ships.
        """
        logger.info("Determining split points...")
        nvessels = len(targets)
        ctr = 0
        for tgt in list(targets.values()):
            logger.debug(
                f"Processing target ship {tgt.mmsi} "
                f"({ctr+1}/{nvessels})"
            )
            ctr += 1
            for track in tgt.tracks:
                track.sort(key=lambda x: x.timestamp)
                _itracks = [] # Intermediary track
                tstartidx = 0
                for i, (msg_t0,msg_t1) in enumerate(pairwise(track)):
                    if self.splitter.is_split_point(msg_t0,msg_t1):
                        _itracks.append(track[tstartidx:i+1])
                        tstartidx = i+1
                        self._n_split_points += 1
            # Only keep tracks with more than one observation
            tgt.tracks = [track for track in _itracks if len(track) > 2]
            # If no tracks are left, remove target ship
            if not tgt.tracks:
                logger.debug(
                    f"Target ship {tgt.mmsi} has no tracks left after filtering."
                )
                del targets[tgt.mmsi]
        logger.info(f"Determined {self._n_split_points} split points.")
        return targets
                        
    def _sp_construct_target_vessels(self,
                                     neighbors: pd.DataFrame,
                                     tpos: TimePosition,
                                     overlap: bool) -> Targets:
        """
        Single-process version of `_construct_target_vessels`.
        
        Walk through the rows of `df` and construct a 
        `TargetVessel` object for every unique MMSI. 
        
        The individual AIS Messages are sorted by date
        and are added to the respective TargetVessel's track attribute.
        
        The `overlap` flag indicates whether we only want to return vessels,
        whose track overlaps with the queried timestamp. If `overlap` is False,
        all vessels whose track is within the time delta of the
        queried timestamp are returned.
        
        If the temporal or spatial difference between two AIS Messages is too large,
        the track of the target ship is split into two tracks.
        
        """
        targets: Targets = {}
        neighbors = neighbors.sort_values(by=BaseColumns.TIMESTAMP.value)
        
        for mmsi,ts,lat,lon,sog,cog in zip(
            neighbors[Msg12318Columns.MMSI.value], 
            neighbors[BaseColumns.TIMESTAMP.value],
            neighbors[Msg12318Columns.LAT.value],  
            neighbors[Msg12318Columns.LON.value],
            neighbors[Msg12318Columns.SPEED.value],
            neighbors[Msg12318Columns.COURSE.value]):
            
            ts: UNIX_TIMESTAMP = ts.to_pydatetime().timestamp() # Convert to unix
            
            msg = AISMessage(
                sender=mmsi,
                timestamp=ts,
                lat=lat,lon=lon,
                COG=cog,SOG=sog
            )
            
            if mmsi not in targets:
                targets[mmsi] = TargetShip(
                    ts = tpos.timestamp if tpos is not None else None,
                    mmsi=mmsi,
                    ship_type=self._get_ship_type(self.data_loader.static_data,mmsi,ts),
                    length=self._get_ship_length(self.data_loader.static_data,mmsi),
                    tracks=[[msg]]
                )
            else:
                # Split track
                if self.splitter.is_split_point(targets[mmsi].tracks[-1][-1],msg):
                    targets[mmsi].tracks.append([])
                v = targets[mmsi]
                v.tracks[-1].append(msg)

        for tgt in targets.values():
            tgt.find_shell() # Find shell (start/end of traj) of target ship
        
        # Remove target ships with only one observation
        self._remove_single_obs(targets)
        
        if overlap:
            self._filter_overlapping(targets,tpos)
            
        return targets
    
    def _filter_overlapping(self, 
                            targets: Targets, 
                            tpos: TimePosition) -> None:
        """
        Remove target ships whose track does not overlap
        with the queried timestamp.
        """
        for tgt in targets.values():
            to_keep = []
            for track in tgt.tracks:
                if self._overlaps_search_date(track,tpos):
                    to_keep.append(track)
            tgt.tracks = to_keep

    def _remove_single_obs(self, targets: Targets) -> Targets:
        """
        Remove tracks that only have a single observation.
        """
        for tgt in targets.values():
            to_keep = []
            for track in tgt.tracks:
                if len(track) >= 2:
                    to_keep.append(track)
            tgt.tracks = to_keep
    
    def _overlaps_search_date(self, 
                              track: Track, 
                              tpos: TimePosition) -> bool:
        """
        Return True if the track of the given vessel
        overlaps with the queried timestamp.
        """
        return (track[0].timestamp < tpos.timestamp < track[-1].timestamp)

    def _get_ship_type(self, 
                       static_msgs: pd.DataFrame,
                       mmsi: int,
                       date: UNIX_TIMESTAMP) -> ShipType:
        """
        Return the ship type of a given MMSI number.

        If more than one ship type is found, the first
        one is returned and a warning is logged.
        """
        st = NeighborhoodTreeSearch._time_filter(static_msgs,date,15)
        st = static_msgs[static_msgs[Msg5Columns.MMSI.value] == mmsi]\
            [Msg5Columns.SHIPTYPE.value].values
        st:np.ndarray = np.unique(st)
        if st.size > 1:
            logger.debug(
                f"More than one ship type found for MMSI {mmsi} "
                f"in a 30 minute window. Found {st}.\n"
                f"Returning the first one.")
        if st.size == 0: return ShipType.NOTAVAILABLE
        return ShipType.from_value(st[0])

    def _get_ship_length(self, 
                         static_msgs: pd.DataFrame,
                         mmsi: int) -> list[int]:
        """
        Return the ship length of a given MMSI number.

        If more than one ship length is found, the largest
        one is returned and a warning is logged.
        """
        raw = static_msgs[static_msgs[Msg5Columns.MMSI.value] == mmsi]\
            [[Msg5Columns.TO_BOW.value,Msg5Columns.TO_STERN.value]].values
        sl:np.ndarray = np.sum(raw,axis=1)
        sl = np.unique(sl)
        if sl.size > 1:
            logger.debug(
                f"More than one ship length found for MMSI {mmsi}. "
                f"Found {sl}.")
            return sl
        return list(sl)
    
    def print_trex_stats(self) -> str:
        """
        Print statistics of the last 
        trajectory extraction.
        """
        metrics = {
            'Number of Raw Messages': self._n_obs_raw,
            'Number of Duplicates': self._n_duplicates,
            'Number of Trajectories': self._n_trajectories,
            'Number of Split-Points': self._n_split_points,
            'Number of Rejoined Tracks': self._n_rejoined_tracks,
        }
        
        # Printout styling
        title = "Trajectory Extraction Summary"
        separator = "-" * 50
        header = f"{'Metric':<30}{'Value':>20}"

        # Printing the title and header
        print(f"{separator}\n{title.center(len(separator))}\n{separator}")
        print(header)
        print(separator)

        # Iterating over the metrics to print each one
        for metric, value in metrics.items():
            print(f"{metric:<30}{value:>20}")

        print(separator)
        
    def reset_stats(self) -> None:
        """
        Reset statistics.
        """
        self._n_split_points = 0
        self._n_rejoined_tracks = 0
        self._n_duplicates = 0
        self._n_obs_raw = 0
        self._n_trajectories = 0
        logger.debug("Trajectory extraction statistics reset.")
        return None
