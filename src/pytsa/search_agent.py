from __future__ import annotations
from dataclasses import dataclass

from datetime import datetime, timedelta
from threading import Thread
from typing import Callable, List, Tuple, Union
from .cell_manager import NOTDETERMINED, LatLonCellManager, UTMCellManager
import warnings

import ciso8601
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
import scipy.linalg
import scipy.optimize
from matplotlib import pyplot as plt
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spilu
from scipy.spatial import cKDTree
import utm
import warnings

from .logger import Loader, logger
from .structs import (
    BoundingBox, Cell, LatLonCell, Point,
    Position, ShellError,TimePosition,
    OUTOFBOUNDS,SUB_TO_ADJ, DataColumns, 
    UTMBoundingBox, UTMCell
)
from .targetship import TargetVessel, AISMessage

# Exceptions
class FileLoadingError(Exception):
    pass


def m2nm(m: float) -> float:
    """Convert meters to nautical miles"""
    return m/1852

def nm2m(nm: float) -> float:
    """Convert nautical miles to meters"""
    return nm*1852

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
    
    def __init__(
        self, 
        datapath: Union[str, List[str]],
        frame: BoundingBox,
        search_radius: float = 0.5, # in nautical miles
        max_tgt_ships: int = 50,
        n_cells: int = 144,
        filter: Callable[[pd.DataFrame],pd.DataFrame] = lambda x: x,
        v = False) -> None:
       
        """ 
        frame: BoundingBox object setting the search space
                for the taget ship extraction process.
                AIS records outside this BoundingBox are 
                not eligible for TargetShip construction.
        datapath: path to (a) csv file(s) containing AIS messages.
                Can be either a single file, or a list of several files.
        search_radius: Radius around agent in which taget vessel 
                search is rolled out
        n_cells: number of cells into which the spatial extent
                of the AIS messages is divided into.
                (See also the `get_cell` function for more info)  
        max_tgt_ships: maximum number of target ships to store

        filter: function to filter out unwanted AIS messages. Default is
                the identity function.
        """

        if not isinstance(datapath,list):
            self.datapath = [datapath]
        else:
            self.datapath = datapath
        
        # Spatial bounding box of current AIS message space
        self.FRAME = frame

        # Maximum number of target ships to extract
        self.max_tgt_ships = max_tgt_ships
        
        # Maximum temporal deviation of target
        # ships from provided time in `init()`
        self.time_delta = 30 # in minutes
        
        # Search radius in [°] around agent
        self.search_radius = search_radius
        
        # Number of cells, into which the spatial extent
        # of AIS records is divided into
        self.n_cells = n_cells
        
        # Init cell manager
        if isinstance(frame,UTMBoundingBox):
            self._utm = True
            self.cell_manager = UTMCellManager(frame,n_cells)
            logger.info("UTM Cell Manager initialized")
        else:
            self._utm = False
            self.cell_manager = LatLonCellManager(frame,n_cells)
            logger.info("LatLon Cell Manager initialized")

        # List of cell-indicies of all 
        # currently buffered cells
        self._buffered_cell_idx = []

        # Custom filter function for AIS messages
        self.filter = filter
        
        self._is_initialized = False

    def init(
            self, 
            tpos: TimePosition, *, 
            supress_warnings=False)-> None:
        """
        tpos: TimePosition object for which 
                TargetShips shall be extracted from 
                AIS records.
        """
        tpos._is_utm = self._utm
        pos = tpos.position
        if not self._is_initialized:
            # Current Cell based on init-position
            self.cell = self.cell_manager.get_cell_by_position(pos)
            
            # Load AIS Messages for specific cell
            self.cell_data = self._load_cell_data(self.cell)
            self._is_initialized = True
            pos = f"{pos.lat:.3f}N, {pos.lon:.3f}E" if isinstance(tpos.position,Position)\
                  else f"{pos.easting:.3f}mE, {pos.northing:.3f}mN"
            logger.info(
                "Target Ship search agent initialized at "
                f"{pos}"
            )
        else:
            if not supress_warnings:
                logger.warn(
                    "TargetShip object is aleady initialized. Re-initializing "
                    "is slow as it reloads cell data. "
                )
                while True:
                    dec = input("Re-initialize anyways? [y/n]: ")
                    if not dec in ["y","n"]:
                        print("Please answer with [y] or [n]")
                    elif dec == "n": pass
                    else: break
                self._is_initialized = False
                self.init(pos)
            else:                
                self._is_initialized = False
                self.init(pos)


    def get_ships(self, tpos: TimePosition) -> List[TargetVessel]:
        """
        Returns a list of target ships
        present in the neighborhood of the given position. 
        
        tpos: TimePosition object of agent for which 
                neighbors shall be found
        """
        # Check if cells need buffering
        self._buffer(tpos.position)
        neigbors = self._get_neighbors(tpos)
        tgts = self._construct_target_vessels(neigbors, tpos)
        # Contruct Splines for all target ships
        tgts = self._construct_splines(tgts)
        return tgts
    
    def _construct_splines(self, tgts: List[TargetVessel]) -> List[TargetVessel]:
        """
        Interpolate all target ship tracks
        """
        for tgt in tgts:
            tgt.fill_rot() # Fill rate of turn if not in data
            tgt.find_shell() # Find shell (start/end of traj) of target ship
            tgt.ts_to_unix() # Convert timestamps to unix
            tgt.construct_splines() # Construct splines
        return tgts

    def _load_cell_data(self, cell: Cell) -> pd.DataFrame:
        """
        Load AIS Messages from a given path or list of paths
        and return only messages that fall inside given `cell`-bounds.
        """
        snippets = []

        # Spatial filter for cell. Since the data is
        # provided in the lat/lon format, we have to
        # convert the cell to lat/lon as well.
        if isinstance(cell,UTMCell):
            cell = cell.to_latlon()
        spatial_filter = (
            f"{DataColumns.LON} > {cell.LONMIN} and "
            f"{DataColumns.LON} < {cell.LONMAX} and "
            f"{DataColumns.LAT} > {cell.LATMIN} and "
            f"{DataColumns.LAT} < {cell.LATMAX}"
        )
        try:
            with Loader(cell):
                for file in self.datapath:
                    df = pd.read_csv(file,sep=",")
                    df = self.filter(df) # Apply custom filter
                    df[DataColumns.TIMESTAMP] = pd.to_datetime(
                        df[DataColumns.TIMESTAMP]).dt.tz_localize(None)
                    snippets.append(df.query(spatial_filter))
        except Exception as e:
            logger.error(f"Error while loading cell data: {e}")
            raise FileLoadingError(e)

        return pd.concat(snippets)

    def _cells_for_buffering(self,pos: Position) -> List[LatLonCell]:

        # Find adjacent cells
        adjacents = self.cell_manager.adjacents(self.cell)
        # Determine current subcell
        subcell = self.cell_manager.get_subcell(pos,self.cell)
        if subcell is NOTDETERMINED:
            return NOTDETERMINED

        # Find which adjacent cells to pre-buffer
        # from the static "SUB_TO_ADJ" mapping
        selection = SUB_TO_ADJ[subcell]

        to_buffer = []
        for direction in selection:
            adj = getattr(adjacents,direction)
            if adj is OUTOFBOUNDS:
                continue
            to_buffer.append(adj)
        
        return to_buffer

    def _buffer(self, pos: Position) -> None:

        # Get cells that need to be pre-buffered
        cells = self._cells_for_buffering(pos)
        # Cancel buffering if vessel is either 
        # in the exact center of the cell or 
        # there are no adjacent cells to buffer.
        if cells is NOTDETERMINED or not cells:
            return

        def buffer():
            self.cell_data = pd.concat(
                self._load_cell_data(cell) for cell in cells
            )
            return

        # Start buffering thread if currently buffered
        # cells are not the same which need buffering
        eligible_for_buffering = [c.index for c in cells]
        if not eligible_for_buffering == self._buffered_cell_idx:
            self._buffered_cell_idx = eligible_for_buffering
            Thread(target=buffer,daemon=True).start()
        return
            
    def _build_kd_tree(self, data: pd.DataFrame) -> cKDTree:
        """
        Build a kd-tree object from the `Lat` and `Lon` 
        columns of a pandas dataframe.

        If the object was initialized with a UTMCellManager,
        the data is converted to UTM before building the tree.
        """
        assert DataColumns.LAT in data and DataColumns.LON in data, \
            "Input dataframe has no `lat` or `lon` columns"
        if isinstance(self.cell_manager,UTMCellManager):
            eastings, northings, *_ = utm.from_latlon(
                data[DataColumns.LAT].values,
                data[DataColumns.LON].values
                )
            return cKDTree(np.column_stack((northings,eastings)))
        else:
            return cKDTree(data[[DataColumns.LAT,DataColumns.LON]])

    def _get_neighbors(self, tpos: TimePosition):
        """
        Return all AIS messages that are no more than
        `self.search_radius` [nm] away from the given position.

        Args:
            tpos: TimePosition object of postion and time for which 
                    neighbors shall be found        

        """
        tpos._is_utm = self._utm
        filtered = self._time_filter(self.cell_data,tpos.timestamp,self.time_delta)
        # Check if filterd result is empty
        if filtered.empty:
            logger.warning("No AIS messages found in time-filtered cell.")
            return filtered
        tree = self._build_kd_tree(filtered)
        # The conversion to degrees is only accurate at the equator.
        # Everywhere else, the distances get smaller as lines of 
        # Longitude are not parallel. Therefore, 
         # this is a convervative estimate.         
        search_radius = (nm2m(self.search_radius) if self._utm 
                         else self.search_radius/60) # Convert to degrees
        d, indices = tree.query(
            list(tpos.position),
            k=self.max_tgt_ships,
            distance_upper_bound=search_radius
        )
        # Sort out any infinite distances
        res = [indices[i] for i,j in enumerate(d) if j != float("inf")]

        return filtered.iloc[res]

    def _construct_target_vessels(
            self, df: pd.DataFrame, tpos: TimePosition) -> List[TargetVessel]:
        """
        Walk through the rows of `df` and construct a 
        `TargetVessel` object for every unique MMSI. 
        
        The individual AIS Messages are sorted by date
        and are added to the respective TargetVessel's track attribute.
        """
        df = df.sort_values(by=DataColumns.TIMESTAMP)
        targets: dict[int,TargetVessel] = {}
        
        for mmsi,ts,lat,lon,sog,cog,turn in zip(
            df[DataColumns.MMSI],df[DataColumns.TIMESTAMP],
            df[DataColumns.LAT], df[DataColumns.LON],
            df[DataColumns.SPEED],df[DataColumns.COURSE],
            df[DataColumns.TURN]):

            msg = AISMessage(
                sender=mmsi,
                timestamp=ts.to_pydatetime(),
                lat=lat,lon=lon,
                COG=cog,SOG=sog,
                ROT=turn,
                _utm=self._utm
            )
            
            if mmsi not in targets:
                targets[mmsi] = TargetVessel(
                    ts = tpos.timestamp,
                    mmsi=mmsi,
                    track=[msg]
                )
            else: 
                v = targets[mmsi]
                v.track.append(msg)

        return self._cleanup_targets(targets,tpos)

    def _cleanup_targets(self, 
            targets: dict[int,TargetVessel], tpos: TimePosition) -> List[TargetVessel]:
        """
        Remove all vessels that have only a single
        observation or whose track lies outside
        the queried timestamp. 
        
        Check for too-slow vessels
        
        """
        for mmsi, target_ship in list(targets.items()):
            # Spline interpolation needs at least 3 points
            if (len(target_ship.track) < 3 or not
                (target_ship.track[0].timestamp < 
                tpos.timestamp < 
                target_ship.track[-1].timestamp) or
                any(v.SOG < .5 for v in target_ship.track)):
                del targets[mmsi]
        return [ship for ship in targets.values()]

    def _time_filter(self, df: pd.DataFrame, date: datetime, delta: int) -> pd.DataFrame:
        """
        Filter a pandas dataframe to only return 
        rows whose `Timestamp` is not more than 
        `delta` minutes apart from imput `date`.
        """
        assert DataColumns.TIMESTAMP in df, "No `timestamp` column found"
        date = pd.Timestamp(date, tz=None)
        dt = pd.Timedelta(delta, unit="minutes")
        mask = (
            (df[DataColumns.TIMESTAMP] > (date-dt)) & 
            (df[DataColumns.TIMESTAMP] < (date+dt))
        )
        return df.loc[mask]