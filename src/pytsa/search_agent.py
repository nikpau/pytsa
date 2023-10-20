from __future__ import annotations

from datetime import datetime
from pathlib import Path
from threading import Thread
from typing import Callable, List, Tuple, Union
from .cell_manager import NOTDETERMINED, LatLonCellManager, UTMCellManager

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import utm

from .logger import Loader, logger
from .structs import (
    BoundingBox, Cell, LatLonCell,
    Position, ShellError,TimePosition,
    OUTOFBOUNDS,SUB_TO_ADJ, Msg12318Columns, 
    Msg5Columns,UTMBoundingBox, UTMCell
)
from .targetship import TargetVessel, AISMessage, InterpolationError

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
        msg12318file: Union[Path,List[Path]],
        frame: BoundingBox,
        msg5file: Union[Path,List[Path]] = None,
        search_radius: float = 0.5, # in nautical miles
        time_delta: int = 30, # in minutes
        max_tgt_ships: int = 50,
        n_cells: int = 144,
        filter: Callable[[pd.DataFrame],pd.DataFrame] = lambda x: x
        ) -> None:
       
        """ 
        frame: BoundingBox object setting the search space
                for the taget ship extraction process.
                AIS records outside this BoundingBox are 
                not eligible for TargetShip construction.
        msg12318file: path to a csv file containing AIS messages
                    of type 1,2,3 and 18.
        search_radius: Radius around agent in which taget vessel 
                search is rolled out
        n_cells: number of cells into which the spatial extent
                of the AIS messages is divided into.
                (See also the `get_cell` function for more info)  
        max_tgt_ships: maximum number of target ships to store

        filter: function to filter out unwanted AIS messages. Default is
                the identity function.
        """

        if not isinstance(msg12318file,list):
            self.msg12318files = [msg12318file]
        else:
            self.msg12318files = msg12318file

        if msg5file is not None:
            logger.info("Initializing SearchAgent with msg5file")
            if not isinstance(msg5file,list):
                self.msg5files = [msg5file]
            else:
                self.msg5files = msg5file
        else:
            logger.info("Initializing SearchAgent without msg5file")
            self.msg5files = None

        # Spatial bounding box of current AIS message space
        self.FRAME = frame

        # Maximum number of target ships to extract
        self.max_tgt_ships = max_tgt_ships
        
        # Maximum temporal deviation of target
        # ships from provided time in `init()`
        self.time_delta = time_delta # in minutes
        
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
            self.msg5_data = self._load_msg5_data()
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
        for i,tgt in enumerate(tgts):
            tgt.fill_rot() # Calculate missing 'rate of turn' values via COG
            tgt.find_shell() # Find shell (start/end of traj) of target ship
            tgt.ts_to_unix() # Convert timestamps to unix
            try:
                tgt.interpolate() # Construct splines
            except InterpolationError as e:
                logger.warn(e)
                del tgts[i]
        return tgts
    
    def _load_msg5_data(self) -> pd.DataFrame:
        """
        Load AIS Messages from a given path or list of paths
        and return only messages that fall inside given `cell`-bounds.
        """
        snippets = []
        for file in self.msg5files:
            logger.info(f"Loading msg5 file '{file}'")
            msg5 = pd.read_csv(
                file,usecols=
                [
                    Msg5Columns.MMSI,
                    Msg5Columns.SHIPTYPE,
                    Msg5Columns.TO_BOW,
                    Msg5Columns.TO_STERN
                ]
            )
            snippets.append(msg5)
        msg5 = pd.concat(snippets)
        msg5 = msg5[msg5[Msg5Columns.MMSI].isin(self.cell_data[Msg12318Columns.MMSI])]
        return msg5

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
            f"{Msg12318Columns.LON} > {cell.LONMIN} and "
            f"{Msg12318Columns.LON} < {cell.LONMAX} and "
            f"{Msg12318Columns.LAT} > {cell.LATMIN} and "
            f"{Msg12318Columns.LAT} < {cell.LATMAX}"
        )
        try:
            with Loader(cell):
                for file in self.msg12318files:
                    msg12318 = pd.read_csv(file,sep=",")
                    msg12318 = self.filter(msg12318) # Apply custom filter
                    msg12318[Msg12318Columns.TIMESTAMP] = pd.to_datetime(
                        msg12318[Msg12318Columns.TIMESTAMP]).dt.tz_localize(None)
                    msg12318 = msg12318.drop_duplicates(
                        subset=[Msg12318Columns.TIMESTAMP,Msg12318Columns.MMSI], keep="first"
                    )
                    snippets.append(msg12318.query(spatial_filter))
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
        assert Msg12318Columns.LAT in data and Msg12318Columns.LON in data, \
            "Input dataframe has no `lat` or `lon` columns"
        if isinstance(self.cell_manager,UTMCellManager):
            eastings, northings, *_ = utm.from_latlon(
                data[Msg12318Columns.LAT].values,
                data[Msg12318Columns.LON].values
                )
            return cKDTree(np.column_stack((northings,eastings)))
        else:
            return cKDTree(data[[Msg12318Columns.LAT,Msg12318Columns.LON]])
        
    def _get_ship_type(self, mmsi: int) -> int:
        """
        Return the ship type of a given MMSI number.

        If more than one ship type is found, the first
        one is returned and a warning is logged.
        """
        st = self.msg5_data[self.msg5_data[Msg5Columns.MMSI] == mmsi]\
            [Msg5Columns.SHIPTYPE].values
        st:np.ndarray = np.unique(st)
        if st.size > 1:
            logger.warning(
                f"More than one ship type found for MMSI {mmsi}. "
                f"Found {st}. Returning {st[0]}.")
            return st[0]
        return st

    def _get_ship_length(self, mmsi: int) -> int:
        """
        Return the ship length of a given MMSI number.

        If more than one ship length is found, the first
        one is returned and a warning is logged.
        """
        raw = self.msg5_data[self.msg5_data[Msg5Columns.MMSI] == mmsi]\
            [[Msg5Columns.TO_BOW,Msg5Columns.TO_STERN]].values
        sl:np.ndarray = np.sum(raw,axis=1)
        sl = np.unique(sl)
        if sl.size > 1:
            logger.warning(
                f"More than one ship length found for MMSI {mmsi}. "
                f"Found {sl}. Returning {sl[0]}.")
            return sl[0]
        return sl

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
        # this is a conservative estimate.         
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
        df = df.sort_values(by=Msg12318Columns.TIMESTAMP)
        targets: dict[int,TargetVessel] = {}
        
        for mmsi,ts,lat,lon,sog,cog in zip(
            df[Msg12318Columns.MMSI],df[Msg12318Columns.TIMESTAMP],
            df[Msg12318Columns.LAT], df[Msg12318Columns.LON],
            df[Msg12318Columns.SPEED],df[Msg12318Columns.COURSE]):

            msg = AISMessage(
                sender=mmsi,
                timestamp=ts.to_pydatetime(),
                lat=lat,lon=lon,
                COG=cog,SOG=sog,
                _utm=self._utm
            )
            
            if mmsi not in targets:
                targets[mmsi] = TargetVessel(
                    ts = tpos.timestamp,
                    mmsi=mmsi,
                    ship_type=self._get_ship_type(mmsi),
                    length=self._get_ship_length(mmsi),
                    track=[msg]
                )
            else: 
                v = targets[mmsi]
                v.track.append(msg)

        return self._cleanup_targets(targets,tpos)

    def _cleanup_targets(self, 
            targets: dict[int,TargetVessel], tpos: TimePosition) -> List[TargetVessel]:
        """
        Flag all vessels with less than 4 points in their track for linear interpolation.
        
        
        Also remove vessels whose track lies outside
        the queried timestamp. 
        
        """
        for mmsi, target_ship in list(targets.items()):
            # A Track needs at least 2 points
            if len(target_ship.track) < 2:
                del targets[mmsi]
                continue
            elif not (
                target_ship.track[0].timestamp < 
                tpos.timestamp < 
                target_ship.track[-1].timestamp) or \
                all(v.SOG < .5 for v in target_ship.track):
                del targets[mmsi]
                continue
            # Spline interpolation needs at least 4 points
            if len(target_ship.track) < 4:
                target_ship.lininterp = True

        return [ship for ship in targets.values()]

    def _time_filter(self, df: pd.DataFrame, date: datetime, delta: int) -> pd.DataFrame:
        """
        Filter a pandas dataframe to only return 
        rows whose `Timestamp` is not more than 
        `delta` minutes apart from imput `date`.
        """
        assert Msg12318Columns.TIMESTAMP in df, "No `timestamp` column found"
        date = pd.Timestamp(date, tz=None)
        dt = pd.Timedelta(delta, unit="minutes")
        mask = (
            (df[Msg12318Columns.TIMESTAMP] > (date-dt)) & 
            (df[Msg12318Columns.TIMESTAMP] < (date+dt))
        )
        return df.loc[mask]