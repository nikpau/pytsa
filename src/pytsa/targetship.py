"""
This module aims at finding all spatio-temporal
neighbors around a given (stripped) AIS Message. 

A provided geographical area will be split into 
evenly-sized grids
"""
from __future__ import annotations
from dataclasses import dataclass

import math
from datetime import datetime, timedelta
from glob import glob
from itertools import pairwise
from threading import Thread
from typing import Callable
from pathlib import Path

import ciso8601
import geopandas as gpd
import numpy as np
import pandas as pd
import scipy.integrate
import scipy.linalg
import scipy.optimize
from matplotlib import pyplot as plt
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spilu
from scipy.spatial import cKDTree

from .logger import Loader, logger
from .structs import (
    BoundingBox, Cell, Point,
    Position, ShellError,TimePosition,
    AdjacentCells, OUTOFBOUNDS,
    SUB_TO_ADJ, DataColumns
)

# Settings for numerical integration
Q_SETTINGS = dict(epsabs=1e-13,epsrel=1e-13,limit=500)

# Type aliases
Latitude = float
Longitude = float

# Constants
PI = np.pi

# Path to geojson files containing the geometry
GEOPATH = Path("data/geometry/combined/")

@dataclass
class AISMessage:
    """
    AIS Message object
    """
    sender: int
    timestamp: datetime
    lat: Latitude
    lon: Longitude
    COG: float # Course over ground
    SOG: float # Speed over ground
    cluster_type: str =  "" # ["EN","EX","PO"]
    label_group: int | None = None

    def __post_init__(self) -> None:
        self.as_array = np.array(
            [self.lat,self.lon,self.COG,self.SOG]
        ).reshape(1,-1)



class TargetVessel:
    
    def __init__(
        self,
        ts: str | datetime, 
        mmsi: int, 
        track: list[AISMessage],
        v=True) -> None:
        
        self.ts = ts
        self.mmsi = mmsi 
        self.track = track
        self.v = v
        
    def observe(self) -> np.ndarray:
        """
        Infers
            - Course over ground (COG),
            - Speed over ground (SOG),
            - Latitude,
            - Longitude,
            
        from two AIS records encompassing the provided
        timestamp `ts`. Those two messages, `m1` and `m2`, 
        which I call their ``shell``, are used to infer the vessels 
        trajectory using a cubic spline interpolation and information 
        about the ships' position and orientation in time:
        
        Let p_A(x) be a thrid degree polynomial with coeficients a ∈ A.
        
        We will construct a spline such that:

            p_a(x1) = y1
            p_a(x2) = y2
            dp_a/dx1 = tan(k1)
            dp_a/dx2 = tan(k2)
            
        with ki, i = 1,2, being the course over ground at
        point (xi,yi) in radians.
        """
        if isinstance(self.ts,str):
            self.ts = ciso8601.parse_datetime(self.ts)
        
        m1, m2 = self._find_shell()
        p1, p2 = Point(m1.lon,m1.lat), Point(m2.lon,m2.lat)
        k1, k2 = m1.COG, m2.COG
        
        # Time difference between queried date
        # and first shell point in seconds
        delta_t = (self.ts - m1.timestamp).seconds
        
        # Time difference between both shell points
        delta_t_shell = (m2.timestamp - m1.timestamp).seconds
        
        # Fraction traveled in between shell points
        frac = delta_t/delta_t_shell

        # Calculate coefficients of 3rd degree polynomial 
        # connecting the shell points under
        # constraints described above. 
        s, s_prime, pl_int, pathlen, _swapped = self._decide(p1,p2,k1,k2)
        
        # From here on, inferred variables will start with "i_"
        
        # Speed over ground is just the average between the
        # two shell speeds weighted by the distance covered on
        # the connecting spline
        i_SOG = frac*m2.SOG + (1-frac)*m1.SOG
        
        # Longitude will be estimated by solving for the 
        # upper integration limit of the path length integral 
        # for the spline such that the length is equal to the
        # the distance traveled on the connecting spline
        if not _swapped:
            i_LON = self._lon_along_path(
                pl_int,p1,frac*pathlen,p1.is_on_left_of(p2)
            )
            i_LAT = s(i_LON)
            
            # Course over ground is the arc tangent
            # of the first derivative of our spline
            # shifted by pi/2 due to spline constraints
            if p1.is_on_left_of(p2):
                i_COG = -np.arctan(s_prime(i_LON)) + PI/2
            else: 
                i_COG = -np.arctan(s_prime(i_LON)) - PI/2
            
            # Transform from [-pi,+pi] rad to [0,360] deg
            i_COG = (i_COG % (2*PI))* 180/PI

        else: 
            i_LAT = self._lat_along_path(
                pl_int,p1,frac*pathlen,p1.is_above_of(p2)
            )
            i_LON = s(i_LAT)
            
            # Course over ground is the arc tangent
            # of the first derivative of our spline
            if not p1.is_above_of(p2):
                i_COG = np.arctan(s_prime(i_LAT))
            else:
                i_COG = np.arctan(s_prime(i_LAT)) - PI
            
            # Transform from [-pi,+pi] rad to [0,360] deg
            i_COG = (i_COG % (2*PI))* 180/PI

        obs = np.array([i_LAT,i_LON,i_COG,i_SOG])

        if self.v:
            self._v(s,obs,m1,m2,_swapped)
        
        return obs
    
    @staticmethod
    def _lon_along_path(f: Callable[[Latitude],float], 
        reference: Point, dist: float, flip_limits: bool)-> Longitude:
        """
        Calculate the Longitude (x-coord) of `f` that lies
        `dist` apart from the `reference` point 
        along the functions' curve
        """
        def pl_nwtn(k):
            if flip_limits:
                lower, upper = reference.x, k
            else: lower, upper = k, reference.x
            _i = scipy.integrate.quad(f,lower,upper,**Q_SETTINGS)[0]
            return _i - dist

        return scipy.optimize.newton(pl_nwtn,x0=reference.x)

    @staticmethod
    def _lat_along_path(f: Callable[[Longitude],float], 
        reference: Point, dist: float, flip_limits: bool)-> Latitude:
        """
        Calculate the Latitude (y-coord) of `f` that lies
        `dist` apart from the `reference` point 
        along the functions' curve
        """
        def pl_nwtn(k):
            if flip_limits:
                lower, upper = k, reference.y
            else: lower, upper = reference.y, k
            _i = scipy.integrate.quad(f,lower,upper,**Q_SETTINGS)[0]
            return _i - dist

        return scipy.optimize.newton(pl_nwtn,x0=reference.y)

    def _decide(
        self,p1: Point,p2: 
        Point,k1: float,k2: float
        ) -> tuple[Callable[[Latitude | Longitude],Latitude | Longitude],
        Callable[[Latitude | Longitude],Latitude | Longitude],
        Callable[[Latitude | Longitude],float],
        bool]:
        """
        Decide whether to use the spline along the
        x (lon) axis or y (lat) axis
        """
        s,sp,pl_int,pathlen = self._spline(p1,p2,k1,k2)
        s2,s2p,pl_int2,pathlen2 = self._spline_swapped(p1,p2,k1,k2)
        
        if pathlen < pathlen2:
            _swapped = False
            return s,sp,pl_int,pathlen, _swapped
        else:
            _swapped = True
            return s2,s2p,pl_int2,pathlen2, _swapped
    
    def _v(self,s,obs,m1:AISMessage,m2:AISMessage, _swapped: bool): 

        f, ax = plt.subplots(1,1)

        p1, p2 = Point(m1.lon,m1.lat), Point(m2.lon,m2.lat)
        ax.scatter([p1.x,p2.x],[p1.y,p2.y],c="#3a5a40",s=72)
        ax.annotate(
            f"Message at {m1.timestamp!s}\nLAT:{m1.lat}°N\nLON:{m1.lon}°E"
            f"\nSOG: {m1.SOG} kn\nCOG: {m1.COG}°",
            textcoords='figure fraction',
            xy=(p1.x,p1.y),xytext=(0.2,0.2),
            arrowprops=dict(facecolor="#6c757d", shrink=0.02,ec="none")
        )
        ax.annotate(
            f"Message at {m2.timestamp!s}\nLAT: {m2.lat}°N\nLON: {m2.lon}°E"
            f"\nSOG: {m2.SOG} kn\nCOG: {m2.COG}°",
            textcoords='figure fraction',
            xy=(p2.x,p2.y),xytext=(0.7,0.7),
            arrowprops=dict(facecolor="#6c757d", shrink=0.02,ec="none")
        )
        if not _swapped:
            v = np.linspace(p1.x-0.001*p1.x,p2.x+0.001*p1.x,3000)
            vals = [s(x) for x in v]
            ax.plot(v,vals,c="#1d3557",linewidth=2)
        else:
            v = np.linspace(p1.y-0.001*p1.y,p2.y+0.001*p1.y,3000)
            vals = [s(y) for y in v]
            ax.plot(vals,v,c="#1d3557",linewidth=2)

        ax.scatter(obs[1],obs[0], c="#e63946", s=72)
        ax.annotate(
            f"Interpolated position at {self.ts!s}"
            f"\nLAT: {obs[0]:.2f}°N\nLON: {obs[1]:.2f}°E\nSOG: "
            f"{obs[3]:.2f} kn\nCOG: {obs[2]:.2f}°",
            xy=(obs[1],obs[0]),
            xytext=(0.5,0.2),
            textcoords='figure fraction',
            arrowprops=dict(facecolor="#6c757d", shrink=0.02,ec="none")
        )
        ax.plot(
            (obs[1],obs[1]+np.sin(_dtr(obs[2]))*0.005),
            (obs[0],obs[0]+np.cos(_dtr(obs[2]))*0.005),
            c="#e63946")
        ax.plot(
            (p1.x,p1.x+np.sin(_dtr(m1.COG))*0.005),
            (p1.y,p1.y+np.cos(_dtr(m1.COG))*0.005),
            c="#e63946")
        ax.plot(
            (p2.x,p2.x+np.sin(_dtr(m2.COG))*0.005),
            (p2.y,p2.y+np.cos(_dtr(m2.COG))*0.005),
            c="#e63946")
        plt.title("Vessel position estimation from AIS Messages")
        ax.set_xlabel("Longitude [°]")
        ax.set_ylabel("Latitude [°]")
        plt.show()


    
    def _find_shell(self) -> tuple[AISMessage,AISMessage]:
        """
        Find the two AIS messages encompassing
        the objects' timestamp `self.ts`. I call this the 
        timestamp shell. The messages returned from this
        function will be referred to as `shell points`
        """
        # The messages are expected to be already sorted
        # by time in ascending order. 
        # We also do not need to check for out-of-range
        # errors, as only messages with at least two track
        # elements are saved.
        for idx, message in enumerate(self.track):
            if message.timestamp > self.ts:
                return self.track[idx-1], self.track[idx]
            
        msg = (
            f"No outer shell for vessel `{self.mmsi}` "
            f"could be found at timestamp `{self.ts!s}`"
        )
        raise ShellError(msg)
    
    @staticmethod
    def _spline(
        p1: Point, p2: Point, k1:float, k2:float
        ) -> tuple[
            Callable[[float],float],
            Callable[[float],float],
            Callable[[float],float],
            float]:
        
        # We need to add 90° to the current course of the vessel
        # as in our local coordinate system a course of 0° would
        # correspond to an infinite slope while a course of 90°
        # corresponds to zero slope. However, we infer the slope
        # via the tangent of the reported course, we must shift it 
        # by pi/2 to meet our requirements.

        # Since this alteration is only needed for the parameter 
        # estimation, we undo the change by substracting pi/2 
        # from the derivative (s_prime) at the end.
        k1, k2 = _dtr2((k1-90)%180), _dtr2((k2-90)%180)
        X = np.array(
            [
                [p1.x**3,p1.x**2,p1.x,1],
                [p2.x**3,p2.x**2,p2.x,1],
                [3*p1.x**2,2*p1.x,1.,0.],
                [3*p2.x**2,2*p2.x,1.,0.]
            ]
        )
        y = np.array(
            [
                [p1.y,p2.y,np.tan(k1),np.tan(k2)]
            ]
        )
        #sol = scipy.linalg.solve(X,y.T)
        X = csc_matrix(X)
        X_inv = spilu(X)
        sol = X_inv.solve(y.T)
        
        #sol2 = np.dot(np.linalg.pinv(X),b.T)
        a3,a2,a1,a0 = sol.flatten().tolist()

        def s(x: float) -> float:
            return a3*x**3+a2*x**2+a1*x+a0
        
        def s_prime(x: float) -> float:
            return 3*a3*x**2+2*a2*x+a1
        
        # Calculate the length of the spline 
        # between its two shell points via the 
        # path lenght integral
        pl_int = lambda x: np.sqrt(1+s_prime(x)**2)
        if p1.is_on_left_of(p2):
            lower,upper = p1.x, p2.x
        else:
            lower,upper = p2.x, p1.x
        pathlen = scipy.integrate.quad(pl_int,lower,upper)[0]
        
        return s, s_prime, pl_int, pathlen
    
    @staticmethod
    def _spline_swapped(
        p1: Point, p2: Point, k1:float, k2:float
        ) -> tuple[
            Callable[[float],float],
            Callable[[float],float],
            Callable[[float],float],
            float]:
        
        """
        Same spline as above with the axes flipped
        """
        k1, k2 = _dtr2(k1%180), _dtr2(k2%180)
        Y = np.array(
            [
                [p1.y**3,p1.y**2,p1.y,1],
                [p2.y**3,p2.y**2,p2.y,1],
                [3*p1.y**2,2*p1.y,1.,0.],
                [3*p2.y**2,2*p2.y,1.,0.]
            ]
        )
        x = np.array(
            [
                [p1.x,p2.x,np.tan(k1),np.tan(k2)]
            ]
        )
        #sol = scipy.linalg.solve(Y,x.T) 
        Y = csc_matrix(Y)  
        Y_inv = spilu(Y)
        sol = Y_inv.solve(x.T)
        
        a3,a2,a1,a0 = sol.flatten().tolist()

        def s(y: float)->float:
            return a3*y**3+a2*y**2+a1*y+a0
        
        def s_prime(y: float)->float:
            return 3*a3*y**2+2*a2*y+a1
        
        # Calculate the length of the spline 
        # between its two shell points via the 
        # path lenght integral
        pl_int = lambda y: np.sqrt(1+s_prime(y)**2)
        if not p1.is_above_of(p2):
            lower,upper = p1.y, p2.y
        else:
            lower,upper = p2.y, p1.y
        pathlen = scipy.integrate.quad(pl_int,lower,upper)[0]
        
        return s, s_prime, pl_int, pathlen
    
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
        datapath: str | list[str],
        frame: BoundingBox,
        search_radius: float = 0.5, # in nautical miles
        max_tgt_ships: int = 50,
        n_cells: int = 144,
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
        """
        self.v = v

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
        self.time_delta = 5
        
        # Search radius in [°] around agent
        self.search_radius = search_radius
        
        # Number of cells, into which the spatial extent
        # of AIS records is divided into
        self.n_cells = n_cells
        
        # Init cell manager 
        self.cell_manager = CellManager(frame,n_cells)

        # List of cell-indicies of all 
        # currently buffered cells
        self._buffered_cell_idx = []
        
        self._is_initialized = False

    def init(self, pos: Position, *, supress_re_init=False)-> None:
        """
        tpos: TimePosition object for which 
                TargetShips shall be extracted from 
                AIS records.
        """
        if not self._is_initialized:
            # Current Cell based on init-position
            self.cell = self.cell_manager.get_cell_by_position(pos)
            
            # Load AIS Messages for specific cell
            self.cell_data = self._load_cell_data(self.cell)
            self._is_initialized = True
            logger.info(
                "Target Ship search agent initialized at "
                f"{pos.lat:.3f}°N, {pos.lon:.3f}°E"
            )
        else:
            if not supress_re_init:
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
            logger.warn(
                "Re-initialization skipped. Use the 'supress_re_init' flag"
                "to override this behavior"
            )

    def get_ships(self, tpos: TimePosition) -> list[TargetVessel]:
        """
        Returns a list of target ships
        present in the neighborhood of the given position. 
        
        tpos: TimePosition object of agent for which 
                neighbors shall be found
        """
        # Check if cells need buffering
        self._buffer(tpos.position)
        neigbors = self._get_neighbors(tpos)
        return self._construct_target_vessels(neigbors, tpos)

    def _load_cell_data(self, cell: BoundingBox) -> pd.DataFrame:
        """
        Load AIS Messages from a given path or list of paths
        and return only messages that fall inside given `cell`-bounds.
        """
        snippets = []
        spatial_filter = (
            f"{DataColumns.LON} > {cell.LONMIN} and "
            f"{DataColumns.LON} < {cell.LONMAX} and "
            f"{DataColumns.LAT} > {cell.LATMIN} and "
            f"{DataColumns.LAT} < {cell.LATMAX}"
        )
        with Loader(cell):
            for file in self.datapath:
                df = pd.read_csv(file,sep=",",usecols=list(range(10)))
                df[DataColumns.TIMESTAMP] = pd.to_datetime(df[DataColumns.TIMESTAMP])
                snippets.append(df.query(spatial_filter))

        return pd.concat(snippets)

    def _cells_for_buffering(self,pos: Position) -> list[Cell]:

        # Find adjacent cells
        adjacents = self.cell_manager.adjacents(self.cell)
        # Determine current subcell
        subcell = self.cell_manager.get_subcell(pos,self.cell)

        # Find which adjacent cells to pre-buffer
        # from the static "SUB_TO_ADJ" mapping
        selection = SUB_TO_ADJ[subcell]

        to_buffer = []
        for direction in selection:
            to_buffer.append(getattr(adjacents,direction))
        
        return to_buffer

    def _buffer(self, pos: Position) -> None:

        # Get cells that need to be pre-buffered
        cells = self._cells_for_buffering(pos)

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
        """
        assert DataColumns.LAT in data and DataColumns.LON in data, \
            "Input dataframe has no `lat` or `lon` columns"
        return cKDTree(data[[DataColumns.LAT,DataColumns.LON]])

    def _get_neighbors(self, tpos: TimePosition):
        """
        Return all AIS messages that are in
        the proximity [±0.5°] of the provided position

        Args:
            tpos: TimePosition object of postion and time for which 
                    neighbors shall be found        

        """
        filtered = self._time_filter(self.cell_data,tpos.timestamp,self.time_delta)
        tree = self._build_kd_tree(filtered)
        d, indices = tree.query(
            [tpos.lat,tpos.lon],
            k=self.max_tgt_ships,
            # The conversion to degrees is only accurate at the equator.
            # Everywhere else, the distances get smaller as lines of 
            # Longitude are not parallel. Therefore, 
            # this is a convervative estimate. 
            distance_upper_bound=self.search_radius/60 # Convert to degrees
        )
        # Sort out any infinite distances
        res = [indices[i] for i,j in enumerate(d) if j != float("inf")]

        return filtered.iloc[res]

    def _construct_target_vessels(
            self, df: pd.DataFrame, tpos: TimePosition) -> list[TargetVessel]:
        """
        Walk through the rows of `df` and construct a 
        `TargetVessel` object for every unique MMSI. 
        
        The individual AIS Messages are sorted by date
        and are added to the respective TargetVessel's track attribute.
        """
        df = df.sort_values(by=DataColumns.TIMESTAMP)
        targets: dict[int,TargetVessel] = {}
        
        for mmsi,ts,lat,lon,sog,cog in zip(
            df[DataColumns.MMSI],df[DataColumns.TIMESTAMP],
            df[DataColumns.LAT], df[DataColumns.LON],
            df[DataColumns.SPEED],df[DataColumns.COURSE]):
            
            if mmsi not in targets:
                targets[mmsi] = TargetVessel(
                    v = self.v,
                    ts = tpos.timestamp,
                    mmsi=mmsi,
                    track=[AISMessage(
                        sender=mmsi,
                        timestamp=ts.to_pydatetime(),
                        lat=lat,lon=lon,
                        COG=cog,SOG=sog
                    )]
                )
            else: 
                v = targets[mmsi]
                v.track.append(AISMessage(
                        sender=mmsi,
                        timestamp=ts.to_pydatetime(),
                        lat=lat,lon=lon,
                        COG=cog,SOG=sog
                    ))
        return self._post_filter(targets,tpos)

    def _post_filter(self, 
            targets: dict[int,TargetVessel], tpos: TimePosition) -> list[TargetVessel]:
        """
        Remove all vessels that have only a single
        observation or whose track lies outside
        the queried timestamp. 

        Furthermore we need to check for too-slow vessels
        as a large temporal gap for slow vessels leads to 
        a very high input senitivity and thus to ill-conditioned
        matricies for the spline interpolation
        """
        for mmsi, target_ship in list(targets.items()):
            if (len(target_ship.track) < 2 or not 
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
        dt = timedelta(minutes=delta)
        mask = (df[DataColumns.TIMESTAMP] > (date-dt)) & (df[DataColumns.TIMESTAMP] < (date+dt))
        return df.loc[mask]

class CellManager:
    """
    Cell Manager object implementing several utility
    functions to generate and manage cells generated from
    a geographical frame:
    
    Args:
        frame (BoundingBox): A geographical area which
                                is to be converted into a 
                                grid of `n_cells` cells.
        n_cells (int):      Number of cells, the provided 
                            frame is divided into.
    """
    def __init__(self, frame: BoundingBox, n_cells: int) -> None:
        
        # Bounding box of the entire frame
        self.frame = frame
        
        # Number of cells, the frame 
        # shall be divided into
        self.n_cells = n_cells
        
        # Dict holding all cells in the
        # current frame
        self.cells: dict[int,Cell] = {}

        # Get the number of rows and colums.
        # Since the grid is always square,
        # they are the same
        self.nrows = self.ncols = self._side_length()
        
        self._setup()
        
    def _setup(self) -> None:
        """
        Create individual Cells from
        provided frame and cell number and assigns
        ascending indices. The first cell
        starts in the North-West:
        
        N
        ^
        |
        |----> E
        
        Example of a 5x5 Grid:
        ------------------------------------
        |  0   |  1   |  2   |  3   |  4   |
        |      |      |      |      |      |
        ------------------------------------
        |  5   |  6   |  7   |  8   |  9   |
        |      |      |      |      |      |            
        ------------------------------------       
        | ...  | ...  | ...  | ...  | ...  |             
                
        """
        latbp, lonbp = self._breakpoints()
        cidx = 0
        latbp= sorted(latbp)[::-1] # Descending Latitudes
        lonbp = sorted(lonbp) # Ascending Longitudes
        for lat_max, lat_min in pairwise(latbp):
            for lon_min, lon_max in pairwise(lonbp):
                self.cells[cidx] = Cell(
                    LATMIN=lat_min,LATMAX=lat_max,
                    LONMIN=lon_min,LONMAX=lon_max,
                    index=cidx
                )
                cidx += 1

    def _side_length(self) -> int:
        """
        Get the side length (number of cells per side) for the
        total number of cells specified during instantiation.

        The number of cells to be constructed `n_cells` must be a 
        square number. If it is not the next closest square number
        will be used instead 
        """        
        root = math.sqrt(self.n_cells)
        if not root%1==0:
            root = round(root)
            logger.info(
                "Provided argument to `n_cells` "
                "is no square number. Rounding to next square."
                f"\nProvided: {self.n_cells}\nUsing: {root**2}"
            )
        return int(root)

            
    def _breakpoints(self) -> tuple[list[float],list[float]]:
        """
        Create breakpoints of latitude and longitude
        coordinates from initial ``frame`` resembling cell borders.
        """
        f = self.frame
        nrow = self.nrows
        ncol = self.ncols

        # Set limiters for cells 
        lat_extent = f.LATMAX-f.LATMIN
        lon_extent = f.LONMAX-f.LONMIN
        # Split into equal sized cells
        csplit = [f.LONMIN + i*(lon_extent/ncol) for i in range(ncol)]
        rsplit = [f.LATMIN + i*(lat_extent/nrow) for i in range(nrow)]
        csplit.append(f.LONMAX) # Add endpoints
        rsplit.append(f.LATMAX) # Add endpoints
        return rsplit, csplit
            
    def inside_frame(self, pos: Position) -> bool:
        """Boolean if the provided position
        is inside the bounds of the cell 
        manager's frame"""
        lat, lon = pos
        return (
            lat > self.frame.LATMIN and lat < self.frame.LATMAX and
            lon > self.frame.LONMIN and lon < self.frame.LONMAX
        )
        
    @staticmethod
    def inside_cell(pos: Position, cell: Cell) -> bool:
        """Boolean if the provided position
        is inside the bounds of the provided cell"""
        lat, lon = pos
        return (
            lat > cell.LATMIN and lat < cell.LATMAX and
            lon > cell.LONMIN and lon < cell.LONMAX
        )
                
    def get_cell_by_index(self, index: int) -> Cell:
        """Get cell by index"""
        assert not index > len(self.cells) 
        return self.cells[index]
    
    def get_cell_by_position(self, pos: Position) -> Cell:
        """Get Cell by Lat-Lon Position"""
        assert self.inside_frame(pos)
        for cell in self.cells.values():
            if self.inside_cell(pos,cell):
                return cell
        raise RuntimeError(
            "Could not locate cell via position.\n"
            f"Cell: {cell!r}\nPosition: {pos!r}"
            )

    def adjacents(self, cell: Cell = None, index: int = None) -> AdjacentCells:
        """
        Return an "AdjacentCells" instance containig all adjacent 
        cells named by cardinal directions for the provided cell or its index.
        If an adjacent cell in any direction is out of bounds
        an "OUTOFBOUNDS" instance will be returned instead of a cell.
        """
        if cell is not None and index is not None:
            raise RuntimeError(
                "Please provide either a cell or an index to a cell, not both."
            )
        if cell is not None:
            index = cell.index

        grid = np.arange(self.nrows**2).reshape(self.ncols,self.nrows)

        # Get row and column of the input cell
        row, col = index//self.nrows, index%self.ncols
        
        # Shorten to avoid clutter
        _cbi = self.get_cell_by_index

        # Calc adjacents for cardinal directions.
        # (Surely not the most efficient, but it works).
        n = _cbi(grid[row-1,col]) if row-1 >= 0 else OUTOFBOUNDS
        s = _cbi(grid[row+1,col]) if row+1 < self.nrows else OUTOFBOUNDS
        e = _cbi(grid[row,col+1]) if col+1 < self.ncols else OUTOFBOUNDS
        w = _cbi(grid[row,col-1]) if col-1 >= 0 else OUTOFBOUNDS

        # Composite directions
        ne = _cbi(grid[row-1,col+1]) if (row-1 >= 0 and col+1 <= self.nrows) else OUTOFBOUNDS
        se = _cbi(grid[row+1,col+1]) if (row+1 < self.nrows and col+1 < self.nrows) else OUTOFBOUNDS
        sw = _cbi(grid[row+1,col-1]) if (row+1 < self.nrows and col-1 >= 0) else OUTOFBOUNDS
        nw = _cbi(grid[row-1,col-1]) if (row-1 >= 0 and col-1 >= 0) else OUTOFBOUNDS

        return AdjacentCells(
            N=n,NE=ne,E=e,SE=se,
            S=s,SW=sw,W=w,NW=nw
        )

    def get_subcell(self,pos: Position, cell: Cell) -> int:
        """
        Each cell will be divided into four subcells
        by bisecting the cell in the middle along each
        axis:

        Example of a singe cell with its four subcells

        N
        ^
        |---------|---------|
        |    1    |    2    |
        |         |         |
        ---------------------
        |    3    |    4    |
        |         |         |
        |---------|---------|-> E

        We will later use these subcells to determine 
        which adjacent cell to pre-buffer.
        """
        assert self.inside_cell(pos,cell)
        lonext = cell.LONMAX-cell.LONMIN # Longitudinal extent of cell
        lonhalf = cell.LONMAX - lonext/2
        latext = cell.LATMAX-cell.LATMIN # Lateral extent of cell
        lathlalf = cell.LATMAX - latext/2
        if pos.lon < lonhalf and pos.lat > lathlalf:
            return 1
        elif pos.lon > lonhalf and pos.lat > lathlalf:
            return 2
        elif pos.lon < lonhalf and pos.lat < lathlalf:
            return 3
        elif pos.lon > lonhalf and pos.lat < lathlalf:
            return 4
        
    def plot_grid(self,*, f: plt.Figure = None, ax: plt.Axes = None) -> None:
        # Load north sea geometry
        if f is None and ax is None:
            f, ax = plt.subplots()
        files = GEOPATH.glob("*.geojson")
        for file in files:
            f = gpd.read_file(file)
            f.plot(ax=ax,color="#283618",markersize=0.5,marker = ".")
            
        lats = [c.LATMIN for c in self.cells.values()]
        lons = [c.LONMIN for c in self.cells.values()]
        lats.append(self.frame.LATMAX)
        lons.append(self.frame.LONMAX)

        # Place grid according to global frame
        ax.hlines(
            lats,self.frame.LONMIN,
            self.frame.LONMAX,colors="#6d6875"
        )
        ax.vlines(
            lons,self.frame.LATMIN,
            self.frame.LATMAX,colors="#6d6875"
        )
        if f is None and ax is None:
            plt.show()
        

def _dtr(angle: float) -> float:
    """Transform from [0,360] to 
    [-180,180] in radians"""
    o = ((angle-180)%360)-180
    return o/180*np.pi

def _dtr2(angle: float) -> float:
    """Transform from [0,360] to 
    [-180,180] in radians and switch
    angle rotation order since python
    rotates counter clockwise while
    heading is provided clockwise"""
    o = ((angle-180)%360)-180
    return -o/180*np.pi
