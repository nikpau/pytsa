"""
This module aims at finding all spatio-temporal
neighbors around a given (stripped) AIS Message. 

A provided geographical area will be split into 
evenly-sized grids
"""
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
import scipy.integrate
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

# Settings for numerical integration
Q_SETTINGS = dict(epsabs=1e-13,epsrel=1e-13,limit=500)



# Type aliases
Latitude = float
Longitude = float

# Constants
PI = np.pi

# Exceptions
class FileLoadingError(Exception):
    pass

class SplineInterpolationError(Exception):
    pass


@dataclass
class AISMessage:
    """
    AIS Message object
    """
    sender: int
    timestamp: datetime
    lat: Latitude
    lon: Longitude
    COG: float # Course over ground [degrees]
    SOG: float # Speed over ground [knots]
    ROT: float # Rate of turn [degrees/minute]
    dROT: float = None # Change of ROT [degrees/minute²]
    _utm: bool = False

    def __post_init__(self) -> None:
        self.easting, self.northing, self.zone_number, self.zone_letter = utm.from_latlon(
            self.lat, self.lon
        )
        self.as_array = np.array(
            [self.northing,self.easting,self.COG,self.SOG]
        ).reshape(1,-1) if self._utm else np.array(
            [self.lat,self.lon,self.COG,self.SOG]
        ).reshape(1,-1)
        self.ROT = self._rot_handler(self.ROT)
    
    def _rot_handler(self, rot: float) -> float:
        """
        Handles the Rate of Turn (ROT) value
        """
        try: rot = float(rot) 
        except: return None
        sign = np.sign(rot)
        if abs(rot) == 127 or abs(rot) == 128:
            return None
        else:
            return sign * (rot / 4.733)**2
        

class TargetVessel:
    
    def __init__(
        self,
        ts: Union[str,datetime], 
        mmsi: int, 
        track: List[AISMessage],
        v=True) -> None:
        
        self.ts = ts
        self.mmsi = mmsi 
        self.track = track
        self.v = v
        self.fail_count = 0
        self._fill_rot()

    def observe(self, skip_linear = False) -> np.ndarray:
        """
        Infers
            - Course over ground (COG) [degrees],
            - Speed over ground (SOG) [knots],
            - Latitude (northing) [degrees] (meters),
            - Longitude (easting) [degrees] (meters),
            - Rate of turn (ROT) [degrees/minute],
            - Change of ROT [degrees/minute²],
            
        from two AIS records encompassing the provided
        timestamp `self.ts`. Those two messages
        are used to perform a spline interpolation.
        If the spline interpolation fails, a linear 
        interpolation is performed instead.

        If `skip_linear` is set to `True`, the linear
        interpolation will be skipped and a
        `SplineInterpolationError` will be raised instead.
        This can be seen as a "strict" mode.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self._observe(skip_linear)

    def _observe(self, skip_linear = False) -> np.ndarray:

        try:
            return self._observe_spline()
        except Exception as e:
            if skip_linear:
                raise SplineInterpolationError(
                    "Linear interpolation was skipped."
                )
            else:
                self.fail_count += 1
                logger.warning(
                    f"Spline interpolation failed: {e}\n"
                    "Falling back to linear interpolation. "
                    f"Fail count: {self.fail_count}"
                    )
                return self._observe_linear()

    def _observe_linear(self) -> np.ndarray:
        """
        Fallback method in case the spline interpolation
        fails. It is a simple linear interpolation between
        the two AIS messages encompassing the provided timestamp.
        """
        if isinstance(self.ts,str):
            self.ts = ciso8601.parse_datetime(self.ts)
        
        m1, m2 = self._find_shell()
        if m1._utm and m2._utm:
            p1, p2 = Point(m1.easting,m1.northing), Point(m2.easting,m2.northing)
        else:
            p1, p2 = Point(m1.lon,m1.lat), Point(m2.lon,m2.lat)
        k1, k2 = m1.COG, m2.COG
        
        # Time difference between queried date
        # and first shell point
        dt = (self.ts - m1.timestamp).total_seconds()
        
        # Time difference between first and second shell point
        dt2 = (m2.timestamp - m1.timestamp).total_seconds()
        
        # Linear interpolation
        i_EAST = p1.x + (p2.x - p1.x) * dt / dt2
        i_NORTH = p1.y + (p2.y - p1.y) * dt / dt2
        
        # Linear interpolation of COG
        i_COG = k1 + (k2 - k1) * dt / dt2
        
        # Linear interpolation of SOG
        i_SOG = m1.SOG + (m2.SOG - m1.SOG) * dt / dt2

        # Linear interpolation of ROT
        if m1.ROT is None or m2.ROT is None:
            i_ROT = 0
        else:
            i_ROT = m1.ROT + (m2.ROT - m1.ROT) * dt / dt2

        # Linear interpolation of dROT
        if m1.dROT is None or m2.dROT is None:
            i_dROT = 0
        else:
            i_dROT = m1.dROT + (m2.dROT - m1.dROT) * dt / dt2

        # Either [Lat,Lon,COG,SOG] or [Northing,Easting,COG,SOG]
        return np.array([i_NORTH,i_EAST,i_COG,i_SOG,i_ROT,i_dROT])
        
    def _observe_spline(self) -> np.ndarray:
        """
        Two AIS messages, `m1` and `m2`, 
        which I call ``shell``, are used to infer the vessels 
        trajectory anywhere betweeen using a cubic spline 
        interpolation and information about the ships' position 
        and orientation in time:
        
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
        if m1._utm and m2._utm:
            p1, p2 = Point(m1.easting,m1.northing), Point(m2.easting,m2.northing)
        else:
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
        s, s_prime, pl_int, pathlen, _swapped = self.spline(p1,p2,k1,k2)
        
        # From here on, inferred variables will start with "i_"
        
        # Speed over ground is the average between the
        # two shell speeds weighted by the distance covered on
        # the connecting spline
        i_SOG = frac*m2.SOG + (1-frac)*m1.SOG

        # Rate of turn is the average between the
        # two shell rates of turn weighted by the distance covered on
        # the connecting spline

        # We need to be careful here because the rate of turn
        # and its first derivative may not be defined for
        # both shell points. If it is not,
        # we will set the rate of turn to 0.
        if m2.ROT is None or m1.ROT is None:
            i_ROT = 0
        else:
            i_ROT = frac*m2.ROT + (1-frac)*m1.ROT

        # First derivative of the rate of turn
        if m2.dROT is None or m1.dROT is None:
            i_dROT = 0
        else:
            i_dROT = frac*m2.dROT + (1-frac)*m1.dROT
        
        # Longitude will be estimated by solving for the 
        # upper integration limit of the path length integral 
        # for the spline such that the length is equal to the
        # the distance traveled on the connecting spline
        if not _swapped:
            i_EAST = self._lon_along_path(
                pl_int,p1,frac*pathlen,p1.is_on_left_of(p2)
            )
            i_NORTH = s(i_EAST)
            
            # Course over ground is the arc tangent
            # of the first derivative of our spline
            # shifted by pi/2 due to spline constraints
            if p1.is_on_left_of(p2):
                i_COG = -np.arctan(s_prime(i_EAST)) + PI/2
            else: 
                i_COG = -np.arctan(s_prime(i_EAST)) - PI/2
            
            # Transform from [-pi,+pi] rad to [0,360] deg
            i_COG = (i_COG % (2*PI))* 180/PI

        else: 
            i_NORTH = self._lat_along_path(
                pl_int,p1,frac*pathlen,p1.is_above_of(p2)
            )
            i_EAST = s(i_NORTH)
            
            # Course over ground is the arc tangent
            # of the first derivative of our spline
            if not p1.is_above_of(p2):
                i_COG = np.arctan(s_prime(i_NORTH))
            else:
                i_COG = np.arctan(s_prime(i_NORTH)) - PI
            
            # Transform from [-pi,+pi] rad to [0,360] deg
            i_COG = (i_COG % (2*PI))* 180/PI

        # Either [Lat,Lon,COG,SOG,...] or [Northing,Easting,COG,SOG,...]
        obs = np.array([i_NORTH,i_EAST,i_COG,i_SOG,i_ROT,i_dROT])

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
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
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

    def spline(
        self,p1: Point,p2: 
        Point,k1: float,k2: float
        ) -> Tuple[Callable[[Union[Latitude,Longitude]],Union[Latitude,Longitude]],
        Callable[[Union[Latitude,Longitude]],Union[Latitude,Longitude]],
        Callable[[Union[Latitude,Longitude]],float],
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


    def _fill_rot(self) -> None:
        """
        Fill out missing rotation data and first 
        derivative of roatation by inferring it from 
        the previous and next AIS messages' headings.
        """
        for idx, msg in enumerate(self.track):
            if idx == 0 or idx == len(self.track)-1:
                continue
            # Fill out missing ROT data
            if msg.ROT is None:
                num = self.track[idx].COG - self.track[idx-1].COG 
                den = (self.track[idx].timestamp - self.track[idx-1].timestamp).seconds()*60
                self.track[idx].ROT = num/den

            # Calculate first derivative of ROT
            num = self.track[idx+1].ROT - self.track[idx].ROT
            den = (self.track[idx+1].timestamp - self.track[idx].timestamp).seconds()*60
            self.track[idx].dROT = num/den
    
    def _find_shell(self) -> Tuple[AISMessage,AISMessage]:
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
        ) -> Tuple[
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
        ) -> Tuple[
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
        return self._construct_target_vessels(neigbors, tpos)

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
                    v = self.v,
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

def m2nm(m: float) -> float:
    """Convert meters to nautical miles"""
    return m/1852

def nm2m(nm: float) -> float:
    """Convert nautical miles to meters"""
    return nm*1852
