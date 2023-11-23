from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import ciso8601
from typing import List, Union
import numpy as np

Latitude  = float
Longitude = float
MMSI = int
UNIX_TIMESTAMP = int

# Empirical quantiles for the
# change in heading [°] between two 
# consecutive messages.
# These values have been obtained
# from the AIS data set.
HQUANTILES = {
    99: [-174.1,163.1],
    95: [-109.3,56.8],
    90: [-44.9,25.1]
}

# Empirical quantiles for the
# change in speed [kn] between two
# consecutive messages.
# These values have been obtained
# from the AIS data set.
SQUANTILES = {
    99: 7.4,
    95: 2.7,
    90: 1.4
}
    
# Empirical quantiles for the
# distance [mi] between two consecutive
# messages.
# These values have been obtained
# from the AIS data set.
DQUANTILES = {
    99: 1.81,
    95: 1.14,
    90: 0.9
}

@dataclass
class Position:
    """
    Position object
    """
    lat: Latitude
    lon: Longitude
    
    def __hash__(self) -> int:
        return hash((self.lat,self.lon))

class NONAME_TYPE:
    pass
NONAME = NONAME_TYPE()

class NOINDEX_TYPE:
    pass
NOINDEX = NOINDEX_TYPE()

class ShellError(Exception):
    pass

@dataclass
class AISMessage:
    """
    AIS Message object
    """
    sender: MMSI
    timestamp: UNIX_TIMESTAMP
    lat: Latitude
    lon: Longitude
    COG: float # Course over ground [degrees]
    SOG: float # Speed over ground [knots]
    ROT: float = None # Rate of turn [degrees/minute]
    dROT: float = None # Change of ROT [degrees/minute²]
    
    # Fields for Trajectory Clustering ------------------
    _cluster_type: str = "" # ["EN","EX","PO"]
    _label_group: Union[int,None] = None

    def __post_init__(self) -> None:

        self.as_array = np.array(
            [self.lat,self.lon,self.COG,self.SOG]
        ).reshape(1,-1)
        self.ROT = self._rot_handler(self.ROT)
    
    def _rot_handler(self, rot: float) -> float:
        """
        Handles the Rate of Turn (ROT) value
        """
        try: 
            rot = float(rot) 
        except: 
            return None
        
        sign = np.sign(rot)
        if abs(rot) == 127 or abs(rot) == 128:
            return None
        else:
            return sign * (rot / 4.733)**2
        

class ShipType(Enum):
    """
    Dataclass to store the type of vessel
    as defined by the AIS standard.
    See here for more information:
    https://coast.noaa.gov/data/marinecadastre/ais/VesselTypeCodes2018.pdf
    """
    WIG = range(20,30) # Wing in ground
    FISHING = 30
    TUGTOW = range(31,33)
    MILITARY = 35
    SAILING = 36
    PLEASURE = 37
    HSC = range(40,50) # High speed craft
    PASSENGER = range(60,70)
    CARGO = range(70,80)
    TANKER = range(80,90)
    OTHER = range(90,100)

@dataclass
class BoundingBox:
    """
    Geographical frame 
    containing longitudinal 
    and lateral bounds to a 
    geographical area
    """
    LATMIN: Latitude
    LATMAX: Latitude
    LONMIN: Longitude
    LONMAX: Longitude
    name: str = NONAME
    number: int = NOINDEX
    
    def __repr__(self) -> str:
        return (
            "<BoundingBox("
            f"LATMIN={self.LATMIN:.3f},"
            f"LATMAX={self.LATMAX:.3f},"
            f"LONMIN={self.LONMIN:.3f},"
            f"LONMAX={self.LONMAX:.3f})>"
        )    
        
    def __str__(self) -> str:
        return self.name
    
    @property
    def center(self) -> Position:
        """
        Return the center of the bounding box
        """
        return self._center()
    
    def _center(self) -> Position:
        """
        Return the center of the bounding box
        """
        return Position(
            (self.LATMIN+self.LATMAX)/2,
            (self.LONMIN+self.LONMAX)/2
        )

    
class TimePosition:
    """
    Time and position object
    """
    def __init__(self,
                 timestamp: Union[datetime, str],
                 lat: Latitude = None,
                 lon: Longitude = None)-> None:
        
        self.timestamp = timestamp
        self.lat = lat
        self.lon= lon

        self.as_array: List[float] = field(default=list)
        self.timestamp = self._validate_timestamp()
        self.timestamp = self.timestamp.timestamp()

        self.as_array = [self.timestamp,self.lat,self.lon]
        
    def _validate_timestamp(self) -> datetime:
        if isinstance(self.timestamp,datetime):
            return self.timestamp
        try:
            return ciso8601.parse_datetime(self.timestamp)
        except ValueError:
            raise ValueError(
                f"Provided date '{self.timestamp}' is not ISO 8601 compliant."
            )
    
    @property
    def position(self) -> Position:
        """Return position as 
        (lat,lon)-namedtuple or 
        (northing,easting)-namedtuple"""
        return Position(self.lat,self.lon)
