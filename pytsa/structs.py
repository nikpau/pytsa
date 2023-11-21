from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import ciso8601
from typing import List, Union
import numpy as np
import utm

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
    
@dataclass
class UTMPosition:
    """
    Position object
    """
    northing: float
    easting: float
    
    def __hash__(self) -> int:
        return hash((self.northing,self.easting))

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
    _utm: bool = False
    
    # Fields for Trajectory Clustering ------------------
    _cluster_type: str = "" # ["EN","EX","PO"]
    _label_group: Union[int,None] = None

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
class LatLonBoundingBox:
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

    
    def to_utm(self) -> UTMBoundingBox:
        """
        Convert bounding box to 
        UTM coordinates
        """
        min_easting, min_northing, zn, zl = utm.from_latlon(
            self.LATMIN, self.LONMIN
        )
        max_easting, max_northing, *_ = utm.from_latlon(
            self.LATMAX, self.LONMAX
        )
        return UTMBoundingBox(
            min_easting=min_easting,
            max_easting=max_easting,
            min_northing=min_northing,
            max_northing=max_northing,
            zone_number=zn,
            zone_letter=zl,
            name=self.name,
            number=self.number
        )

@dataclass
class UTMBoundingBox:
    """
    Bounding box using 
    UTM coordinates
    """
    min_easting: float
    max_easting: float
    min_northing: float
    max_northing: float
    zone_number: int
    zone_letter: str
    name: str = NONAME
    number: int = NOINDEX
    
    def __repr__(self) -> str:
        return (
            "<UTMBoundingBox("
            f"min_easting={self.min_easting:.3f},"
            f"max_easting={self.max_easting:.3f},"
            f"min_northing={self.min_northing:.3f},"
            f"max_northing={self.max_northing:.3f},"
            f"zone_number={self.zone_number},"
            f"zone_letter={self.zone_letter})>"
        )
        
    def __str__(self) -> str:
        return self.name

    def to_latlon(self) -> LatLonBoundingBox:
        """
        Convert bounding box to 
        latitude and longitude
        """
        latmin, lonmin = utm.to_latlon(
            self.min_easting, self.min_northing,
            self.zone_number, self.zone_letter
        )
        latmax, lonmax = utm.to_latlon(
            self.max_easting, self.max_northing,
            self.zone_number, self.zone_letter
        )
        return LatLonBoundingBox(
            LATMIN=latmin,
            LATMAX=latmax,
            LONMIN=lonmin,
            LONMAX=lonmax,
            name=self.name
        )
# Bounding boxes ------------------------------------
BoundingBox = Union[LatLonBoundingBox, UTMBoundingBox]

class TimePosition:
    """
    Time and position object
    """
    def __init__(
            self,
            timestamp: Union[datetime, str],
            lat: Latitude = None,
            lon: Longitude = None,
            easting: float = None,
            northing: float = None,
            as_utm: bool = False
            )-> None:
        
        self.timestamp = timestamp
        self.lat = lat
        self.lon= lon
        self.easting = easting
        self.northing = northing

        self.as_array: List[float] = field(default=list)
        self.timestamp = self._validate_timestamp()
        self.timestamp = self.timestamp.timestamp()

        self._is_utm = as_utm
        if self.easting is None or self.northing is None:
            self.easting, self.northing, *_ = utm.from_latlon(
                    self.lat, self.lon
            )
        if self._is_utm:
            self.as_array = [self.timestamp,self.easting,self.northing]
        else:
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
        if self._is_utm:
            return UTMPosition(self.northing,self.easting)
        else:
            return Position(self.lat,self.lon)
