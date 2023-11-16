from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import ciso8601
from typing import List, Union
import utm

Latitude  = float
Longitude = float

Position = namedtuple("Position", ["lat","lon"])
UTMPosition = namedtuple("UTMPosition", ["northing","easting"])

class NONAME_TYPE:
    pass
NONAME = NONAME_TYPE()

class NOINDEX_TYPE:
    pass
NOINDEX = NOINDEX_TYPE()

class ShellError(Exception):
    pass

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