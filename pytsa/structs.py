from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import ciso8601
from typing import List, Union

Latitude  = float
Longitude = float
MMSI = int
UNIX_TIMESTAMP = int | float

class NONAME_TYPE:
    pass
NONAME = NONAME_TYPE()

class NOINDEX_TYPE:
    pass
NOINDEX = NOINDEX_TYPE()

class ShellError(Exception):
    pass

# Bins for length-division of the quantiles
LENGTH_BINS = [0.,25.,50.,75.,100.,125.,150.,175.,200.,float("inf")]

def _mflatten(l):
    """
    Flatten a mixed list
    of numbers and lists
    """
    for el in l:
        if isinstance(el, (list,range)):
            yield from _mflatten(el)
        else:
            yield el

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
    second: int = None # Senders second of the minute
    ROT: float = None # Rate of turn [degrees/minute]
    dROT: float = None # Change of ROT [degrees/minute²]

    def __repr__(self) -> str:
        return (
            f"<AISMessage("
            f"sender={self.sender},"
            f"timestamp={self.timestamp},"
            f"lat={self.lat},"
            f"lon={self.lon},"
            f"COG={self.COG},"
            f"SOG={self.SOG},"
            f"ROT={self.ROT},"
            f"dROT={self.dROT})>"
        ) 

    def __hash__(self) -> int:
        """
        Timestamp and lat/lon are used to uniquely
        identify an AIS message.
        """
        return hash(self.lat + self.lon)
    
    def __eq__(self, other: AISMessage) -> bool:
        """
        Two AIS messages are equal if they have the same
        timestamp or the same lat/lon.
        """
        return self.lat == other.lat and self.lon == other.lon
    
    def __ne__(self, other: AISMessage) -> bool:
        return not self.__eq__(other)

# Track type
Track = list[AISMessage]
class ShipType(Enum):
    """
    Dataclass to store the type of vessel
    as defined by the AIS standard.
    See here for more information:
    https://coast.noaa.gov/data/marinecadastre/ais/VesselTypeCodes2018.pdf
    """
    __order__ = (
        "NOTAVAILABLE WIG FISHING TUGTOW MILITARY "
        "SAILING PLEASURE HSC PASSENGER "
        "CARGO TANKER OTHER"
    )
    NOTAVAILABLE = 0
    WIG = range(20,30) # Wing in ground
    FISHING = 30
    TUGTOW = [31,32,52]
    MILITARY = 35
    SAILING = 36
    PLEASURE = 37
    HSC = range(40,50) # High speed craft
    PASSENGER = range(60,70)
    CARGO = range(70,80)
    TANKER = range(80,90)
    OTHER = list(_mflatten([33,34,50,51,list(range(52,60)),list(range(90,100))]))
    
    def from_value(value: int) -> ShipType:
        """
        Return the ship type from a value
        """
        for st in ShipType:
            if isinstance(st.value, (list,range)):
                if value in st.value:
                    return st
            else:         
                if value == st.value:
                    return st
        raise ValueError(f"Ship type {value} not found.")

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
    
    @property
    def aspect_ratio(self) -> float:
        """
        Return the aspect ratio of the bounding box
        """
        return self._aspect_ratio()
    
    def _center(self) -> Position:
        """
        Return the center of the bounding box
        """
        return Position(
            (self.LATMIN+self.LATMAX)/2,
            (self.LONMIN+self.LONMAX)/2
        )

    def _aspect_ratio(self) -> float:
        """
        Return the aspect ratio of the bounding box
        """
        return (self.LONMAX-self.LONMIN)/(self.LATMAX-self.LATMIN)

    def contains(self, message: AISMessage) -> bool:
        """
        Check if a position is within the bounding box
        """
        return (
            self.LATMIN <= message.lat <= self.LATMAX and
            self.LONMIN <= message.lon <= self.LONMAX
        )

@dataclass
class Position:
    """
    Position object for a geographical point.
    """
    lat: Latitude
    lon: Longitude
    
    def __hash__(self) -> int:
        return hash((self.lat,self.lon))
    
    @property
    def as_list(self) -> List[float]:
        return [self.lat,self.lon]
    
class TimePosition(Position):
    """
    Position object with a timestamp.
    
    Parameters
    ----------
    timestamp : Union[datetime, str, UNIX_TIMESTAMP]
        Timestamp of the observation. Must be either
        a datetime object, a string in ISO 8601 format
        or a UNIX timestamp.
    lat : Latitude
        Latitude of the observation.
    lon : Longitude
        Longitude of the observation.
    """
    def __init__(self,
                 timestamp: Union[datetime, str, UNIX_TIMESTAMP],
                 lat: Latitude = None,
                 lon: Longitude = None)-> None:
        
        self.timestamp = timestamp
        self.lat = lat
        self.lon = lon

        self.timestamp = self._validate_timestamp()
        self.timestamp = int(self.timestamp.timestamp())

    def _validate_timestamp(self) -> datetime:
        if isinstance(self.timestamp,datetime):
            return self.timestamp
        try:
            return ciso8601.parse_datetime(self.timestamp)
        except ValueError:
            try: # Try to parse as UNIX timestamp
                return datetime.fromtimestamp(self.timestamp)
            except:
                raise ValueError(
                    f"Provided date '{self.timestamp}' is neither "
                    "ISO 8601 compliant nor a UNIX timestamp."
                )
    
    @property
    def position(self) -> Position:
        """Return position as 
        (lat,lon)-namedtuple or 
        (northing,easting)-namedtuple"""
        return Position(self.lat,self.lon)
