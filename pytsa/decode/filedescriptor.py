"""
File descriptors for the AIS data files.

This is mostly user-defined and depends on the
data source. The default values are for data set
acquired from the European Maritime Safety Agency (EMSA).

If you are using a different data set, please
change the values accordingly.
"""
from dataclasses import dataclass

class BaseColumns:
    """
    For the decoder to work, both the
    Dynamic and Static source files
    must feature the following identical 
    column names.
    """
    TIMESTAMP: str = "timestamp"
    MESSAGE_ID: str = "message_id"

@dataclass
class Msg12318Columns(BaseColumns):
    """
    Data columns for message 1,2,3 and 18
    source files.
    """
    RAW_MESSAGE: str = "raw_message"
    ORIGINATOR: str = "originator"
    MMSI: str = "MMSI"
    LAT: str = "lat"
    LON: str = "lon"
    SPEED: str = "speed"
    COURSE: str = "course"
    TURN: str = "turn"

@dataclass
class Msg5Columns(BaseColumns):
    """
    Data columns for message 5
    source files. 
    """
    RAW_MESSAGE1: str = "raw_message1"
    RAW_MESSAGE2: str = "raw_message2"
    ORIGINATOR: str = "originator"
    MMSI: str = "MMSI"
    SHIPTYPE: str = "ship_type"
    SHIPNAME: str = "shipname"
    CALLSIGN: str = "callsign"
    TO_BOW: str = "to_bow"
    TO_STERN: str = "to_stern"
    TO_PORT: str = "to_port"
    TO_STARBOARD: str = "to_starboard"