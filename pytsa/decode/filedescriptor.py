"""
File descriptors for the AIS data files.

This is mostly user-defined and depends on the
data source. The default values are for data set
acquired from the European Maritime Safety Agency (EMSA).

If you are using a different data set, please
change the values accordingly.
"""
from enum import Enum

class BaseColumns(Enum):
    """
    For the decoder to work, both the
    Dynamic and Static source files
    must feature the following identical 
    column names.
    """
    __order__ = (
        "TIMESTAMP MESSAGE_ID "
        "RAW_MESSAGE RAW_MESSAGE1 "
        "RAW_MESSAGE2"
    )
    TIMESTAMP: str = "timestamp"
    MESSAGE_ID: str = "message_id"
    RAW_MESSAGE: str = "raw_message"
    RAW_MESSAGE1: str = "raw_message1"
    RAW_MESSAGE2: str = "raw_message2"

class Msg12318Columns(Enum):
    """
    Data columns for message 1,2,3 and 18
    source files.
    """
    __order__ = (
        "MMSI LAT LON SPEED COURSE "
        "TURN"
    )
    MMSI: str = "MMSI"
    LAT: str = "lat"
    LON: str = "lon"
    SPEED: str = "speed"
    COURSE: str = "course"
    TURN: str = "turn"

class Msg5Columns(Enum):
    """
    Data columns for message 5
    source files. 
    """
    __order__ = (
        "MMSI SHIPTYPE SHIPNAME CALLSIGN "
        "TO_BOW TO_STERN TO_PORT TO_STARBOARD"
    )
    MMSI: str = "MMSI"
    SHIPTYPE: str = "ship_type"
    SHIPNAME: str = "shipname"
    CALLSIGN: str = "callsign"
    TO_BOW: str = "to_bow"
    TO_STERN: str = "to_stern"
    TO_PORT: str = "to_port"
    TO_STARBOARD: str = "to_starboard"