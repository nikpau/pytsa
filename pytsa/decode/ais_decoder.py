"""
This module defines some structures and functions 
to decode, sort, and export AIS Messages.

The structures are specific to the EMSA dataset
being analyzed with it. 
"""
import os
from typing import Any, List, Tuple, Dict

import numpy as np
import pandas as pd
import pyais as ais
import pyais.messages as msg
import multiprocessing as mp
from .filedescriptor import (
    BaseColumns, Msg12318Columns, Msg5Columns
)

class StructuralError(Exception):
    pass

# Message type tuples
_STATIC_TYPES = (5,)
_DYNAMIC_TYPES = (1,2,3,18,)

# Slot names for message fields
MSG5SLOTS = msg.MessageType5.__slots__
MSG12318SLOTS = msg.MessageType1.__slots__
MSGSLOTS = MSG5SLOTS | MSG12318SLOTS

# Default value for missing data
_NA = "NA"

# Decoding plan
class DynamicDecoder:
    type = "dynamic"
    
    def __call__(self, df: pd.DataFrame) -> Any:
        return self._decode_dynamic_messages(df)

    def _decode_dynamic_messages(self, df: pd.DataFrame) -> List[ais.ANY_MESSAGE]:
        """
        Decode AIS messages of types 1,2,3,18 
        supplied as a pandas Series object.
        """
        messages = df[Msg12318Columns.RAW_MESSAGE]
        # Split at exclamation mark and take the last part
        raw = messages.str.split("!",expand=True).iloc[:,-1:]
        # Since we split on the exclamation mark we need to
        # re-add it at the front of the message
        raw = "!" + raw
        return [ais.decode(val) for val in raw.values.ravel()]

class StaticDecoder:
    type = "static"
    
    def __call__(self, df: pd.DataFrame) -> Any:
        return self._decode_static_messages(df)
    
    def _decode_static_messages(self, df: pd.DataFrame):
        """
        Decode AIS messages of type 5 
        supplied as a pandas DataFrame.
        """
        msg1 = df[Msg5Columns.RAW_MESSAGE1]
        msg2 = df[Msg5Columns.RAW_MESSAGE2]
        raw1 = msg1.str.split("!",expand=True).iloc[:,-1:]
        raw2 = msg2.str.split("!",expand=True).iloc[:,-1:]
        raw1, raw2 = "!" + raw1, "!" + raw2
        return [ais.decode(*vals) for vals in 
                zip(raw1.values.ravel(),raw2.values.ravel())]

# Type alias
Decoder = DynamicDecoder | StaticDecoder

def _extract_fields(messages: List[ais.ANY_MESSAGE],
                    fields: tuple) -> Dict[str,np.ndarray]:
    out = np.empty((len(messages),len(fields)),dtype=object)
    for i, msg in enumerate(messages):
        out[i] = [getattr(msg,field,_NA) for field in fields]
    return dict(zip(fields,out.T))

def _get_decoder(dataframe: pd.DataFrame) -> Tuple[Decoder,MSGSLOTS]:
    """
    Returns a message-specific decoding function
    based on message types present in the dataframe.

    Note: Since input dataframes will be processed 
    at once, they can only contain either position 
    record messages (types 1,2,3,18), or static messages
    (type 5) but not both.
    """
    types = dataframe[BaseColumns.MESSAGE_ID]
    if all(k in dataframe for k in (Msg5Columns.RAW_MESSAGE1, 
        Msg5Columns.RAW_MESSAGE2)):
        # Maybe type 5 
        if all(b in _STATIC_TYPES for b in types.unique()):
            return StaticDecoder, MSG5SLOTS
        else: raise StructuralError(
                "Assumed type-5-only dataframe, but found "
                f"messages of types {types.unique()}"
        )
    elif all(b in _DYNAMIC_TYPES for b in types.unique()):
        return DynamicDecoder, MSG12318SLOTS
    else: raise StructuralError(
            "Found not processable combination "
            "of message types. Need either type-5-only dataframe "
            "or type-1-2-3-18-only dataframe. Found messages "
            f"of types {types.unique()}"
    )
    
# Pandas file operations 
#
# The pipeline will first read-in the csv-file
# as a pandas dataframe, decode the raw AIS
# message, and save the extracted information 
# as new columns in the existing file. 
def decode_from_file(source: str,
                     dest: str,
                     save_to_file = True) -> None:  
    df  = pd.read_csv(source,sep=",",quotechar='"',
                      encoding="utf-8",index_col=False)
    decoder, fields = _get_decoder(df)
    if decoder.type == "dynamic":
        # Drop messages with newline characters
        # since they are not valid AIS messages
        df = df.drop(df[df[Msg12318Columns.RAW_MESSAGE].str.contains(r"\n")].index)
    else:
        # Drop messages with newline characters
        # since they are not valid AIS messages
        df = df.drop(df[df[Msg5Columns.RAW_MESSAGE1].str.contains(r"\n")].index)
        df = df.drop(df[df[Msg5Columns.RAW_MESSAGE2].str.contains(r"\n")].index)
    decoded = decoder(df)
    df["DECODE_START"] = "||"
    df = df.assign(**_extract_fields(decoded,fields))
    return df.to_csv(dest) if save_to_file else df


from pathlib import Path
def mpdecode(source: Path, dest: Path, njobs: int = 16) -> None:
    """
    Decode AIS messages in parallel.
    """
    files = list(source.rglob("*.csv")) 
    with mp.Pool(processes=njobs) as pool:
        pool.starmap(decode_from_file, 
                     [(file, f"{dest.as_posix()}/{'/'.join(file.parts[len(source.parts):])}")
                      for file in files])
    return