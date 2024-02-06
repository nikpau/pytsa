"""
Utility functions for pytsa
"""
from datetime import datetime
from itertools import cycle
import math
from pathlib import Path
from threading import Thread
import time
from typing import Callable, Generator
import pandas as pd
import psutil
import vincenty as _vincenty
import ciso8601
from .logger import logger
from .decode import decode_from_file
from .decode.filedescriptor import (
    BaseColumns, Msg12318Columns, Msg5Columns   
)

def m2nm(m: float) -> float:
    """Convert meters to nautical miles"""
    return m/1852

def nm2m(nm: float) -> float:
    """Convert nautical miles to meters"""
    return nm*1852

def s2h(s: float) -> float:
    """Convert seconds to hours"""
    return s/3600

def mi2nm(mi: float) -> float:
    """Convert miles to nautical miles"""
    return mi/1.151

def vincenty(lon1, lat1, lon2, lat2, miles = True) -> float:
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees)
    """
    p1 = (lat1,lon1)
    p2 = (lat2,lon2)
    return _vincenty.vincenty_inverse(p1,p2,miles=miles)

def haversine(lon1, lat1, lon2, lat2, miles = True):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = (
        math.sin(dlat/2)**2 + math.cos(lat1) * 
        math.cos(lat2) * math.sin(dlon/2)**2
        )
    c = 2 * math.asin(math.sqrt(a)) 
    r = 3956 if miles else 6371 # Radius of earth in kilometers or miles
    return c * r

def greater_circle_distance(lon1, lat1, lon2, lat2, miles = True, method = "haversine"):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees)
    """
    assert method in ["haversine", "vincenty"], "Invalid method: {}".format(method)
    if method == "haversine":
        return haversine(lon1, lat1, lon2, lat2, miles=miles)
    return vincenty(lon1, lat1, lon2, lat2, miles=miles)

def heading_change(h1,h2):
    """
    calculate the change between two headings
    such that the smallest angle is returned.
    """

    diff = abs(h1-h2)
    if diff > 180:
        diff = 360 - diff
    if (h1 + diff) % 360 == h2:
        return diff
    else:
        return -diff
    
class DataLoader:
    """
    Data loader for AIS data.
    
    Takes a list of paths to static and dynamic data files
    provides utilities to decode and load the data.
    """
    
    # Fraction of the file to load
    chunkfrac = 1/3
    
    dynamic_columns = [
        BaseColumns.TIMESTAMP.value,
        Msg12318Columns.MMSI.value,
        Msg12318Columns.LAT.value,
        Msg12318Columns.LON.value,
        Msg12318Columns.SPEED.value,
        Msg12318Columns.COURSE.value
    ]
    
    static_columns = [
        Msg5Columns.MMSI.value,
        Msg5Columns.SHIPTYPE.value,
        Msg5Columns.TO_BOW.value,
        Msg5Columns.TO_STERN.value
    ]
    
    def __init__(self, 
                 dynamic_paths: list[Path],
                 static_paths: list[Path],
                 pre_processor: Callable[[pd.DataFrame],pd.DataFrame]) -> None:
        
        self.preprocessor = pre_processor
        
        # Check that we only have csv files
        assert all(
            [f.suffix == ".csv" for f in dynamic_paths + static_paths]
        ), "Only csv files are supported."
        
        # Check for same size and same order
        self.sdyn, self.sstat = self.align_data_files(
            dynamic_paths, static_paths
        )
        
        # Flag to indicate if the data has
        # been loaded into memory entirely.
        # Only used for the `load` method.
        self.loaded = False
    
    @staticmethod
    def _date_transformer(datefile: Path) -> datetime:
        """
        Transform a date string in the format YYYY_MM_DD
        to a datetime object.
        """
        return ciso8601.parse_datetime(datefile.stem.replace("_", "-"))

    @staticmethod
    def align_data_files(dyn: list[Path], 
                         stat: list[Path]
                         ) -> tuple[list[Path], list[Path]]:
        """
        Align dynamic and static data files by sorting them
        according to their date, and removing all files that
        are not present in both lists.

        This function assumes that the dynamic and static data files
        are named according to the following convention:
        - dynamic data files: YYYY_MM_DD.csv
        - static data files: YYYY_MM_DD.csv
        
        In case your input data files are not named according to this
        convention, you are advised to either rename them accordingly
        or adapt the `_date_transformer` function, which is used to
        sort the files by date.
        """
        if len(dyn) != len(stat):

            print(
                "Number of dynamic and static messages do not match."
                f"Dynamic: {len(dyn)}, static: {len(stat)}\n"
                "Processing only common files."
            )
            # Find the difference
            d = set([f.stem for f in dyn])
            s = set([f.stem for f in stat])
            
            # Find all files that are in d and s
            common = list(d.intersection(s))
            
            # Remove all files that are not in common
            dyn = [f for f in dyn if f.stem in common]
            stat = [f for f in stat if f.stem in common]
            
        # Sort the files by date
        dyn = sorted(dyn, key=DataLoader._date_transformer)
        stat = sorted(stat, key=DataLoader._date_transformer)

        assert all([d.stem == s.stem for d,s in zip(dyn, stat)]),\
            "Dynamic and static messages are not in the same order."

        return dyn, stat
    
    def _dynamic_preprocessor(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the dynamic messages.
        """
        # Apply custom filter
        df = self.preprocessor(df).copy()
        # Convert timestamp to datetime
        df[BaseColumns.TIMESTAMP.value] = pd.to_datetime(
                df[BaseColumns.TIMESTAMP.value])
        # Drop duplicates form multiple base stations
        df = df.drop_duplicates(
                subset=[
                    BaseColumns.TIMESTAMP.value,
                    Msg12318Columns.MMSI.value
                ], keep="first"
            )
        return df
    
    def from_raw(self, raw_dyn: Path, raw_stat: Path) -> tuple[pd.DataFrame,pd.DataFrame]:
        """
        Return a DataFrame containing the decoded dynamic and static
        messages.
        
        Decoding the raw AIS messages will result in a 
        DataFrame containing all fields of a NMEA message.
        Since we are only interested in a subset of the fields,
        we select only those fields and return them as a new
        DataFrame.
        
        Note that the file is NOT decoded in chunks, but rather
        as a whole.
        """
        # Decode dynamic data
        dyn = decode_from_file(raw_dyn, None, save_to_file=False)
        dyn = self._dynamic_preprocessor(dyn)
        # Decode static data
        stat = decode_from_file(raw_stat, None, save_to_file=False)
        return (
            dyn[[DataLoader.dynamic_columns]], 
            stat[[DataLoader.static_columns]]
        )
        
    def load(self) -> None:
        """
        Loads all data into memory.
        """
        self.dynamic_data = pd.DataFrame()
        self.static_data = pd.DataFrame()
        for dyn_path, stat_path in zip(self.sdyn,self.sstat):
            logger.info(f"Loading {stat_path.stem} and {dyn_path.stem}")
            d = pd.read_csv(dyn_path,sep=",",usecols=self.dynamic_columns)
            d = self._dynamic_preprocessor(d)
            s = pd.read_csv(stat_path,sep=",",usecols=self.static_columns)
            self.dynamic_data = pd.concat([self.dynamic_data,d])
            self.static_data = pd.concat([self.static_data,s])
        logger.info("Done.")
        
        self.loaded = True
    
    def iterate_chunks(self,
                   decode: bool = False
                ) -> Generator[
                      tuple[pd.DataFrame,pd.DataFrame], 
                      None, 
                      None
                    ]:
        """
        Yield chunks of the dynamic and static
        messages.
        
        If decode is True, the raw AIS messages
        will be decoded and the decoded messages
        will be returned as a DataFrame.
        
        """
        dyn_options = dict(
            sep=",",
            usecols=self.dynamic_columns
        )
        stat_options = dict(
            sep=",",
            usecols=self.static_columns
        )
        for dyn_path, stat_path in zip(self.sdyn,self.sstat):
            
            # Count rows of both files
            with open(dyn_path) as f, open(stat_path) as g:
                dyn_rows = sum(1 for _ in f)
                stat_rows = sum(1 for _ in g)
             
            chunksize_stat = int(stat_rows * DataLoader.chunkfrac) + 1
            chunksize_dyn = int(dyn_rows * DataLoader.chunkfrac) + 1
            dyn_options["chunksize"] = chunksize_dyn
            stat_options["chunksize"] = chunksize_stat
            
            if decode:
                # Decode dynamic data
                yield self.from_raw(dyn_path, stat_path)
            else:
                # Create a generator of pandas DataFrames
                dyniter = pd.read_csv(dyn_path,**dyn_options)
                statiter = pd.read_csv(stat_path,**stat_options)
                
                for i, (dc,sc) in enumerate(zip(dyniter,statiter)):
                    logger.info(
                        f"Processing chunk {i+1} of file "
                        f"{dyn_path.stem}"
                    )
                    dc = self._dynamic_preprocessor(dc)            
                    yield dc, sc

# DEPRECATED v ================================================================
class Loader:
    def __init__(self, bb):
        """
        A loader-like context manager

        Args:
            desc (str, optional): The loader's description. Defaults to "Loading...".
            end (str, optional): Final print. Defaults to "Done!".
            timeout (float, optional): Sleep time between prints. Defaults to 0.1.
        """
        self.desc = (
            f"Buffering area from {bb.LATMIN:.3f}°N-{bb.LATMAX:.3f}°N "
            f"and {bb.LONMIN:.3f}°E-{bb.LONMAX:.3f}°E"
        )
        self.timeout = 0.1

        self._thread = Thread(target=self._animate, daemon=True)
        self.steps = ["⢿", "⣻", "⣽", "⣾", "⣷", "⣯", "⣟", "⡿"]
        self.done = False

    def start(self):
        self.t_start = time.perf_counter()
        self._thread.start()
        return self

    def _animate(self):
        for c in cycle(self.steps):
            if self.done:
                break
            print(f"{self.desc} {c}", flush=True, end="\r")
            time.sleep(self.timeout)

    def __enter__(self):
        self.start()

    def stop(self):
        self.done = True
        print(" "*100, end = "\r")
        logger.info(f"{self.desc}")
        self.t_end = time.perf_counter()
        logger.info(f"Cell Buffering completed in [{(self.t_end-self.t_start):.1f} s]")

    def __exit__(self, exc_type, exc_value, tb):
        # handle exceptions with those variables ^
        self.stop()

# Context manager that continuously shows memory usage
# while running the code inside the context.
class MemoryLoader:
    def __init__(self):
        self.timeout = 0.2

        self._thread = Thread(target=self._show_memory_usage, daemon=True)
        self.done = False

    def start(self):
        self.t_start = time.perf_counter()
        self._thread.start()
        return self

    def _show_memory_usage(self):
        """
        Prints memory usage every `timeout` seconds
        """
        while True:
            if self.done:
                break
            print(
                f"Memory usage: {psutil.virtual_memory().percent}% "
                f"[{psutil.virtual_memory().used/1e9:.2f} GB]", 
                end="\r")
            time.sleep(self.timeout)

    def __enter__(self):
        self.start()

    def stop(self):
        self.done = True
        print(" "*100, end = "\r")
        self.t_end = time.perf_counter()
        print(
            f"Loading took {self.t_end - self.t_start:.2f} seconds \n"
            f"Memory usage: {psutil.virtual_memory().percent}% "
            f"[{psutil.virtual_memory().used/1e9:.2f} GB]"
            )

    def __exit__(self, exc_type, exc_value, tb):
        # handle exceptions with those variables ^
        self.stop()