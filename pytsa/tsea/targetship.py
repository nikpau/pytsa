"""
This module aims at finding all spatio-temporal
neighbors around a given (stripped) AIS Message. 

A provided geographical area will be split into 
evenly-sized grids
"""
from __future__ import annotations

from datetime import datetime
from typing import List, Union

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline, interp1d
from ..structs import AISMessage, MMSI, Track
from ..logger import logger

# Type aliases
Latitude = float
Longitude = float

# Constants
PI = np.pi
class Track:
    """
    Object containing a ships' track,
    and an optional interpolated version of it.
    """
    def __init__(self, messages: List[AISMessage]) -> None:
        self.messages = messages
        self._interpolation: _TrackInterpolator | None = None
        self._interpolation_failed = False
        
    def __getitem__(self, index: int) -> AISMessage:
        return self.messages[index]
    
    def __len__(self) -> int:
        return len(self.messages)
    
    def __iter__(self) -> AISMessage:
        return iter(self.messages)
    
    def interpolate_at(self, timestamp: int) -> np.ndarray:
        """
        Interpolate the track at the given timestamp.
        """
        if self._interpolation is None:
            raise InterpolationError(
                "Track is not interpolated. Did you call ship.interpolate()?."
            )
        return self._interpolation.interpolate_at(timestamp)
    
    def _apply_interpolation(self, mode: str) -> None:
        """
        Apply interpolation to the track.
        """
        assert mode in ["linear","spline","auto"],\
            "Mode must be either 'auto', 'linear' or 'spline'."
        
        try:
            if mode == "auto":
                self._interpolation = (
                    _TrackInterpolatorSpline(self.messages)
                    if len(self) > 3 # Need at least 4 points for spline
                    else _TrackInterpolatorLinear(self.messages)
                )
            elif mode == "spline":
                self._interpolation = _TrackInterpolatorSpline(self.messages)
            else:
                self._interpolation = _TrackInterpolatorLinear(self.messages)
        except Exception as e:
            logger.error(
                f"Could not interpolate track: {e}."
            )
            self._interpolation_failed = True


# Exceptions
class OutofTimeBoundsError(Exception):
    pass
class InterpolationError(Exception):
    pass

class _TrackInterpolator:
    def interpolate_at(self, timestamp: int) -> np.ndarray:
        """
        Interpolate the track at the given timestamp.
        """
        return np.array([
            self.lat(timestamp),
            self.lon(timestamp),
            self.COG(timestamp),
            self.SOG(timestamp)
        ])

class _TrackInterpolatorSpline(_TrackInterpolator):
    """
    Class for performing separate univariate
    spline interpolation on a given AIS track.

    This class provides the following attributes:
        - lat: Spline interpolation of latitude
        - lon: Spline interpolation of longitude
        - northing: Spline interpolation of northing
        - easting: Spline interpolation of easting
        - COG: Spline interpolation of course over ground
        - SOG: Spline interpolation of speed over ground

        All splines are univariate splines, with time as the
        independent variable.
    """
    def __init__(self, track: List[AISMessage]) -> None:
        self.track = track
        self._attach_splines()

    def _attach_splines(self) -> None:
        """
        Perform spline interpolation on the
        AIS track and attach the results to
        the class as attributes.
        """
        timestamps = [int(msg.timestamp) for msg in self.track]

        self.lat = InterpolatedUnivariateSpline(
            timestamps, [msg.lat for msg in self.track]
        )
        self.lon = InterpolatedUnivariateSpline(
            timestamps, [msg.lon for msg in self.track]
        )
        # Add the 360° complement to correctly interpolate
        # course changes from 360° to 0°
        complement360 = np.rad2deg(
            np.unwrap(
                np.deg2rad([msg.COG for msg in self.track])
            )
        )
        self.COG = InterpolatedUnivariateSpline(
            timestamps, complement360
        )
        self.SOG = InterpolatedUnivariateSpline(
            timestamps, [msg.SOG for msg in self.track]
        )

class _TrackInterpolatorLinear(_TrackInterpolator):
    """
    Linear interpolation of a given AIS track.
    """
    def __init__(self, track: List[AISMessage]) -> None:
        self.track = track
        self._attach_linear()

    def _attach_linear(self) -> None:
        """
        Perform linear interpolation on the
        AIS track and attach the results to
        the class as attributes.
        """
        timestamps = [msg.timestamp for msg in self.track]
        
        self.lat = interp1d(
            timestamps, [msg.lat for msg in self.track]
        )
        self.lon = interp1d(
            timestamps, [msg.lon for msg in self.track],
            
        )
        # Add the 360° complement to correctly interpolate
        # course changes from 360° to 0°
        complement360 = np.rad2deg(
            np.unwrap(
                np.deg2rad([msg.COG for msg in self.track])
            )
        )
        self.COG = interp1d(
            timestamps, complement360,
            bounds_error=False, fill_value=None
        )
        self.SOG = interp1d(
            timestamps, [msg.SOG for msg in self.track]
        )
class TargetShip:
    """
    Central object for the pytsa package. It holds
    a collection of functions for working with the 
    AIS messages belonging to it.
    
    Attributes:
        - ts: Timestamp for which to observe the vessel
        - mmsi: Maritime Mobile Service Identity
        - track: List of AIS messages belonging to the
                target vessel
        - ship_type: Ship type of the target vessel
        - length: Length of the target vessel
                
    Methods:
        - interpolate(): Construct smoothed univariate splines
                for northings, eastings, course over ground,
                speed over ground, rate of turn and change of
                rate of turn contained in the vessels' track.
                
                If the caller wants to use linear interpolation
                instead of splines, the lininterp flag can be set
                to True.
                
        - observe_at_query(): Returns the 6-tuple of [northing,
                easting, COG, SOG, ROT, dROT] at the timestamp 
                the object was initialized with.
                The timestamp should lie within the
                track's temporal bounds.
                
                Currently, there is no check 
                for the timestamp being within the track's
                temporal bounds, which can lead to errors.
                If you want to use this function, please
                make sure that the timestamp is within bounds.
                
                You can also pass a timestamp to the function
                to observe the vessel at a different timestamp.
                Please note that there is still no check for
                the timestamp being within the track's bounds.
                
        - observe_interval(): Returns an array of 7-tuples of [northing,
                easting, COG, SOG, ROT, dROT, timestamp] for 
                the track between the given start and end timestamps,
                with the given interval in [seconds].
                
        - fill_rot(): Fill out missing rotation data and first
                derivative of rotation by inferring it from the
                previous and next AIS messages' headings.
        
        - overwrite_rot(): Overwrite the ROT and dROT values with
                the values from COG.
        
        - find_shell(): Find the two AIS messages encompassing
                the objects' track elements and save them
                as attributes.
        
        - ts_to_unix(): Convert the vessel's timestamp for
                each track element to unix time.
    """    
    
    def __init__(
        self,
        ts: Union[int, None], 
        mmsi: MMSI, 
        tracks: List[Track],
        ship_type: list[int] = None,
        ship_length: float = None,
        ) -> None:
        
        self.ts = ts
        self.mmsi = mmsi 
        self.tracks = tracks
        self.ship_type = ship_type
        self.ship_length = ship_length
        
        # Indicator whether a trajectory extraction
        # has been applied to the vessel's track
        self._trex_applied = False 
        
        self.lininterp = False # Linear interpolation flag
        
    def interpolate(self,mode: str) -> None:
        """
        Construct splines for the target vessel
        """
        assert mode in ["linear","spline","auto"],\
            "Mode must be either 'auto', 'linear', 'spline'."
        if not self.tracks:
            logger.warning(
                f"Empty track for vessel {self.mmsi}."
            )
            return
        for track in self.tracks:
            track._apply_interpolation(mode)
        return 

    def observe(self) -> np.ndarray:
        """
        Infers
            - Latitude,
            - Longitude,
            - Course over ground (COG) [degrees],
            - Speed over ground (SOG) [knots],
            
        from an interpolated trajectory for the timestamp, 
        the object was initialized with.

        """
        if self.ts is None:
            msg = (
                "`observe_at()` method not available for use with `get_all_ships()`.\n"
                "Please use `observe_interval()` or instatiante the object through `get_ships()`." 
                )
            logger.error(msg)
            raise NotImplementedError(msg)
        # Convert query timestamp to unix time
        if isinstance(self.ts, datetime):
            ts = self.ts.timestamp()
        else: ts = self.ts
            
        # Check if the query timestamp is within the
        # track's timestamps
        for i,track in enumerate(self.tracks):
            if self._is_in_interval(ts,track):
                break
            if i == len(self.tracks)-1:
                raise OutofTimeBoundsError(
                    "Query timestamp is not within the track's timestamps."
                )

        # Return the observed values from the splines
        # at the given timestamp
        # Returns a 1x4 array:
        # [northing, easting, COG, SOG]
        return self.tracks[i].interpolate_at(ts)

    def observe_interval(
            self, 
            start: datetime, 
            end: datetime, 
            interval: int
            ) -> np.ndarray:
        """
        Infers
            - Northing (meters),
            - Easting (meters),
            - Course over ground (COG) [degrees],
            - Speed over ground (SOG) [knots],

        from univariate splines for the track between
        the given start and end timestamps, with the
        given interval in [seconds].
        """
        assert self.interpolation is not None,\
            "Interpolation has not been run. Call interpolate() first."
        # Convert query timestamps to unix time
        if isinstance(start, datetime):
            start = start.timestamp()
            end = end.timestamp()

        # Check if the start timestamp is within the
        # track's timestamps
        for i,track in enumerate(self.tracks):
            if self._is_in_interval(start,track):
                break
            if i == len(self.tracks)-1:
                raise OutofTimeBoundsError(
                    "Start timestamp is not within the track's timestamps. "
                    "MMSI: {}".format(self.mmsi)
                )
        # Check if the end timestamp is within the
        # track's timestamps
        for j,track in enumerate(self.tracks):
            if self._is_in_interval(end,track):
                break
            if j == len(self.tracks)-1:
                raise OutofTimeBoundsError(
                    "End timestamp is not within the track's timestamps. "
                    "MMSI: {}".format(self.mmsi)
                )

        # Create a list of timestamps between the start and end
        # timestamps, with the given interval
        timestamps = np.arange(start, end, interval)

        # Start and end points of the interval
        # lie within the same track
        if i == j:
            # Return the observed values from the interpolation
            # at the given timestamps
            # Returns a Nx4 array:
            # [northing, easting, COG, SOG, timestamp]
            preds: np.ndarray = np.concatenate([
                self.tracks[i].interpolate_at(timestamps),
                timestamps.reshape(-1,1)
            ], axis=1) 
            return preds.T
        # Start and end points of the interval
        # lie in different tracks
        else:
            logger.warning(
                "Start and end points of the interval lie in different tracks. "
                "Results may be inaccurate. MMSI: %s".format(self.mmsi)
            )
            # Combine the tracks into one list
            tracks = self.tracks[i:j+1]
            tracks = [msg for track in tracks for msg in track]
            interp_ = _TrackInterpolatorSpline(tracks) if not self.lininterp else _TrackInterpolatorLinear(tracks)
            # Return the observed values from the interpolation
            # at the given timestamps
            # Returns a Nx4 array:
            # [northing, easting, COG, SOG, timestamp]
            preds: np.ndarray = np.concatenate([
                interp_.interpolate_at(timestamps),
                timestamps.reshape(-1,1)
            ], axis=1)
            return preds.T
            
    
    def _is_in_interval(self, query: int, msgs: Track) -> bool:
        """
        Check if the query timestamp is within the
        given interval.
        """
        return (query >= msgs[0].timestamp) and (query <= msgs[-1].timestamp)


    def find_shell(self) -> None:
        """
        Find the two AIS messages encompassing
        the objects' track elements and save them
        as attributes.
        """ 
        self.lower, self.upper = [], []
        for track in self.tracks:
            self.lower.append(track[0])
            self.upper.append(track[-1])


Targets = dict[MMSI,TargetShip]

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
