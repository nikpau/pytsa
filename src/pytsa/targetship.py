"""
This module aims at finding all spatio-temporal
neighbors around a given (stripped) AIS Message. 

A provided geographical area will be split into 
evenly-sized grids
"""
from __future__ import annotations
from dataclasses import dataclass

from datetime import datetime
from typing import Union, List

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
import utm


# Settings for numerical integration
Q_SETTINGS = dict(epsabs=1e-13,epsrel=1e-13,limit=500)



# Type aliases
Latitude = float
Longitude = float

# Constants
PI = np.pi

# Exceptions

class OutofTimeBoundsError(Exception):
    pass


class TrackSplines:
    """
    Class for performing separate univariate
    spline interpolation on a given AIS track.

    This class provides the following attributes:
        - spl_northing: Spline interpolation of northing
        - spl_easting: Spline interpolation of easting
        - spl_COG: Spline interpolation of course over ground
        - spl_SOG: Spline interpolation of speed over ground
        - spl_ROT: Spline interpolation of rate of turn
        - spl_dROT: Spline interpolation of change of rate of turn

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
        timestamps = [msg.timestamp for msg in self.track]

        self.spl_northing = UnivariateSpline(
            timestamps, [msg.northing for msg in self.track]
        )
        self.spl_easting = UnivariateSpline(
            timestamps, [msg.easting for msg in self.track]
        )
        self.spl_COG = UnivariateSpline(
            timestamps, [msg.COG for msg in self.track]
        )
        self.spl_SOG = UnivariateSpline(
            timestamps, [msg.SOG for msg in self.track]
        )
        self.spl_ROT = UnivariateSpline(
            timestamps, [msg.ROT for msg in self.track]
        )
        self.spl_dROT = UnivariateSpline(
            timestamps, [msg.dROT for msg in self.track]
        )


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
        track: List[AISMessage]
        ) -> None:
        
        self.ts = ts
        self.mmsi = mmsi 
        self.track = track
        self._fill_rot()
        self._ts_to_unix()
        self.splines = TrackSplines(self.track)

    def observe_at_query(self) -> np.ndarray:
        """
        Infers
            - Northing (meters),
            - Easting (meters),
            - Course over ground (COG) [degrees],
            - Speed over ground (SOG) [knots],
            - Rate of turn (ROT) [degrees/minute],
            - Change of ROT [degrees/minute²],
            
        from univariate splines for the timestamp, 
        the object was initialized with.

        Inferred variables start with "i_".
        """
        # Convert query timestamp to unix time
        ts = self.ts.timestamp()

        # Return the observed values from the splines
        # at the given timestamp
        # Returns a 1x6 array:
        # [northing, easting, COG, SOG, ROT, dROT]
        return np.array([
            self.splines.spl_northing(ts),
            self.splines.spl_easting(ts),
            self.splines.spl_COG(ts),
            self.splines.spl_SOG(ts),
            self.splines.spl_ROT(ts),
            self.splines.spl_dROT(ts),
        ])

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
            - Rate of turn (ROT) [degrees/minute],
            - Change of ROT [degrees/minute²],

        from univariate splines for the track between
        the given start and end timestamps, with the
        given interval in [seconds].
        """
        # Convert query timestamps to unix time
        start = start.timestamp()
        end = end.timestamp()

        # Check if the interval boundary is within the
        # track's timestamps
        if start < self.lower.timestamp.timestamp():
            raise OutofTimeBoundsError(
                "Start timestamp is before the track's first timestamp."
            )
        if end > self.upper.timestamp.timestamp():
            raise OutofTimeBoundsError(
                "End timestamp is after the track's last timestamp."
            )

        # Create a list of timestamps between the start and end
        # timestamps, with the given interval
        timestamps = np.arange(start, end, interval)

        # Return the observed values from the splines
        # at the given timestamps
        # Returns a Nx7 array:
        # [northing, easting, COG, SOG, ROT, dROT, timestamp]
        preds: np.ndarray = np.array([
            self.splines.spl_northing(timestamps),
            self.splines.spl_easting(timestamps),
            self.splines.spl_COG(timestamps),
            self.splines.spl_SOG(timestamps),
            self.splines.spl_ROT(timestamps),
            self.splines.spl_dROT(timestamps),
            timestamps
        ])
        return preds.T


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
    
    def _find_shell(self) -> None:
        """
        Find the two AIS messages encompassing
        the objects' track elements and save them
        as attributes.
        """ 
        self.lower = self.track[0]
        self.upper = self.track[-1]

    def _ts_to_unix(self) -> None:
        """
        Convert the vessel's timestamp for 
        each track element to unix time.
        """
        for msg in self.track:
            msg.timestamp = msg.timestamp.timestamp()

class TrajectoryMatcher:
    """
    Class for matching trajectories of two vessels
    """

    def __init__(self, vessel1: TargetVessel, vessel2: TargetVessel) -> None:
        self.vessel1 = vessel1
        self.vessel2 = vessel2
    
    def _start(self) -> None:
        """
        Find the starting point included in both trajectories
        """
        if self.vessel1.lower.timestamp < self.vessel2.lower.timestamp:
            self.start = self.vessel2.lower.timestamp
        else:
            self.start = self.vessel1.lower.timestamp

    def _end(self) -> None:
        """
        Find the end point included in both trajectories
        """
        if self.vessel1.upper.timestamp > self.vessel2.upper.timestamp:
            self.end = self.vessel2.upper.timestamp
        else:
            self.end = self.vessel1.upper.timestamp

    def observe_interval(self,interval: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Retruns the trajectories of both vessels
        between the start and end points, with the
        given interval in [seconds].
        """
        obs_vessel1 = self.vessel1.observe_interval(
            self.start, self.end, interval
        )
        obs_vessel2 = self.vessel2.observe_interval(
            self.start, self.end, interval
        )
        return obs_vessel1, obs_vessel2
        


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
