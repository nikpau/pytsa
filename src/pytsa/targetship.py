"""
This module aims at finding all spatio-temporal
neighbors around a given (stripped) AIS Message. 

A provided geographical area will be split into 
evenly-sized grids
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import List, Union

import numpy as np
import utm
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.interpolate import InterpolatedUnivariateSpline, interp1d
from pytsa.structs import ShipType

# Settings for numerical integration
Q_SETTINGS = dict(epsabs=1e-13,epsrel=1e-13,limit=500)



# Type aliases
Latitude = float
Longitude = float
MMSI = int

# Constants
PI = np.pi

# Exceptions

class OutofTimeBoundsError(Exception):
    pass

class InterpolationError(Exception):
    pass


class TrackSplines:
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
        - ROT: Spline interpolation of rate of turn
        - dROT: Spline interpolation of change of rate of turn

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
        self.northing = InterpolatedUnivariateSpline(
            timestamps, [msg.northing for msg in self.track]
        )
        self.easting = InterpolatedUnivariateSpline(
            timestamps, [msg.easting for msg in self.track]
        )
        self.COG = InterpolatedUnivariateSpline(
            timestamps, [msg.COG for msg in self.track]
        )
        self.SOG = InterpolatedUnivariateSpline(
            timestamps, [msg.SOG for msg in self.track]
        )
        self.ROT = InterpolatedUnivariateSpline(
            timestamps, [msg.ROT for msg in self.track]
        )
        self.dROT = InterpolatedUnivariateSpline(
            timestamps, [msg.dROT for msg in self.track]
        )

class TrackLinear:
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
            timestamps, [msg.lon for msg in self.track]
        )
        self.northing = interp1d(
            timestamps, [msg.northing for msg in self.track]
        )
        self.easting = interp1d(
            timestamps, [msg.easting for msg in self.track]
        )
        self.COG = interp1d(
            timestamps, [msg.COG for msg in self.track]
        )
        self.SOG = interp1d(
            timestamps, [msg.SOG for msg in self.track]
        )
        self.ROT = interp1d(
            timestamps, [msg.ROT for msg in self.track]
        )
        # Derivative of ROT is not available in linear interpolation
        # as there are too few points to perform numerical differentiation.
        # Instead, we set it to zero.
        self.dROT = interp1d(
            timestamps, [0.0 for _ in self.track]
        )


@dataclass
class AISMessage:
    """
    AIS Message object
    """
    sender: MMSI
    timestamp: datetime | float
    lat: Latitude
    lon: Longitude
    COG: float # Course over ground [degrees]
    SOG: float # Speed over ground [knots]
    ROT: float = None # Rate of turn [degrees/minute]
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
        try: 
            rot = float(rot) 
        except: 
            return None
        
        sign = np.sign(rot)
        if abs(rot) == 127 or abs(rot) == 128:
            return None
        else:
            return sign * (rot / 4.733)**2
        

class TargetVessel:
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
        ts: Union[str,datetime], 
        mmsi: MMSI, 
        tracks: List[List[AISMessage]],
        ship_type: ShipType = None,
        length: float = None
        ) -> None:
        
        self.ts = ts
        self.mmsi = mmsi 
        self.tracks = tracks
        self.ship_type = ship_type
        self.length = length
        self.lininterp = False # Linear interpolation flag
        
        # Indicate if the interpolation function have 
        # been attached to the object
        self.interpolation: list[TrackSplines] | list[TrackLinear] | None = None
    
    def interpolate(self,mode: str) -> None:
        """
        Construct splines for the target vessel
        """
        assert mode in ["linear","spline","auto"],\
            "Mode must be either 'linear', 'spline' or 'auto'."
        self.interpolation = []
        for track in self.tracks:
            if mode == "auto":
                try:
                    if self.lininterp:
                        self.interpolation.append(TrackLinear(track))
                    else:
                        self.interpolation.append(TrackSplines(track))
                except Exception as e:
                    raise InterpolationError(
                        f"Could not interpolate the target vessel trajectory:\n{e}."
                    )
            elif mode == "linear":
                try:
                    self.interpolation.append(TrackLinear(track))
                except Exception as e:
                    raise InterpolationError(
                        f"Could not interpolate the target vessel trajectory:\n{e}."
                    )
            elif mode == "spline":
                try:
                    self.interpolation.append(TrackSplines(track))
                except Exception as e:
                    raise InterpolationError(
                        f"Could not interpolate the target vessel trajectory:\n{e}."
                    )

    def observe_at_query(self, time: datetime | str | None = None) -> np.ndarray:
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

        """
        assert self.interpolation is not None,\
            "Interpolation has not been run. Call interpolate() first."
        # Convert query timestamp to unix time
        if time is not None:
            ts = time.timestamp()
        else:
            ts = self.ts.timestamp()
            
        # Check if the query timestamp is within the
        # track's timestamps
        for i,track in enumerate(self.tracks):
            if self._is_in_interval(ts,[msg.timestamp for msg in track]):
                break
            if i == len(self.tracks)-1:
                raise OutofTimeBoundsError(
                    "Query timestamp is not within the track's timestamps."
                )

        # Return the observed values from the splines
        # at the given timestamp
        # Returns a 1x6 array:
        # [northing, easting, COG, SOG, ROT, dROT]
        return np.array([
            self.interpolation[i].northing(ts),
            self.interpolation[i].easting(ts),
            # Take the modulo 360 to ensure that the
            # course over ground is in the interval [0,360]
            self.interpolation[i].COG(ts) % 360,
            self.interpolation[i].SOG(ts),
            self.interpolation[i].ROT(ts),
            self.interpolation[i].dROT(ts),
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
        assert self.interpolation is not None,\
            "Interpolation has not been run. Call interpolate() first."
        # Convert query timestamps to unix time
        if isinstance(start, datetime):
            start = start.timestamp()
            end = end.timestamp()

        # Check if the interval boundary is within the
        # track's timestamps
        if start < self.lower.timestamp:
            raise OutofTimeBoundsError(
                "Start timestamp is before the track's first timestamp."
            )
        if end > self.upper.timestamp:
            raise OutofTimeBoundsError(
                "End timestamp is after the track's last timestamp."
            )
        
        # Convert interval from seconds to milliseconds
        #interval = interval * 1000

        # Create a list of timestamps between the start and end
        # timestamps, with the given interval
        timestamps = np.arange(start, end, interval)

        # Return the observed values from the splines
        # at the given timestamps
        # Returns a Nx7 array:
        # [northing, easting, COG, SOG, ROT, dROT, timestamp]
        preds: np.ndarray = np.array([
            self.interpolation.northing(timestamps),
            self.interpolation.easting(timestamps),
            # Take the modulo 360 of the COG to get the
            # heading to be between 0 and 360 degrees
            self.interpolation.COG(timestamps) % 360,
            self.interpolation.SOG(timestamps),
            self.interpolation.ROT(timestamps),
            self.interpolation.dROT(timestamps),
            timestamps
        ])
        return preds.T
    
    def _is_in_interval(self, query: int, timestamps: list[int]) -> bool:
        """
        Check if the query timestamp is within the
        given interval.
        """
        return (query >= timestamps[0]) and (query <= timestamps[-1])


    def fill_rot(self) -> None:
        """
        Fill out missing rotation data and first 
        derivative of roatation by inferring it from 
        the previous and next AIS messages' headings.
        """
        for idx, msg in enumerate(self.tracks):
            if idx == 0:
                self.tracks[idx].ROT = 0.0
                continue
            # Fill out missing ROT data
            if msg.ROT is None:
                num = self.tracks[idx].COG - self.tracks[idx-1].COG 
                den = (self.tracks[idx].timestamp - self.tracks[idx-1].timestamp).seconds/60
                if den == 0:
                    self.tracks[idx].ROT = 0.0
                else:
                    self.tracks[idx].ROT = num/den

        for idx, msg in enumerate(self.tracks):
            if idx == 0 or idx == len(self.tracks)-1:
                self.tracks[idx].dROT = 0.0
                continue
            # Calculate first derivative of ROT
            num = self.tracks[idx+1].ROT - self.tracks[idx].ROT
            den = (self.tracks[idx+1].timestamp - self.tracks[idx].timestamp).seconds/60 # Minutes
            if den == 0:
                self.tracks[idx].dROT = 0.0
            else:
                self.tracks[idx].dROT = num/den


    def overwrite_rot(self) -> None:
        """
        Overwrite the ROT and dROT values with
        the values from COG.

        Note that the timestamps are already in
        unix timestamps, so we need to divide by 60
        to get the minutes.
        """
        for idx, msg in enumerate(self.tracks):
            if idx == 0:
                continue

            num = self.tracks[idx].COG - self.tracks[idx-1].COG 
            den = (self.tracks[idx].timestamp - self.tracks[idx-1].timestamp)/60 # Minutes
            if den == 0:
                msg.ROT = 0.0
            else:
                msg.ROT = num/den

        for idx, msg in enumerate(self.tracks):
            if idx == 0 or idx == len(self.tracks)-1:
                continue
            # Calculate first derivative of ROT
            num = self.tracks[idx+1].ROT - self.tracks[idx].ROT
            den = (self.tracks[idx+1].timestamp - self.tracks[idx].timestamp)/60
            if den == 0:
                msg.dROT = 0.0
            else:
                msg.dROT = num/den

        self.interpolate()

    
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

    def ts_to_unix(self) -> None:
        """
        Convert the vessel's timestamp for 
        each track element to unix time.
        """
        for track in self.tracks:
            for msg in track:
                msg.timestamp = msg.timestamp.timestamp()

class TrajectoryMatcher:
    """
    Class for matching trajectories of two vessels.

    The trajectories are matched if they overlap
    for a given threshold.

    The trajectories are disjoint if they do not
    overlap at all.
    """

    def __init__(
            self, 
            vessel1: TargetVessel, 
            vessel2: TargetVessel,
            threshold: float = 20 # [min] Minimun overlap between trajectories
            ) -> None:
        self.vessel1 = vessel1
        self.vessel2 = vessel2

        self.vessel1.find_shell()
        self.vessel2.find_shell()

        if self._disjoint():
            self.disjoint_trajectories = True
            self.overlapping_trajectories = False
        else:
            self.disjoint_trajectories = False
            self._start()
            self._end()
            if self._overlapping(threshold):
                self.overlapping_trajectories = True
            else:
                self.overlapping_trajectories = False

    def _overlapping(self, threshold: int) -> bool:
        """
        Check if the trajectories overlap for a given threshold
        """
        return (self.end - self.start) > threshold * 60
    
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
    
    def _disjoint(self) -> bool:
        """
        Check if the trajectories are disjoint on the time axis
        """
        return (
            (self.vessel1.upper.timestamp < self.vessel2.lower.timestamp) or
            (self.vessel2.upper.timestamp < self.vessel1.lower.timestamp) or 
            (self.vessel1.lower.timestamp > self.vessel2.upper.timestamp) or
            (self.vessel2.lower.timestamp > self.vessel1.upper.timestamp)
        )

    def observe_interval(self,interval: int) -> TrajectoryMatcher:
        """
        Retruns the trajectories of both vessels
        between the start and end points, with the
        given interval in [seconds].
        """
        if self.disjoint_trajectories:
            raise ValueError(
                "Trajectories are disjoint on the time scale."
            )
        
        obs_vessel1 = self.vessel1.observe_interval(
            self.start, self.end, interval
        )
        obs_vessel2 = self.vessel2.observe_interval(
            self.start, self.end, interval
        )

        self.obs_vessel1 = obs_vessel1
        self.obs_vessel2 = obs_vessel2
        
        return self

    def plot(self, every: int = 10, path: str = None) -> None:
        """
        Plot the trajectories of both vessels
        between the start and end points.
        """
        
        n = every
        v1color = "#d90429"
        v2color = "#2b2d42"

        # Check if obs_vessel1 and obs_vessel2 are defined
        try:
            obs_vessel1 = self.obs_vessel1
            obs_vessel2 = self.obs_vessel2
        except AttributeError:
            raise AttributeError(
                "Nothing to plot. "
                "Please run observe_interval() before plotting."
            )

        # Plot trajectories and metrics
        fig = plt.figure(layout="constrained",figsize=(16,10))
        gs = GridSpec(4, 2, figure=fig)
        
        ax1 = fig.add_subplot(gs[0:2, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 1])
        ax4 = fig.add_subplot(gs[2, 0])
        ax5 = fig.add_subplot(gs[3, 0])
        ax6 = fig.add_subplot(gs[2, 1])
        ax7 = fig.add_subplot(gs[3, 1])

        # Custom xticks for time
        time_tick_locs = obs_vessel1[:,6][::10]
        # Make list of HH:MM for each unix timestamp
        time_tick_labels = [datetime.fromtimestamp(t).strftime('%H:%M') for t in time_tick_locs]

        # Plot trajectories in easting-northing space
        v1p = ax1.plot(obs_vessel1[:,1], obs_vessel1[:,0],color = v1color)[0]
        v2p = ax1.plot(obs_vessel2[:,1], obs_vessel2[:,0],color=v2color)[0]

        for x,y,n in zip(obs_vessel1[:,1][::n], 
                         obs_vessel1[:,0][::n],
                         np.arange(len(obs_vessel1[:,0][::n]))):
            ax1.text(x,y,n,fontsize=8,color=v1color)
        
        for x,y,n in zip(obs_vessel2[:,1][::n],
                        obs_vessel2[:,0][::n],
                        np.arange(len(obs_vessel2[:,0][::n]))):
             ax1.text(x,y,n,fontsize=8,color=v2color)
            
        
        v1s = ax1.scatter(obs_vessel1[:,1][::n], obs_vessel1[:,0][::n],color = v1color)
        v2s = ax1.scatter(obs_vessel2[:,1][::n], obs_vessel2[:,0][::n],color=v2color)
        ax1.set_title("Trajectories")
        ax1.set_xlabel("Easting [m]")
        ax1.set_ylabel("Northing [m]")
        ax1.legend(
            [(v1p,v1s),(v2p,v2s)],
            [f"Vessel {self.vessel1.mmsi}", f"Vessel {self.vessel2.mmsi}"],
            fontsize=8
        )

        # Plot easting in time-space
        v1ep = ax2.plot(obs_vessel1[:,6],obs_vessel1[:,1],color = v1color)[0]
        v1es = ax2.scatter(obs_vessel1[:,6][::n],obs_vessel1[:,1][::n],color=v1color)
        v2ep = ax2.plot(obs_vessel2[:,6],obs_vessel2[:,1],color = v2color)[0]
        v2es = ax2.scatter(obs_vessel2[:,6][::n],obs_vessel2[:,1][::n],color=v2color)

        # Original trajectories for both vessels
        v1esp = ax2.scatter(
            [m.timestamp for m in self.vessel1.tracks],
            [m.easting for m in self.vessel1.tracks],color = v1color,marker="x"
        )
        v2esp = ax2.scatter(
            [m.timestamp for m in self.vessel2.tracks],
            [m.easting for m in self.vessel2.tracks],color=v2color,marker="x"
        )
        ax2.set_xticks(time_tick_locs)
        ax2.set_xticklabels(time_tick_labels, rotation=45)
        
        ax2.set_title("Easting")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Easting [m]")
        ax2.legend(
            [(v1ep,v1es),(v2ep,v2es),(v1esp),(v2esp)],
            [
                f"Vessel {self.vessel1.mmsi}", 
                f"Vessel {self.vessel2.mmsi}",
                f"Vessel {self.vessel1.mmsi} raw data", 
                f"Vessel {self.vessel2.mmsi} raw data"
            ],
            fontsize=8
        )

        # Plot northing in time-space
        v1np = ax3.plot(obs_vessel1[:,6],obs_vessel1[:,0],color=v1color)[0]
        v1ns = ax3.scatter(obs_vessel1[:,6][::n],obs_vessel1[:,0][::n],color=v1color)
        v2np = ax3.plot(obs_vessel2[:,6],obs_vessel2[:,0],color=v2color)[0]
        v2ns = ax3.scatter(obs_vessel2[:,6][::n],obs_vessel2[:,0][::n],color=v2color)

        # Original trajectories for both vessels
        v1nsp = ax3.scatter(
            [m.timestamp for m in self.vessel1.tracks],
            [m.northing for m in self.vessel1.tracks],color=v1color,marker="x"
        )
        v2nsp = ax3.scatter(
            [m.timestamp for m in self.vessel2.tracks],
            [m.northing for m in self.vessel2.tracks],color=v2color,marker="x"
        )
        ax3.set_xticks(time_tick_locs)
        ax3.set_xticklabels(time_tick_labels, rotation=45)
        
        ax3.set_title("Northing")
        ax3.set_xlabel("Time")
        ax3.set_ylabel("Nothing [m]")
        ax3.legend(
            [(v1np,v1ns),(v2np,v2ns),(v1nsp),(v2nsp)],
            [
                f"Vessel {self.vessel1.mmsi}", 
                f"Vessel {self.vessel2.mmsi}",
                f"Vessel {self.vessel1.mmsi} raw data",
                f"Vessel {self.vessel2.mmsi} raw data"
            ],
            fontsize=8
        )
        
        # Plot COG in time-space
        v1cp = ax4.plot(obs_vessel1[:,6],obs_vessel1[:,2],color=v1color)[0]
        v1cs = ax4.scatter(obs_vessel1[:,6][::n],obs_vessel1[:,2][::n],color=v1color)
        v2cp = ax4.plot(obs_vessel2[:,6],obs_vessel2[:,2],color=v2color)[0]
        v2cs = ax4.scatter(obs_vessel2[:,6][::n],obs_vessel2[:,2][::n],color=v2color)

        # Original trajectories for both vessels
        v1csp = ax4.scatter(
            [m.timestamp for m in self.vessel1.tracks],
            [m.COG for m in self.vessel1.tracks],color=v1color,marker="x"
        )
        v2csp = ax4.scatter(
            [m.timestamp for m in self.vessel2.tracks],
            [m.COG for m in self.vessel2.tracks],color=v2color,marker="x"
        )
        ax4.set_xticks(time_tick_locs)
        ax4.set_xticklabels(time_tick_labels, rotation=45)

        ax4.set_title("Course over Ground")
        ax4.set_xlabel("Time")
        ax4.set_ylabel("Course over Ground [deg]")
        ax4.legend(
            [(v1cp,v1cs),(v2cp,v2cs),(v1csp),(v2csp)],
            [
                f"Vessel {self.vessel1.mmsi}", 
                f"Vessel {self.vessel2.mmsi}",
                f"Vessel {self.vessel1.mmsi} raw data",
                f"Vessel {self.vessel2.mmsi} raw data"
            ],
            fontsize=8
        )
        
        # Plot SOG in time-space
        v1sp = ax5.plot(obs_vessel1[:,6],obs_vessel1[:,3],color=v1color)[0]
        v1ss = ax5.scatter(obs_vessel1[:,6][::n],obs_vessel1[:,3][::n],color=v1color)
        v2sp = ax5.plot(obs_vessel2[:,6],obs_vessel2[:,3],color=v2color)[0]
        v2ss = ax5.scatter(obs_vessel2[:,6][::n],obs_vessel2[:,3][::n],color=v2color)

        # Original trajectories for both vessels
        v1ssp = ax5.scatter(
            [m.timestamp for m in self.vessel1.tracks],
            [m.SOG for m in self.vessel1.tracks],color=v1color,marker="x"
        )
        v2ssp = ax5.scatter(
            [m.timestamp for m in self.vessel2.tracks],
            [m.SOG for m in self.vessel2.tracks],color=v2color,marker="x"
        )
        ax5.set_xticks(time_tick_locs)
        ax5.set_xticklabels(time_tick_labels, rotation=45)

        ax5.set_title("Speed over Ground")
        ax5.set_xlabel("Time")
        ax5.set_ylabel("Speed over Ground [knots]")
        ax5.legend(
            [(v1sp,v1ss),(v2sp,v2ss),(v1ssp),(v2ssp)],
            [
                f"Vessel {self.vessel1.mmsi}", 
                f"Vessel {self.vessel2.mmsi}",
                f"Vessel {self.vessel1.mmsi} raw data",
                f"Vessel {self.vessel2.mmsi} raw data"
            ],
            fontsize=8
        )

        # Plot ROT in time-space
        v1rp = ax6.plot(obs_vessel1[:,6],obs_vessel1[:,4],color=v1color)[0]
        v1rs = ax6.scatter(obs_vessel1[:,6][::n],obs_vessel1[:,4][::n],color=v1color)
        v2rp = ax6.plot(obs_vessel2[:,6],obs_vessel2[:,4],color=v2color)[0]
        v2rs = ax6.scatter(obs_vessel2[:,6][::n],obs_vessel2[:,4][::n],color=v2color)

        # Original trajectories for both vessels
        v1rsp = ax6.scatter(
            [m.timestamp for m in self.vessel1.tracks],
            [m.ROT for m in self.vessel1.tracks],color=v1color,marker="x"
        )
        v2rsp = ax6.scatter(
            [m.timestamp for m in self.vessel2.tracks],
            [m.ROT for m in self.vessel2.tracks],color=v2color,marker="x"
        )
        ax6.set_xticks(time_tick_locs)
        ax6.set_xticklabels(time_tick_labels, rotation=45)

        ax6.set_title("Rate of Turn")
        ax6.set_xlabel("Time")
        ax6.set_ylabel("Rate of Turn [deg/min]")
        ax6.legend(
            [(v1rp,v1rs),(v2rp,v2rs),(v1rsp),(v2rsp)],
            [
                f"Vessel {self.vessel1.mmsi}", 
                f"Vessel {self.vessel2.mmsi}",
                f"Vessel {self.vessel1.mmsi} raw data",
                f"Vessel {self.vessel2.mmsi} raw data"
            ],
            fontsize=8
        )

        # Plot dROT in time-space
        v1drp = ax7.plot(obs_vessel1[:,6],obs_vessel1[:,5],color=v1color)[0]
        v1drs = ax7.scatter(obs_vessel1[:,6][::n],obs_vessel1[:,5][::n],color=v1color)
        v2drp = ax7.plot(obs_vessel2[:,6],obs_vessel2[:,5],color=v2color)[0]
        v2drs = ax7.scatter(obs_vessel2[:,6][::n],obs_vessel2[:,5][::n],color=v2color)

        # Original trajectories for both vessels
        v1drsp = ax7.scatter(
            [m.timestamp for m in self.vessel1.tracks],
            [m.dROT for m in self.vessel1.tracks],color=v1color,marker="x"
        )
        v2drsp = ax7.scatter(
            [m.timestamp for m in self.vessel2.tracks],
            [m.dROT for m in self.vessel2.tracks],color=v2color,marker="x"
        )
        ax7.set_xticks(time_tick_locs)
        ax7.set_xticklabels(time_tick_labels, rotation=45)

        ax7.set_title("Derivative of Rate of Turn")
        ax7.set_xlabel("Time")
        ax7.set_ylabel("Rate of Turn [$deg/min^2$]")
        ax7.legend(
            [(v1drp,v1drs),(v2drp,v2drs),(v1drsp),(v2drsp)],
            [
                f"Vessel {self.vessel1.mmsi}", 
                f"Vessel {self.vessel2.mmsi}",
                f"Vessel {self.vessel1.mmsi} raw data",
                f"Vessel {self.vessel2.mmsi} raw data"
            ],
            fontsize=8
        )
        
        plt.suptitle("Trajectories")
        plt.tight_layout()

        if path is None:
            # Make directory if it does not exist
            if not os.path.exists("~/aisout/plots"):
                os.makedirs("~/aisout/plots")
            savepath = f"~/aisout/plots/"
        else:
            savepath = path

        fname = f"trajectories_{self.vessel1.mmsi}_{self.vessel2.mmsi}.png"

        # Check if file exists and if so, append number to filename
        if os.path.exists(f"{savepath}/{fname}"):
            i = 1
            while os.path.exists(f"{savepath}/{fname}"):
                fname = f"trajectories_{self.vessel1.mmsi}_{self.vessel2.mmsi}_{i}.png"
                i += 1

        plt.savefig(
            f"{savepath}/{fname}",
            dpi=300
        )
        plt.close()
        


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
