"""
Trajectory Splitter.
====================

This module contains the class for splitting trajectories
according to a set of rules.
"""
import numpy as np
import multiprocessing as mp
from copy import deepcopy

from ..logger import logger
from ..tsea.search_agent import Targets, Track
from ..tsea.targetship import TargetShip, AISMessage
from .rules import Recipe

def print_rejection_rate(n_rejected: int, n_total: int) -> None:
    logger.info(
        f"Filtered {n_total} trajectories. "
        f"{(n_rejected)/n_total*100:.2f}% rejected."
    )

class Inspector:
    """
    The Inspector takes a dictionary of rules 
    and applies them to the trajectories of 
    a `Targets` type dictionary passed to it.
    
    The dict passed to the inspector
    is expected to have the following structure:
        dict[MMSI,TargetVessel]
    and will most likely be the output of the
    :meth:`SearchAgent.get_all_ships()` method.
    
    For how to create a recipe, see the 
    :mod:`pytsa.trajectories.rules` module.
    """
    def __init__(self, data: Targets, recipe: Recipe) -> None:
        self.data = data
        self.condition = recipe.cook()
        self.rejected: Targets = {}
        self.accepted: Targets = {}
    
    def inspect(self, njobs: int = 4) -> tuple[Targets,Targets]:
        """
        Inspects TargetShips in `data` and returns two dictionaries:
        - Accepted: Trajectories evalutating to False for the recipe
        - Rejected: Trajectories evalutating to True for the recipe
        
        The accepted and rejected dictionaries can contain the same MMSIs, 
        if the target ship has multiple tracks, and only some of them
        meet the criteria.

        **NOTE**:
        n > 1 is only recommended for smaller datasets, as the overhead
        of splitting the data into chunks and recombining it can be
        significant.
        
        """
        if njobs == 1:
            a,r,_n = self._inspect_impl(self.data)
            # Number of target ships after filtering
            n_rejected = sum(len(r.tracks) for r in r.values())
            print_rejection_rate(n_rejected,_n)
            return a,r
        # Split the target ships into `njobs` chunks
        items = list(self.data.items())
        mmsis, target_ships = zip(*items)
        mmsi_chunks = np.array_split(mmsis,njobs)
        target_ship_chunks = np.array_split(target_ships,njobs)
        chunks = []
        for mmsi_chunk, target_ship_chunk in zip(mmsi_chunks,target_ship_chunks):
            chunks.append(dict(zip(mmsi_chunk,target_ship_chunk)))
        
        with mp.Pool(njobs) as pool:
            results = pool.map(self._inspect_impl,chunks)
        accepted, rejected, _n = zip(*results)
        a_out, r_out = {}, {}
        for a,r in zip(accepted,rejected):
            a_out.update(a)
            r_out.update(r)
            
        # Number of target ships after filtering
        n_rejected = sum(len(r.tracks) for r in r_out.values())
        print_rejection_rate(n_rejected,sum(_n))
        
        return a_out, r_out
    
    def _inspect_impl(self, targets: Targets) -> tuple[Targets,Targets,int]:
        """
        Inspector implementation.
        """
        nships = len(targets)
        _n = 0 # Number of trajectories before split
        for i, (_,target_ship) in enumerate(targets.items()):
            logger.info(f"Inspecting target ship {i+1}/{nships}")
            for track in target_ship.tracks:
                _n += 1
                if self.condition(track):
                    self.reject_track(target_ship,track)
                else:
                    self.accept_track(target_ship,track)
        return self.accepted, self.rejected, _n
    
    def reject_track(self,
                     vessel: TargetShip,
                     track: Track) -> None:
        """
        Reject a track.
        """
        self._copy_track(vessel,self.rejected,track)
        
    def accept_track(self,
                     vessel: TargetShip,
                     track: Track) -> None:
        """
        Accept a track.
        """
        self._copy_track(vessel,self.accepted,track)        
    
    def _copy_track(self,
                    vessel: TargetShip, 
                    target: Targets,
                    track: Track) -> None:
        """
        Copy a track from one TargetVessel object to another,
        and delete it from the original.
        """
        if vessel.mmsi not in target:
            target[vessel.mmsi] = deepcopy(vessel)
            target[vessel.mmsi].tracks = []
        target[vessel.mmsi].tracks.append(track)

# Utility functions for calculating
# the adapted average smoothness
def cosine_of_angle_between(msg_t0: AISMessage,
                            msg_t1: AISMessage,
                            msg_t2: AISMessage) -> float:
    """
    Return the cosine of the angle between the track
    of three AIS Messages.
    """
    p1 = (msg_t0.lon,msg_t0.lat)
    p2 = (msg_t1.lon,msg_t1.lat)
    p3 = (msg_t2.lon,msg_t2.lat)
    
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    
    nom = np.dot(v1,v2)
    den = np.linalg.norm(v1) * np.linalg.norm(v2)
    return nom / den

def angle_between(msg_t0: AISMessage,
                  msg_t1: AISMessage,
                  msg_t2: AISMessage) -> float:
        """
        Return the angle between the track
        of three AIS Messages.
        """
        _cos = cosine_of_angle_between(msg_t0,msg_t1,msg_t2)        
        return np.arccos(round(_cos,6)) # Round to avoid floating point errors
    
def average_smoothness(track: Track) -> float:
    """
    Calculate the average smoothness of a navigational 
    track.

    This function computes the average smoothness of a 
    path represented  by a list of AISMessage objects.  
    It evaluates the smoothness based on the angles 
    formed at each point along the path, where each 
    angle is determined by a sequence of three consecutive 
    AISMessage points. The smoothness is a measure of 
    how straight or curved the path is, with larger 
    angles indicating a smoother path.

    The angle at each point is normalized by dividing it 
    by π, with the function 'angle_between' used to 
    calculate these angles. Since 'angle_between' returns 
    values from 0 to π, the normalized angles will range 
    from 0 (representing a U-turn)  to 1 (representing a 
    straight line).

    Parameters:
    - track (Track): A list of AISMessage objects 
      representing the navigational path. Each AISMessage 
      contains positional data necessary for angle calculation.

    Returns:
    - float: The average smoothness of the track, 
      represented as a float. This value is the mean 
      of the normalized angles along the track, where 
      a larger values indicates a smoother track.

    Note:
    - The function assumes that the track has at least three 
       AISMessage points to form at least one angle. If the 
       track has fewer than three points, the behavior of the 
       function is unspecified.
    """
    angles = []
    if len(track) < 3:
        raise ValueError(
            "Average smoothness requires at "
            "least three messages per track. "
            "{} were given".format(len(track))
        )
    for i in range(1,len(track)-1):
        ang = angle_between(track[i-1],track[i],track[i+1])
        angles.append((ang / np.pi)**2)
    return np.mean(angles)
