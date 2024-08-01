"""
Trajectory Inspector.
====================

This module contains the Inspector class, which is used to filter
trajectories based on a set of rules. The rules are defined in the
:mod:`pytsa.trajectories.rules` module.

The module also contains a set of functions that can be used to
calculate the smoothness and change of course of a navigational
track. These functions can be used as rules in the Inspector class.

"""
import numpy as np
import multiprocessing as mp
from copy import deepcopy

from ..logger import logger
from ..structs import Track
from ..tsea.targetship import TargetShip, Targets
from .rules import Recipe

def print_rejection_rate(n_rejected: int, n_total: int) -> None:
    if n_total == 0:
        logger.warning("No trajectories to filter.")
        return
    logger.info(
        f"Filtered {n_total} trajectories. "
        f"{(n_rejected)/n_total*100:.2f}% rejected."
    )

class Inspector:
    """
    The Inspector is the central class for assessing 
    properties of trajectories, based on a set of 
    rules bundled in a Recipe object. The Inspector
    class applies the rules to a set of target ships
    and returns the accepted and rejected trajectories.
    
    Parameters:
    - data (Targets): A dictionary of TargetShip objects, 
        where the keys are the MMSI numbers of the target 
        ships.
        
    - recipe (Recipe): A Recipe object containing the rules
        to be applied to the trajectories.
    """
    def __init__(self, data: Targets, recipe: Recipe) -> None:
        self.data = data
        self.condition = recipe.cook()
        self.rejected: Targets = {}
        self.accepted: Targets = {}
    
    def inspect(self, njobs: int = 1) -> tuple[Targets,Targets]:
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
                try:
                    if self.condition(track):
                        self.reject_track(target_ship,track)
                    else:
                        self.accept_track(target_ship,track)
                except Exception as e:
                    logger.warning(
                        f"Track from ship {target_ship.mmsi} skipped: {e}"
                    )
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

def cosine_of_angle_between(track: Track) -> float:
    """
    Return the cosines for all three consecutive
    AIS Messages in a track.
    """
    positions = np.array([(msg.lon,msg.lat) for msg in track])
    v1 = positions[1:-1] - positions[:-2] # p1 - p0 
    v2 = positions[2:] - positions[1:-1] # p2 - p1
    
    # Dot product
    dot_product = np.einsum('ij,ij->i',v1,v2)
    # Norms
    norms = np.linalg.norm(v1,axis=1) * np.linalg.norm(v2,axis=1)
    
    return dot_product / norms

def angle_between(track: Track) -> float:
    """
    Return the angles between each three consecutive
    AIS Messages in a track.
    """
    _cos = cosine_of_angle_between(track)        
    return np.arccos(np.round(_cos,6)) # Round to avoid floating point errors
    
# def average_smoothness(track: Track) -> float:
#     """
#     Calculate the average smoothness of a navigational 
#     track.

#     This function computes the average smoothness of a 
#     path represented  by a list of AISMessage objects.  
#     It evaluates the smoothness based on the angles 
#     formed at each point along the path, where each 
#     angle is determined by a sequence of three consecutive 
#     AISMessage points. The smoothness is a measure of 
#     how straight or curved the path is, with larger 
#     angles indicating a smoother path.

#     The angle at each point is normalized by dividing it 
#     by π, with the function 'angle_between' used to 
#     calculate these angles. Since 'angle_between' returns 
#     values from 0 to π, the normalized angles will range 
#     from 0 (representing a U-turn)  to 1 (representing a 
#     straight line).

#     Parameters:
#     - track (Track): A list of AISMessage objects 
#       representing the navigational path. Each AISMessage 
#       contains positional data necessary for angle calculation.

#     Returns:
#     - float: The average smoothness of the track, 
#       represented as a float. This value is the mean 
#       of the normalized angles along the track, where 
#       a larger values indicates a smoother track.

#     Note:
#     - The function assumes that the track has at least three 
#        AISMessage points to form at least one angle. If the 
#        track has fewer than three points, the behavior of the 
#        function is unspecified.
#     """
#     if len(track) < 3:
#         raise ValueError(
#             "Average smoothness requires at "
#             "least three messages per track. "
#             "{} were given".format(len(track))
#         )
#     angles = angle_between(track)
#     normalized_angles = (angles / np.pi)
#     return np.mean(normalized_angles)

def average_absolute_change_of_course(track: Track, degrees: bool = False) -> float:
    """
    Calculate the average absolute change of course 
    of a navigational track.

    This function computes the average absolute change 
    of course of a path represented by a list of AISMessage 
    objects. It evaluates the change of course based on 
    the angles formed at each point along the path, where 
    each angle is determined by a sequence of three 
    consecutive AISMessage points. The change of course 
    is a measure of how much the path changes direction 
    at each point, with larger angles indicating a greater 
    change of course.

    Parameters:
    - track (Track): A list of AISMessage objects
        representing the navigational path. Each AISMessage 
        contains positional data necessary for angle calculation.
    
    - degrees (bool): A boolean flag indicating whether
        the output should be in degrees or radians. If 
        True, the output will be in degrees; if False, 
        the output will be in radians. The default value 
        is False.

    Returns:
    - float: The average absolute change of course of the 
      track in degrees or radians. This value is the mean 
      of the absolute differences between consecutive 
      normalized angles along the track, where a larger 
      value indicates a greater change of course.

    Note:
    - The function requires that the track has at least three 
       AISMessage points to form at least one angle.
    """
    assert len(track) >= 3, "Track must have at least three messages."
    aacog = np.mean(angle_between(track))
    return np.degrees(aacog) if degrees else aacog