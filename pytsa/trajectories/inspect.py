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
import math
import multiprocessing as mp
from copy import deepcopy

from ..logger import logger
from ..structs import Track
from ..tsea.targetship import TargetShip, Targets
from .rules import Recipe

def print_rejection_rate(n_rejected: int, n_total: int) -> None:
    """Log the percentage of trajectories rejected during inspection.

    Args:
        n_rejected: Number of trajectories classified as rejected.
        n_total: Number of trajectories considered for classification.

    A warning is logged instead of a percentage when there are no
    trajectories to inspect.
    """
    if n_total == 0:
        logger.warning("No trajectories to filter.")
        return
    logger.info(
        f"Filtered {n_total} trajectories. "
        f"{(n_rejected)/n_total*100:.2f}% rejected."
    )

def _inspect_chunk(args: tuple) -> tuple:
    """Inspect the ships in one multiprocessing work chunk.

    Args:
        args: A ``(chunk_data, condition)`` pair, where ``chunk_data`` is a
            mapping of MMSI values to target ships and ``condition`` returns
            ``True`` for tracks that must be rejected.

    Returns:
        A tuple containing mappings of accepted and rejected ships and the
        number of ships processed. A ship is included in a mapping when at
        least one of its tracks receives that classification.

    Exceptions raised while evaluating an individual track are logged and do
    not prevent the remaining tracks in the chunk from being inspected.
    """
    chunk_data, condition = args
    accepted = {}
    rejected = {}
    n_total = len(chunk_data)
    for i, (mmsi, target_ship) in enumerate(chunk_data.items()):
        logger.info(f"Inspecting target ship {i+1}/{n_total}")
        for track in target_ship.tracks:
            try:
                if condition(track):
                    rejected[mmsi] = target_ship
                else:
                    accepted[mmsi] = target_ship
            except Exception as e:
                logger.warning(
                    f"Track from ship {target_ship.mmsi} skipped: {e}"
                )
    return accepted, rejected, n_total

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
        """Create an inspector for ``data`` using the rules in ``recipe``.

        The recipe is cooked once so its resulting predicate can be reused by
        sequential and multiprocessing inspection. The accepted and rejected
        mappings are retained for the legacy ``_inspect_impl`` pathway.
        """
        self.data = data
        self.condition = recipe.cook()
        self.rejected: Targets = {}
        self.accepted: Targets = {}

    def inspect(self, njobs: int = 1) -> tuple:
        """Classify tracks in the input ships as accepted or rejected.

        A track is rejected when the cooked recipe predicate returns ``True``;
        it is accepted when the predicate returns ``False``. Set ``njobs`` to
        one to process the full mapping in the current process, or to a larger
        value to divide ships across a process pool.

        Args:
            njobs: Number of worker processes. ``1`` performs sequential
                inspection.

        Returns:
            A pair ``(accepted, rejected)`` of MMSI-to-``TargetShip``
            mappings. A ship may occur in both mappings when it has tracks
            with different classifications.
        """
        if njobs == 1:
            a, r, _n = _inspect_chunk((self.data, self.condition))
            n_rejected = sum(len(ship.tracks) for ship in r.values())
            print_rejection_rate(n_rejected, _n)
            return a, r

        # Split whole ships between workers so no process handles the same ship.
        items = list(self.data.items())
        if not items:
            print_rejection_rate(0, 0)
            return {}, {}

        chunk_size = math.ceil(len(items) / njobs)
        
        # Workers receive only their ship subset and the cooked rule predicate.
        tasks = [
            (dict(items[i : i + chunk_size]), self.condition) 
            for i in range(0, len(items), chunk_size)
        ]
        
        with mp.Pool(njobs) as pool:
            results = pool.map(_inspect_chunk, tasks)
            
        accepted, rejected, _n = zip(*results)
        
        # Each worker returns independent mappings, which can be merged by MMSI.
        a_out, r_out = {}, {}
        for a, r in zip(accepted, rejected):
            a_out.update(a)
            r_out.update(r)
            
        n_rejected = sum(len(r_ship.tracks) for r_ship in r_out.values())
        print_rejection_rate(n_rejected, sum(_n))
        
        return a_out, r_out
    
    def _inspect_impl(self, targets: Targets) -> tuple[Targets,Targets,int]:
        """
        Inspect ``targets`` using the instance's stateful track-copy helpers.

        This legacy implementation builds ``self.accepted`` and
        ``self.rejected`` incrementally. Each input track is evaluated once;
        evaluation errors are logged and leave that track out of both outputs.

        Returns:
            The accepted mapping, the rejected mapping, and the number of
            tracks considered before filtering.
        """
        nships = len(targets)
        _n = 0 # Number of tracks considered before classification.
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
        Copy ``track`` into the rejected output for ``vessel``.

        The source vessel and its original track collection are not modified.
        """
        self._copy_track(vessel,self.rejected,track)
        
    def accept_track(self,
                     vessel: TargetShip,
                     track: Track) -> None:
        """
        Copy ``track`` into the accepted output for ``vessel``.

        The source vessel and its original track collection are not modified.
        """
        self._copy_track(vessel,self.accepted,track)        
    
    def _copy_track(self,
                    vessel: TargetShip, 
                    target: Targets,
                    track: Track) -> None:
        """
        Add a track to a classification output without mutating its source.

        A deep copy of ``vessel`` is created on first use for its MMSI, then
        its copied ``tracks`` collection is populated only with tracks assigned
        to ``target``. The original vessel remains unchanged.
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