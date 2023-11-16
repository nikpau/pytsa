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
from ..tsea.search_agent import Targets
from ..tsea.targetship import TargetVessel, AISMessage
from .rules import Recipe

class TrajectorySplitter:
    """
    The trajectory splitter class takes a 
    dictionary of rules and applies them
    to the trajectories of the TargetVessels
    passed to it.
    
    The TargetVessel dict passed to the splitter
    is expected to have the following structure:
        dict[MMSI,TargetVessel]
    and will most likely be the output of the
    :meth:`SearchAgent.get_ships` method.
    """
    def __init__(self, data: Targets, recipe: Recipe) -> None:
        self.data = data
        self.recipe = recipe.cook()
    
    def split(self, njobs: int = 4) -> tuple[Targets,Targets]:
        """
        Split the given target ships' trajectories into two groups:
        - Accepted: Trajectories returning True for the recipe
        - Rejected: Trajectories returning False for the recipe
        
        The accepted and rejected dictionaries can contain the same MMSIs, 
        if the target ship has multiple tracks, and only some of them
        meet the criteria.
        
        """
        def print_rejetion_rate(n_rejected: int, n_total: int) -> None:
            logger.info(
                f"Filtered {n_total} trajectories. "
                f"{(n_rejected)/n_total*100:.2f}% rejected."
            )
        if njobs == 1:
            a,r,_n = self._split_impl(self.data)
            # Number of target ships after filtering
            n_rejected = sum(len(r.tracks) for r in r.values())
            print_rejetion_rate(n_rejected,_n)
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
            results = pool.map(self._split_impl,chunks)
        accepted, rejected, _n = zip(*results)
        a_out, r_out = {}, {}
        for a,r in zip(accepted,rejected):
            a_out.update(a)
            r_out.update(r)
            
        # Number of target ships after filtering
        n_rejected = sum(len(r.tracks) for r in r_out.values())
        print_rejetion_rate(n_rejected,sum(_n))
        
        return a_out, r_out
    

    def _split_impl(self, targets: Targets) -> tuple[Targets,Targets,int]:
        """
        Also remove vessels whose track lies outside
        the queried timestamp.
        
        `overlap_tpos` is a boolean flag indicating whether
        we only want to return vessels, whose track overlaps
        with the queried timestamp. If `overlap_tpos` is False,
        all vessels whose track is within the time delta of the
        queried timestamp are returned.
        
        """
        rejected: Targets = {}
        accepted: Targets = {}


        nships = len(targets)
        _n = 0 # Number of trajectories before split
        for i, (_,target_ship) in enumerate(targets.items()):
            logger.info(f"Filtering target ship {i+1}/{nships}")
            for track in target_ship.tracks:
                _n += 1
                if self.recipe(track):
                    self._copy_track(target_ship,rejected,track)
                else:
                    self._copy_track(target_ship,accepted,track)
        return accepted, rejected, _n
                        
    
    def _copy_track(self,
                    vessel: TargetVessel, 
                    target: Targets,
                    track: list[AISMessage]) -> None:
        """
        Copy a track from one TargetVessel object to another,
        and delete it from the original.
        """
        if vessel.mmsi not in target:
            target[vessel.mmsi] = deepcopy(vessel)
            target[vessel.mmsi].tracks = []
        target[vessel.mmsi].tracks.append(track)
