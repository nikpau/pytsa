__doc__="""Module for defining trajectory splitting rules.
All rule functions must have the following signature:
    def rule_name(track: Track, *args, **kwargs) -> bool:
        ...
        
Rules must be defined such that they return True if the track is to be
rejected, and False if the track is to be accepted. 
        
To make a recipe for the TrajectorySplitter class, you are expected to
fix the rule function's arguments, such that only a one-argument function
remains. It is recommended to use the `functools.partial` function for this.
        
Once you have defined a set of rule functions, you can create a recipe
for the TrajectorySplitter class by passing them to the Recipe class.

Example:
    from functools import partial
    from pytsa.trajectories import rules
    
    # Define a recipe
    recipe = rules.Recipe(
        partial(rules.too_few_obs, n=100),
        partial(rules.too_small_span, span=0.1)
    )
    
    # Cook the recipe
    cooked = recipe.cook()
    
The `cooked` function can now be passed to the TrajectorySplitter class
to perform the trajectory splitting.

"""
import utm
import numpy as np

from typing import Callable
from inspect import signature
from functools import partial
from scipy.spatial import ConvexHull

from ..structs import Track
from ..logger import logger

class Recipe:
    """
    Rule recipe class.
    =================
    
    This class is used to define a recipe for the TrajectorySplitter class.
    
    """
    Rule = Callable[[Track], bool]
    def __init__(self, *funcs: Rule) -> None:
        self.funcs = funcs
        for func in self.funcs:
            _check_signature(func)
        
    def cook(self) -> Callable[[Track], bool]:
        """
        Cook the recipe into a function that can be passed to the
        TrajectorySplitter class.
        """
        return self.cooked
            
    def cooked(self,track: Track) -> bool:
        return all(func(track) for func in self.funcs)

# Signature checker-------------------------------------------------------------
def _check_signature(func) -> None:
    """
    Check if the given function has the correct signature.
    """
    if not callable(func):
        raise TypeError(f"Expected a callable, got {type(func)}")
    sig = signature(func)
    if arg := next(iter(sig.parameters)) != "track":
        raise TypeError(
            f"Expected a function with parameter `track` as first argument, "
            f"got {arg}"
        )
    if sig.parameters["track"].annotation != Track:
        raise TypeError(
            f"Expected a function with parameter `track` of type "
            f"Track, got {sig.parameters['track'].annotation}"
        )
    if sig.return_annotation != bool:
        raise TypeError(
            f"Expected a function with return type bool, "
            f"got {sig.return_annotation}"
        )

# Rules -----------------------------------------------------------------------

def too_few_obs(track: Track, n: int) -> bool:
    """
    Return True if the length of the track of the given vessel
    is smaller than `n`.
    """
    return len(track) < n

def spatial_deviation(track: Track, sd: tuple | float) -> bool:
    """
    Return True if the summed standard deviation of lat/lon 
    of the track of the given vessel is smaller than `sd`,
    or if `sd` is a tuple, if the standard deviation is
    within the range of `sd`.
    Unit of `sd` is [°].

    Note that the standard deviation is no accurate measure
    of the spatial deviation, as it does not take into account
    the actual distances between the points. It is only a
    rough measure of the spread of the points.
    Accuracy decreases rapidly with proximity to the poles.

    (Not used in the paper)
    """
    assert isinstance(sd,(tuple,float))
    if isinstance(sd,float):
        assert sd > 0
        lower, upper = 0, sd
    else:
        assert all(s > 0 for s in sd)
        lower, upper = sd
    sdlon = np.std([v.lon for v in track])
    sdlat = np.std([v.lat for v in track])
    return lower <= sdlon + sdlat <= upper

def too_small_span(track: Track, span: float) -> bool:
    """
    Return True if the lateral and longitudinal span
    of the track of the given vessel is smaller than `span`.
    (Not used in the paper)
    """
    lat_span = np.ptp([v.lat for v in track])
    lon_span = np.ptp([v.lon for v in track])
    return lat_span > span and lon_span > span

def convex_hull_area(track: Track, area: float | tuple) -> bool:
    """
    Reject a track if the area of the convex hull of the
    track is smaller than `area`. If `area` is a tuple,
    reject the track if the area is not within the range
    of the tuple.
    """
    assert len(track) > 2, "Need at least 3 points to calculate convex hull"
    res = utm.from_latlon(
        np.array([p.lat for p in track]),
        np.array([p.lon for p in track])
    )
    points = np.array([res[0],res[1]]).T
    # Convex hull calculation can fail for 
    # some tracks, e.g. if all points are 
    # colinear. In this case, we reject the track.
    try:
        _cvharea = ConvexHull(points).area
        if isinstance(area,float):
            return _cvharea < area
        elif isinstance(area,tuple):
            lower, upper = area
            return not lower < _cvharea < upper
        else:
            raise TypeError(
                f"Expected area to be a float or tuple, "
                f"got {type(area)}"
            )
    except Exception as e:
        logger.error(
            f"Error in convex hull calculation: {e}"
            "Rejecting track."
            )
        return True

# Example recipe---------------------------------------------------------------
ExampleRecipe = Recipe(
    partial(too_few_obs, n=100),
    partial(convex_hull_area, area=3e4)
)