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
from typing import Callable
import numpy as np
from inspect import signature
from functools import partial

from ..tsea.search_agent import Track

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
    """
    assert isinstance(sd,(tuple,float))
    if isinstance(sd,float):
        assert sd > 0
        lower, upper = 0, sd
    else:
        assert all(s > 0 for s in sd)
        lower, upper = sd
    sdlon = np.sqrt(np.var([v.lon for v in track]))
    sdlat = np.sqrt(np.var([v.lat for v in track]))
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

# Example recipe---------------------------------------------------------------
ExampleRecipe = Recipe(
    partial(too_few_obs, n=100),
    partial(spatial_deviation, sd=0.1)
)