"""
Plotting functions for Empirical Cumulative 
Distribution Functions (ECDFs) of the split
point metrics.

All functions in this module are take an instance
of the :class:`pytsa.tsea.search_agent.SearchAgent`
as their only argument.
"""
import sys
from typing import Generator
import numpy as np
import matplotlib.patches as patches
from functools import partial
if sys.version_info >= (3,10):
    from itertools import pairwise
else:
    from more_itertools import pairwise

from . import (
    PLOT_FOLDER,
    COLORWHEEL,
    plt
)
from ..tsea.search_agent import SearchAgent, Targets
from ..tsea.targetship import TargetShip
from ..structs import AISMessage, BoundingBox
from ..tsea import split

def iter_msg(sa: SearchAgent, 
             skip_tsplit: bool = False
            ) -> Generator[tuple[AISMessage,AISMessage],None,None]:
    """
    Iterate over all messages in the 
    trajectories extacted by the `SearchAgent`
    and yield the raw messages in a pairwise
    fashion.
    """
    ships: Targets = sa.extract_all(skip_tsplit=skip_tsplit)
    for ship in ships.values():
        for track in ship.tracks:
            for msg1, msg2 in pairwise(track):
                yield (msg1,msg2)

# Export raw and split messages iterators         
iter_msg_raw = partial(iter_msg,skip_tsplit=True)
iter_msg_tsplit = partial(iter_msg,skip_tsplit=False)

def plot_time_diffs(sa: SearchAgent):
    """
    Plot the time difference between two consecutive
    messages for data in the `SearchAgent`.
    """
    f, ax = plt.subplots(1,1,figsize=(6,4))
    ax: plt.Axes
    time_diffs = []
                
    for msg1, msg2 in iter_msg_raw(sa):
        time_diffs.append(msg2.timestamp - msg1.timestamp)
        
    # Quantiles of time diffs
    qs = np.quantile(
        time_diffs,
        [0.99,0.95,0.90]
    )

    # Boxplot of time diffs
    ax.boxplot(
        time_diffs,
        vert=False,
        showfliers=False,
        patch_artist=True,
        widths=0.5,
        boxprops=dict(facecolor=COLORWHEEL[0], color=COLORWHEEL[0]),
        medianprops=dict(color=COLORWHEEL[1]),
        whiskerprops=dict(color=COLORWHEEL[1]),
        capprops=dict(color=COLORWHEEL[1]),
        flierprops=dict(color=COLORWHEEL[1], markeredgecolor=COLORWHEEL[1])
    )
    # Remove y ticks
    ax.set_yticks([])
    
    # Legend with heading
    ax.legend(
        handles=[
            patches.Patch(
                color=COLORWHEEL[0],
                label=f"1% larger than {qs[0]:.2f} s"
            ),
            patches.Patch(
                color=COLORWHEEL[0],
                label=f"5% larger than {qs[1]:.2f} s"
            ),
            patches.Patch(
                color=COLORWHEEL[0],
                label=f"10% larger than {qs[2]:.2f} s"
            )
        ]
    )
    
    ax.set_xlabel("Time difference [s]")
    
    ax.set_title(
        "Time difference between two consecutive messages",fontsize=10
    )
    
    plt.tight_layout()
    plt.savefig(f"{PLOT_FOLDER}/time_diffs.pdf")
    
def plot_reported_vs_calculated_speed(sa:SearchAgent):
    """
    Plot the reported speed against the
    calculated speed.
    """
    
    f, ax = plt.subplots(1,1,figsize=(6,4))
    ax: plt.Axes
    speeds = []
    
    for msg1, msg2 in iter_msg_raw(sa):
        rspeed = split.avg_speed(msg1,msg2)
        cspeed = split.speed_from_position(msg1,msg2)
        speeds.append(rspeed - cspeed)
    
    # Boxplot of speeds
    ax.boxplot(
        speeds,
        vert=False,
        labels=["Speed"],
        showfliers=False,
        patch_artist=True,
        widths=0.5,
        boxprops=dict(facecolor=COLORWHEEL[0], color=COLORWHEEL[0]),
        medianprops=dict(color=COLORWHEEL[1]),
        whiskerprops=dict(color=COLORWHEEL[1],width=1.5),
        capprops=dict(color=COLORWHEEL[1]),
        flierprops=dict(color=COLORWHEEL[1], markeredgecolor=COLORWHEEL[1])
    )
    
    # Quantiles of speeds
    s_qs = np.quantile(
        speeds,
        [0.005,0.995,0.025,0.975,0.05,0.95]
    )
    q_labels_h = [
        f"99% within [{s_qs[0]:.2f} kn,{s_qs[1]:.2f} kn]",
        f"95% within [{s_qs[2]:.2f} kn,{s_qs[3]:.2f} kn]",
        f"90% within [{s_qs[4]:.2f} kn,{s_qs[5]:.2f} kn]"
    ]

    # Legend with heading
    ax.legend(handles=[
        patches.Patch(color=COLORWHEEL[0],label=q_labels_h[0]),
        patches.Patch(color=COLORWHEEL[0],label=q_labels_h[1]),
        patches.Patch(color=COLORWHEEL[0],label=q_labels_h[2])
    ])
    
    ax.set_xlabel("Difference [kn]")
    
    ax.set_title(
        "Difference between reported and calculated speed [kn]",fontsize=10
    )
    
    plt.tight_layout()
    plt.savefig(f"{PLOT_FOLDER}/reported_vs_calculated_speed.pdf")
    
def plot_heading_and_speed_changes(sa: SearchAgent):
    """
    Plot the changes in heading and speed between
    two consecutive messages.
    
    For historical reasons, this function does two
    things at once.
    Feel free to split this function into two separate.
    """
    
    def _heading_change(h1,h2):
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
    f, ax = plt.subplots(1,1,figsize=(8,5))

    heading_changes = []
    speed_changes = []
                
    for msg1, msg2 in iter_msg_raw(sa):
        _chheading = _heading_change(
            msg1.COG,
            msg2.COG
        )
        _chspeed = abs(msg2.SOG - msg1.SOG)
        heading_changes.append(_chheading)
        speed_changes.append(_chspeed)

    # Quantiles of heading changes
    h_qs = np.quantile(
        heading_changes,
        [0.005,0.995,0.025,0.975,0.05,0.95]
    )
    q_labels_h = [
        f"99% within [{h_qs[0]:.2f}°,{h_qs[1]:.2f}°]",
        f"95% within [{h_qs[2]:.2f}°,{h_qs[3]:.2f}°]",
        f"90% within [{h_qs[4]:.2f}°,{h_qs[5]:.2f}°]"
    ]
    # Heading Quantiles as vertical lines
    hl11 = ax.axvline(h_qs[0],color=COLORWHEEL[0],label=q_labels_h[0],ls="--")
    hl21 = ax.axvline(h_qs[2],color=COLORWHEEL[0],label=q_labels_h[1],ls="-.")
    hl31 = ax.axvline(h_qs[4],color=COLORWHEEL[0],label=q_labels_h[2],ls=":")
    
    # Histogram of heading changes
    ax.hist(
        heading_changes,
        bins=100,
        density=True,
        alpha=0.8,
        color=COLORWHEEL[0]
    )
    
    # Quantiles of speed changes
    s_qs = np.quantile(
        speed_changes,
        [0.99,0.95,0.90]
    )
    
    q_labels_s = [
        f"99% smaller than {s_qs[0]:.2f} kn",
        f"95% smaller than {s_qs[1]:.2f} kn",
        f"90% smaller than {s_qs[2]:.2f} kn"
    ]
    
    
    # Legend with heading
    ax.legend(handles=[hl11,hl21,hl31])
    ax.set_xlabel("Change in heading [°]")
    ax.set_ylabel("Density")
    ax.set_title(
        "Change in heading between two consecutive messages",fontsize=10
    )
    plt.tight_layout()
    plt.savefig(f"{PLOT_FOLDER}/heading_changes.pdf")
    plt.close()
    
    # Plot speed changes
    f, ax = plt.subplots(1,1,figsize=(8,5))

    # Speed Quantiles as vertical lines
    sl1 = ax.axvline(s_qs[0],color=COLORWHEEL[0],label=q_labels_s[0],ls="--")
    sl2 = ax.axvline(s_qs[1],color=COLORWHEEL[0],label=q_labels_s[1],ls="-.")
    sl3 = ax.axvline(s_qs[2],color=COLORWHEEL[0],label=q_labels_s[2],ls=":")
    
    # Histogram of speed changes
    ax.hist(
        speed_changes,
        bins=200,
        density=True,
        alpha=0.8,
        color=COLORWHEEL[0]
    )
    
    ax.legend(handles=[sl1,sl2,sl3])
    ax.set_xlabel("Absolute change in speed [knots]")
    ax.set_ylabel("Density")
    ax.set_title(
        "Change in speed between two consecutive messages",fontsize=10
    )
    ax.set_xlim(-0.2,4)
    
    plt.tight_layout()
    plt.savefig(f"{PLOT_FOLDER}/speed_changes.pdf")
    
