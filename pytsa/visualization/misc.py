"""
Miscellaneous visualization functions which
have been used throughout the paper.

Some functions, like the heatmap, are not
used in the paper, but are included here
anyways, as they might be useful for future
work.
"""
import pickle
from pytsa import BoundingBox, TargetShip
from pytsa.trajectories import inspect
from pytsa.tsea.search_agent import Targets
from glob import glob
import geopandas as gpd
import numpy as np

from . import plt, PLOT_FOLDER, mpl, COLORWHEEL_MAP
from ..data.geometry import __path__ as geometry_path

def plot_coastline(extent: BoundingBox, 
                   ax: plt.Axes = None,
                   save_plot: bool = False,
                   return_figure: bool = False) -> plt.Figure | None:
    """
    Plots the coastline of the search area.
    
    Currently only supports the coastline of the
    North Sea and the Baltic Sea.
    
    Additional coastlines can be added by adding
    the corresponding .json file to the `data/geometry`
    folder.
    
    Parameters
    ----------
    extent : BoundingBox
        The extent of the search area.
    ax : plt.Axes, optional
        The axis on which to plot the coastline.
        If None, a new figure will be created.
    save_plot : bool, optional
        Whether to save the plot to the `PLOT_FOLDER`.
    return_figure : bool, optional
        Whether to return the figure.
        
    Returns
    -------
    plt.Figure | None
        The figure if `return_figure` is True,
        None otherwise.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(15,15))
    coasts = glob(f"{geometry_path[0]}/*.json")
    for coast in coasts:
        gdf = gpd.read_file(coast)
        gdf.crs = 'epsg:3395' # Mercator projection
        gdf.plot(ax=ax, color="#007d57", alpha=0.8,linewidth=2)
        
    # Crop the plot to the extent
    ax.set_xlim(extent.LONMIN, extent.LONMAX)
    ax.set_ylim(extent.LATMIN, extent.LATMAX)
    
    if save_plot:
        plt.savefig(f"{PLOT_FOLDER}/coastline.png", dpi=300)
    return None if not return_figure else fig

def binned_heatmap(targets: Targets, 
                   bb: BoundingBox,
                   npixels: int) -> None:
    """
    Creates a 2D heatmap of messages 
    in the search area with resolution
    `npixels x npixels`.
    
    Parameters
    ----------
    targets : Targets
        The targets to be plotted.
    bb : BoundingBox
        The search area.
    npixels : int
        The number of pixels per dimension.
        
    """
    # Create a grid of pixels
    x = np.linspace(bb.LONMIN,bb.LONMAX,npixels)
    y = np.linspace(bb.LATMIN,bb.LATMAX,npixels)
    xx, yy = np.meshgrid(x,y)
    
    # Count the number of messages in each pixel
    counts = np.zeros((npixels,npixels))
    for ship in targets.values():
        for track in ship.tracks:
            for msg in track:
                # Find the closest pixel
                i = np.argmin(np.abs(x - msg.lon))
                j = np.argmin(np.abs(y - msg.lat))
                counts[j,i] += 1
    
    # Plot the heatmap
    fig, ax = plt.subplots(figsize=(10,10))
    ax: plt.Axes

    # Add coastline redered to an image
    # and plot it on top of the heatmap
    plot_coastline(bb,ax=ax)
    
    # Mask the pixels with no messages
    counts = np.ma.masked_where(counts == 0,counts)

    # Log transform of counts to avoid
    # spots with many messages to dominate
    # the plot
    counts = np.vectorize(
        lambda x: np.log(np.log(x+1)+1))(counts)
    
    cmap = mpl.colormaps["Reds"]
    cmap.set_bad(alpha=0)
    ax.grid(False)
    ax.pcolormesh(xx,yy,counts,cmap=cmap)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Heatmap of messages")
    
    
    plt.tight_layout()
    plt.savefig(f"{PLOT_FOLDER}/headmap.png",dpi=300)
    
def plot_trajectories_on_map(ships: dict[int,TargetShip], 
                             extent: BoundingBox) -> None:
    """
    Plot the trajectories of the ships in `ships`
    on a map. The map is cropped to the extent
    of the `BoundingBox`.
    
    Coastal geometry is only available for the
    North Sea and the Baltic Sea. (See `plot_coastline`).
    """
    fig, ax = plt.subplots(figsize=(10,10))
    idx = 0
    plot_coastline(extent=extent,ax=ax)
    
    for ship in ships.values():
        for track in ship.tracks:
            idx += 1
            ax.plot(
                [p.lon for p in track],
                [p.lat for p in track],
                alpha=0.5, linewidth=0.3, marker = "x", markersize = 0.5,
                color = COLORWHEEL_MAP[idx % len(COLORWHEEL_MAP)]
            )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.title(f"Extracted Trajectories", size = 14)
    plt.tight_layout()
    plt.savefig(f"aisstats/out/trajectories_map.png",dpi=600)
    plt.close()

def pixelmap_average_smoothness(ships: dict[int,TargetShip],
                                sds: np.ndarray = np.linspace(0,0.1,101),
                                minlens: np.ndarray = np.linspace(0,100,101)) -> None:
    """
    Figure 14 in the paper. 
    
    Calulates the average smoothness of the
    each track in `ships` and saves the results
    to a file.
    
    The average smoothness is calculated for
    each track individually and then assigned
    to its corrensponding bin in the `sds` and
    `minlens` arrays.
    
    Parameters
    ----------
    ships : dict[int,TargetShip]
        The ships to be analyzed.
    sds : np.ndarray
        The $\sigma_{\mathrm{ssd}}$ bins 
        to be analyzed.
    minlens : np.ndarray
        The minimum length bins to be analyzed.
    """
    def online_average(avg, new, n):
        return avg + (new - avg) / n

    smthness = np.full((len(minlens),len(sds)),np.nan)
    
    # count for running average
    counts = np.full((len(minlens),len(sds)),0)
    for ship in ships.values():
        for track in ship.tracks:
            length = len(track)
            if length < 3:
                print("Track too short")
                continue
            sd = np.std([p.lon for p in track]) + np.std([p.lat for p in track])
            s = inspect.average_smoothness(track)
            
            # Find the index of the minimum length
            minlen_idx = np.argmin(np.abs(minlens-length))
            sd_idx = np.argmin(np.abs(sds-sd))
            
            # If there is no value yet, set it
            if counts[minlen_idx,sd_idx] == 0:
                smthness[minlen_idx,sd_idx] = s
                counts[minlen_idx,sd_idx] = 1
                continue
            
            # Update the running average
            counts[minlen_idx,sd_idx] += 1
            smthness[minlen_idx,sd_idx] = online_average(
                smthness[minlen_idx,sd_idx], 
                s,
                counts[minlen_idx,sd_idx]
            )
    
    # Plot the results -----------------------------------
    fig, ax = plt.subplots(figsize=(6,6))

    CMAP_NAME = "gist_ncar_r"

    # White background
    ax.set_facecolor("white")

    ax.pcolormesh(
        smthness**2,
        cmap=CMAP_NAME,
        antialiased=True,
        shading="flat",
        vmin=np.nanmin(smthness), 
        vmax=np.nanmax(smthness)
    )
    ax.set_aspect("equal")

    # Horizontal line at y=3
    ax.axhline(3, color="black", linewidth=1)

    # Grid off
    ax.grid(False)

    # Add colorbar
    cbat = fig.colorbar(
        mpl.cm.ScalarMappable(
            norm=mpl.colors.Normalize(
                vmin=np.nanmin(smthness), 
                vmax=np.nanmax(smthness)
            ),
            cmap=CMAP_NAME
        ),
        ax=ax,
        fraction=0.046,
        pad=0.04,
        shrink=0.8,
    )

    # Label colorbar
    cbat.set_label(r"$\bar{s}(\mathcal{T}^>)$",fontsize=14)

    # Set axis ticks
    ax.set_xticks(np.linspace(0,len(sds),11))

    # Y ticks (We add 3 to the end of the array to get a tick at y=3)
    yt1 = np.linspace(0,len(minlens),11)
    yt2 = 3
    yt = np.atleast_1d(np.append(yt1,yt2))
    yt.sort()
    # Add y ticks
    ax.set_yticks(yt)

    # Set axis tick labels
    ax.set_xticklabels(
        [f"{sd:.2f}" for sd in np.linspace(0,sds[-1],11)],
        fontsize=12,
        rotation=45
    )
    labelnums = np.append(np.linspace(0,minlens[-1],11),3)
    labelnums.sort()
    ax.set_yticklabels(
        [f"{minlen:.0f}" for minlen in labelnums],
        fontsize=12
    )

    # Axes labels
    ax.set_xlabel(r"$\sigma_{\mathrm{ssd}}$",fontsize=18)
    ax.set_ylabel(r"$n_{msg}$",fontsize=18)

    plt.tight_layout()
    plt.savefig(f"{PLOT_FOLDER}/pixelmap_avg_smoothness.pdf")
    
def ssd_range_comparison(ships: dict[int,TargetShip]) -> None:
    """
    Figure 12 in the paper.
    
    Walks through the trajectories of the ships
    in `ships` and plots 100 trajectories for each
    standard deviation range.
    The standard deviation range is defined by
    iterating over the `sds` array with a 
    rolling window of size 2.
    
    An inset is added to each plot, showing the
    center region of the trajectory, as for some
    ranges, the trajectories are too dense to
    distinguish individual trajectories.
    """
    
    sds = np.array([0,0.01,0.02,0.03,0.04,0.05,0.1,0.2,0.3])
    
    for ship in ships.values():
        # Center trajectory
        # Get mean of lat and lon
        for track in ship.tracks:
            latmean = sum(p.lat for p in track) / len(track)
            lonmean = sum(p.lon for p in track) / len(track)
            # Subtract mean from all positions
            for msg in track:
                msg.lat -= latmean
                msg.lon -= lonmean
    
    ships: list[TargetShip] = list(ships.values())
            
    # Plot trajectories for different sds
    ncols = 4
    div,mod = divmod(len(sds)-1,ncols)
    nrows = div + 1 if mod else div
    fig, axs = plt.subplots(nrows=nrows,ncols=ncols,figsize=(4*ncols,8))
    subpos = [0.55,0.55,0.4,0.4]
    axs: dict[int,plt.Axes] # type hint
    niter = 0
    idx = 0
    for row in range(nrows):
        for col in range(ncols):
            
            trnr = 0
            # Find trajectories whose standard deviation
            # is within the current range
            lons, lats = [], []
            
            np.random.shuffle(ships)
            for ship in ships:
                # Get standard deviation of lat and lon
                for track in ship.tracks:
                    lo = [p.lon for p in track]
                    la = [p.lat for p in track]
                    latstd = np.std(la)
                    lonstd = np.std(lo)
                    # Check if within range
                    if sds[idx] <= (latstd + lonstd) <= sds[idx+1]:
                        if trnr == 100:
                            break
                        trnr += 1
                        lons.append(lo)
                        lats.append(la)
                        

            inset = axs[row,col].inset_axes(subpos)
            
            # Add lines for x and y axes
            inset.axhline(0, color='k', linewidth=0.5)
            inset.axvline(0, color='k', linewidth=0.5)
            
            axs[row,col].axhline(0, color='k', linewidth=0.5)
            axs[row,col].axvline(0, color='k', linewidth=0.5)

            for la, lo in zip(lats,lons):
                axs[row,col].plot(
                    lo,
                    la,
                    alpha=0.5, 
                    marker = "x", 
                    markersize = 0.65, 
                    color = COLORWHEEL_MAP[niter % len(COLORWHEEL_MAP)],
                    linewidth = 0.6
                )
            
                # Plot center region in inset
                inset.plot(    
                    lo,la,
                    color = COLORWHEEL_MAP[niter % len(COLORWHEEL_MAP)],
                    linewidth = 1,
                    marker = "x",
                    markersize = 1
                )
                
                    
                # inset.set_axes_locator(ip)
                inset.set_xlim(-0.02,0.02)
                inset.set_ylim(-0.02,0.02)
                
                niter += 1
                
            axs[row,col].set_xlabel("Longitude")
            axs[row,col].set_ylabel("Latitude")
            axs[row,col].set_title(
                "$\sigma_{ssd}\in$"
                f"[{sds[row*ncols+col]:.2f},{sds[row*ncols+col+1]:.2f}]",
                fontsize=16
            )

            # Set limits
            axs[row,col].set_xlim(-0.25,0.65)
            axs[row,col].set_ylim(-0.25,0.65)
            
            idx += 1
            
    plt.tight_layout()
    
    # Pdfs are rather large, so to save space
    # png can be used instead.
    plt.savefig(f"aisstats/out/trjitter.pdf")
    plt.close()