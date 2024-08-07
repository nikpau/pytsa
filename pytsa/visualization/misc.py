"""
Miscellaneous visualization functions which
have been used throughout the paper.

Some functions, like the heatmap, are not
used in the paper, but are included here
anyways, as they might be useful for future
work.
"""
import requests
import utm
import geopandas as gpd
import numpy as np

from glob import glob
from pathlib import Path
from pytsa.structs import Track
from scipy.spatial import ConvexHull
from osm2geojson import json2geojson
from pytsa.trajectories import inspect
from pytsa import BoundingBox, TargetShip
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from . import plt, PLOT_FOLDER, mpl, COLORWHEEL_MAP
from ..data.geometry import __path__ as geometry_path
from ..tsea.targetship import Targets

def _check_duplicate_file_name(filename: Path) -> Path:
    """
    Check if a file with the same name
    already exists in the `PLOT_FOLDER`.
    
    If it does, append a number to the filename
    and return the new filename.
    
    Parameters
    ----------
    filename : str
        The filename to be checked.
        
    Returns
    -------
    str
        The new filename.
    """
    if not (PLOT_FOLDER / filename).exists():
        return PLOT_FOLDER / filename
    i = 1
    while (PLOT_FOLDER / f"{filename}_{i}").exists():
        i += 1
    return PLOT_FOLDER / f"{filename}_{i}"

def _cvh_area(track: Track) -> float:
    """
    Calculate the area of the convex hull
    of a track.
    """
    res = utm.from_latlon(
        np.array([p.lat for p in track]),
        np.array([p.lon for p in track])
    )
    points = np.array([res[0],res[1]]).T
    return ConvexHull(points).area

def plot_coastline(extent: BoundingBox, 
                   ax: plt.Axes = None,
                   save_plot: bool = False,
                   return_figure: bool = False,
                   detail_lvl: int = 4) -> plt.Figure | None:
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
    detail_lvl : int, optional
        The level of detail of the road network:
        1: motorways
        2: motorways and primary roads
        3: motorways, primary and secondary roads
        4: motorways, primary, secondary and tertiary roads
        
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
        gdf.plot(ax=ax, color="#00657d", alpha=0.8,linewidth=2)
    
        queries = [
        get_overpass_roads_motorway(extent),
        get_overpass_roads_primary(extent),
        get_overpass_roads_secondary(extent),
        get_overpass_roads_tertiary(extent),
    ]
        
    # Additional query for overpass API
    level = 0
    for query, color, width in zip(
        queries, 
        ["#DB3123","#dba119","#bfa246","#999999"],
        [1,1,0.5,0.3]
        
        ):
        url = f"https://overpass-api.de/api/interpreter?data={query}"
        r = requests.get(url)
        data = r.json()
        data = json2geojson(data)
        gdf = gpd.GeoDataFrame.from_features(data["features"])
        gdf.plot(ax=ax, color=color, linewidth=width)
        level += 1
        if level == detail_lvl:
            break
        
    # Crop the plot to the extent
    ax.set_xlim(extent.LONMIN, extent.LONMAX)
    ax.set_ylim(extent.LATMIN, extent.LATMAX)
    
    if save_plot:
        plt.savefig(_check_duplicate_file_name("coastline.png"), dpi=300)
    return None if not return_figure else fig

def binned_heatmap(targets: Targets, 
                   bb: BoundingBox,
                   npixels: int,
                   title: str = None) -> None:
    """
    Creates a 2D heatmap of messages 
    in the search area with resolution
    `npixels x npixels`.
    
    The targets do not need to come exactly
    from the same area as the bounding box,
    as the heatmap will be cropped to the
    bounding box.
    
    Parameters
    ----------
    targets : Targets
        The target ships to be analyzed.
    bb : BoundingBox
        Spatial extent to be plotted on the heatmap.
    npixels : int
        The number of pixels per dimension.
    title : str
        The title of the heatmap.
        
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
                if not bb.contains(msg):
                    continue
                # Find the closest pixel
                i = np.argmin(np.abs(x - msg.lon))
                j = np.argmin(np.abs(y - msg.lat))
                counts[j,i] += 1
    
    # Plot the heatmap
    fig, ax = plt.subplots(figsize=(10,10))
    ax: plt.Axes

    # Add coastline redered to an image
    # and plot it on top of the heatmap
    plot_coastline(bb,ax=ax,detail_lvl=1)
    
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
    pcm = ax.pcolormesh(xx,yy,counts,cmap=cmap)
    
    
    # Small inset Colorbar
    cbaxes = inset_axes(ax, width="40%", height="2%", loc=4, borderpad = 2)
    cbaxes.grid(False)
    cbar = fig.colorbar(pcm,cax=cbaxes, orientation="horizontal")
    cbar.set_label(r"Route density ($n_{msg}$)",color="black")

    newticks = np.linspace(1,counts.max(),3)
    cbar.set_ticks(
        ticks = newticks,
        labels = [f"{(np.exp(np.exp(t))-np.exp(1))/np.exp(1):.0f}" for t in newticks]
    )
    
    cbar.ax.xaxis.set_ticks_position("top")
    cbar.ax.xaxis.set_tick_params(color="black")
    plt.setp(plt.getp(cbar.ax.axes, 'xticklabels'), color="black") 
    
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    if title:
        ax.set_title(title)
    else:
        ax.set_title("Heatmap of messages")
    
    
    plt.tight_layout()
    plt.savefig(_check_duplicate_file_name("heatmap.png"),dpi=300)
    
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
    plt.savefig(_check_duplicate_file_name("trajectories_map.png"),dpi=600)
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
    plt.savefig(_check_duplicate_file_name("pixelmap_avg_smoothness.pdf"))
    
def cvh_range_comparison(ships: dict[int,TargetShip]) -> None:
    """
    Figure 13 in the paper.
    
    Walks through the trajectories of the ships
    in `ships` and plots 100 trajectories for each
    convex hull range.
    
    An inset is added to each plot, showing the
    center region of the trajectory, as for some
    ranges, the trajectories are too dense to
    distinguish individual trajectories.
    """
    areas = np.array(
        [0,1000,5000,10000,20000,50000,100_000,200_000,300_000]
    )
    
    for ship in ships.values():
        # We need to calculate the convex hull area
        # for each track in the ship before
        # de-meaning the trajectories as the
        # UTM transform will not work on flipping
        # coorinate signs.
        ship.cvhareas = []
        for track in ship.tracks:
            try:
                ship.cvhareas.append(_cvh_area(track))
            except:
                ship.cvhareas.append(-1)
            # Center trajectory
            # Get mean of lat and lon
            latmean = sum(p.lat for p in track) / len(track)
            lonmean = sum(p.lon for p in track) / len(track)
            # Subtract mean from all positions
            for msg in track:
                msg.lat -= latmean
                msg.lon -= lonmean
    
    ships: list[TargetShip] = list(ships.values())
            
    # Plot trajectories for different sds
    ncols = 4
    div,mod = divmod(len(areas)-1,ncols)
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
                for i, track in enumerate(ship.tracks):
                    lo = [p.lon for p in track]
                    la = [p.lat for p in track]
                    hullarea = ship.cvhareas[i]
                    if hullarea == -1:
                        continue
                    # Check if within range
                    if areas[idx] <= hullarea <= areas[idx+1]:
                        if trnr == 50:
                            break
                        trnr += 1
                        lons.append(lo)
                        lats.append(la)
                        
            if row != 1:
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
            
                if row != 1:
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
                "$A^{C}\in$"
                f"[{areas[row*ncols+col]},{areas[row*ncols+col+1]}]$m^2$",
                fontsize=16
            )

            # Set limits
            axs[row,col].set_xlim(-0.25,0.65)
            axs[row,col].set_ylim(-0.25,0.65)
            
            idx += 1
            
    plt.tight_layout()
    plt.savefig(_check_duplicate_file_name("cvhjitter.pdf"))
    plt.close()    

# OSM Overpass API cals for primary, secondary and tertiary roads
def get_overpass_roads_motorway(bb: BoundingBox) -> str:
    bbstr = f"{bb.LATMIN},{bb.LONMIN},{bb.LATMAX},{bb.LONMAX}"
    return f"""
        [out:json][timeout:100];
        // fetch only larger roads and their relations within the bounding box
        (
        way["highway"~"motorway|trunk"]({bbstr});
        relation["highway"~"motorway|trunk"]({bbstr});
        );
        out geom;
        """

def get_overpass_roads_primary(bb: BoundingBox) -> str:
    bbstr = f"{bb.LATMIN},{bb.LONMIN},{bb.LATMAX},{bb.LONMAX}"
    return f"""
        [out:json][timeout:100];
        // fetch only larger roads and their relations within the bounding box
        (
        way["highway"~"primary"]({bbstr});
        relation["highway"~"primary"]({bbstr});
        );
        out geom;
        """

def get_overpass_roads_secondary(bb: BoundingBox) -> str:
    bbstr = f"{bb.LATMIN},{bb.LONMIN},{bb.LATMAX},{bb.LONMAX}"
    return f"""
        [out:json][timeout:100];
        // fetch only larger roads and their relations within the bounding box
        (
        way["highway"~"secondary"]({bbstr});
        relation["highway"~"secondary"]({bbstr});
        );
        out geom;
        """

def get_overpass_roads_tertiary(bb: BoundingBox) -> str:
    bbstr = f"{bb.LATMIN},{bb.LONMIN},{bb.LATMAX},{bb.LONMAX}"
    return f"""
        [out:json][timeout:100];
        // fetch only larger roads and their relations within the bounding box
        (
        way["highway"~"tertiary"]({bbstr});
        relation["highway"~"tertiary"]({bbstr});
        );
        out geom;
        """

def get_overpass_roads_all(bb: BoundingBox) -> str:
    bbstr = f"{bb.LATMIN},{bb.LONMIN},{bb.LATMAX},{bb.LONMAX}"
    return f"""
        [out:json][timeout:100];
        // fetch only larger roads and their relations within the bounding box
        (
        way["highway"~"motorway|trunk|primary|secondary|tertiary|residential"]({bbstr});
        relation["highway"~"motorway|trunk|primary|secondary|tertiary|residential"]({bbstr});
        );
        out geom;
        """