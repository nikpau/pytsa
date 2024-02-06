"""
Module for visualizing both the ECDFs of the split point metrics,
and the extracted trajectories.

This init file handles the creation of the `plots`
folder, where the plots are saved, houses
helper functions for the plotting functions
and defines the default plot aesthetics.
"""
from typing import Optional
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import colorsys

# Plotting folder -------------------------------------------
def _create_default_plot_folder() -> Path:
    """
    Creates the default plot folder.
    """
    path = Path("plots")
    if not path.exists():
        path.mkdir()
    return path

def register_plot_dir(path: Optional[str]) -> None:
    """
    Change the default plot folder.
    """
    global PLOT_FOLDER
    path = Path(path)
    if not path.exists():
        path.mkdir()
    PLOT_FOLDER = path

# Export the default plot folder
PLOT_FOLDER = _create_default_plot_folder()

# Plotting helpers ------------------------------------------
cc = mpl.colors.ColorConverter.to_rgb

def scale_lightness(rgb, scale_l):
    """
    Scale the lightness of an rgb color.
    """
    # Convert rgb to hls (hue, lightness, saturation)
    h, l, s = colorsys.rgb_to_hls(*rgb)
    # Manipulate h, l, s values and return as rgb
    return colorsys.hls_to_rgb(h, min(1, l * scale_l), s = s)

# Colorwheels-----------------------------------------------
COLORWHEEL = [
    "#264653",
    "#2a9d8f",
    "#e9c46a",
    "#f4a261",
    "#e76f51",
    "#E45C3A",
    "#732626"
]
COLORWHEEL_DARK = [scale_lightness(cc(c), 0.6) for c in COLORWHEEL]

COLORWHEEL2 = [
    "#386641", 
    "#6a994e", 
    "#a7c957", 
    "#f2e8cf", 
    "#bc4749"
]
COLORWHEEL2_DARK = [scale_lightness(cc(c), 0.6) for c in COLORWHEEL2]

COLORWHEEL3 = [
    "#335c67",
    "#fff3b0",
    "#e09f3e",
    "#9e2a2b",
    "#540b0e"
]
COLORWHEEL_MAP = [
    "#0466c8",
    "#0353a4",
    "#023e7d",
    "#002855",
    "#001845",
    "#001233",
    "#33415c",
    "#5c677d",
    "#7d8597",
    "#979dac"
]

# Matplotlib settings ---------------------------------------
plt.style.use('bmh')
plt.rcParams["font.family"] = "monospace"