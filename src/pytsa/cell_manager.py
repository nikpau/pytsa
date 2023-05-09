
from .structs import (
    LatLonBoundingBox,UTMBoundingBox, 
    LatLonCell,UTMCell, OUTOFBOUNDS,
    Position,AdjacentCells, UTMPosition,
)
from typing import Tuple, List
import math
from .logger import logger
import numpy as np 
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path

class _NOT_DETERMINED_TYPE:
    pass
NOTDETERMINED = _NOT_DETERMINED_TYPE()

# Path to geojson files containing the geometry
GEOPATH = Path("data/geometry/combined/")

class LatLonCellManager:
    """
    Cell Manager object implementing several utility
    functions to generate and manage cells generated from
    a geographical frame:
    
    Args:
        frame (BoundingBox): A geographical area which
                                is to be converted into a 
                                grid of `n_cells` cells.
        n_cells (int):      Number of cells, the provided 
                            frame is divided into.
    """
    def __init__(self, frame: LatLonBoundingBox, n_cells: int) -> None:
        
        # Bounding box of the entire frame
        self.frame = frame
        
        # Number of cells, the frame 
        # shall be divided into
        self.n_cells = n_cells
        
        # Dict holding all cells in the
        # current frame
        self.cells: dict[int,LatLonCell] = {}

        # Get the number of rows and colums.
        # Since the grid is always square,
        # they are the same
        self.nrows = self.ncols = self._side_length()
        
        self._setup()
        
    def _setup(self) -> None:
        """
        Create individual Cells from
        provided frame and cell number and assigns
        ascending indices. The first cell
        starts in the North-West:
        
        N
        ^
        |
        |----> E
        
        Example of a 5x5 Grid:
        ------------------------------------
        |  0   |  1   |  2   |  3   |  4   |
        |      |      |      |      |      |
        ------------------------------------
        |  5   |  6   |  7   |  8   |  9   |
        |      |      |      |      |      |            
        ------------------------------------       
        | ...  | ...  | ...  | ...  | ...  |             
                
        """
        latbp, lonbp = self._breakpoints()
        cidx = 0
        latbp= sorted(latbp)[::-1] # Descending Latitudes
        lonbp = sorted(lonbp) # Ascending Longitudes
        for lat_max, lat_min in zip(latbp,latbp[1:]):
            for lon_min, lon_max in zip(lonbp,lonbp[1:]):
                self.cells[cidx] = LatLonCell(
                    LATMIN=lat_min,LATMAX=lat_max,
                    LONMIN=lon_min,LONMAX=lon_max,
                    index=cidx
                )
                cidx += 1

    def _side_length(self) -> int:
        """
        Get the side length (number of cells per side) for the
        total number of cells specified during instantiation.

        The number of cells to be constructed `n_cells` must be a 
        square number. If it is not the next closest square number
        will be used instead 
        """        
        root = math.sqrt(self.n_cells)
        if not root%1==0:
            root = round(root)
            logger.info(
                "Provided argument to `n_cells` "
                "is no square number. Rounding to next square."
                f"\nProvided: {self.n_cells}\nUsing: {root**2}"
            )
        return int(root)

            
    def _breakpoints(self) -> Tuple[List[float],List[float]]:
        """
        Create breakpoints of latitude and longitude
        coordinates from initial ``frame`` resembling cell borders.
        """
        f = self.frame
        nrow = self.nrows
        ncol = self.ncols

        # Set limiters for cells 
        lat_extent = f.LATMAX-f.LATMIN
        lon_extent = f.LONMAX-f.LONMIN
        # Split into equal sized cells
        csplit = [f.LONMIN + i*(lon_extent/ncol) for i in range(ncol)]
        rsplit = [f.LATMIN + i*(lat_extent/nrow) for i in range(nrow)]
        csplit.append(f.LONMAX) # Add endpoints
        rsplit.append(f.LATMAX) # Add endpoints
        return rsplit, csplit
            
    def inside_frame(self, pos: Position) -> bool:
        """Boolean if the provided position
        is inside the bounds of the cell 
        manager's frame"""
        lat, lon = pos
        return (
            lat > self.frame.LATMIN and lat < self.frame.LATMAX and
            lon > self.frame.LONMIN and lon < self.frame.LONMAX
        )
        
    @staticmethod
    def inside_cell(pos: Position, cell: LatLonCell) -> bool:
        """Boolean if the provided position
        is inside the bounds of the provided cell"""
        lat, lon = pos
        return (
            lat > cell.LATMIN and lat < cell.LATMAX and
            lon > cell.LONMIN and lon < cell.LONMAX
        )
                
    def get_cell_by_index(self, index: int) -> LatLonCell:
        """Get cell by index"""
        assert not index > len(self.cells) 
        return self.cells[index]
    
    def get_cell_by_position(self, pos: Position) -> LatLonCell:
        """Get Cell by Lat-Lon Position"""
        assert self.inside_frame(pos)
        for cell in self.cells.values():
            if self.inside_cell(pos,cell):
                return cell
        raise RuntimeError(
            "Could not locate cell via position.\n"
            f"Cell: {cell!r}\nPosition: {pos!r}"
            )

    def adjacents(self, cell: LatLonCell = None, index: int = None) -> AdjacentCells:
        """
        Return an "AdjacentCells" instance containig all adjacent 
        cells named by cardinal directions for the provided cell or its index.
        If an adjacent cell in any direction is out of bounds
        an "OUTOFBOUNDS" instance will be returned instead of a cell.
        """
        if cell is not None and index is not None:
            raise RuntimeError(
                "Please provide either a cell or an index to a cell, not both."
            )
        if cell is not None:
            index = cell.index

        grid = np.arange(self.nrows**2).reshape(self.ncols,self.nrows)

        # Get row and column of the input cell
        row, col = index//self.nrows, index%self.ncols
        
        # Shorten to avoid clutter
        _cbi = self.get_cell_by_index

        # Calc adjacents for cardinal directions.
        # (Surely not the most efficient, but it works).
        n = _cbi(grid[row-1,col]) if row-1 >= 0 else OUTOFBOUNDS
        s = _cbi(grid[row+1,col]) if row+1 < self.nrows else OUTOFBOUNDS
        e = _cbi(grid[row,col+1]) if col+1 < self.ncols else OUTOFBOUNDS
        w = _cbi(grid[row,col-1]) if col-1 >= 0 else OUTOFBOUNDS

        # Composite directions
        ne = _cbi(grid[row-1,col+1]) if (row-1 >= 0 and col+1 <= self.nrows) else OUTOFBOUNDS
        se = _cbi(grid[row+1,col+1]) if (row+1 < self.nrows and col+1 < self.nrows) else OUTOFBOUNDS
        sw = _cbi(grid[row+1,col-1]) if (row+1 < self.nrows and col-1 >= 0) else OUTOFBOUNDS
        nw = _cbi(grid[row-1,col-1]) if (row-1 >= 0 and col-1 >= 0) else OUTOFBOUNDS

        return AdjacentCells(
            N=n,NE=ne,E=e,SE=se,
            S=s,SW=sw,W=w,NW=nw
        )

    def get_subcell(self,pos: Position, cell: LatLonCell) -> int:
        """
        Each cell will be divided into four subcells
        by bisecting the cell in the middle along each
        axis:

        Example of a singe cell with its four subcells

        N
        ^
        |---------|---------|
        |    1    |    2    |
        |         |         |
        ---------------------
        |    3    |    4    |
        |         |         |
        |---------|---------|-> E

        We will later use these subcells to determine 
        which adjacent cell to pre-buffer.
        """
        assert self.inside_cell(pos,cell)
        lonext = cell.LONMAX-cell.LONMIN # Longitudinal extent of cell
        lonhalf = cell.LONMAX - lonext/2
        latext = cell.LATMAX-cell.LATMIN # Lateral extent of cell
        lathlalf = cell.LATMAX - latext/2
        if pos.lon < lonhalf and pos.lat > lathlalf:
            return 1
        elif pos.lon > lonhalf and pos.lat > lathlalf:
            return 2
        elif pos.lon < lonhalf and pos.lat < lathlalf:
            return 3
        elif pos.lon > lonhalf and pos.lat < lathlalf:
            return 4
        else: return NOTDETERMINED # Vessel is exacty in the middle of the cell
        
    def plot_grid(self,*, f: plt.Figure = None, ax: plt.Axes = None) -> None:
        # Load north sea geometry
        if f is None and ax is None:
            f, ax = plt.subplots()
        files = GEOPATH.glob("*.geojson")
        for file in files:
            f = gpd.read_file(file)
            f.plot(ax=ax,color="#283618",markersize=0.5,marker = ".")
            
        lats = [c.LATMIN for c in self.cells.values()]
        lons = [c.LONMIN for c in self.cells.values()]
        lats.append(self.frame.LATMAX)
        lons.append(self.frame.LONMAX)

        # Place grid according to global frame
        ax.hlines(
            lats,self.frame.LONMIN,
            self.frame.LONMAX,colors="#6d6875"
        )
        ax.vlines(
            lons,self.frame.LATMIN,
            self.frame.LATMAX,colors="#6d6875"
        )
        if f is None and ax is None:
            plt.show()

class UTMCellManager(LatLonCellManager):
    """
    Reimplementation of the CellManager class
    for UTM coordinates.
    """
    def __init__(self, frame: UTMBoundingBox, n_cells: int) -> None:
        
        # Bounding box of the entire frame
        self.frame = frame
        
        # Number of cells, the frame 
        # shall be divided into
        self.n_cells = n_cells
        
        # Dict holding all cells in the
        # current frame
        self.cells: dict[int,UTMCell] = {}

        # Get the number of rows and colums.
        # Since the grid is always square,
        # they are the same
        self.nrows = self.ncols = self._side_length()
        
        self._setup()
        
    def _setup(self) -> None:
        """
        Create individual Cells from
        provided frame and cell number and assigns
        ascending indices. The first cell
        starts in the North-West:
        
        N
        ^
        |
        |----> E
        
        Example of a 5x5 Grid:
        ------------------------------------
        |  0   |  1   |  2   |  3   |  4   |
        |      |      |      |      |      |
        ------------------------------------
        |  5   |  6   |  7   |  8   |  9   |
        |      |      |      |      |      |            
        ------------------------------------       
        | ...  | ...  | ...  | ...  | ...  |             
                
        """
        eastbp, northbp = self._breakpoints()
        cidx = 0
        eastbp= sorted(eastbp)[::-1] # Descending Eastings
        northbp = sorted(northbp) # Ascending Northings
        for east_max, east_min in zip(eastbp,eastbp[1:]):
            for north_min, north_max in zip(northbp,northbp[1:]):
                self.cells[cidx] = UTMCell(
                    min_easting=east_min,max_easting=east_max,
                    min_northing=north_min,max_northing=north_max,
                    zone_letter=self.frame.zone_letter,
                    zone_number=self.frame.zone_number,
                    index=cidx
                )
                cidx += 1

    def _breakpoints(self) -> Tuple[List[float],List[float]]:
        """
        Create breakpoints of latitude and longitude
        coordinates from initial ``frame`` resembling cell borders.
        """
        f = self.frame
        nrow = self.nrows
        ncol = self.ncols

        # Set limiters for cells 
        east_extent = f.max_easting-f.min_easting
        north_extent = f.max_northing-f.min_northing
        # Split into equal sized cells
        csplit = [f.min_northing + i*(north_extent/ncol) for i in range(ncol)]
        rsplit = [f.min_easting + i*(east_extent/nrow) for i in range(nrow)]
        csplit.append(f.max_northing) # Add endpoints
        rsplit.append(f.max_easting) # Add endpoints
        return rsplit, csplit
            
    def inside_frame(self, pos: UTMPosition) -> bool:
        """Boolean if the provided position
        is inside the bounds of the cell 
        manager's frame"""
        northing, easting = pos
        return (
            easting > self.frame.min_easting and easting < self.frame.max_easting and
            northing > self.frame.min_northing and northing < self.frame.max_northing
        )
        
    @staticmethod
    def inside_cell(pos: UTMPosition, cell: UTMCell) -> bool:
        """Boolean if the provided position
        is inside the bounds of the provided cell"""
        northing, easting = pos
        return (
            easting > cell.min_easting and easting < cell.max_easting and
            northing > cell.min_northing and northing < cell.max_northing
        )
    
    def get_cell_by_position(self, pos: Position) -> LatLonCell:
        """Get Cell by Lat-Lon Position"""
        assert self.inside_frame(pos)
        for cell in self.cells.values():
            if self.inside_cell(pos,cell):
                return cell
        raise RuntimeError(
            "Could not locate cell via position.\n"
            f"Cell: {cell!r}\nPosition: {pos!r}"
            )
    
    def get_subcell(self,pos: UTMPosition, cell: UTMCell) -> int:
        """
        Each cell will be divided into four subcells
        by bisecting the cell in the middle along each
        axis:

        Example of a singe cell with its four subcells

        N
        ^
        |---------|---------|
        |    1    |    2    |
        |         |         |
        ---------------------
        |    3    |    4    |
        |         |         |
        |---------|---------|-> E

        We will later use these subcells to determine 
        which adjacent cell to pre-buffer.
        """
        assert self.inside_cell(pos,cell)
        northext = cell.max_northing-cell.min_northing # Longitudinal extent of cell
        northhalf = cell.max_northing - northext/2
        eastext = cell.max_easting-cell.min_easting # Lateral extent of cell
        easthalf = cell.max_easting - eastext/2
        if pos.northing < northhalf and pos.easting > easthalf:
            return 1
        elif pos.northing > northhalf and pos.easting > easthalf:
            return 2
        elif pos.northing < northhalf and pos.easting < easthalf:
            return 3
        elif pos.northing > northhalf and pos.easting < easthalf:
            return 4
        else: return NOTDETERMINED # Vessel is exacty in the middle of the cell
        
    def plot_grid(self,*, f: plt.Figure = None, ax: plt.Axes = None) -> None:
        # Load north sea geometry
        if f is None and ax is None:
            f, ax = plt.subplots()
        files = GEOPATH.glob("*.geojson")
        for file in files:
            f = gpd.read_file(file)
            f.plot(ax=ax,color="#283618",markersize=0.5,marker = ".")
            
        lats = [c.min_easting for c in self.cells.values()]
        lons = [c.min_northing for c in self.cells.values()]
        lats.append(self.frame.max_easting)
        lons.append(self.frame.max_northing)

        # Place grid according to global frame
        ax.hlines(
            lats,self.frame.min_northing,
            self.frame.max_northing,colors="#6d6875"
        )
        ax.vlines(
            lons,self.frame.min_easting,
            self.frame.max_easting,colors="#6d6875"
        )
        if f is None and ax is None:
            plt.show()

        