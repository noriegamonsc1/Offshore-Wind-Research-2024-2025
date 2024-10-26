# wind_field_plotter.py

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from loguru import logger

def plot_wind_field(owi_speed, owi_dir, lat, lon, title, output_path=False, quiver_density=20):
    """
    Plot the wind field using the provided speed, direction, latitude, and longitude data.
    
    Parameters:
        owi_speed (ndarray): Array of wind speed data.
        owi_dir (ndarray): Array of wind direction data.
        lat (ndarray): Array of latitude values.
        lon (ndarray): Array of longitude values.
        title (str): Title for the plot.
        output_path (str or bool): Path to save the plot or False to display on screen.
        quiver_density (int): Density of quiver arrows.
    """
    fig = plt.figure(figsize=(15, 15))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Draw Wind Speed Data
    carto_map = ax.contourf(lon, lat, owi_speed, levels=100,
                            transform=ccrs.PlateCarree(), cmap='jet')

    # Title
    ax.set_title(title, fontsize=16, fontweight='bold', loc='left')

    # Extents with Margin
    lon_margin, lat_margin = 0.1, 0.1  # Adjust as necessary
    ax.set_xlim(np.amin(lon) - lon_margin, np.amax(lon) + lon_margin)
    ax.set_ylim(np.amin(lat) - lat_margin, np.amax(lat) + lat_margin)

    # Mask land regions and add coastlines
    land = cfeature.NaturalEarthFeature('physical', 'land', '10m',
                                        edgecolor='face', facecolor='w')
    ax.add_feature(land)
    ax.coastlines(resolution='10m', linewidth=1.5, color='black')

    # Colorbar
    cbar = plt.colorbar(carto_map, ax=ax, shrink=0.5, pad=0.05)
    cbar.set_label('SAR Wind Speed [m.s-1]', fontsize=15, labelpad=25)

    # Calculate and Draw Wind Vectors
    dx = np.cos(np.radians(owi_dir))  # u component
    dy = np.sin(np.radians(owi_dir))  # v component
    ax.quiver(lon, lat, dx, dy, angles='xy', color='black', scale=40, width=0.002, regrid_shape=quiver_density)

    # Draw gridlines and set label formatting
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.6, linestyle=':')
    gl.xformatter, gl.yformatter = LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    gl.xlabels_top, gl.ylabels_right = False, False
    gl.xlabel_style = gl.ylabel_style = {'color': 'black', 'weight': 'bold', 'size': 10}

    # Show or save the plot
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        logger.info(f"Plot saved to {output_path}.")
    else:
        plt.show()
