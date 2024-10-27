# wind_field_plotter.py

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from loguru import logger
from pathlib import Path
from offshore_wind_nj.data_loader import extract_datetime_from_filename, data_files, all_arrays
from offshore_wind_nj.config import FIGURES_DIR

def plot_wind_field(idx, save=False, quiver_density=20):
    """
    Plot the wind field using wind speed, direction, latitude, and longitude data.
    
    Parameters:
        idx (int): Index of the data file to plot in all_arrays.
        save (bool): If True, saves the plot; if False, displays on screen.
        quiver_density (int): Density of quiver arrows.
    """
    # Get the variables speed, direction, latitude, and longitude by index
    owi_speed = all_arrays[idx][0]
    owi_dir = all_arrays[idx][1]
    lat = all_arrays[idx][2]
    lon = all_arrays[idx][3]

    # Extract filename, date, start time, and end time from data file name
    filename, date, start_time, end_time = extract_datetime_from_filename(data_files[idx])
    title = f"Sentinel-1 Ocean Wind Field (OWI)\n(Start: {start_time}, End: {end_time} on {date})"

    fig = plt.figure(figsize=(15, 15))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Draw Wind Speed Data
    carto_map = ax.contourf(lon, lat, owi_speed, levels=100,
                            transform=ccrs.PlateCarree(), cmap='jet')

    # Title and Labels
    ax.set_title(title, fontsize=16, fontweight='bold', loc='left')

    # Set extent with margin
    lon_margin, lat_margin = 0.1, 0.1
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

    # Calculate and draw wind vectors
    dx = np.cos(np.radians(owi_dir))
    dy = np.sin(np.radians(owi_dir))
    ax.quiver(lon, lat, dx, dy, angles='xy', color='black', scale=40, width=0.002, regrid_shape=quiver_density)

    # Draw gridlines and set label formatting
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.6, linestyle=':')
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xlabel_style = {'color': 'black', 'weight': 'bold', 'size': 10}
    gl.ylabel_style = {'color': 'black', 'weight': 'bold', 'size': 10}

    # Define output path with PNG extension
    output_path = FIGURES_DIR / f"{filename}.png"

    # Show or save the plot
    if save:
        plt.savefig(output_path, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Plot saved to {output_path}.")
    else:
        plt.show()

def plot_wind_field_by_arrays(array, quiver_density=20):
    """
    Plot the wind field using wind speed, direction, latitude, and longitude data.
    
    Parameters:
        idx (int): Index of the data file to plot in all_arrays.
        save (bool): If True, saves the plot; if False, displays on screen.
        quiver_density (int): Density of quiver arrows.
    """
    # Get the variables speed, direction, latitude, and longitude by index
    owi_speed = array[0]
    owi_dir = array[1]
    lat = array[2]
    lon = array[3]

    # Extract filename, date, start time, and end time from data file name
    # filename, date, start_time, end_time = extract_datetime_from_filename(data_files[idx])
    title = f"Sentinel-1 Ocean Wind Field (OWI)\n(Start: , End:  on )"

    fig = plt.figure(figsize=(15, 15))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Draw Wind Speed Data
    carto_map = ax.contourf(lon, lat, owi_speed, levels=100,
                            transform=ccrs.PlateCarree(), cmap='jet')

    # Title and Labels
    ax.set_title(title, fontsize=16, fontweight='bold', loc='left')

    # Set extent with margins
    lon_margin, lat_margin = 0.1, 0.1
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

    # Calculate and draw wind vectors
    dx = np.cos(np.radians(owi_dir))
    dy = np.sin(np.radians(owi_dir))
    ax.quiver(lon, lat, dx, dy, angles='xy', color='black', scale=40, width=0.002, regrid_shape=quiver_density)

    # Draw gridlines and set label formatting
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.6, linestyle=':')
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xlabel_style = {'color': 'black', 'weight': 'bold', 'size': 10}
    gl.ylabel_style = {'color': 'black', 'weight': 'bold', 'size': 10}

    # Define output path with PNG extension
    # output_path = FIGURES_DIR / f"{filename}.png"

    # Show or save the plot
    plt.show()
