# data_loader.py

import numpy as np
from pathlib import Path
from loguru import logger
import offshore_wind_nj.config as config
import re
from datetime import datetime
from typing import Tuple
from offshore_wind_nj.data_cleaning import fill_zeros, find_zeros, find_nan, fill_nan

# Create a list of all .npz files in the processed data directory
data_files = list(config.PROCESSED_DATA_DIR.glob('*.npz'))

# Global variable to store all loaded arrays
all_arrays = []
removed_files = []

def load_data(input_files):
    """
    Load and preprocess data from a list of input .npz files.
    
    Parameters:
        input_files (list): List of paths to the input .npz files.
    """
    global all_arrays, removed_files  # Declare that we are using the global variable
    all_arrays = []  # Clear the list before loading new data
    removed_files = []
    for input_file in input_files:
        with np.load(input_file) as data:
            owi_speed = data['owiSpeed']
            owi_dir = data['owiDir']
            lat = data['lat']
            lon = data['lon']

            # if np.any(owi_speed == 0):
            #     # Skip if entire array is zero; otherwise, clean and add
            #     if np.all(owi_speed == 0):
            #         data_files.remove(input_file)
            #         removed_files.append(input_file)
            #         # logger.info(f"{len(data_files)} Total data_files after removing index {i}, {removed_files}")
            #         # continue  # Skip if all values are zero
            #     else:
            #         clean = (owi_speed, owi_dir, lat, lon)
            #         all_arrays.append((owi_speed, owi_dir, lat, lon))
            #         # all_arrays.append(fill_zeros(clean))
            #         # logger.info(f"{len(data_files)} Total data_files after cleaning index {i}, {len(all_arrays)}")
            # else:
            all_arrays.append((owi_speed, owi_dir, lat, lon))
    process_arrays()

def process_arrays():
    '''
    Process the loaded arrays to fill zeros and remove any that are invalid
    '''
    global all_arrays, removed_files
    list_removed = []
    
    speed_indices = [i for i, arr in enumerate(all_arrays) if find_zeros(arr)]

    # Create a list to hold indices of elements to remove
    indices_to_remove = []

    for i in speed_indices:
        if np.all(all_arrays[i][0] == 0):
            indices_to_remove.append(i)
            removed_files = data_files[i]  # Store the removed file info
            list_removed.append(removed_files)

    # Fill zeros for remaining arrays
    for i in speed_indices:
        if i not in indices_to_remove:
            all_arrays[i] = fill_zeros(all_arrays[i])
    
    # Now remove elements in reverse order to avoid index shifting
    for i in sorted(indices_to_remove, reverse=True):
        all_arrays.pop(i)
        data_files.pop(i)
    
            

def load_single_data(input_file):
    """
    Load and preprocess data from a single input .npz file.
    
    Parameters:
        input_file (str): Path to the input .npz file.
        
    Returns:
        tuple: A tuple containing (owi_speed, owi_dir, lat, lon).
    """
    logger.info(f"Loading data from {input_file}...")
    with np.load(input_file) as data:
        owi_speed = data['owiSpeed']
        owi_dir = data['owiDir']
        lat = data['lat']
        lon = data['lon']
    logger.success(f"Data loaded successfully from {input_file}.")
    return owi_speed, owi_dir, lat, lon

def extract_datetime_from_filename(file_path: Path) -> Tuple[str, str, str]:
    """
    Extracts the filename and formats the start and end datetime from a given file path.
    
    Args:
        file_path (Path): The full path to the file.

    Returns:
        Tuple[str, str, str]: The filename without the extension, the start datetime as 'YYYY-MM-DD HH:MM:SS',
                              and the end datetime as 'YYYY-MM-DD HH:MM:SS'.
    """
    # Extract the filename without the extension
    filename = Path(file_path).stem
    
    # Split filename into parts based on underscore
    parts = re.split('_', filename)
    
    try:
        # Convert the start and end timestamp parts to datetime objects and format them
        date = datetime.strptime(parts[5], '%Y%m%dT%H%M%S').strftime('%Y-%m-%d')
        start_time = datetime.strptime(parts[5], '%Y%m%dT%H%M%S').strftime('%H:%M:%S')
        end_time = datetime.strptime(parts[6], '%Y%m%dT%H%M%S').strftime('%H:%M:%S')
    except (IndexError, ValueError) as e:
        raise ValueError(f"Filename '{filename}' does not contain valid timestamps or has an unexpected format.") from e
    
    return filename, date, start_time, end_time


if __name__ == "__main__":
    print(f"{__name__} run as a module")
else:
    # data_files = list(config.PROCESSED_DATA_DIR.glob('*.npz'))
    
    # Call the load_data function to populate all_arrays
    load_data(data_files)
    print(f'There are {len(data_files)} files')