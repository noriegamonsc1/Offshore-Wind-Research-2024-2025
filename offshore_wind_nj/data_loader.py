# data_loader.py

import numpy as np
from pathlib import Path
from loguru import logger
import offshore_wind_nj.config as config

# Create a list of all .npz files in the processed data directory
data_files = list(config.PROCESSED_DATA_DIR.glob('*.npz'))

# Global variable to store all loaded arrays
all_arrays = []

def load_data(input_files):
    """
    Load and preprocess data from a list of input .npz files.
    
    Parameters:
        input_files (list): List of paths to the input .npz files.
    """
    global all_arrays  # Declare that we are using the global variable
    all_arrays = []  # Clear the list before loading new data
    
    for input_file in input_files:
        # logger.info(f"Loading data from {input_file}...")
        with np.load(input_file) as data:
            owi_speed = data['owiSpeed']
            owi_dir = data['owiDir']
            lat = data['lat']
            lon = data['lon']
            all_arrays.append((owi_speed, owi_dir, lat, lon))
        # logger.success(f"Data loaded successfully from {input_file}.")

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

if __name__ == "__main__":
    print(f"{__name__} run as a module")
else:
    data_files = list(config.PROCESSED_DATA_DIR.glob('*.npz'))
    
    # Call the load_data function to populate all_arrays
    load_data(data_files)
    print(f'There are {len(data_files)} files')