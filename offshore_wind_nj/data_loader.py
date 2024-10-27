import numpy as np
from pathlib import Path
from loguru import logger
import offshore_wind_nj.config as config
import re
from datetime import datetime
from offshore_wind_nj.data_cleaning import fill_zeros
from typing import List, Tuple

def load_data(input_files: List[Path]) -> Tuple[List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], List[Path]]:
    """
    Load and preprocess data from a list of input .npz files.
    
    Parameters:
        input_files (list of Path): Paths to the input .npz files.
        
    Returns:
        Tuple: all_arrays (List of tuples with loaded data), cleaned_files (List of files with zeros).
    """
    all_arrays = []
    cleaned_files = []
    
    for input_file in input_files:
        with np.load(input_file) as data:
            owi_speed = data['owiSpeed']
            owi_dir = data['owiDir']
            lat = data['lat']
            lon = data['lon']

            # Check for zero-speed points
            if np.any(owi_speed == 0):
                cleaned_files.append(input_file)
                clean_data = (owi_speed, owi_dir, lat, lon)
                filled_data = fill_zeros(clean_data)  # Fill zeros
                all_arrays.append(filled_data)
            else:
                all_arrays.append((owi_speed, owi_dir, lat, lon))
    
    return all_arrays, cleaned_files

def load_single_data(input_file: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and preprocess data from a single input .npz file.
    
    Parameters:
        input_file (Path): Path to the input .npz file.
        
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
    Extracts filename and formats start and end datetime from a given file path.
    
    Args:
        file_path (Path): Full path to the file.
    
    Returns:
        Tuple[str, str, str]: Filename without extension, start date, start and end times.
    """
    filename = file_path.stem
    parts = re.split('_', filename)
    
    try:
        date = datetime.strptime(parts[5], '%Y%m%dT%H%M%S').strftime('%Y-%m-%d')
        start_time = datetime.strptime(parts[5], '%Y%m%dT%H%M%S').strftime('%H:%M:%S')
        end_time = datetime.strptime(parts[6], '%Y%m%dT%H%M%S').strftime('%H:%M:%S')
    except (IndexError, ValueError) as e:
        logger.error(f"Filename '{filename}' does not contain valid timestamps or has an unexpected format.")
        raise ValueError("Filename format issue") from e
    
    return filename, date, start_time, end_time

if __name__ == "__main__":
    print("Module run independently.")
else:
    data_files = list(config.PROCESSED_DATA_DIR.glob('*.npz'))
    all_arrays, cleaned_files = load_data(data_files)
    print(f'{len(cleaned_files)} files required cleaning.')
