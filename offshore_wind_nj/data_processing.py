from offshore_wind_nj.data_loader import all_arrays, extract_datetime_from_filename
from offshore_wind_nj.data_cleaning import find_nan, fill_nan
import numpy as np
from sklearn.preprocessing import StandardScaler


# Global variables
flattened_data_list = []
scaled_data_list = []

def fill_nan_all_arrays(all_arrays):
    """
    This function takes the list all_arrays and fills NaN values where needed.
    Returns a list with filled values.
    
    Parameters:
    - all_arrays: list of numpy arrays containing data with possible NaN values.

    Returns:
    - List with NaN values filled in.
    """
    nan_indices = [i for i, arr in enumerate(all_arrays) if find_nan(arr)]

    for i in nan_indices:
        all_arrays[i] = fill_nan(all_arrays[i])
    return all_arrays

def flatten_data(all_arrays, data_files=None, mask=False):
    """
    Flattens the given arrays of wind speed, direction (converted to Cartesian coordinates), latitude, and longitude.
    
    Parameters:
    - all_arrays: iterable of tuples (speed, direction, lat, lon)
        Each tuple contains numpy arrays of shape (H, W).
    - data_files: list of filenames, optional
        List of data files, used for extracting dates.
    - mask: bool, optional (default is False)
        If True, apply a NaN mask to remove invalid values before flattening.
    
    Returns:
    - flattened_data_list: list of numpy arrays
        Each entry is a 2D numpy array of shape (Total_valid_pixels, 5) if mask=True, or (Total_pixels, 5) if mask=False.
    - dates_arrays (optional): list of dates
        List of extracted dates if data_files is provided.
    """
    flattened_data_list = []
    
    for speed, direction, lat, lon in all_arrays:
        if mask:
            # Apply NaN mask to filter out invalid values
            valid_mask = ~np.isnan(speed)
            speed_valid = speed[valid_mask]
            direction_valid = direction[valid_mask]
            lat_valid = lat[valid_mask]
            lon_valid = lon[valid_mask]
        else:
            # Use entire arrays without masking
            speed_valid = speed.flatten()
            direction_valid = direction.flatten()
            lat_valid = lat.flatten()
            lon_valid = lon.flatten()
        
        # Convert direction to Cartesian coordinates
        dir_cos = np.cos(np.radians(direction_valid))
        dir_sin = np.sin(np.radians(direction_valid))
        
        # Combine data into a single array for each set of (speed, direction, lat, lon) 11/07/24: added the direction_valid to the list
        flattened_data = np.column_stack((speed_valid, direction_valid, dir_cos, dir_sin, lat_valid, lon_valid))
        flattened_data_list.append(flattened_data)
    
    if data_files is not None:
        dates_arrays = [extract_datetime_from_filename(file) for file in data_files]
        return flattened_data_list, dates_arrays
    else:
        return flattened_data_list

def scaler_flattened_data(flattened_data_list):
    """
    Concatenates all flattened arrays, fits a StandardScaler, and scales each array.
    
    Parameters:
    - flattened_data_list: list of numpy arrays
        Each entry is a 2D numpy array of shape (Num_pixels, 5).
    
    Returns:
    - scaled_data_list: list of numpy arrays
        Each entry is a scaled version of the corresponding array in `flattened_data_list`.
    - scaler: StandardScaler object
        The fitted scaler, useful for applying the same scaling to new data.
    """
    # Step 1: Concatenate all flattened arrays into a single array
    all_flattened_data = np.concatenate(flattened_data_list, axis=0)  # Shape: (Total_pixels, 5)
    
    # Step 2: Fit the scaler on the combined data
    scaler = StandardScaler()
    scaler.fit(all_flattened_data)

    # # Step 3: Scale each individual flattened array
    # scaled_data_list = [scaler.transform(data) for data in flattened_data_list]

    return scaler

def scale_flat_data(flattened_data_list, scaler):
    scaled_data_list = []
    error_indices = []
    
    for idx, data in enumerate(flattened_data_list):
        try:
            # Attempt to scale the data
            scaled_data = scaler.transform(data)  # Wrap data in a list for 2D shape (1, n)
            scaled_data_list.append(scaled_data)  # Add scaled data without extra list
        except ValueError:
            # Track the index if there's an issue with this data point
            error_indices.append(idx)
    
    return scaled_data_list, error_indices


def reshape_scaled_data(scaled_data_list, all_arrays): # This is for the Conv Autoencoder (remember that this fills the nan values)
    """
    Reshapes each scaled data array back to its original spatial dimensions.

    Parameters:
    - scaled_data_list: list of numpy arrays
        Each array is a flattened, scaled version of the original data, with shape (Total_pixels, 5).
    - all_arrays: list of tuples (speed, direction, lat, lon)
        Each tuple contains numpy arrays of shape (H, W).

    Returns:
    - reshaped_scaled_data_list: list of numpy arrays
        Each array is reshaped to (H, W, 5), where (H, W) corresponds to the original shape of each dataset.
    """
    reshaped_scaled_data_list = []
    for idx, scaled_data in enumerate(scaled_data_list):
        # Get the original shape (H, W) of the current array
        original_shape = all_arrays[idx][0].shape  # Assuming speed has the shape (H, W)
        # Reshape scaled data back to (H, W, 5)
        reshaped_data = scaled_data.reshape(original_shape[0], original_shape[1], 5)
        reshaped_scaled_data_list.append(reshaped_data)

    return reshaped_scaled_data_list

# Load your all_arrays data here before flattening and scaling
# all_arrays = [...]  # Replace with your actual loading logic
# flatten_data(all_arrays)  # Flatten the data immediately upon import
# scale_flattened_data()  # Scale the flattened data immediately upon import

# # Optionally, you could print the scaled data for verification
# print("Scaled data list:", len(scaled_data_list))
