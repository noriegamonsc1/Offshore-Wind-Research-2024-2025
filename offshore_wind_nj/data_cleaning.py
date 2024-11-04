from scipy.spatial import cKDTree
import numpy as np
# from offshore_wind_nj.data_loader import all_arrays

def find_zeros(arr):
    return np.any(arr[0] == 0)

def find_nan(arr):
    return np.any(np.isnan(arr[0]))


def idw_interpolation(coords, values, target_coords, k=4, p=2):
    tree = cKDTree(coords)
    distances, indices = tree.query(target_coords, k=k)
    
    # Calculate weights
    weights = 1 / (distances ** p)
    weights /= weights.sum(axis=1, keepdims=True)
    
    # Interpolated values
    return np.sum(values[indices] * weights, axis=1)

def circular_idw_interpolation(coords, angles, target_coords, k=4, p=2):
    # Convert angles to unit vectors
    sin_vals = np.sin(np.radians(angles))
    cos_vals = np.cos(np.radians(angles))
    
    sin_interp = idw_interpolation(coords, sin_vals, target_coords, k, p)
    cos_interp = idw_interpolation(coords, cos_vals, target_coords, k, p)
    
    # Calculate interpolated angle
    interpolated_angles = np.degrees(np.arctan2(sin_interp, cos_interp)) % 360
    return interpolated_angles

def fill_zeros(array):
    """
    Fill missing or zero-speed values using IDW interpolation.
    
    Parameters:
        array : tuple of np.ndarray
            A tuple containing speed, direction, latitude, and longitude arrays.
    
    Returns:
        tuple of np.ndarray: Filled speed, filled direction, latitude, and longitude arrays.
    """
    speed, direction, lat, lon = array
    
    # Identify pixels with speed 0 and direction 180
    mask = speed == 0
    
    # Get valid points (where speed != 0 and direction != 180)
    valid_data_mask = ~mask
    valid_speed = speed[valid_data_mask]
    valid_direction = direction[valid_data_mask]
    x, y = np.meshgrid(np.arange(speed.shape[1]), np.arange(speed.shape[0]))
    valid_coords = np.column_stack((x[valid_data_mask], y[valid_data_mask]))
    target_coords = np.column_stack((x[mask], y[mask]))
    
    # Interpolate values using IDW
    filled_speed = speed.copy()
    filled_direction = direction.copy()
    filled_speed[mask] = idw_interpolation(valid_coords, valid_speed, target_coords)
    filled_direction[mask] = circular_idw_interpolation(valid_coords, valid_direction, target_coords)
    
    return filled_speed, filled_direction, lat, lon

def fill_nan(array):
    """
    Fill missing or zero-speed values using IDW interpolation.
    
    Parameters:
        array : tuple of np.ndarray
            A tuple containing speed, direction, latitude, and longitude arrays.
    
    Returns:
        tuple of np.ndarray: Filled speed, filled direction, latitude, and longitude arrays.
    """
    speed, direction, lat, lon = array
    
    # Identify pixels with speed nan
    mask = np.isnan(speed)
    
    # Get valid points (where speed != 0 and direction != 180)
    valid_data_mask = ~mask
    valid_speed = speed[valid_data_mask]
    valid_direction = direction[valid_data_mask]
    x, y = np.meshgrid(np.arange(speed.shape[1]), np.arange(speed.shape[0]))
    valid_coords = np.column_stack((x[valid_data_mask], y[valid_data_mask]))
    target_coords = np.column_stack((x[mask], y[mask]))
    
    # Interpolate values using IDW
    filled_speed = speed.copy()
    filled_direction = direction.copy()
    filled_speed[mask] = idw_interpolation(valid_coords, valid_speed, target_coords)
    filled_direction[mask] = circular_idw_interpolation(valid_coords, valid_direction, target_coords)
    
    return filled_speed, filled_direction, lat, lon

# def idw_interpolation(coords, values, target_coords, k=4, p=2):
#     tree = cKDTree(coords)
#     distances, indices = tree.query(target_coords, k=k)
    
#     # Calculate weights
#     weights = 1 / (distances ** p)
#     weights /= weights.sum(axis=1, keepdims=True)
    
#     # Interpolated values
#     return np.sum(values[indices] * weights, axis=1)

# def fill_zeros(array):
#     """
#     Fill missing or zero-speed values using IDW interpolation.
    
#     Parameters:
#         array : tuple of np.ndarray
#             A tuple containing speed, direction, latitude, and longitude arrays.
    
#     Returns:
#         tuple of np.ndarray: Filled speed, filled direction, latitude, and longitude arrays.
#     """
#     speed, direction, lat, lon = array
    
#     # Identify pixels with speed 0 and direction 180
#     mask = speed == 0
    
#     # Get valid points (where speed != 0 and direction != 180)
#     valid_data_mask = ~mask
#     valid_speed = speed[valid_data_mask]
#     valid_direction = direction[valid_data_mask]
#     x, y = np.meshgrid(np.arange(speed.shape[1]), np.arange(speed.shape[0]))
#     valid_coords = np.column_stack((x[valid_data_mask], y[valid_data_mask]))
#     target_coords = np.column_stack((x[mask], y[mask]))
    
#     # Interpolate values using IDW
#     filled_speed = speed.copy()
#     filled_direction = direction.copy()
#     filled_speed[mask] = idw_interpolation(valid_coords, valid_speed, target_coords)
#     filled_direction[mask] = idw_interpolation(valid_coords, valid_direction, target_coords)
    
#     return filled_speed, filled_direction, lat, lon

# def clean_list(all_arrays):
#     speed_indices = [i for i, arr in enumerate(all_arrays) if find_zeros]
# def fill_zeros(array):
#     """
#     Fill missing or zero-speed values using IDW interpolation.
    
#     Parameters:
#         array : tuple of np.ndarray
#             A tuple containing speed, direction, latitude, and longitude arrays.
    
#     Returns:
#         tuple of np.ndarray: Filled speed, filled direction, latitude, and longitude arrays.
#     """
#     speed, direction, lat, lon = array
    
#     # Identify pixels with speed 0, regardless of direction
#     mask = speed == 0
    
#     # Get valid points (where speed != 0)
#     valid_data_mask = ~mask
#     valid_speed = speed[valid_data_mask]
    
#     if valid_speed.size == 0:
#         # If no valid speed values are found, return the original arrays
#         return speed, direction, lat, lon
    
#     # Prepare coordinates for interpolation
#     x, y = np.meshgrid(np.arange(speed.shape[1]), np.arange(speed.shape[0]))
#     valid_coords = np.column_stack((x[valid_data_mask], y[valid_data_mask]))
#     target_coords = np.column_stack((x[mask], y[mask]))

#     # Interpolate values using IDW
#     filled_speed = speed.copy()
#     filled_direction = direction.copy()

#     # Only fill the speed values at mask locations
#     filled_speed[mask] = idw_interpolation(valid_coords, valid_speed, target_coords)

#     # Optionally, fill direction values if required (based on your logic)
#     # If you want to keep the original direction for speed=0, uncomment the next line:
#     # filled_direction[mask] = direction[mask]  # Retain original direction for speed=0

#     return filled_speed, filled_direction, lat, lon
