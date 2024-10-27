from scipy.spatial import cKDTree
import numpy as np

def idw_interpolation(coords, values, target_coords, k=4, p=2):
    tree = cKDTree(coords)
    distances, indices = tree.query(target_coords, k=k)
    
    # Calculate weights
    weights = 1 / (distances ** p)
    weights /= weights.sum(axis=1, keepdims=True)
    
    # Interpolated values
    return np.sum(values[indices] * weights, axis=1)

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
    mask = (speed == 0) & (direction == 180)
    
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
    filled_direction[mask] = idw_interpolation(valid_coords, valid_direction, target_coords)
    
    return filled_speed, filled_direction, lat, lon
