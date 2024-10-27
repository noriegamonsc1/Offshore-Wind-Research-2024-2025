'''1. Identify Missing or Zero-Speed Values
First, identify the pixels with speed values of 0 and direction of 180 that need to be filled. Create a mask to locate these points.
'''
import numpy as np

# Assuming speed and direction are 2D numpy arrays with shape (167, 255)
mask = (speed == 0) & (direction == 180)  # Boolean mask where these conditions are true

'''2. Extract Coordinates and Valid Data Points
Use the lat and lon arrays to create coordinates, then filter out the valid data points (where speed and direction are not zero or 180) and the target points that need filling.

'''
# Create mesh grid for coordinates
x, y = np.meshgrid(np.arange(speed.shape[1]), np.arange(speed.shape[0]))

# Get valid points (where speed != 0 and direction != 180)
valid_mask = ~mask
valid_speed = speed[valid_mask]
valid_direction = direction[valid_mask]
valid_coords = np.column_stack((x[valid_mask], y[valid_mask]))

# Get coordinates for points that need filling
target_coords = np.column_stack((x[mask], y[mask]))

'''3. Apply Nearest-Neighbor Interpolation
With scipy, you can perform nearest-neighbor interpolation using scipy.interpolate.griddata. This method fills the missing values by referencing the nearest valid data point geographically.
'''
from scipy.interpolate import griddata

# Perform nearest-neighbor interpolation for speed and direction separately
interpolated_speed = griddata(valid_coords, valid_speed, target_coords, method='nearest')
interpolated_direction = griddata(valid_coords, valid_direction, target_coords, method='nearest')

# Fill the missing values in the original arrays
filled_speed = speed.copy()
filled_direction = direction.copy()

filled_speed[mask] = interpolated_speed
filled_direction[mask] = interpolated_direction

'''
4. Optional: Apply Inverse Distance Weighting (IDW) for Smoother Interpolation
If a smoother result is preferred, IDW can be implemented to weigh nearby points based on distance.
'''
from scipy.spatial import cKDTree

def idw_interpolation(coords, values, target_coords, k=4, p=2):
    tree = cKDTree(coords)
    distances, indices = tree.query(target_coords, k=k)
    
    # Calculate weights
    weights = 1 / (distances ** p)
    weights /= weights.sum(axis=1, keepdims=True)
    
    # Interpolated values
    return np.sum(values[indices] * weights, axis=1)

# Apply IDW interpolation
interpolated_speed_idw = idw_interpolation(valid_coords, valid_speed, target_coords)
interpolated_direction_idw = idw_interpolation(valid_coords, valid_direction, target_coords)

# Update the arrays with IDW-interpolated values
filled_speed[mask] = interpolated_speed_idw
filled_direction[mask] = interpolated_direction_idw

'''
5. Verify Results
'''
import matplotlib.pyplot as plt

# Plot the original and filled speed arrays for comparison
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Speed")
plt.imshow(speed, cmap='jet')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.title("Filled Speed")
plt.imshow(filled_speed, cmap='jet')
plt.colorbar()
plt.show()

