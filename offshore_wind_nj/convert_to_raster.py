import arcpy
from arcpy import env
from arcpy.sa import *

env.workspace = "your_workspace_directory"
spatial_ref = arcpy.SpatialReference(4326)  # Set spatial reference (e.g., WGS 84)

# Convert NumPy array to raster
raster = arcpy.NumPyArrayToRaster(owiSpeed, lower_left_corner, cell_size, value_to_nodata)
raster.spatialReference = spatial_ref

# Save raster to file
raster.save("output_raster.tif")
