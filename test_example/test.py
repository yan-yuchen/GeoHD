from GeoHD import *
import os


# Define the name of the folder to be created
folder_name = 'output'
# Check if the folder exists, if not, create it
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
    print(f"Folder '{folder_name}' has been created.")
else:
    print(f"Folder '{folder_name}' already exists.")


# Visualization of hotspots on real maps:
visualize_shapefile('../data/crash.shp', output_image_path='../output/custom_image.png')


# Analytic Plane Point Patterns: Ripley G, Ripley F, Ripley J, Ripley K, Ripley L, etc. through the plotting function.
plot_g_function('../data/crash.shp')


# Realization of kernel density analysis with fixed bandwidth:
input_file_path = '../data/crash.shp'  
output_data_path = '../output/density_data.npy'
density_raster = process_shapefile(input_file_path)
plot_density_raster(density_raster,output_data_path, *gpd.read_file(input_file_path).total_bounds)


# Kernel density analysis for realizing adaptive bandwidth:
shp_file = '../data/crash.shp'  
output_data_path = '../output/AKDE_density_data.npy'
adaptiveKDE(shp_file,output_data_path)

# Hotspot Identification:
density_data_path = '../output/AKDE_density_data.npy'
hotspots = extract_hotspots(density_data_path)
visualize_hotspots(np.load(density_data_path), hotspots)

# The study area was divided into a quadrilateral (hexagonal) grid and fast visualization was achieved based on the density of point data within the divided area.
create_cell_zones('../data/area.shp', '../data/crash.shp')
create_hex_grid_zones('../data/area.shp', '../data/crash.shp')
create_cell_heatmap('../data/area.shp', '../data/crash.shp')
create_hexagonal_heatmap('../data/area.shp', '../data/crash.shp')
