from GeoHD import *
import geopandas as gpd
import os

def test_plot_density_raster():
    """
    Test plotting density raster with fixed bandwidth.
    """
    # Create output folder if it doesn't exist
    folder_name = 'output'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' has been created.")
    else:
        print(f"Folder '{folder_name}' already exists.")
        
    # Provide test data paths
    input_file_path = './test_data/crash.shp'
    output_data_path = './output/density_data.npy'
    density_raster = process_shapefile(input_file_path)
    plot_density_raster(density_raster, output_data_path, *gpd.read_file(input_file_path).total_bounds)

if __name__ == "__main__":
    test_plot_density_raster()
