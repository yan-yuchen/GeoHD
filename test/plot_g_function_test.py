from GeoHD import *
import os

def test_plot_g_function():
    """
    Test plotting G function.
    """
    # Create output folder if it doesn't exist
    folder_name = 'output'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' has been created.")
    else:
        print(f"Folder '{folder_name}' already exists.")
        
    # Provide test data paths
    shapefile_path = './test_data/crash.shp'
    plot_g_function(shapefile_path)

if __name__ == "__main__":
    test_plot_g_function()
