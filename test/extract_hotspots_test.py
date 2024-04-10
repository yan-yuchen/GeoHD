from GeoHD import *
import numpy as np
import os

def test_extract_hotspots():
    """
    Test extracting hotspots and visualizing them.
    """
    # Create output folder if it doesn't exist
    folder_name = 'output'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' has been created.")
    else:
        print(f"Folder '{folder_name}' already exists.")
        
    # Provide test data paths
    density_data_path = './output/AKDE_density_data.npy'
    hotspots = extract_hotspots(density_data_path)
    visualize_hotspots(np.load(density_data_path), hotspots)

if __name__ == "__main__":
    test_extract_hotspots()
