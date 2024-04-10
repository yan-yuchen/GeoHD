from GeoHD import *
import os

def test_adaptive_kde():
    """
    Test adaptive kernel density estimation.
    """
    # Create output folder if it doesn't exist
    folder_name = 'output'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' has been created.")
    else:
        print(f"Folder '{folder_name}' already exists.")
        
    # Provide test data paths
    shp_file = './test_data/crash.shp'
    output_data_path = './output/AKDE_density_data.npy'
    adaptiveKDE(shp_file, output_data_path)

if __name__ == "__main__":
    test_adaptive_kde()
