from GeoHD import *
import os

def test_visualize_shapefile():
    """
    Test visualizing shapefile on a real map.
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
    output_image_path = './output/custom_image.png'
    visualize_shapefile(shapefile_path, output_image_path)

if __name__ == "__main__":
    test_visualize_shapefile()
