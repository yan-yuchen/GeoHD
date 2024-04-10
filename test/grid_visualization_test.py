from GeoHD import *
import os

def test_grid_visualization():
    """
    Test grid visualization functionality.
    """
    # Create output folder if it doesn't exist
    folder_name = 'output'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' has been created.")
    else:
        print(f"Folder '{folder_name}' already exists.")
        
    # Provide test data paths
    area_shapefile_path = './test_data/area.shp'
    crash_shapefile_path = './test_data/crash.shp'
    
    # Test creating cell zones
    create_cell_zones(area_shapefile_path, crash_shapefile_path)
    
    # Test creating hexagonal grid zones
    create_hex_grid_zones(area_shapefile_path, crash_shapefile_path)
    
    # Test creating cell heatmap
    create_cell_heatmap(area_shapefile_path, crash_shapefile_path)
    
    # Test creating hexagonal heatmap
    create_hexagonal_heatmap(area_shapefile_path, crash_shapefile_path)

if __name__ == "__main__":
    test_grid_visualization()
