import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
import warnings

def is_valid_shapefile(file_path):
    """
    Check if the file exists and is a valid shapefile.

    Parameters:
    - file_path: Path to the shapefile (.shp).

    Returns:
    - bool: True if the file is valid, False otherwise.
    """
    try:
        gpd.read_file(file_path)
        return True
    except (FileNotFoundError, gpd.errors.NeedMandatoryFile):
        return False

def visualize_shapefile(file_path, output_image_path='./output/point_data.png'):
    """
    Visualize a shapefile with a map background using GeoPandas and Contextily.

    Parameters:
    - file_path: Path to the shapefile (.shp).
    - output_image_path: Path to save the visualization image (default: './output/point_data.png').

    Raises:
    - FileNotFoundError: If the specified shapefile does not exist.
    - ValueError: If the output_image_path is not provided.

    Returns:
    - None
    """
    # Check if the specified file is a valid shapefile
    if not is_valid_shapefile(file_path):
        raise FileNotFoundError("The provided file is not a valid shapefile or does not exist.")

    # Read the shapefile
    points_gdf = gpd.read_file(file_path)

    # Display the first few rows of the point data
    print("First few rows of point data:")
    print(points_gdf.head())

    # Create a new figure and axis for plotting
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot the point data
    points_gdf.plot(ax=ax, markersize=20, color='red', marker='o', alpha=0.7)

    # Add a basemap using contextily
    ctx.add_basemap(ax, crs=points_gdf.crs.to_string(), source=ctx.providers.CartoDB.Positron)

    # Set the axis limits based on the point data extent
    ax.set_xlim(points_gdf.total_bounds[0], points_gdf.total_bounds[2])
    ax.set_ylim(points_gdf.total_bounds[1], points_gdf.total_bounds[3])

    # Add a title to the plot
    ax.set_title('Visualization of Point Data with Map Background')

    # Display the plot
    plt.show()

    # Save the visualization image
    if not output_image_path:
        raise ValueError("Please provide a path to save the visualization image.")
    fig.savefig(output_image_path)
    print(f"Visualization image saved at: {output_image_path}")

# Example usage:
if __name__ == "__main__":
    import os
    # Define the name of the folder to be created
    folder_name = 'output'
    # Check if the folder exists, if not, create it
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' has been created.")
    else:
        print(f"Folder '{folder_name}' already exists.")
        
    visualize_shapefile('./data/crash.shp')
    visualize_shapefile('./data/crash.shp', output_image_path='./output/custom_image.png')
