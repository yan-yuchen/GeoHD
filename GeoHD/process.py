import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter

def calculate_kernel_density(points_gdf, bandwidth):
    """
    Calculate kernel density estimation from GeoDataFrame points.

    Parameters:
        points_gdf (geopandas.GeoDataFrame): GeoDataFrame containing point geometries.
        bandwidth (float): Bandwidth for kernel density estimation.

    Returns:
        gaussian_kde: Kernel density estimation object.
    """
    # Extract coordinates from GeoDataFrame
    coords = np.array(points_gdf.geometry.apply(lambda geom: (geom.x, geom.y)).tolist()).T

    # Calculate kernel density estimation
    kde = gaussian_kde(coords, bw_method=bandwidth)
    return kde

def generate_density_raster(kde, xmin, ymin, xmax, ymax, pixel_size):
    """
    Generate density raster from kernel density estimation.

    Parameters:
        kde (scipy.stats.gaussian_kde): Kernel density estimation object.
        xmin (float): Minimum x-coordinate of the bounding box.
        ymin (float): Minimum y-coordinate of the bounding box.
        xmax (float): Maximum x-coordinate of the bounding box.
        ymax (float): Maximum y-coordinate of the bounding box.
        pixel_size (float): Pixel size for density raster.

    Returns:
        numpy.ndarray: Density raster array.
    """
    x, y = np.meshgrid(
        np.arange(xmin, xmax, pixel_size),
        np.arange(ymin, ymax, pixel_size)
    )
    xy = np.vstack([x.ravel(), y.ravel()])
    z = kde(xy).reshape(x.shape)
    return z

def process_shapefile(input_file_path, bandwidth=0.2, pixel_size=0.001):
    """
    Process a shapefile to generate a kernel density estimation raster.

    Parameters:
        input_file_path (str): Path to the input shapefile.
        bandwidth (float): Bandwidth for kernel density estimation. Default is 0.2.
        pixel_size (float): Pixel size for density raster. Default is 50.

    Returns:
        numpy.ndarray: Smoothed density raster array.
    """
    # Read the shapefile
    points_gdf = gpd.read_file(input_file_path)

    # Get bounding box of the point data
    xmin, ymin, xmax, ymax = points_gdf.total_bounds

    # Calculate kernel density estimation
    kde = calculate_kernel_density(points_gdf, bandwidth)

    # Generate density raster
    density_raster = generate_density_raster(kde, xmin, ymin, xmax, ymax, pixel_size)

    # Smooth the density raster using Gaussian filter
    smoothed_density = gaussian_filter(density_raster, sigma=1.5)

    return smoothed_density

def plot_density_raster(smoothed_density,output_data_path, xmin, ymin, xmax, ymax):
    """
    Plot the density raster.

    Parameters:
        smoothed_density (numpy.ndarray): Smoothed density raster array.
        xmin (float): Minimum x-coordinate of the bounding box.
        ymin (float): Minimum y-coordinate of the bounding box.
        xmax (float): Maximum x-coordinate of the bounding box.
        ymax (float): Maximum y-coordinate of the bounding box.
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(smoothed_density, extent=[xmin, xmax, ymin, ymax], origin='lower', cmap='Purples')
    plt.colorbar(label='Density')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Kernel Density Estimation')
    plt.show()

    # Save density data as NumPy array
    # output_data_path = './output/density_data.npy'
    np.save(output_data_path, smoothed_density)
    print(f"Density data saved at: {output_data_path}")

def kernel_density_estimation_from_shapefile(shapefile_path, bandwidth=0.1, grid_resolution=100):
    # Read shapefile using geopandas
    gdf = gpd.read_file(shapefile_path)
    
    # Extract points from geometry
    points = gdf.geometry.apply(lambda geom: (geom.x, geom.y)).tolist()
    points = np.array(points)

    # Define the range of the grid
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)
    x_range = np.linspace(x_min, x_max, grid_resolution)
    y_range = np.linspace(y_min, y_max, grid_resolution)

    # Create grid
    X, Y = np.meshgrid(x_range, y_range)

    # Initialize density estimation grid
    density = np.zeros_like(X)

    # Perform kernel density estimation
    for point in points:
        # Calculate distances from each point to every grid point
        distances = np.sqrt((X - point[0])**2 + (Y - point[1])**2)
        # Apply Gaussian kernel
        kernel_values = 1 / (2 * np.pi * bandwidth**2) * np.exp(-0.5 * (distances / bandwidth)**2)
        # Update density grid
        density += kernel_values

    # Plot density map
    plt.figure(figsize=(8, 6))
    plt.imshow(np.rot90(density), cmap=plt.cm.jet, extent=[x_min, x_max, y_min, y_max])
    plt.colorbar(label='Density')
    plt.scatter(points[:, 0], points[:, 1], color='k', s=5, alpha=0.5)  # Plot original points
    plt.title('Kernel Density Estimation')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()


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
        
    input_file_path = './data/crash.shp'  # Replace with the actual path to your shapefile
    output_data_path = './output/density_data.npy'
    density_raster = process_shapefile(input_file_path)
    plot_density_raster(density_raster,output_data_path, *gpd.read_file(input_file_path).total_bounds)
