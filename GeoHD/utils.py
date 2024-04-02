import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter

def extract_hotspots(density_data_path, window_size=10, threshold=0):
    """
    Extract hotspots from density data using a window analysis approach.

    Parameters:
        density_data_path (str): Path to the .npy file containing density data.
        window_size (int): Size of the window for maximum filter (default is 10).
        threshold (int): Threshold value for extreme regions (default is 0).

    Returns:
        hotspots (ndarray): Array containing coordinates of hotspots.
    """
    # Load density data from the saved .npy file
    density_data = np.load(density_data_path)

    # Step 1: Window analysis to get maximum values within each window
    max_density_surface = maximum_filter(density_data, size=window_size)

    # Step 2: Algebraic subtraction to get non-negative surface
    non_negative_surface = max_density_surface - density_data

    # Step 3: Reclassification algorithm to classify into extreme and non-extreme regions
    extreme_region = non_negative_surface == threshold

    # Step 4: Extract hotspots from extreme regions
    hotspots = np.argwhere(extreme_region)

    return hotspots

def visualize_hotspots(density_data, hotspots):
    """
    Visualize hotspots on the density data.

    Parameters:
        density_data (ndarray): Density data array.
        hotspots (ndarray): Array containing coordinates of hotspots.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(density_data, cmap='Purples', origin='lower')
    plt.colorbar(label='Density')
    plt.scatter(hotspots[:, 1], hotspots[:, 0], color='red', s=5, label='Hotspots')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Hotspot Extraction')
    plt.legend()
    plt.show()

# Example usage:
if __name__ == "__main__":
    density_data_path = './output/AKDE_density_grid.npy'
    hotspots = extract_hotspots(density_data_path)
    visualize_hotspots(np.load(density_data_path), hotspots)
