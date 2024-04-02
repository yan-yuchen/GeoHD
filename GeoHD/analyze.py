import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import pointpats
import warnings

def plot_g_function(data_path):
    """
    Plot the G Function for spatial point pattern analysis.

    Parameters:
    - data_path: Path to the shapefile (.shp) containing spatial point data.

    Returns:
    - None
    """
    # Read the data from the shapefile
    data = gpd.read_file(data_path)
    points = np.array(data.geometry.apply(lambda geom: (geom.x, geom.y)).tolist())

    # Compute G Function using pointpats library
    g_test = pointpats.g_test(points, support=10, keep_simulations=True)
    middle_95pct = np.percentile(g_test.simulations, q=(2.5, 97.5), axis=0)

    # Plotting
    plt.fill_between(g_test.support, *middle_95pct, color='lightgrey', label='simulated')
    plt.plot(g_test.support, g_test.statistic, color='orangered', label='observed')
    plt.scatter(g_test.support, g_test.statistic, cmap='viridis', c=g_test.pvalue < .01)

    plt.legend()
    plt.xlabel('Distance')
    plt.ylabel('G Function')
    plt.title('G Function Plot')
    plt.show()

def plot_f_function(data_path):
    """
    Plot the F Function for spatial point pattern analysis.

    Parameters:
    - data_path: Path to the shapefile (.shp) containing spatial point data.

    Returns:
    - None
    """
    # Read the data from the shapefile
    data = gpd.read_file(data_path)
    points = np.array(data.geometry.apply(lambda geom: (geom.x, geom.y)).tolist())

    # Compute G Function and F Function using pointpats library
    g_test = pointpats.g_test(points, support=10, keep_simulations=True)
    f_test = pointpats.f_test(points, support=g_test.support, keep_simulations=True, hull='convex')

    # Plotting
    plt.plot(f_test.support, f_test.simulations.T, alpha=.01, color='k')
    plt.plot(f_test.support, f_test.statistic, color='red')

    plt.scatter(f_test.support, f_test.statistic, cmap='viridis', c=f_test.pvalue < .05, zorder=4)

    plt.xlabel('Distance')
    plt.ylabel('F Function')
    plt.title('F Function Plot')
    plt.show()

def plot_j_function(data_path):
    """
    Plot the J Function for spatial point pattern analysis.

    Parameters:
    - data_path: Path to the shapefile (.shp) containing spatial point data.

    Returns:
    - None
    """
    # Read the data from the shapefile
    data = gpd.read_file(data_path)
    points = np.array(data.geometry.apply(lambda geom: (geom.x, geom.y)).tolist())

    # Compute J Function using pointpats library
    jp1 = pointpats.j_test(points, support=20)

    # Plotting
    plt.plot(jp1.support, jp1.statistic, color='orangered')
    plt.axhline(1, linestyle=':', color='k')
    plt.xlabel('Distance')
    plt.ylabel('J Function')
    plt.title('J Function Plot')
    plt.show()

def plot_k_function(data_path):
    """
    Plot the K Function for spatial point pattern analysis.

    Parameters:
    - data_path: Path to the shapefile (.shp) containing spatial point data.

    Returns:
    - None
    """
    # Read the data from the shapefile
    data = gpd.read_file(data_path)
    points = np.array(data.geometry.apply(lambda geom: (geom.x, geom.y)).tolist())

    # Compute K Function using pointpats library
    k_test = pointpats.k_test(points, keep_simulations=True)

    # Plotting
    plt.plot(k_test.support, k_test.simulations.T, color='k', alpha=.01)
    plt.plot(k_test.support, k_test.statistic, color='orangered')

    plt.scatter(k_test.support, k_test.statistic, cmap='viridis', c=k_test.pvalue < .05, zorder=4)

    plt.xlabel('Distance')
    plt.ylabel('K Function')
    plt.title('K Function Plot')
    plt.show()

def plot_l_function(data_path):
    """
    Plot the L Function for spatial point pattern analysis.

    Parameters:
    - data_path: Path to the shapefile (.shp) containing spatial point data.

    Returns:
    - None
    """
    # Read the data from the shapefile
    data = gpd.read_file(data_path)
    points = np.array(data.geometry.apply(lambda geom: (geom.x, geom.y)).tolist())

    # Compute L Function using pointpats library
    l_test = pointpats.l_test(points, keep_simulations=True)

    # Plotting
    plt.plot(l_test.support, l_test.simulations.T, color='k', alpha=.01)
    plt.plot(l_test.support, l_test.statistic, color='orangered')

    plt.scatter(l_test.support, l_test.statistic, cmap='viridis', c=l_test.pvalue < .05, zorder=4)

    plt.xlabel('Distance')
    plt.ylabel('L Function')
    plt.title('L Function Plot')
    plt.show()

# Example usage:
# Replace './data/crash.shp' with the path to your shapefile
# plot_g_function('./data/crash.shp')
# plot_f_function('./data/crash.shp')
# plot_j_function('./data/crash.shp')
# plot_k_function('./data/crash.shp')
# plot_l_function('./data/crash.shp')
