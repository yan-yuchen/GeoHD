import geopandas as gpd
import numpy as np
import math
import matplotlib.pyplot as plt
import time

# Adaptive Kernel Density Estimation (AKDE) is an advanced non-parametric method used to estimate the probability density function of a random variable.
# It is an enhancement over the traditional Kernel Density Estimation (KDE) that adaptively adjusts the bandwidth of the kernel to better capture the local characteristics of the data.
# This method is particularly useful for datasets that have complex structures or exhibit features across different scales.


def adaptiveKDE(shp_file_path,output_data_path):
    """
    Calculate the Kernel Density Estimation (KDE) for a given set of points.
    
    :param points_gdf: A GeoDataFrame object containing point data.
    :param bandwidth: The bandwidth used for the calculation of KDE.
    :return: A KDE object that can be used to estimate the probability density of the data points.
    
    KDE is a widely used technique in spatial analysis to smooth and visualize the distribution of spatial points.
    The bandwidth parameter is crucial in KDE as it controls the trade-off between bias and variance in the estimation.
    A smaller bandwidth leads to a more varied and detailed estimate (low bias, high variance), while a larger bandwidth results in a smoother estimate (high bias, low variance).
    
    The calculation of KDE involves the following steps:
    1. Extract the coordinates from the GeoDataFrame and form a two-dimensional array.
    2. Apply the Gaussian kernel function to each point, which is centered at the point and has a spread determined by the bandwidth.
    3. Sum the contributions from all points to obtain the estimated density at any location.
    4. Normalize the resulting density estimate so that it integrates to 1 over the entire space.
    
    The `gaussian_kde` function from the `scipy.stats` module is a convenient tool for computing KDE efficiently.
    It provides an object-oriented interface that allows for multiple evaluations of the density function at different locations.
    """
    # Read Shapefile and extract required columns
    gdf = gpd.read_file(shp_file_path)
    data = gdf[['geometry']].reset_index(drop=True)
    data['x'] = data['geometry'].x
    data['y'] = data['geometry'].y
    data['ID'] = range(1, len(data) + 1)

    # Record start time
    time_start = time.time()
    
    # Initialize parameters
    a, delta_a, e_a = 0.5, 0.1, 0.005
    points = {}
    x = []
    y = []
    dis = []
    
    # Iterate through each row in the GeoDataFrame
    for idx, row in data.iterrows():
        points[row['ID']] = [row['x'], row['y']]
        x.append(row['x'])
        y.append(row['y'])

    x = np.array(x)
    y = np.array(y)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    for i in range(len(x)):
        dis.append(math.pow(math.pow(x[i] - x_mean, 2) + math.pow(y[i] - y_mean, 2), 0.5))
    dis_array = np.array(dis)
    q1 = np.percentile(dis_array, 25)
    q3 = np.percentile(dis_array, 75)
    dis_std = np.std(dis_array)
    n = len(x)

    delta = min(dis_std, (q3 - q1) / 1.34)
    h = 1.06 * delta * pow(len(x), -0.2)
    print("Initial bandwidth (h):", h)
    
    delta_h = h / 10
    e_h = delta_h / 20
    iters = 0
    max_iters = 12
    
    while (delta_a > e_a or delta_h > e_h) and max_iters > iters:
        iters += 1
        g = [1, 1, 1]
        h_list = [h - delta_h, h, h + delta_h]
        for i in points:
            fi = [0, 0, 0]
            for j in points:
                dij_square = pow(points[i][0] - points[j][0], 2) + pow(points[i][1] - points[j][1], 2)
                for h_idx in range(len(h_list)):
                    h1 = h_list[h_idx]
                    fi[h_idx] += pow(math.e, -dij_square / (2 * pow(h1, 2)))
            for h_idx in range(len(h_list)):
                h1 = h_list[h_idx]
                fi[h_idx] = 1 / (2 * math.pi * n * pow(h1, 2)) * fi[h_idx]
                if len(points[i]) < 10:
                    points[i].append(fi[h_idx])
                else:
                    points[i][2 + h_idx] = fi[h_idx]
                if fi[h_idx] != 0:
                    g[h_idx] += math.log(fi[h_idx])
        
        a_list = [a - delta_a, a, a + delta_a]
        for h_idx in range(len(h_list)):
            g[h_idx] = pow(math.e, g[h_idx] / n)
            for i in points:
                if h_idx == 0:
                    a1 = a_list[0]
                    hi = pow(points[i][2 + h_idx] / g[h_idx], -a1) * h_list[h_idx]
                    if len(points[i]) < 10:
                        points[i].append(hi)
                    else:
                        points[i][5] = hi
                elif h_idx == 1:
                    for a_idx in range(len(a_list)):
                        a1 = a_list[a_idx]
                        hi = pow(points[i][2 + h_idx] / g[h_idx], -a1) * h_list[h_idx]
                        if len(points[i]) < 10:
                            points[i].append(hi)
                        else:
                            points[i][6 + a_idx] = hi
                else:
                    a1 = a_list[2]
                    hi = pow(points[i][2 + h_idx] / g[h_idx], -a1) * h_list[h_idx]
                    if len(points[i]) < 10:
                        points[i].append(hi)
                    else:
                        points[i][9] = hi
        
        L = [0, 0, 0, 0, 0]
        for i in points:
            fi = np.zeros(5)
            for j in points:
                if i != j:
                    dij_square = pow(points[i][0] - points[j][0], 2) + pow(points[i][1] - points[j][1], 2)
                    for fi_idx in range(len(fi)):
                        fi[fi_idx] += pow(math.e, -dij_square / (2 * pow(points[j][5 + fi_idx], 2))) / pow(points[j][5 + fi_idx], 2)
            for fi_idx in range(len(fi)):
                fi[fi_idx] = 1 / (2 * math.pi * (n - 1)) * fi[fi_idx]
                if fi[fi_idx] != 0:
                    L[fi_idx] += math.log(fi[fi_idx])
        
        max_L = max(L)
        if max_L == L[2]:
            delta_a = delta_a / 2
            delta_h = delta_h / 2
        else:
            idx = L.index(max_L)
            if idx == 0:
                a = a - delta_a
                h = h - delta_h
            elif idx == 1:
                a = a - delta_a
            elif idx == 3:
                a = a + delta_a
            elif idx == 4:
                a = a + delta_a
                h = h + delta_h
    
    time_end = time.time()
    print('totally cost', time_end - time_start)
    print("the parameter of h:" + str(h) + ",the parameter of a:" + str(a))
    
    plot_density_grid(points, h,output_data_path)

def plot_density_grid(points, h, output_data_path,resolution=1000):
    """
    Plot the density estimation grid image.

    Parameters:
        points (dict): Dictionary containing points data.
        h (float): Bandwidth parameter for KDE.
        resolution (int): Grid resolution for plotting. Default is 1000.
    """
    x = np.array([points[p][0] for p in points])
    y = np.array([points[p][1] for p in points])
    x_min, x_max = min(x), max(x)
    y_min, y_max = min(y), max(y)
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                         np.linspace(y_min, y_max, resolution))
    density_values = np.zeros_like(xx)
    for i in range(len(x)):
        density_values += np.exp(-((xx - x[i])**2 + (yy - y[i])**2) / (2 * h**2)) \
                          / (2 * np.pi * h**2)
        
    np.save(output_data_path, density_values)
    print(f"Density data saved at: {output_data_path}")
    
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, density_values, cmap='Purples')
    plt.colorbar(label='Density')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Adaptive KDE Density Estimation')
    plt.legend()
    plt.show()

def adaptive_kernel_density_clustering(points_shapefile, bandwidth):
    """
    Perform clustering on a set of points from a shapefile using the adaptive kernel density algorithm and plot the clusters.

    Args:
    points_shapefile (str): Path to the shapefile containing point data.
    bandwidth (float): Bandwidth parameter for the kernel density estimation.
    """
    def gaussian_kernel(distance, bandwidth):
        return np.exp(-0.5 * (distance / bandwidth)**2) / (np.sqrt(2 * np.pi) * bandwidth)

    def plot_clusters(points_gdf, cluster_labels):
        """
        Plot the clusters.

        Args:
        points_gdf (GeoDataFrame): GeoDataFrame containing point data.
        cluster_labels (list): List of cluster labels for each point.
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
        for cluster_id in np.unique(cluster_labels):
            cluster_points = points_gdf.geometry[cluster_labels == cluster_id]
            ax.scatter([point.x for point in cluster_points], [point.y for point in cluster_points], c=colors[cluster_id % len(colors)], label=f'Cluster {cluster_id}')
        ax.legend()
        plt.title('Clustering using Adaptive Kernel Density Algorithm')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.show()

    def compute_density_map(points_gdf, coordinates, bandwidth):
        """
        Compute density map for the given points.

        Args:
        points_gdf (GeoDataFrame): GeoDataFrame containing point data.
        coordinates (array): Array of point coordinates.
        bandwidth (float): Bandwidth parameter for the kernel density estimation.

        Returns:
        array: Density map for the points.
        """
        # Ensure coordinates is a 2D array
        if len(coordinates.shape) == 1:
            coordinates = coordinates.reshape(-1, 1)

        density_map = np.zeros(len(points_gdf))
        for i in range(len(points_gdf)):
            for j in range(len(points_gdf)):
                if i != j:  # Avoid self-distance
                    # Extract x and y coordinates from tuples
                    x1, y1 = coordinates[i]
                    x2, y2 = coordinates[j]
                    # Calculate Euclidean distance
                    distance = np.linalg.norm((x2 - x1, y2 - y1))
                    kernel_values = gaussian_kernel(distance, bandwidth)
                    density_map[i] += kernel_values
        return density_map


    def find_cluster_centers(points_gdf, density_map, bandwidth):
        """
        Find cluster centers using the density map.

        Args:
        points_gdf (GeoDataFrame): GeoDataFrame containing point data.
        density_map (array): Density map for the points.
        bandwidth (float): Bandwidth parameter for the kernel density estimation.

        Returns:
        list: List of cluster centers.
        """
        cluster_centers = []
        while len(points_gdf) > 0:
            max_density_index = np.argmax(density_map)
            cluster_center = points_gdf.iloc[max_density_index]
            cluster_centers.append(cluster_center)
            points_gdf = points_gdf.drop(max_density_index)
            density_map = np.delete(density_map, max_density_index)
            for i, point in points_gdf.iterrows():
                distance = np.linalg.norm(np.array(cluster_center.geometry.coords[0]) - np.array(point.geometry.coords[0]))
                point_density = gaussian_kernel(distance, bandwidth)
                density_map[i] -= point_density
        return cluster_centers

    def assign_points_to_clusters(points_gdf, cluster_centers, bandwidth):
        """
        Assign points to clusters based on cluster centers.

        Args:
        points_gdf (GeoDataFrame): GeoDataFrame containing point data.
        cluster_centers (list): List of cluster centers.
        bandwidth (float): Bandwidth parameter for the kernel density estimation.

        Returns:
        array: Cluster labels for each point.
        """
        cluster_labels = np.zeros(len(points_gdf), dtype=int)
        for i, point in points_gdf.iterrows():
            if cluster_labels[i] == 0:  # Only assign labels to unassigned points
                min_distance = np.inf
                for j, center in enumerate(cluster_centers):
                    distance = np.linalg.norm(np.array(center.geometry.coords[0]) - np.array(point.geometry.coords[0]))
                    if distance < min_distance:
                        min_distance = distance
                        cluster_labels[i] = j + 1  # Use 1-based indexing for cluster labels
        return cluster_labels

    # Load shapefile point data
    try:
        points_gdf = gpd.read_file(points_shapefile)
    except FileNotFoundError:
        print("Error: File not found.")
        return
    except Exception as e:
        print("Error:", e)
        return

    # Check if the bandwidth value is valid
    if bandwidth <= 0:
        print("Error: Bandwidth must be greater than zero.")
        return

    # Extract coordinates from GeoDataFrame
    coordinates = np.array(points_gdf.geometry.apply(lambda point: point.coords[0]))

    # Compute density map
    density_map = compute_density_map(points_gdf, coordinates, bandwidth)

    # Find cluster centers
    cluster_centers = find_cluster_centers(points_gdf, density_map, bandwidth)

    # Assign points to clusters
    cluster_labels = assign_points_to_clusters(points_gdf, cluster_centers, bandwidth)

    # Plot clusters
    plot_clusters(points_gdf, cluster_labels)



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

    shp_file = './data/crash.shp'  # Replace with your Shapefile path
    output_data_path = './output/AKDE_density_data.npy'
    adaptiveKDE(shp_file,output_data_path)
